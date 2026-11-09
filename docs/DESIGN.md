# Design decisions

This document explains *why* the pipeline is built the way it is. Each section answers a question
an industrial vision engineer would ask in a design review.

## 1. Why an auto-encoder (unsupervised) rather than a supervised detector like YOLO?

On a production line, defects are rare and diverse: collecting and annotating hundreds of
examples of every defect type is slow and never exhaustive. An auto-encoder trained **only on
good parts** learns what "normal" looks like and flags anything it cannot reconstruct, including
defect types never seen before. Trade-off: it says *where* something is abnormal, not *what* the
defect is. In production, both approaches are often combined (anomaly detection to sort good/bad,
then a supervised classifier on the rejected parts).

## 2. Down-scale the frame, or tile it at full resolution?

Both are implemented (`model.input_size` in `config.yaml`) and were measured on the same test set
(same trained-on-good-only protocol, threshold at 1% false rejects):

| Mode | Image AUROC | Crack detection | Stain AUROC |
|---|---|---|---|
| 256 → 128 down-scale (default) | **0.946** | 34% | **0.906** |
| 2x2 full-resolution tiles of 128 | 0.871 | **48%** | 0.646 |

Tiling keeps thin cracks at native contrast, so it detects more of them. But down-sampling also
*averages out the texture noise*, which helps every other defect, and a 128 px tile is too small
to see a large diffuse stain as abnormal. On this surface the down-scaled mode wins overall and
is 4x cheaper. On a part where hairline cracks are the critical defect, tiling would be the
better choice: this is a per-application decision, which is why it is a configuration option.

## 3. Why blur the signed residual *before* squaring?

The auto-encoder cannot reproduce random surface texture, so the residual (input - reconstruction)
is dominated by texture noise. That noise is zero-mean: averaging it locally (Gaussian blur)
cancels it. A defect has a coherent sign (a scratch is consistently darker or brighter), so it
survives the blur. Squaring first would turn the texture into positive energy that no blur
removes.

## 4. Why several scales, normalized on good parts?

A scratch is best seen with a small blur (sigma 1), a diffuse stain with a large one (sigma 4).
Each scale is divided by its typical level on defect-free validation images, so all scales are
comparable and one single threshold works for every defect size.

## 5. How is the threshold chosen without any defect example?

The validation set contains only good parts. It is split in two: the first half fixes the scale
normalization, the second half gives the score distribution of good parts, and the threshold is
its 99.5th percentile. By construction, about 0.5% of good parts would be rejected: the
false-reject rate is a **design parameter**, chosen by the quality team, not a side effect.

## 6. Why read the stream in a separate thread with a 1-frame buffer?

If inference is slower than the camera, an RTSP client that reads frames in order accumulates
latency without bound (the verdict would arrive seconds after the part has left). The reader
thread always keeps only the **latest** frame; the inference loop drops stale frames instead of
falling behind.

## 7. Why debounce alerts (N consecutive frames + cooldown)?

A part stays under the camera for several frames. Requiring 3 consecutive anomalous frames
removes single-frame glitches (reflection, compression artifact), and the cooldown prevents one
defective part from generating dozens of alerts.

## 8. Why ONNX Runtime in the Docker image instead of PyTorch?

The inference image does not need autograd: ONNX Runtime is a much smaller dependency, starts
faster, and the same `.onnx` file can be compiled into a TensorRT engine on an NVIDIA Jetson
(`trtexec --onnx=models/autoencoder.onnx --fp16`). `scripts/export_onnx.py` checks that the ONNX
output matches PyTorch to within 1e-4.

## Known limitations

* The dataset is synthetic (brushed-metal texture with generated defects). The layout follows
  MVTec AD, so the real dataset can be dropped in `data/` without code changes.
* Thin cracks remain the hardest class (see per-type results in the README).
* The model gives a per-pixel anomaly map, not a defect class.
