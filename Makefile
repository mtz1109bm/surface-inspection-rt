PY ?= python

.PHONY: all data train eval export bench video test run up

all: data train export eval bench video

data:
	$(PY) scripts/make_synthetic_dataset.py

train:
	$(PY) scripts/train.py

export:
	$(PY) scripts/export_onnx.py

eval:
	$(PY) scripts/evaluate.py --backend onnx

bench:
	$(PY) scripts/benchmark.py

video:
	$(PY) scripts/make_demo_video.py

test:
	$(PY) -m pytest -q

run:  # local run on the demo video, no Docker needed
	STREAM_SOURCE=demo/line1.mp4 $(PY) -m inspector.web

up:
	docker compose up --build
