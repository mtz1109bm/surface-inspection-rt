# Inference image: ONNX Runtime only (no PyTorch) -> small image, fast start.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inspector/ inspector/
COPY config.yaml .
COPY models/autoencoder.onnx models/threshold.json models/

ENV INFERENCE_BACKEND=onnx
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["python", "-m", "inspector.web"]
