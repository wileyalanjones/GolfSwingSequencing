FROM node:20-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt 

RUN mkdir -p weights && curl -fL https://github.com/wileyalanjones/GolfSwingSequencing/releases/download/model-v1/model.weights.h5 -o weights/model.weights.h5

COPY golfswing/ golfswing/
COPY webapp/ webapp/
COPY --from=frontend /frontend/dist frontend/dist

ENV MODEL_PATH=/app/weights/model.weights.h5

CMD ["sh", "-c", "uvicorn webapp.main:app --host 0.0.0.0 --port ${PORT:-8000}"]