
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/service/src

WORKDIR /service

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-core.txt requirements-api.txt ./

RUN pip install --no-cache-dir -r requirements-api.txt

COPY src ./src

COPY artifacts/model.joblib ./artifacts/model.joblib
COPY artifacts/model_metadata.json ./artifacts/model_metadata.json

RUN useradd --create-home --uid 10001 apiuser \
    && chown -R apiuser:apiuser /service

USER apiuser

EXPOSE 8000

CMD ["uvicorn", "bankpredict.api:app", "--host", "0.0.0.0", "--port", "8000"]
