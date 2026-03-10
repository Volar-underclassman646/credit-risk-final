FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/
COPY data/processed/ data/processed/

EXPOSE 8000

CMD ["uvicorn", "src.api.scoring_api:app", "--host", "0.0.0.0", "--port", "8000"]
