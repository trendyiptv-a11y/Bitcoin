FROM python:3.11-slim

# Evităm .pyc & buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dependențe OS de bază (dacă pandas/yfinance au nevoie de ele)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiem requirements și instalăm
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiem restul codului
COPY . .

# Expunem portul pentru uvicorn
EXPOSE 8000

# Comanda de rulare: API FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
