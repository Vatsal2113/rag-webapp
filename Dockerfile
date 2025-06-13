# Dockerfile
FROM python:3.11-slim

# system deps for Tesseract + Poppler
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr libtesseract-dev poppler-utils && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
EXPOSE 5000

CMD ["gunicorn","-k","gevent","-w","4","app:app","--bind","0.0.0.0:5000"]
