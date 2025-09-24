FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create persistent data dir (Render attaches a disk here)
RUN mkdir -p /data
ENV DATA_DIR=/data

COPY app ./app
ENV PYTHONPATH=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
