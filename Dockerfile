# Agent Reliability Platform — Docker Image for Render (512MB RAM free tier)
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    USE_LIGHTWEIGHT_PREDICTOR=true

# Copy requirements file
COPY requirements.txt .

# Install lightweight production dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose default port
EXPOSE 8000

# Start FastAPI server bound to Render's assigned PORT
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
