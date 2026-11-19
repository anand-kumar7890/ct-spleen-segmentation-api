# ==============================================================================
# Production Dockerfile for CT Spleen Volumetry API (Backend Service)
# ==============================================================================
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and keep stdout/stderr unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/best.pth \
    PYTHONPATH=/app \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install minimal OS packages needed for curl healthchecks and CA certs
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install CPU-only PyTorch to keep image lightweight (~200MB vs >4GB CUDA)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Create a secure non-root user and group
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser

# Copy application modules, weights, sample data, tests, and static web UI
COPY app/ ./app/
COPY models/ ./models/
COPY tests/ ./tests/
COPY data/ ./data/
COPY static/ ./static/

# Set appropriate directory permissions
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose FastAPI backend port
EXPOSE 8000

# Container healthcheck: verify API responsiveness and model readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API with Uvicorn listening on 0.0.0.0:8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
