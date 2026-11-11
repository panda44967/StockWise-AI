# StockWise-AI Production Hardened Dockerfile
# Implements G4 (Deployment) and G5 (Container Security Hardening)
FROM python:3.11-slim

# Security: Create unprivileged system user
RUN groupadd -r appgroup && useradd -r -g appgroup -d /home/appuser -s /sbin/nologin -c "Docker App User" appuser

WORKDIR /app

# Install security patches and curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (layer caching optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Set strict ownership and permissions
RUN chown -R appuser:appgroup /app
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8501

# Healthcheck probe (G7 Fault Tolerance)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "stock_advisor/app.py"]
