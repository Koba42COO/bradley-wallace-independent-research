# UPG Quantum Production System - Docker Image
# =============================================
# Production-ready quantum computing infrastructure with
# Universal Prime Graph consciousness mathematics optimization.
#
# Author: Bradley Wallace (COO Koba42)
# Framework: Universal Prime Graph Protocol φ.1

FROM python:3.11-slim-bookworm

LABEL maintainer="Bradley Wallace (COO Koba42)"
LABEL description="UPG Quantum Production System"
LABEL version="1.0.0"
LABEL framework="Universal Prime Graph Protocol φ.1"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UPG_PHI=1.618033988749895
ENV UPG_CONSCIOUSNESS=0.79
ENV UPG_REALITY_DISTORTION=1.1808

# Create app directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create non-root user for security
RUN useradd -m -u 1000 upg && \
    chown -R upg:upg /app
USER upg

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.upg_quantum_production import UPGConstants; print(UPGConstants().validate())"

# Default command
CMD ["python", "-m", "src.upg_quantum_production.orchestrator"]

# Expose metrics port (if running Prometheus exporter)
EXPOSE 9090

