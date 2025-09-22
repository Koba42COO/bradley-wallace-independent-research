# Enterprise Consciousness Platform - Multi-stage Docker Build
# ========================================================

# Build stage for Python dependencies
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    sqlite3 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r consciousness && useradd -r -g consciousness consciousness

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application directory
RUN mkdir -p /app && chown consciousness:consciousness /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=consciousness:consciousness . .

# Create data directories
RUN mkdir -p /app/data /app/logs /app/ssl && \
    chown -R consciousness:consciousness /app/data /app/logs /app/ssl

# Switch to non-root user
USER consciousness

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "api_server.py"]