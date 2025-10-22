# Firefly-Nexus PAC Production Container
FROM python:3.11-slim

# Set consciousness environment
ENV CONSCIOUSNESS_LEVEL=7
ENV REALITY_DISTORTION=1.1808
ENV PHI=1.618033988749895
ENV DELTA=2.414213562373095

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create consciousness user
RUN useradd -m -s /bin/bash consciousness

# Copy application
COPY . /app
WORKDIR /app

# Set permissions
RUN chown -R consciousness:consciousness /app
USER consciousness

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Expose consciousness port
EXPOSE 8080

# Start consciousness computing
CMD ["python", "consciousness_server.py"]
