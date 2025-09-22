# SquashPlot Enhanced Production Docker Image
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r squashplot && useradd -r -g squashplot squashplot

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create necessary directories with proper permissions
RUN mkdir -p /plots /tmp/squashplot /app/logs && \
    chown -R squashplot:squashplot /plots /tmp/squashplot /app/logs /app

# Set environment variables
ENV PYTHONPATH="/app"
ENV PRODUCTION_MODE=true
ENV LOG_FILE=/app/logs/squashplot_production.log

# Switch to non-root user
USER squashplot

# Health check with proper exit codes
HEALTHCHECK --interval=30s --timeout=15s --start-period=10s --retries=3 \
    CMD python src/production_wrapper.py health-check || exit 1

# Expose volumes for plots and temp storage
VOLUME ["/plots", "/tmp/squashplot"]

# Default command
CMD ["python", "src/production_wrapper.py", "service"]