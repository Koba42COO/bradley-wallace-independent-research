# Benchmark System Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy benchmark system code
COPY unified_framework_benchmark.py .
COPY benchmark_analysis_report.py .
COPY benchmark_report_1757104877.json ./results/

# Create results directory
RUN mkdir -p results

# Set environment variables
ENV PYTHONPATH=/app
ENV BENCHMARK_ENV=production

# Expose port for benchmark API
EXPOSE 8087

# Default command
CMD ["python3", "unified_framework_benchmark.py"]

