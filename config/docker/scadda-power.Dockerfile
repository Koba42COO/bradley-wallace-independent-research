# SCADDA Power Optimization Framework Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY scadda_power_optimizer.py .
COPY scadda_power_demo_fixed.py .
COPY gpu_abstraction_layer.py .
COPY scadda_power_optimization_results.json ./results/

# Create results directory
RUN mkdir -p results

# Set environment variables
ENV PYTHONPATH=/app
ENV SCADDA_ENV=production

# Expose port for potential API
EXPOSE 8082

# Default command
CMD ["python3", "scadda_power_demo_fixed.py"]

