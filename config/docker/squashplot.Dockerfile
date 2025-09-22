# SquashPlot Chia System Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Chia and GPU support
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

# Install optional GPU monitoring (will work if available)
RUN pip install --no-cache-dir gputil || echo "GPUtil not available, continuing without GPU monitoring"

# Copy Chia/SquashPlot application code
COPY squashplot_chia_system.py .
COPY f2_gpu_optimizer.py .
COPY gpu_abstraction_layer.py .
COPY chia_plot_size_analyzer.py .
COPY plot_analysis_results.json ./results/

# Create necessary directories
RUN mkdir -p results plots temp

# Set environment variables
ENV PYTHONPATH=/app
ENV CHIA_ENV=production

# Expose port for potential web interface
EXPOSE 8083

# Default command
CMD ["python3", "squashplot_chia_system.py"]

