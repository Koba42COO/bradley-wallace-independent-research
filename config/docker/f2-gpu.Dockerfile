# F2 GPU Optimizer Dockerfile with GPU support
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for GPU and CUDA
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    libssl-dev \
    libffi-dev \
    python3-dev \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install GPU-specific packages
RUN pip install --no-cache-dir \
    cupy-cuda11x \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    || echo "GPU packages installation failed, continuing with CPU-only mode"

# Copy GPU optimization code
COPY f2_gpu_optimizer.py .
COPY gpu_abstraction_layer.py .
COPY test_m3_gpu.py .

# Create necessary directories
RUN mkdir -p results temp

# Set environment variables for GPU
ENV PYTHONPATH=/app
ENV GPU_ENV=production
ENV CUDA_VISIBLE_DEVICES=all

# Expose port for GPU monitoring
EXPOSE 8084

# Default command
CMD ["python3", "f2_gpu_optimizer.py"]

