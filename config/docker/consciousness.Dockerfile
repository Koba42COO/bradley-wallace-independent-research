# Consciousness Frameworks Dockerfile
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

# Install additional ML/AI packages
RUN pip install --no-cache-dir \
    scikit-learn \
    tensorflow \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    || echo "ML packages installation partially failed, continuing"

# Copy consciousness framework code
COPY consciousness_elysia_framework.py .
COPY consciousness_homomorphic_encryption.py .
COPY consciousness_ruleset_enforcer.py .
COPY elysia_integration_test.py .

# Create results directory
RUN mkdir -p results

# Set environment variables
ENV PYTHONPATH=/app
ENV CONSCIOUSNESS_ENV=production

# Expose port for consciousness API
EXPOSE 8085

# Default command
CMD ["python3", "consciousness_elysia_framework.py"]

