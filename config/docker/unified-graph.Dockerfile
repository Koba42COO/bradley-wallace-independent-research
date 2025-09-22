# Unified Wallace Elysia Graph Framework Dockerfile
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
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install graph and ML packages
RUN pip install --no-cache-dir \
    networkx \
    matplotlib \
    plotly \
    scikit-learn \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    || echo "Graph/ML packages installation partially failed, continuing"

# Copy unified graph framework code
COPY unified_wallace_elysia_graph_framework.py .
COPY consciousness_elysia_framework.py .
COPY wallace_research_suite/christopher_wallace_validation_framework.py .

# Create results directory
RUN mkdir -p results graphs

# Set environment variables
ENV PYTHONPATH=/app
ENV GRAPH_ENV=production

# Expose port for graph API
EXPOSE 8086

# Default command
CMD ["python3", "unified_wallace_elysia_graph_framework.py"]

