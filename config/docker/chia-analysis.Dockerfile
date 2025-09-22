# Chia Analysis System Dockerfile
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
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Chia analysis code
COPY chia_comprehensive_study.py .
COPY chia_developer_communities.py .
COPY chia_lisp_advanced_study.py .
COPY chia_plot_size_analyzer.py .
COPY chia_study_complete.json ./results/
COPY chia_communities_complete.json ./results/

# Create results directory
RUN mkdir -p results

# Set environment variables
ENV PYTHONPATH=/app
ENV CHIA_ANALYSIS_ENV=production

# Expose port for analysis API
EXPOSE 8088

# Default command
CMD ["python3", "chia_comprehensive_study.py"]

