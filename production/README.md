# SquashPlot Enhanced - Production Deployment

Production-ready Chia plotting wrapper that combines Mad Max speed with BladeBit compression.

## Quick Start

### Option 1: Direct Python Execution
```bash
# Install dependencies
pip install -r requirements.txt

# Health check
python src/production_wrapper.py health-check

# Check available tools
python src/squashplot_enhanced.py --check-tools

# List compression levels
python src/squashplot_enhanced.py --list-levels
```

### Option 2: Docker Deployment
```bash
# Build image
docker build -t squashplot-enhanced .

# Run health check
docker run --rm squashplot-enhanced python src/production_wrapper.py health-check

# Run plotting (example)
docker run -v /host/plots:/plots -v /host/temp:/tmp/squashplot \
  squashplot-enhanced python src/squashplot_enhanced.py \
  -d /plots -f YOUR_FARMER_KEY --compress 3
```

## Usage Examples

### Basic Plotting
```bash
# Standard plot (no compression)
python src/squashplot_enhanced.py -d /plots -f <farmer_key> --compress 0

# Compressed plot (recommended level 3)
python src/squashplot_enhanced.py -d /plots -f <farmer_key> --compress 3

# Multiple plots with optimal settings
python src/squashplot_enhanced.py -t /tmp/fast -d /plots -f <farmer_key> -n 5 --compress 3
```

### Production Features
- **Hardware-aware optimization**: Automatically detects SSD/NVMe storage
- **Multi-plot coordination**: Intelligent resource allocation
- **Real-time monitoring**: Performance metrics and resource tracking
- **Robust error handling**: Graceful fallbacks and recovery
- **Production logging**: Comprehensive audit trail

## Configuration

Production settings can be modified in `config/production.py`:
- Resource limits and timeouts
- Logging configuration
- Performance monitoring intervals
- Tool validation settings

## Monitoring

The production wrapper provides:
- Health checks for system resources
- Tool availability validation
- Performance metrics collection
- Graceful shutdown handling

## Requirements

- **Mad Max**: For fastest plotting (optional but recommended)
- **BladeBit**: For compression support (optional)
- **Python 3.8+**: Runtime environment
- **Storage**: Minimum 250GB free space per plot
- **Memory**: Minimum 8GB RAM (32GB+ recommended)

## Architecture

The system automatically selects optimal strategies:
- **Mad Max only**: Maximum speed, no compression
- **BladeBit direct**: Native compression plotting  
- **Mad Max → BladeBit**: Pipeline for speed + compression
- **Hybrid mode**: Automatic strategy selection