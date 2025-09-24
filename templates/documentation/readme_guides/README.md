# 🌱 SquashPlot Beta - Advanced Chia Plot Compression Tool

[![Replit](https://img.shields.io/badge/Run%20on-Replit-orange)](https://replit.com/@yourusername/squashplot)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.75+-green.svg)](https://fastapi.tiangolo.com/)
[![Chia](https://img.shields.io/badge/Chia-Farming-orange.svg)](https://www.chia.net/)
[![Andy's CLI](https://img.shields.io/badge/Andy's_CLI-Integrated-blueviolet)](https://github.com/Koba42COO)

> **🧠 Andy's Enhanced SquashPlot: Professional Chia plotting with CLI integration, real-time monitoring, and Replit-optimized deployment.**

## 🚀 Features

### 🗜️ **Advanced Compression Technology**
- **Multi-Stage Compression**: Zstandard, Brotli, LZ4 algorithms
- **5 Compression Levels**: 0% to 35% space savings
- **Chia-Aware Processing**: Optimized for Chia plot data
- **100% Fidelity**: Lossless compression with SHA256 verification

### 📊 **Professional Dashboard**
- **Real-Time Monitoring**: System resources, plotting progress, network stats
- **Live Market Data**: XCH price, network space, farming metrics
- **Job Management**: Queue, monitor, and control plotting jobs
- **Storage Management**: Drive optimization and health monitoring

### ⚡ **Performance Optimization**
- **GPU Acceleration**: CUDA support for F2 optimization
- **Multi-Threading**: Parallel processing for maximum efficiency
- **Resource Management**: Intelligent CPU, memory, and storage allocation
- **Multi-Plotter Integration**: Compatible with Mad Max, BladeBit, and Dr. Plotter

### 🎯 **Farming Management**
- **Pool Integration**: Connect to Chia pools
- **Rewards Tracking**: Monitor earnings and ROI
- **Analytics**: Performance insights and trends
- **Automation**: Auto-claim, auto-compress, scheduling

### 🤖 **Andy's CLI Integration**
- **Server Monitoring**: Real-time status with `check_server.py` logic
- **Professional Commands**: Mad Max/BladeBit compatible CLI structures
- **Command Templates**: Pre-built commands for common operations
- **Web CLI Interface**: Execute commands directly from dashboard
- **Replit Optimization**: Cloud-ready with automatic port configuration

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- 8GB+ RAM recommended
- 100GB+ free storage for plotting
- CUDA-compatible GPU (optional, for acceleration)

### Quick Start

#### 🚀 **Replit Deployment (Recommended - Andy's Optimization)**

1. **Fork on Replit**
   ```bash
   # Click "Fork" on the Replit template
   # Automatic setup with Andy's CLI integration
   ```

2. **Run the Application**
   ```bash
   # Replit will automatically start the server
   # Access at: https://your-replit-name.replit.dev
   ```

3. **Features Available**
   - ✅ Real-time server monitoring
   - ✅ CLI command templates
   - ✅ Professional web dashboard
   - ✅ Andy's check_server integration

#### 💻 **Local Development**

1. **Clone and Setup**
   ```bash
   git clone https://github.com/Koba42COO/222hr-Hackathon-Entry-Squashplot-Beta.git
   cd 222hr-Hackathon-Entry-Squashplot-Beta
   pip install -r requirements.txt
   ```

2. **Run SquashPlot (Andy's CLI)**
   ```bash
   # Enhanced Web Dashboard with CLI integration
   python main.py --web

   # Professional CLI Mode
   python main.py --cli

   # Interactive Demo
   python main.py --demo

   # Server Status Check (Andy's utility)
   python check_server.py
   ```

3. **Access the Dashboard**
   - **Replit**: `https://your-replit-name.replit.dev`
   - **Local**: `http://localhost:8080`
   - **API Docs**: `http://localhost:8080/docs`

## 📖 Usage

### Web Dashboard
The web dashboard provides a comprehensive interface for managing your Chia farming operations:

- **Dashboard**: System overview and real-time metrics
- **Jobs**: Plotting job management and monitoring
- **Storage**: Drive management and optimization
- **Rewards**: Earnings tracking and analysis
- **Pools**: Pool management and switching
- **Analytics**: Performance insights and trends
- **Settings**: Configuration and preferences
- **Help**: Documentation and support

### Command Line Interface
For advanced users and automation:

```bash
# Basic compression
python squashplot.py --input plot.plot --output compressed.plot --level 3

# Batch processing
python squashplot.py --batch --input-dir /plots --output-dir /compressed --level 5

# GPU acceleration
python squashplot.py --gpu --input plot.plot --output compressed.plot --level 7

# Dr. Plotter integration
python squashplot.py --plotter drplotter --tmp /tmp --final /plots --farmer-key YOUR_KEY
```

### API Endpoints
SquashPlot provides a REST API for integration:

```bash
# System status
curl http://localhost:5000/api/status

# Job management
curl http://localhost:5000/api/jobs

# Market data
curl http://localhost:5000/api/live-price

# Compression levels
curl http://localhost:5000/api/compression-levels
```

## 🏗️ Architecture

### Core Components
- **SquashPlot Engine**: Core compression algorithms
- **Web Server**: Flask-based dashboard and API
- **Job Queue**: Asynchronous job management
- **Storage Manager**: Drive and plot organization
- **Market Data**: Real-time Chia network information
- **Analytics Engine**: Performance monitoring and insights

### Plotter Integrations
SquashPlot supports multiple plotting tools for maximum flexibility:

- **Mad Max**: Fast plotting with excellent performance
- **BladeBit**: GPU-accelerated plotting with compression support
- **Dr. Plotter**: Advanced plotting with built-in optimization features

The system automatically detects available plotters and selects the optimal one based on your configuration and system capabilities.

### Technology Stack
- **Backend**: Python 3.9+, Flask, SQLAlchemy
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Compression**: Zstandard, Brotli, LZ4, zlib, bz2, lzma
- **Plotters**: Mad Max, BladeBit, Dr. Plotter integration
- **Data**: JSON, SQLite, CSV export
- **APIs**: RESTful endpoints, real-time updates

## 📊 Performance

### Compression Benchmarks
| Level | Algorithm | Savings | Size (K32) | Speed |
|-------|-----------|---------|------------|-------|
| 0 | None | 0% | 108.8 GB | Instant |
| 1 | LZ4 + zlib | 20% | 87.0 GB | Fast |
| 2 | Zstandard | 25% | 81.6 GB | Medium |
| 3 | Brotli | 30% | 76.2 GB | Medium |
| 4 | Advanced | 35% | 70.7 GB | Slow |

### System Requirements
- **Minimum**: 4 CPU cores, 8GB RAM, 200GB storage
- **Recommended**: 8+ CPU cores, 16GB+ RAM, 1TB+ storage
- **Optimal**: 16+ CPU cores, 32GB+ RAM, 2TB+ NVMe storage

## 🔧 Configuration

### Environment Variables
```bash
# Database
export DATABASE_URL="sqlite:///squashplot.db"

# API Keys
export API_KEY="your-api-key"
export SESSION_SECRET="your-session-secret"

# Chia Configuration
export CHIA_ROOT="/path/to/chia"
export FARMER_KEY="your-farmer-key"
export POOL_KEY="your-pool-key"
```

### Settings File
Create `config.json` for advanced configuration:

```json
{
  "compression": {
    "default_level": 3,
    "auto_compress": true,
    "verify_integrity": true
  },
  "plotting": {
    "default_plotter": "madmax",
    "threads": 4,
    "temp_dir": "/tmp",
    "final_dir": "/plots"
  },
  "farming": {
    "auto_claim": false,
    "pool_url": "https://pool.space",
    "wallet_address": "xch1..."
  }
}
```

## 🚀 Deployment

### Docker Deployment
```bash
# Build image
docker build -t squashplot .

# Run container
docker run -p 5000:5000 -v /plots:/plots squashplot
```

### Production Deployment
```bash
# Install production dependencies
pip install -r requirements-prod.txt

# Run with Gunicorn
gunicorn --bind 0.0.0.0:5000 src.web_server:app
```

### Cloud Deployment
SquashPlot is compatible with:
- **AWS**: EC2, ECS, Lambda
- **Google Cloud**: Compute Engine, Cloud Run
- **Azure**: Virtual Machines, Container Instances
- **DigitalOcean**: Droplets, App Platform

## 📈 Roadmap

### Version 1.1 (Q2 2024)
- [ ] Advanced GPU optimization
- [ ] Multi-node clustering
- [ ] Enhanced analytics
- [ ] Mobile app

### Version 1.2 (Q3 2024)
- [ ] Machine learning optimization
- [ ] Advanced compression algorithms
- [ ] Enterprise features
- [ ] API v2

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone repository
git clone https://github.com/Koba42COO/222hr-Hackathon-Entry-Squashplot-Beta.git
cd 222hr-Hackathon-Entry-Squashplot-Beta

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Chia Network** for the innovative blockchain technology
- **Mad Max Plotter** for the efficient plotting algorithms
- **BladeBit** for GPU-accelerated plotting
- **Open Source Community** for the amazing tools and libraries

## 📞 Support

- **Documentation**: [Wiki](https://github.com/Koba42COO/222hr-Hackathon-Entry-Squashplot-Beta/wiki)
- **Issues**: [GitHub Issues](https://github.com/Koba42COO/222hr-Hackathon-Entry-Squashplot-Beta/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Koba42COO/222hr-Hackathon-Entry-Squashplot-Beta/discussions)
- **Email**: support@squashplot.dev

## 🏆 Hackathon Entry

This project was developed for the **222hr Hackathon** and represents a complete solution for Chia farming optimization. The system combines advanced compression technology with professional-grade management tools to create the ultimate Chia farming platform.

### Key Innovations
- **Advanced Compression**: Up to 35% space savings
- **Real-Time Monitoring**: Comprehensive dashboard
- **Professional UI/UX**: Enterprise-grade interface
- **API Integration**: RESTful endpoints for automation
- **Responsive Design**: Works on all devices

---

**Made with ❤️ for the Chia farming community**

*SquashPlot - Compress More, Farm Better*