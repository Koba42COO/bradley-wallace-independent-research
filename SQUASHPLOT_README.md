# 🍃 SquashPlot - Advanced Chia Blockchain Farming Optimization System

> **Complete Farming Management & GPU-Accelerated Plotting for Chia Blockchain**
>
> SquashPlot is a comprehensive farming optimization system designed specifically for Chia blockchain farming operations. Features advanced F2 optimization, GPU acceleration, intelligent plot management, and real-time monitoring.

---

## 🎯 **System Overview**

### **Core Components**
- **🌾 Farming Manager**: Real-time farming monitoring and optimization
- **🚀 F2 GPU Optimizer**: Advanced plotting with performance profiles
- **💾 Disk Optimizer**: Intelligent storage management and balancing
- **🌐 Web Dashboard**: Real-time monitoring and control interface
- **🤖 Automation Engine**: Automated maintenance and optimization

### **Key Features**
- ✅ **F2 Optimization Algorithm** with GPU acceleration
- ✅ **Performance Profiles**: Speed, Cost, Middle (balanced)
- ✅ **Real-time Monitoring** and analytics
- ✅ **Intelligent Plot Management** and distribution
- ✅ **Web Dashboard** for remote monitoring
- ✅ **Automated Maintenance** and optimization
- ✅ **Resource Optimization** and cost analysis

---

## 🚀 **Quick Start**

### **Installation**
```bash
# Clone the repository
git clone https://github.com/your-repo/squashplot.git
cd squashplot

# Install dependencies
pip install -r requirements.txt

# Optional GPU support
pip install cupy GPUtil
```

### **Basic Usage**
```bash
# Start farming monitoring
python squashplot_chia_system.py --plot-dirs /plots1 /plots2 --mode middle

# GPU-accelerated plotting
python f2_gpu_optimizer.py --temp-dirs /temp1 /temp2 --final-dirs /plots1 --profile speed --num-plots 4

# Disk optimization
python squashplot_disk_optimizer.py --plot-dirs /plots1 /plots2 --analyze

# Web dashboard
python squashplot_dashboard.py --plot-dirs /plots1 /plots2 --host 0.0.0.0 --port 5000
```

---

## ⚡ **Performance Profiles**

### **Speed Profile** 🚀
- **GPU Acceleration**: Maximum utilization
- **CPU Threads**: All available cores
- **Memory Usage**: High (80% of available)
- **Batch Processing**: 4 parallel plots
- **Best For**: Fastest plotting, higher electricity costs

### **Cost Profile** 💰
- **GPU Acceleration**: Disabled
- **CPU Threads**: Minimal (1-2 cores)
- **Memory Usage**: Conservative (30% of available)
- **Batch Processing**: 1 plot at a time
- **Best For**: Minimum electricity costs, slower plotting

### **Middle Profile** ⚖️
- **GPU Acceleration**: Conditional (if available)
- **CPU Threads**: Balanced (50% of cores)
- **Memory Usage**: Moderate (50% of available)
- **Batch Processing**: 2 parallel plots
- **Best For**: Balanced performance and cost optimization

---

## 🛠️ **Component Details**

### **1. Farming Manager (`squashplot_chia_system.py`)**

#### **Features**
- Real-time farming statistics monitoring
- Plot file scanning and quality assessment
- Resource utilization tracking
- Automated optimization based on selected mode
- Alert system for farming issues

#### **Usage**
```bash
python squashplot_chia_system.py \
    --chia-root ~/chia-blockchain \
    --plot-dirs /mnt/plots1 /mnt/plots2 /mnt/plots3 \
    --mode speed \
    --export farming_report.json
```

#### **Key Metrics**
- Total plots and active plots
- Farming efficiency and proof discovery
- System resource utilization
- Plot quality scores and distribution

---

### **2. F2 GPU Optimizer (`f2_gpu_optimizer.py`)**

#### **Features**
- F2 optimization algorithm with GPU acceleration
- Customizable performance profiles
- Real-time GPU utilization monitoring
- Cost-benefit analysis and reporting
- Intelligent resource allocation

#### **Usage**
```bash
python f2_gpu_optimizer.py \
    --chia-root ~/chia-blockchain \
    --temp-dirs /mnt/temp1 /mnt/temp2 \
    --final-dirs /mnt/plots1 /mnt/plots2 \
    --profile speed \
    --num-plots 8 \
    --farmer-key YOUR_FARMER_KEY \
    --pool-key YOUR_POOL_KEY \
    --output optimization_results.json
```

#### **GPU Requirements**
- NVIDIA GPU with CUDA support
- Minimum 4GB VRAM (8GB recommended)
- CUDA 11.0+ compatible drivers

#### **Performance Metrics**
- Plotting speed (plots per hour)
- GPU utilization percentage
- Cost per plot calculation
- Resource efficiency scoring

---

### **3. Disk Optimizer (`squashplot_disk_optimizer.py`)**

#### **Features**
- Intelligent plot distribution analysis
- Disk health monitoring and scoring
- Automated migration planning
- Space optimization and cleanup
- RAID and storage pool management

#### **Usage**
```bash
# Analyze disk configuration
python squashplot_disk_optimizer.py \
    --plot-dirs /mnt/plots1 /mnt/plots2 /mnt/plots3 \
    --analyze

# Perform optimization
python squashplot_disk_optimizer.py \
    --plot-dirs /mnt/plots1 /mnt/plots2 \
    --optimize \
    --output disk_optimization.json

# Execute migrations (dry run first!)
python squashplot_disk_optimizer.py \
    --plot-dirs /mnt/plots1 /mnt/plots2 \
    --migrate \
    --dry-run
```

#### **Health Metrics**
- Disk usage and free space monitoring
- Plot distribution balance scoring
- Disk health status assessment
- Migration priority calculation

---

### **4. Web Dashboard (`squashplot_dashboard.py`)**

#### **Features**
- Real-time farming statistics visualization
- Interactive performance charts
- Alert system and notification center
- Remote optimization mode switching
- Export capabilities for reporting

#### **Starting the Dashboard**
```bash
python squashplot_dashboard.py \
    --chia-root ~/chia-blockchain \
    --plot-dirs /mnt/plots1 /mnt/plots2 \
    --mode middle \
    --host 0.0.0.0 \
    --port 5000
```

#### **Access**
- **URL**: http://localhost:5000
- **Features**: Live charts, alerts, optimization controls
- **Export**: Ctrl+E to export dashboard data

---

## 📊 **Monitoring & Analytics**

### **Real-time Metrics**
- **Farming Statistics**: Plots, proofs, efficiency
- **System Resources**: CPU, memory, GPU usage
- **Disk Health**: Usage, balance, health scores
- **Performance Trends**: Historical data analysis

### **Alert System**
- CPU/memory usage warnings
- Disk space critical alerts
- Farming efficiency notifications
- Health status monitoring

### **Reporting**
- JSON export for all components
- Performance trend analysis
- Cost-benefit reporting
- Optimization recommendations

---

## 🔧 **Configuration**

### **Environment Setup**
```bash
# Create virtual environment
python -m venv squashplot_env
source squashplot_env/bin/activate  # Linux/Mac
# or
squashplot_env\Scripts\activate     # Windows

# Install requirements
pip install -r requirements.txt
```

### **Configuration File**
```json
{
    "chia_root": "~/chia-blockchain",
    "plot_directories": ["/mnt/plots1", "/mnt/plots2"],
    "temp_directories": ["/mnt/temp1", "/mnt/temp2"],
    "optimization_mode": "middle",
    "min_free_space_gb": 100,
    "rebalance_threshold": 0.1,
    "dashboard_host": "0.0.0.0",
    "dashboard_port": 5000
}
```

---

## 🎯 **Use Cases**

### **Home Farming Setup**
```bash
# Basic monitoring
python squashplot_chia_system.py --plot-dirs ~/plots --mode cost

# Occasional optimization
python f2_gpu_optimizer.py --temp-dirs ~/temp --final-dirs ~/plots --profile middle --num-plots 2
```

### **Commercial Farming Operation**
```bash
# Full monitoring suite
python squashplot_dashboard.py --plot-dirs /srv/plots1 /srv/plots2 /srv/plots3 --mode speed

# Regular optimization
python squashplot_disk_optimizer.py --plot-dirs /srv/plots* --optimize

# GPU plotting farm
python f2_gpu_optimizer.py --temp-dirs /srv/temp1 /srv/temp2 --final-dirs /srv/plots* --profile speed --num-plots 16
```

### **Enterprise Chia Farm**
```bash
# Multi-server monitoring
# Deploy dashboard on central server
# Configure distributed plot directories
# Set up automated optimization schedules
# Monitor across multiple physical locations
```

---

## 📈 **Performance Benchmarks**

### **Plotting Performance**

| Profile | GPU | CPU Cores | Plots/Hour | Cost/kWh | Efficiency |
|---------|-----|-----------|------------|----------|------------|
| Speed   | ✅  | All       | 2.5-3.0   | High     | Highest   |
| Middle  | ✅  | 50%       | 1.5-2.0   | Medium   | Balanced  |
| Cost    | ❌  | 1-2       | 0.8-1.2   | Low      | Lowest    |

### **System Requirements**

#### **Minimum**
- CPU: 4 cores
- RAM: 8GB
- Storage: 1TB
- OS: Linux/Windows/Mac

#### **Recommended**
- CPU: 8+ cores
- RAM: 16GB+
- GPU: NVIDIA RTX 3060+
- Storage: 10TB+ NVMe SSD
- OS: Linux (Ubuntu 20.04+)

#### **Enterprise**
- CPU: 16+ cores
- RAM: 64GB+
- GPU: NVIDIA RTX 3080+ or A-series
- Storage: 100TB+ enterprise storage
- Network: 10Gbps connectivity

---

## 🚨 **Troubleshooting**

### **Common Issues**

#### **GPU Not Detected**
```bash
# Check GPU status
nvidia-smi

# Install CUDA toolkit
# Ensure GPU drivers are installed
# Check GPUtil installation: pip install GPUtil
```

#### **High Memory Usage**
```bash
# Switch to cost profile
python squashplot_chia_system.py --mode cost

# Reduce batch size in GPU optimizer
python f2_gpu_optimizer.py --num-plots 1
```

#### **Disk Space Issues**
```bash
# Run disk optimizer
python squashplot_disk_optimizer.py --plot-dirs /plots --optimize

# Check disk health
python squashplot_disk_optimizer.py --analyze
```

#### **Plotting Failures**
```bash
# Verify Chia installation
chia version

# Check temp/final directory permissions
ls -la /temp /plots

# Monitor system resources during plotting
python squashplot_chia_system.py --mode cost
```

---

## 📚 **API Reference**

### **Farming Manager API**
```python
from squashplot_chia_system import ChiaFarmingManager, OptimizationMode

# Initialize
manager = ChiaFarmingManager(
    chia_root="~/chia-blockchain",
    plot_directories=["/plots1", "/plots2"],
    optimization_mode=OptimizationMode.MIDDLE
)

# Start monitoring
manager.start_monitoring()

# Get stats
stats = manager.get_farming_report()

# Stop monitoring
manager.stop_monitoring()
```

### **GPU Optimizer API**
```python
from f2_gpu_optimizer import F2GPUOptimizer, PerformanceProfile

# Initialize
optimizer = F2GPUOptimizer(
    chia_root="~/chia-blockchain",
    temp_dirs=["/temp1"],
    final_dirs=["/plots1"],
    profile=PerformanceProfile.SPEED
)

# Run optimization
results = optimizer.optimize_f2_plotting(
    num_plots=4,
    farmer_key="your_farmer_key",
    pool_key="your_pool_key"
)
```

---

## 🤝 **Contributing**

### **Development Setup**
```bash
# Fork and clone
git clone https://github.com/your-fork/squashplot.git
cd squashplot

# Create feature branch
git checkout -b feature/new-optimization

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Submit pull request
```

### **Code Standards**
- PEP 8 compliance
- Type hints for all functions
- Comprehensive docstrings
- Unit tests for new features
- Documentation updates

---

## 📄 **License**

**MIT License**

Copyright (c) YYYY STREET NAME (Koba42 Corp)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

---

## 📞 **Support & Contact**

### **Technical Support**
- **Documentation**: See individual component READMEs
- **Issues**: GitHub Issues for bug reports
- **Discussions**: GitHub Discussions for questions
- **Performance**: Benchmark results in `/benchmarks/`

### **Business Inquiries**
- **Email**: user@domain.com
- **Company**: Koba42 Corp
- **Website**: https://vantaxsystems.com

### **Community**
- **Discord**: Join our farming optimization community
- **Forum**: Chia farming optimization discussions
- **Newsletter**: Monthly performance updates and tips

---

## 🎉 **Acknowledgments**

**Special thanks to:**
- Chia Network for the innovative blockchain platform
- NVIDIA for GPU technology enabling faster plotting
- The Chia farming community for sharing knowledge
- VantaX Systems for computational proof validation

---

## 🔄 **Changelog**

### **Version 1.0.0** (Current)
- ✅ Complete farming management system
- ✅ F2 optimization with GPU acceleration
- ✅ Performance profiles (Speed/Cost/Middle)
- ✅ Real-time monitoring and web dashboard
- ✅ Disk optimization and health monitoring
- ✅ Automated plot management and migration
- ✅ Comprehensive analytics and reporting

### **Upcoming Features**
- 🔄 Multi-GPU support and optimization
- 🔄 Cloud integration for distributed farming
- 🔄 Advanced machine learning predictions
- 🔄 Mobile app for remote monitoring
- 🔄 Integration with major Chia pools

---

**🍃 SquashPlot - Optimizing Chia Farming for Maximum Efficiency**

**Built by Bradley Wallace - COO, Koba42 Corp**  
*Advanced farming optimization through intelligent automation*  
*GPU-accelerated plotting with F2 optimization algorithms*  
*Real-time monitoring and cost-effective resource management*

**🌟 Choose Your Profile: Speed 🚀 | Cost 💰 | Middle ⚖️**  
**🌾 Farm Smarter, Not Harder!** ✨🤖🌾
