# SquashPlot on Replit - Complete Setup Guide
==============================================

This guide will help you set up and run SquashPlot on Replit for development, testing, and demonstration purposes.

## 🚀 Quick Start

### 1. Fork this Replit
```bash
# Click "Fork" on this Replit to create your own copy
# Or create a new Replit and clone this repository
```

### 2. Install Dependencies
```bash
# Replit will automatically install dependencies from requirements.txt
# If needed, you can manually install:
pip install -r requirements.txt
```

### 3. Run SquashPlot
```bash
# Start the web interface (recommended)
python main.py --web

# Or run the command-line interface
python main.py --cli

# Or run the interactive demo
python main.py --demo
```

### 4. Access Your App
- **Web Interface**: Click the "Open in new tab" button in Replit
- **URL**: `https://your-replit-name.replit.dev`
- **Port**: 8080 (automatically configured)

## 📁 Project Structure

```
squashplot-replit/
├── main.py                    # Main entry point for Replit
├── squashplot.py             # Core SquashPlot compression engine
├── whitelist_signup.py       # Pro version whitelist management
├── compression_validator.py  # Compression validation and testing
├── squashplot_web_interface.html  # Professional web interface
├── requirements.txt          # Python dependencies
├── replit.nix               # Replit system dependencies
├── .replit                  # Replit configuration
├── REPLIT_README.md         # This file
├── README.md                # General SquashPlot documentation
├── twitter_bio.txt          # Social media content
├── squashplot_web_interface_old.html  # Backup of old interface
└── SQUASHPLOT_TECHNICAL_WHITEPAPER.md  # Technical documentation
```

## 🛠️ Features Available

### Basic Version (FREE)
- ✅ 42% compression ratio
- ✅ 2x processing speed
- ✅ Proven multi-stage algorithms
- ✅ 100% farming compatibility
- ✅ Web interface
- ✅ CLI interface

### Pro Version (Whitelist)
- 🚀 Up to 70% compression ratio
- 🚀 Up to 2x faster processing
- 🚀 Enhanced algorithms
- 🚀 Priority support
- 🚀 Advanced features

## 🎯 How to Use

### Web Interface (Recommended)
1. Run `python main.py --web`
2. Click "Open in new tab" in Replit
3. Use the professional web interface:
   - Switch between Basic/Pro versions
   - Generate compressed plots
   - Connect Chia wallets
   - Calculate ROI
   - Run performance benchmarks

### Command Line Interface
```bash
# Basic compression
python squashplot.py --input plot.dat --output compressed.dat

# Pro compression (requires whitelist)
python squashplot.py --pro --input plot.dat --output compressed.dat

# Run benchmark
python squashplot.py --benchmark

# Request Pro access
python squashplot.py --whitelist-request user@domain.com
```

### Interactive Demo
```bash
python main.py --demo
```

## 🔧 Configuration

### Environment Variables
```bash
# Set in Replit Secrets or .replit file
PYTHONPATH=.
FLASK_ENV=development
```

### Custom Port
```bash
# Change port in .replit file or use:
python main.py --web --port 3000
```

## 🧪 Testing & Validation

### Run Compression Tests
```bash
# Test basic compression
python compression_validator.py --size 50

# Test Pro compression
python compression_validator.py --size 50 --pro
```

### Whitelist Management
```bash
# Add user to whitelist
python whitelist_signup.py --add user@domain.com

# Check whitelist status
python whitelist_signup.py --check user@domain.com

# View statistics
python whitelist_signup.py --stats
```

## 🚀 Deployment

### On Replit
1. **Fork** this repository
2. **Run** `python main.py --web`
3. **Share** the generated URL

### Custom Domain (Optional)
1. Go to Replit settings
2. Add custom domain
3. Configure DNS records
4. Your app will be available at your custom domain

## 🐛 Troubleshooting

### Common Issues

**"Module not found" errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

**"Port already in use":**
```bash
# Change port in .replit file
[ports]
"3000" = { public = true }
```

**"Permission denied":**
```bash
# Make sure files are executable
chmod +x main.py
```

**"Memory limit exceeded":**
```bash
# Replit has memory limits - use smaller test files
# Reduce --size parameter in tests
```

### Debug Mode
```bash
# Enable verbose logging
python main.py --web 2>&1 | tee debug.log
```

## 🔧 Advanced Configuration

### Custom Dependencies
Edit `requirements.txt` to add new packages:
```txt
# Add new package
new-package>=1.0.0
```

### Environment Setup
Add to `.replit` file:
```toml
[env]
CUSTOM_VAR = "value"
```

### File Upload Size
For larger plot files, you may need to adjust Replit's limits.

## 📊 Performance on Replit

### Expected Performance
- **CPU**: Shared across users (variable)
- **Memory**: 512MB - 2GB (depending on plan)
- **Storage**: 10GB free tier
- **Network**: Good for small files

### Optimization Tips
- Use smaller test files for demos
- Cache results when possible
- Use efficient algorithms
- Monitor memory usage

## 🤝 Contributing

### Development Setup
1. Fork this Replit
2. Make changes
3. Test thoroughly
4. Submit pull request

### Code Style
```bash
# Format code
black .

# Check style
flake8 .

# Type checking
mypy .
```

## 📚 Documentation

### Available Docs
- `README.md` - General SquashPlot documentation
- `SQUASHPLOT_TECHNICAL_WHITEPAPER.md` - Technical details
- `REPLIT_README.md` - This file

### API Documentation
- Web interface provides interactive documentation
- CLI provides `--help` for all commands

## 🔐 Security Notes

### Replit Specific
- Use Replit Secrets for sensitive data
- Don't commit API keys to version control
- Use HTTPS for production deployments

### SquashPlot Security
- All compression is lossless
- Data integrity verified with SHA256
- No external data transmission
- Local processing only

## 🎯 Use Cases

### Development & Testing
- Test compression algorithms
- Validate performance claims
- Demonstrate features to stakeholders

### Education & Learning
- Learn about Chia farming
- Understand compression techniques
- Study blockchain technology

### Production Use
- Small-scale Chia farming
- Educational demonstrations
- Proof-of-concept deployments

## 📞 Support

### Replit Issues
- Check Replit status page
- Review Replit documentation
- Contact Replit support

### SquashPlot Issues
- Check documentation
- Run diagnostic tests
- Review error logs

## 🎉 Success!

Once set up, you'll have:
- ✅ Professional web interface
- ✅ Working CLI tools
- ✅ Interactive demos
- ✅ Complete documentation
- ✅ Ready for development or demonstration

**Happy coding with SquashPlot on Replit!** 🚀🧠💎
