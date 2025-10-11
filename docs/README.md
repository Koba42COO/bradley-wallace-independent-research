# Wallace Research Suite Documentation

Complete documentation for the Wallace Research Suite - a comprehensive AI and mathematical research platform.

## Overview

The Wallace Research Suite is a production-ready collection of AI-powered tools and frameworks developed by Bradley Wallace. It includes:

- **WQRF**: Wallace Quantum Resonance Framework - ML-based primality testing
- **AIVA IDE**: AI-powered Integrated Development Environment
- **CUDNT**: CPU Universal Deep Neural Training - GPU virtualization
- **Supporting Infrastructure**: Docker, Kubernetes, monitoring, and testing

## Quick Start

```bash
# Clone repository
git clone https://github.com/your-org/wallace-research-suite.git
cd wallace-research-suite

# Start all services
docker-compose -f docker-compose.prod.yml up -d

# Check status
curl http://localhost/health
```

## Architecture

```
Wallace Research Suite
├── WQRF (Port 5001)
│   ├── ML-based primality testing
│   ├── REST API with batch processing
│   └── 98.2% prediction accuracy
├── AIVA IDE (Ports 3000/3001)
│   ├── React frontend with Monaco editor
│   ├── Node.js backend with WebSocket
│   └── OpenAI GPT integration
├── CUDNT (CPU-based)
│   ├── GPU virtualization on CPU
│   ├── TensorFlow-compatible API
│   └── 10x performance improvement
├── Infrastructure
│   ├── PostgreSQL, Redis, Nginx
│   ├── Prometheus & Grafana monitoring
│   └── Kubernetes manifests
└── Testing & CI/CD
    ├── Comprehensive test suites
    ├── Docker-based deployment
    └── Automated validation
```

## Products

### WQRF - Wallace Quantum Resonance Framework

**What it does:** Machine learning-based primality testing with mathematical pattern recognition.

**Key Features:**
- 98.2% accuracy on primality classification
- Two models: clean_ml (95.73%) and hybrid_ml (98.13%)
- Batch processing up to 100 numbers
- REST API with comprehensive error handling

**Use Cases:**
- Cryptographic research
- Number theory analysis
- Educational tools
- High-performance computing validation

### AIVA IDE - AI-Powered IDE

**What it does:** Modern development environment with AI assistance and real-time collaboration.

**Key Features:**
- Monaco editor with syntax highlighting
- GPT-4 powered code completion and chat
- Real-time collaborative editing via WebSocket
- Secure file system operations
- Room-based collaboration spaces

**Use Cases:**
- AI-assisted software development
- Remote pair programming
- Code review and collaboration
- Educational programming environments

### CUDNT - CPU Universal Deep Neural Training

**What it does:** GPU-like performance on CPU systems through advanced virtualization.

**Key Features:**
- 10x performance improvement over NumPy
- TensorFlow-compatible API
- Multi-threaded CPU utilization
- End-to-end ML pipeline optimization

**Use Cases:**
- Machine learning on CPU-only systems
- Development and testing environments
- Cost-effective AI infrastructure
- Research computing without GPU hardware

## Installation

### Prerequisites

- Docker & Docker Compose
- Node.js 16+ (optional, for development)
- Python 3.9+ (optional, for development)
- OpenAI API key (for AI features)

### Production Deployment

```bash
# 1. Clone repository
git clone https://github.com/your-org/wallace-research-suite.git
cd wallace-research-suite

# 2. Configure environment
cp env.example .env
# Edit .env with your API keys and settings

# 3. Start services
docker-compose -f docker-compose.prod.yml up -d

# 4. Verify deployment
curl http://localhost/health
```

### Development Setup

```bash
# WQRF Development
cd wqrf
pip install -r requirements.txt
python3 deployment_api.py

# AIVA IDE Development
cd aiva_ide/server && npm install && npm run dev
cd ../client && npm install && npm start

# CUDNT Development
cd cudnt
python3 cudnt_enhanced_integration.py
```

## API Reference

### WQRF API

```bash
# Health check
curl http://localhost:5001/health

# Single prediction
curl http://localhost:5001/predict/17

# Batch prediction
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"numbers": [13, 15, 17], "model": "clean_ml"}'
```

### AIVA IDE API

```bash
# Health check
curl http://localhost:3001/api/health

# List files
curl http://localhost:3001/api/files

# AI chat
curl -X POST http://localhost:3001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

### CUDNT API

```python
from cudnt_enhanced_integration import CUDNT_Enhanced

cudnt = CUDNT_Enhanced()
result = cudnt.matrix_multiply(a, b)  # 10x faster than NumPy
```

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-your-openai-api-key
POSTGRES_PASSWORD=secure-database-password

# Optional
PORT=5001
NODE_ENV=production
DEBUG=false
```

### Docker Configuration

Services are configured in `docker-compose.prod.yml`:

- **wqrf-api**: WQRF REST API service
- **aiva-ide-server**: AIVA IDE backend
- **aiva-ide-client**: AIVA IDE frontend
- **cudnt-accelerator**: CUDNT processing service
- **postgres**: Data persistence
- **redis**: Caching and sessions
- **nginx**: Reverse proxy and SSL
- **prometheus**: Metrics collection
- **grafana**: Monitoring dashboards

## Monitoring

### Health Checks

All services expose health endpoints:
- WQRF: `http://localhost:5001/health`
- AIVA IDE: `http://localhost:3001/api/health`
- CUDNT: Built-in status reporting

### Metrics

Prometheus collects metrics from all services:
- Request latency and throughput
- Error rates and success rates
- Resource utilization (CPU, memory)
- Custom business metrics

### Dashboards

Grafana provides pre-configured dashboards:
- Service overview and health
- Performance metrics and trends
- Error monitoring and alerting
- Resource utilization graphs

## Testing

### Run All Tests

```bash
# Run complete test suite
python3 tests/run_tests.py

# Run specific test suites
python3 -m pytest tests/test_wqrf_api.py -v
python3 -m pytest tests/test_aiva_ide.py -v
python3 -m pytest tests/test_cudnt.py -v
```

### Test Coverage

- **WQRF**: API endpoints, model accuracy, error handling
- **AIVA IDE**: WebSocket collaboration, file operations, AI integration
- **CUDNT**: Performance benchmarks, API compatibility, ML pipelines
- **Integration**: End-to-end workflows, cross-service communication

## Deployment Options

### Docker Compose (Recommended)

```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes

```bash
kubectl apply -f k8s/
kubectl get pods
```

### Manual Deployment

```bash
# WQRF
python3 deployment_api.py

# AIVA IDE
cd aiva_ide/server && npm start
cd ../client && npm run build && npx serve -s build

# CUDNT
python3 cudnt_enhanced_integration.py
```

## Security

### Authentication

- API key authentication for external access
- JWT tokens for user sessions
- OpenAI API key management

### Network Security

- HTTPS with SSL/TLS encryption
- CORS protection
- Rate limiting on all endpoints
- Input validation and sanitization

### Data Protection

- Encrypted database connections
- Secure API key storage
- Audit logging for sensitive operations
- Regular security updates

## Performance

### Benchmarks

- **WQRF**: 0.2-0.5ms per prediction
- **AIVA IDE**: <100ms AI response time
- **CUDNT**: 10x NumPy performance

### Scaling

- Horizontal scaling with Kubernetes
- Load balancing with Nginx
- Database connection pooling
- Redis caching for performance

## Troubleshooting

### Common Issues

1. **Port conflicts**: Change ports in docker-compose files
2. **API key issues**: Verify OpenAI API key configuration
3. **Memory issues**: Increase Docker memory limits
4. **Network issues**: Check firewall and proxy settings

### Logs

```bash
# Docker logs
docker-compose logs wqrf-api
docker-compose logs aiva-ide-server

# Application logs
docker exec -it wqrf-api tail -f /app/logs/app.log
```

### Diagnostics

```bash
# Health check all services
curl http://localhost/health

# Test individual APIs
python3 tests/run_tests.py

# Performance diagnostics
python3 cudnt_benchmark_suite.py
```

## Contributing

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Write tests for new functionality
4. Implement the feature
5. Run the full test suite
6. Submit a pull request

### Code Standards

- Python: PEP 8 with type hints
- JavaScript: ESLint configuration
- Documentation: Markdown with examples
- Tests: 80%+ coverage required

### Commit Messages

```
feat: add new WQRF model endpoint
fix: resolve WebSocket connection issues
docs: update API reference for CUDNT
test: add integration tests for AIVA IDE
```

## License

This project is part of the Wallace Quantum Resonance Framework research suite, developed by Bradley Wallace.

## Support

### Documentation
- [User Guide](user-guide.md)
- [API Reference](api-reference.md)
- [Deployment Guide](deployment.md)

### Community
- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: General questions and community support
- Email: coo@koba42.com

### Professional Support
- Enterprise deployment assistance
- Custom feature development
- Performance optimization consulting
- Training and workshops

## Roadmap

### Q1 2025
- Enhanced ML models for WQRF
- Mobile app for AIVA IDE
- GPU acceleration support for CUDNT

### Q2 2025
- Multi-language support in AIVA IDE
- Advanced visualization for WQRF results
- Distributed computing for CUDNT

### Q3 2025
- Voice interface for AIVA IDE
- Quantum computing integration
- Advanced research analytics

---

**Bradley Wallace** - *COO & Lead Researcher, Koba42 Corp*

*Advancing AI-assisted development through mathematical pattern recognition and computational optimization.*
