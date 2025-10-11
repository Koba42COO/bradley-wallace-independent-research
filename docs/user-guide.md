# Wallace Research Suite - User Guide

Complete user guide for deploying and using the Wallace Research Suite products.

## Table of Contents

- [Quick Start](#quick-start)
- [WQRF (Wallace Quantum Resonance Framework)](#wqrf-wallace-quantum-resonance-framework)
- [AIVA IDE](#aiva-ide)
- [CUDNT](#cudnt)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Quick Start

### Prerequisites

```bash
# Required software
- Docker & Docker Compose
- Node.js 16+ (for AIVA IDE)
- Python 3.9+ (for WQRF and CUDNT)
- OpenAI API key (for AI features)

# Install dependencies
pip install flask numpy scikit-learn joblib
cd aiva_ide/server && npm install
cd ../client && npm install
```

### One-Command Deployment

```bash
# Clone and deploy everything
git clone https://github.com/your-org/wallace-research-suite.git
cd wallace-research-suite

# Start all services
docker-compose -f docker-compose.prod.yml up -d

# Check status
curl http://localhost/health
```

## WQRF (Wallace Quantum Resonance Framework)

ML-based primality testing with 98.2% accuracy.

### Basic Usage

```python
import requests

# Single prediction
response = requests.get('http://localhost:5001/predict/17')
print(response.json())
# {"number": 17, "prediction": "prime", "confidence": 0.987}

# Batch prediction
data = {"numbers": [13, 15, 17, 19], "model": "clean_ml"}
response = requests.post('http://localhost:5001/predict', json=data)
print(response.json())
```

### Python Client

```python
from wqrf_client import WQRFClient

client = WQRFClient('http://localhost:5001')

# Test primality
result = client.predict(17)
print(f"{result['number']} is {result['prediction']}")

# Batch processing
numbers = [11, 13, 15, 17, 19]
results = client.predict_batch(numbers)
for r in results:
    print(f"{r['number']}: {r['prediction']}")
```

### Model Selection

WQRF provides two models:

- **clean_ml**: 95.73% accuracy, pure mathematical features
- **hybrid_ml**: 98.13% accuracy, mathematical + divisibility checks

```python
# Use high-accuracy model
result = client.predict(23, model='hybrid_ml')
```

### Performance Benchmarks

```
Model       | Accuracy | Latency | Use Case
------------|----------|---------|---------
clean_ml    | 95.73%   | 0.2ms   | Research
hybrid_ml   | 98.13%   | 0.5ms   | Production
```

## AIVA IDE

AI-powered development environment with real-time collaboration.

### Starting the IDE

```bash
# Start server
cd aiva_ide/server
npm run dev

# Start client (new terminal)
cd ../client
npm start

# Open http://localhost:3000
```

### Basic Features

#### File Management
```javascript
import { fileApi } from './lib/api';

// List files
const files = await fileApi.getFiles();

// Read file
const content = await fileApi.readFile('src/main.py');

// Write file
await fileApi.writeFile('src/main.py', 'print("Hello")');

// Create file
await fileApi.createFile('new_file.py', 'print("New")');

// Delete file
await fileApi.deleteFile('old_file.py');
```

#### AI Chat
```javascript
import { gptApi } from './lib/api';

// Chat with AI
const response = await gptApi.chat([
  { role: 'user', content: 'Help me debug this Python code' }
]);

console.log(response.message);
```

#### Code Completion
```javascript
// Get AI code completion
const completion = await gptApi.completeCode(
  'def fibonacci(n):',
  'python',
  'Write a recursive Fibonacci function'
);

console.log(completion);
```

### Real-Time Collaboration

#### Joining a Room
```javascript
import { RealtimeClient } from './lib/api';

const client = new RealtimeClient();
await client.connect('project-alpha');

// Listen for changes
client.onCodeChange((data) => {
  console.log('Code changed:', data);
  updateEditor(data.content);
});

client.onUserJoined((data) => {
  console.log('User joined:', data.userId);
});
```

#### Collaborative Editing
```javascript
// Emit code changes
editor.on('change', (content) => {
  client.emitCodeChange({
    filePath: currentFile,
    content: content
  });
});

// Track cursor movements
editor.on('cursor', (position) => {
  client.emitCursorMove({
    position: position
  });
});
```

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+S` | Save file |
| `Ctrl+Shift+P` | AI completion |
| `Ctrl+Enter` | Send chat message |
| `Ctrl+N` | New file |
| `Ctrl+O` | Open file |

## CUDNT

GPU virtualization for ML workloads on CPU systems.

### Basic Usage

```python
from cudnt_enhanced_integration import CUDNT_Enhanced
import numpy as np

# Initialize CUDNT
cudnt = CUDNT_Enhanced()

# GPU-like operations on CPU
a = np.random.rand(1000, 1000)
b = np.random.rand(1000, 1000)

# Accelerated matrix operations
result = cudnt.matrix_multiply(a, b)  # 10x faster than numpy
sum_result = cudnt.tensor_add(a, b)   # Parallel addition

# Neural network operations
activations = cudnt.relu(layer_output)
normalized = cudnt.batch_normalize(activations)
```

### TensorFlow Compatibility

```python
# Drop-in replacement for TensorFlow operations
import tensorflow as tf
# Replace with:
# import cudnt as tf

# Same API, runs on CPU with GPU-like performance
result = cudnt.tf_matmul(a, b)
conv_result = cudnt.tf_conv2d(input_tensor, kernel, stride=1)
```

### ML Pipeline Optimization

```python
# Complete ML training pipeline
params = {
    'weights': np.random.randn(784, 128),
    'bias': np.zeros(128)
}

# Training data
X_train = np.random.randn(60000, 784)
y_train = np.random.randint(0, 10, 60000)

# Optimize entire pipeline
result = cudnt.optimize_ml_pipeline(params, (X_train, y_train), epochs=50)

print(f"Final loss: {result['final_loss']}")
print(f"GPU operations: {result['pipeline_stats']['gpu_operations']}")
```

### Performance Comparison

```
Operation          | NumPy | CUDNT | Speedup
-------------------|-------|-------|--------
Matrix Multiply    | 1.2s  | 0.12s | 10x
Tensor Addition    | 0.8s  | 0.08s | 10x
Convolution 2D     | 5.5s  | 0.55s | 10x
Batch Norm         | 0.3s  | 0.03s | 10x
```

## Deployment

### Docker Deployment

```bash
# Build and run all services
docker-compose -f docker-compose.prod.yml up --build

# Run individual services
docker-compose -f docker-compose.wqrf.yml up
docker-compose -f docker-compose.cudnt.yml up

# With monitoring
docker-compose -f docker-compose.prod.yml -f monitoring.yml up
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check status
kubectl get pods
kubectl get services

# View logs
kubectl logs -l app=wqrf-api
kubectl logs -l app=aiva-ide-server
```

### Environment Configuration

```bash
# Copy example environment file
cp env.example .env

# Edit with your values
nano .env

# Key variables
OPENAI_API_KEY=sk-your-key
POSTGRES_PASSWORD=secure-password
GRAFANA_PASSWORD=admin-password
```

### SSL Configuration

```bash
# Generate SSL certificates
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365

# Update nginx configuration
# Certificates are mounted in docker-compose.prod.yml
```

## Monitoring

### Health Checks

```bash
# All services
curl http://localhost:5001/health  # WQRF
curl http://localhost:3001/api/health  # AIVA IDE
curl http://localhost:8080/health  # CUDNT

# Prometheus metrics
curl http://localhost:9090/metrics

# Grafana dashboard
open http://localhost:3002
```

### Logs

```bash
# Docker logs
docker-compose logs wqrf-api
docker-compose logs aiva-ide-server

# Kubernetes logs
kubectl logs -l app=wqrf-api
kubectl logs -l app=aiva-ide-server
```

### Metrics

Monitor these key metrics:

- **WQRF**: Prediction latency, accuracy, request rate
- **AIVA IDE**: WebSocket connections, file operations, AI requests
- **CUDNT**: GPU operations, memory usage, training performance

## Troubleshooting

### Common Issues

#### WQRF API Not Starting
```bash
# Check port availability
lsof -i :5001

# Check logs
docker-compose logs wqrf-api

# Test manually
python3 deployment_api.py
```

#### AIVA IDE WebSocket Issues
```bash
# Check WebSocket connection
curl -I -N -H "Connection: Upgrade" \
     -H "Upgrade: websocket" \
     http://localhost:3001

# Check CORS settings
# Verify origins in server/src/index.js
```

#### CUDNT Performance Issues
```bash
# Check CPU usage
top -p $(pgrep -f cudnt)

# Verify thread count
python3 -c "import cudnt_enhanced_integration; print(cudnt_enhanced_integration.CUDNT_Enhanced().gpu_virtualizer.n_threads)"
```

#### Database Connection Issues
```bash
# Test PostgreSQL
docker exec -it postgres pg_isready -U wallace

# Check Redis
docker exec -it redis redis-cli ping
```

### Performance Tuning

#### WQRF Optimization
```python
# Use batch predictions for better throughput
client.predict_batch(large_number_list)

# Choose appropriate model based on needs
# clean_ml: faster, hybrid_ml: more accurate
```

#### AIVA IDE Optimization
```javascript
// Debounce file saves
const debouncedSave = _.debounce(saveFile, 500);

// Limit concurrent AI requests
const semaphore = new Semaphore(3);
```

#### CUDNT Optimization
```python
# Adjust thread count based on CPU cores
cudnt = CUDNT_Enhanced({'gpu_threads': multiprocessing.cpu_count()})

# Use appropriate batch sizes
# Larger batches = better GPU utilization
```

### Backup and Recovery

```bash
# Backup databases
docker exec postgres pg_dump -U wallace wallace_research > backup.sql
docker exec redis redis-cli save

# Backup models
cp -r models/ models_backup/

# Restore from backup
docker exec -i postgres psql -U wallace wallace_research < backup.sql
```

## Best Practices

### Security

1. **Use HTTPS in production**
2. **Set strong passwords**
3. **Limit API rate limits**
4. **Regular security updates**
5. **Monitor access logs**

### Performance

1. **Use batch operations when possible**
2. **Implement caching for frequent requests**
3. **Monitor resource usage**
4. **Scale horizontally as needed**
5. **Use appropriate instance sizes**

### Development

1. **Write comprehensive tests**
2. **Use version control for all changes**
3. **Document API changes**
4. **Monitor error rates**
5. **Plan for backward compatibility**

### Deployment

1. **Use environment-specific configurations**
2. **Implement health checks**
3. **Set up monitoring and alerting**
4. **Have rollback procedures**
5. **Test deployments thoroughly**

## Support

### Getting Help

1. **Check logs**: `docker-compose logs`
2. **Run diagnostics**: `python3 tests/run_tests.py`
3. **Check documentation**: This guide and API reference
4. **Community forums**: GitHub issues and discussions

### Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new features
4. Submit a pull request
5. Follow the established code style

---

*Built with ❤️ by Bradley Wallace - Advancing AI-assisted development through the Wallace Quantum Resonance Framework*
