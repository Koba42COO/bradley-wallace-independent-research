# Wallace Research Suite - API Reference

Complete API documentation for all Wallace Research Suite products.

## Table of Contents

- [WQRF API](#wqrf-api)
- [AIVA IDE API](#aiva-ide-api)
- [CUDNT API](#cudnt-api)
- [Common Patterns](#common-patterns)
- [Error Handling](#error-handling)
- [Authentication](#authentication)

## WQRF API

The Wallace Quantum Resonance Framework provides machine learning-based primality testing.

**Base URL:** `http://localhost:5001`

### Health Check

Get the health status of the WQRF API service.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "primality_testing_api",
  "version": "1.0.0",
  "models_available": ["clean_ml", "hybrid_ml"],
  "timestamp": "2025-01-15T10:30:00.000Z"
}
```

### Single Prediction

Predict primality for a single number.

```http
GET /predict/{number}
```

**Parameters:**
- `number` (integer): Number to test for primality (must be ≥ 2)
- `model` (optional, query): Model to use ("clean_ml" or "hybrid_ml", default: "clean_ml")

**Example:**
```bash
curl "http://localhost:5001/predict/17?model=clean_ml"
```

**Response:**
```json
{
  "number": 17,
  "prediction": "prime",
  "confidence": 0.987,
  "model": "clean_ml",
  "accuracy_rating": "95.73%",
  "timestamp": "2025-01-15T10:30:00.000Z"
}
```

### Batch Prediction

Predict primality for multiple numbers.

```http
POST /predict
```

**Request Body:**
```json
{
  "numbers": [13, 15, 17, 19, 23],
  "model": "clean_ml"
}
```

**Parameters:**
- `numbers` (array): Array of integers to test (max 100)
- `model` (optional): Model to use ("clean_ml" or "hybrid_ml")

**Response:**
```json
{
  "model": "clean_ml",
  "results": [
    {
      "number": 13,
      "prediction": "prime",
      "confidence": 0.945,
      "model": "clean_ml",
      "accuracy_rating": "95.73%"
    },
    {
      "number": 15,
      "prediction": "composite",
      "confidence": 0.892,
      "model": "clean_ml",
      "accuracy_rating": "95.73%"
    }
  ],
  "batch_size": 5,
  "processed": 5,
  "errors": 0,
  "timestamp": "2025-01-15T10:30:00.000Z"
}
```

### API Information

Get detailed information about the API and available models.

```http
GET /info
```

**Response:**
```json
{
  "service": "primality_testing_api",
  "version": "1.0.0",
  "models": {
    "clean_ml": {
      "status": "available",
      "accuracy": "95.73%",
      "complexity": "O(log n)",
      "features": 31,
      "use_case": "Research and general screening"
    },
    "hybrid_ml": {
      "status": "available",
      "accuracy": "98.13%",
      "complexity": "O(k) k=20",
      "features": 71,
      "use_case": "High-reliability applications"
    }
  },
  "endpoints": {
    "GET /health": "Health check and available models",
    "GET /predict/<number>": "Predict primality for single number",
    "POST /predict": "Batch prediction (JSON: {\"numbers\": [...], \"model\": \"...\"})",
    "GET /info": "Detailed API information"
  },
  "limitations": [
    "Numbers must be integers ≥ 2",
    "Batch size limited to 100 numbers",
    "Models trained on numbers up to 20,000",
    "Probabilistic results (not deterministic like AKS/ECPP)"
  ]
}
```

## AIVA IDE API

The AIVA IDE provides AI-powered development environment with real-time collaboration.

**Base URL:** `http://localhost:3001/api`

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "aiva-ide-server",
  "version": "1.0.0",
  "websocket_clients": 5,
  "active_rooms": 3,
  "timestamp": "2025-01-15T10:30:00.000Z"
}
```

### File System Operations

#### List Files

```http
GET /files?path={path}
```

**Parameters:**
- `path` (optional): Directory path to list

**Response:**
```json
{
  "files": [
    {
      "name": "script.py",
      "path": "src/script.py",
      "type": "file",
      "size": 1024,
      "modified": "2025-01-15T10:30:00.000Z"
    },
    {
      "name": "utils",
      "path": "src/utils",
      "type": "directory",
      "size": null,
      "modified": "2025-01-15T10:25:00.000Z"
    }
  ],
  "path": "src"
}
```

#### Read File

```http
GET /files/{path}
```

**Response:**
```json
{
  "path": "src/script.py",
  "content": "print('Hello World')",
  "size": 1024,
  "modified": "2025-01-15T10:30:00.000Z"
}
```

#### Write File

```http
POST /files/{path}
```

**Request Body:**
```json
{
  "content": "print('Updated content')"
}
```

**Response:**
```json
{
  "success": true,
  "path": "src/script.py",
  "size": 1025,
  "modified": "2025-01-15T10:31:00.000Z"
}
```

#### Create File

```http
POST /files
```

**Request Body:**
```json
{
  "path": "new_file.py",
  "content": "print('New file')"
}
```

**Response:**
```json
{
  "success": true,
  "path": "new_file.py",
  "size": 18,
  "created": "2025-01-15T10:32:00.000Z"
}
```

#### Delete File

```http
DELETE /files/{path}
```

**Response:**
```json
{
  "success": true,
  "path": "old_file.py",
  "deleted": "2025-01-15T10:33:00.000Z"
}
```

### AI Operations

#### Chat with AI

```http
POST /chat
```

**Request Body:**
```json
{
  "messages": [
    {"role": "user", "content": "Help me debug this Python code"}
  ],
  "model": "gpt-4"
}
```

**Response:**
```json
{
  "message": "I'd be happy to help you debug your Python code...",
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 150,
    "total_tokens": 175
  },
  "model": "gpt-4"
}
```

#### Code Completion

```http
POST /complete
```

**Request Body:**
```json
{
  "code": "def fibonacci(n):",
  "language": "python",
  "context": "Complete this recursive Fibonacci function"
}
```

**Response:**
```json
{
  "completion": "\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)",
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 35,
    "total_tokens": 80
  }
}
```

### WebSocket Events

**WebSocket URL:** `ws://localhost:3001`

#### Join Room
```json
{
  "type": "join-room",
  "roomId": "project-alpha"
}
```

#### Leave Room
```json
{
  "type": "leave-room",
  "roomId": "project-alpha"
}
```

#### Code Change
```json
{
  "type": "code-change",
  "roomId": "project-alpha",
  "content": "print('Updated code')",
  "filePath": "main.py",
  "userId": "user123",
  "timestamp": "2025-01-15T10:30:00.000Z"
}
```

#### Cursor Movement
```json
{
  "type": "cursor-move",
  "roomId": "project-alpha",
  "position": {"line": 5, "column": 10},
  "userId": "user123"
}
```

#### User Events
```json
{
  "type": "user-joined",
  "userId": "user456",
  "timestamp": "2025-01-15T10:30:00.000Z"
}
```

```json
{
  "type": "user-left",
  "userId": "user456",
  "timestamp": "2025-01-15T10:31:00.000Z"
}
```

## CUDNT API

The CUDNT (CPU Universal Deep Neural Training) provides GPU-like operations on CPU systems.

### Python API

```python
from cudnt_enhanced_integration import CUDNT_Enhanced

# Initialize CUDNT
cudnt = CUDNT_Enhanced()

# Basic tensor operations
a = np.random.rand(100, 100)
b = np.random.rand(100, 100)

result = cudnt.tensor_add(a, b)  # GPU-like addition
result = cudnt.matrix_multiply(a, b)  # Accelerated matrix multiplication

# TensorFlow-like API
result = cudnt.tf_add(a, b)
result = cudnt.tf_matmul(a, b)
result = cudnt.tf_conv2d(input_tensor, kernel)
result = cudnt.tf_batch_norm(tensor)
result = cudnt.tf_relu(tensor)

# ML Pipeline Optimization
params = {'weights': np.random.rand(10, 5), 'bias': np.random.rand(5)}
result = cudnt.optimize_ml_pipeline(params, (X_train, y_train), epochs=10)

# System Status
status = cudnt.get_system_status()
```

### Key Methods

#### Tensor Operations
- `tensor_add(a, b)` - Parallel tensor addition
- `matrix_multiply(a, b)` - Optimized matrix multiplication
- `convolution_2d(input, kernel)` - 2D convolution
- `batch_normalize(tensor)` - Batch normalization
- `relu(tensor)` - ReLU activation

#### ML Operations
- `optimize_ml_pipeline(params, data, epochs)` - End-to-end ML training
- `gradient_step(params, gradients, lr)` - Parameter updates
- `_forward_pass(X, params)` - Neural network forward pass
- `_compute_loss_and_gradients(pred, target, params)` - Loss and gradients

#### TensorFlow Compatibility
- `tf_add(a, b)` - TensorFlow-style addition
- `tf_matmul(a, b)` - TensorFlow-style matrix multiplication
- `tf_conv2d(input, kernel)` - TensorFlow-style convolution
- `tf_batch_norm(tensor)` - TensorFlow-style batch normalization
- `tf_relu(tensor)` - TensorFlow-style ReLU

## Common Patterns

### Error Handling

All APIs follow consistent error handling patterns:

```json
{
  "error": "Descriptive error message",
  "details": "Additional error information",
  "timestamp": "2025-01-15T10:30:00.000Z"
}
```

### Rate Limiting

APIs include built-in rate limiting:
- WQRF: 100 requests per minute
- AIVA IDE: 500 requests per minute
- File operations: 50 operations per minute

### Pagination

For large result sets:

```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "limit": 50,
    "total": 250,
    "has_more": true
  }
}
```

## Authentication

### API Keys

For services requiring authentication:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.wallace-research.com/wqrf/predict/17
```

### Environment Variables

Required environment variables:

```bash
# OpenAI (for AIVA IDE)
OPENAI_API_KEY=sk-your-openai-key

# Database
POSTGRES_PASSWORD=your-db-password

# Application
PORT=3001
NODE_ENV=production
```

## WebSocket Security

WebSocket connections support:

- **Origin validation**: Only allowed origins can connect
- **Room isolation**: Users only see events from joined rooms
- **Rate limiting**: Message rate limits per connection
- **Timeout handling**: Automatic cleanup of stale connections

## Monitoring

All services expose metrics endpoints:

- **Health**: `/health` - Service health status
- **Metrics**: `/metrics` - Prometheus metrics (when enabled)
- **Info**: `/info` - Detailed service information

## Versioning

API versioning follows semantic versioning:

- **Major**: Breaking changes (e.g., v2.0.0)
- **Minor**: New features (e.g., v1.1.0)
- **Patch**: Bug fixes (e.g., v1.0.1)

Backwards compatibility is maintained within major versions.
