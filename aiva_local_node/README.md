# AIVA Local Node - Modular AI Assistant

A complete, modular AI assistant that runs entirely on your local infrastructure with full data control.

## 🏗️ Architecture Overview

This project has been reorganized into a clean, modular architecture:

```
aiva_local_node/
├── src/                          # Core Python modules
│   ├── __init__.py              # Package exports
│   ├── config.py                # Configuration management
│   ├── auth.py                  # Authentication utilities
│   ├── models.py                # Pydantic data models
│   ├── inference_engine.py      # vLLM model inference
│   ├── retrieval.py             # Context retrieval & embeddings
│   ├── identity.py              # Identity pack management
│   ├── tools.py                 # Tool definitions & execution
│   ├── middleware.py            # Request logging & rate limiting
│   └── inference_server.py     # FastAPI server (endpoints only)
├── scripts/                      # Setup and utility scripts
├── models/                       # Downloaded AI models
├── data/                         # Persistent data storage
├── logs/                         # System logs
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🎯 Key Features

### **Privacy & Control**
- ✅ **Zero Data Exfiltration**: Everything runs locally
- ✅ **No External APIs**: No calls to OpenAI or cloud services
- ✅ **Full Data Ownership**: Your conversations stay private
- ✅ **Auditable**: Complete logging and monitoring

### **AI Capabilities**
- ✅ **Chat Interface**: Natural language AI assistant
- ✅ **Code Completion**: Intelligent inline suggestions
- ✅ **Tool Integration**: File operations, web search, code execution
- ✅ **Context Memory**: Long-term conversation memory
- ✅ **Identity Preservation**: Behavioral testing and consistency

### **Production Ready**
- ✅ **Modular Design**: Clean separation of concerns
- ✅ **Rate Limiting**: Request throttling and abuse prevention
- ✅ **Error Handling**: Comprehensive error management
- ✅ **Monitoring**: Health checks and performance metrics

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.11+
python3 --version

# Git for model downloads
git --version

# Optional: CUDA for GPU acceleration
nvidia-smi  # Check GPU availability
```

### Installation
```bash
# Clone and setup
cd aiva_local_node
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create environment file
cp .env.example .env
# Edit .env with your configuration
```

### Download Models
```bash
# Option 1: Mixtral 8x7B (Recommended - General purpose)
cd models
git lfs install
git clone https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1

# Option 2: CodeLlama 13B (Coding focused)
git clone https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf
```

### Start the Server
```bash
# Start AIVA local node
python -m src.inference_server

# Or use the convenience script
./scripts/start.sh

# Server will be available at http://localhost:8000
```

### Install Identity Pack
```bash
# Install the AIVA personality and behavioral tests
curl -X POST http://localhost:8000/v1/identity/install \
  -H "Content-Type: application/json" \
  -d '{"base_dir": ".", "overwrite": false}'
```

## 📁 Module Guide

### `config.py` - Configuration Management
Centralized configuration for all components:
```python
from src.config import AIVA_SYSTEM_PROMPT, API_KEY, CONTEXT_DIR
```

### `models.py` - Data Models
Pydantic models for API requests/responses:
```python
from src.models import ChatCompletionRequest, CompletionResponse
```

### `inference_engine.py` - Model Inference
Handles vLLM model loading and text generation:
```python
from src.inference_engine import generate_chat_completion
response = await generate_chat_completion(messages)
```

### `retrieval.py` - Context Memory
Vector-based context retrieval and knowledge search:
```python
from src.retrieval import retrieve_context
context = retrieve_context("python error handling")
```

### `tools.py` - Tool System
Function calling capabilities for external operations:
```python
from src.tools import execute_tool
result = await execute_tool("read_file", {"path": "example.txt"})
```

### `identity.py` - Personality Management
Identity pack installation and behavioral testing:
```python
from src.identity import install_identity_pack, eval_identity_checks
```

### `auth.py` & `middleware.py` - Security
Authentication and request processing middleware.

## 🔧 Configuration

### Environment Variables (`.env`)
```bash
# Model Configuration
MODEL_PATH=models/Mixtral-8x7B-Instruct-v0.1
MODEL_NAME=mixtral-8x7b-instruct

# Server Configuration
HOST=0.0.0.0
PORT=8000
API_KEY=your_secure_api_key_here

# Context & Memory
CONTEXT_DIR=./context
AIVA_CONTEXT_DIR=./context

# Security
AIVA_API_KEY=your_api_key
AIVA_LOG_REQUESTS=true
AIVA_RATE_LIMIT_PER_MIN=120

# Performance
GPU_MEMORY_UTILIZATION=0.9
TENSOR_PARALLEL_SIZE=1
```

### Identity Pack Configuration
```bash
# Custom system prompt
AIVA_SYSTEM_PROMPT="Your custom AIVA personality..."

# Context directory for knowledge base
AIVA_CONTEXT_DIR=./my_knowledge

# Behavioral testing
# (Edit src/identity.py AIVA_BEHAV_TESTS for custom tests)
```

## 🛠️ API Endpoints

### Core Inference
- `POST /v1/chat/completions` - Chat completion
- `POST /v1/completions` - Text completion
- `POST /v1/embeddings` - Text embeddings

### Tool System
- `POST /v1/tools/call` - Execute tools directly
- `GET /v1/tools` - List available tools

### Identity Management
- `POST /v1/identity/install` - Install identity pack
- `GET /v1/identity/info` - Get identity status
- `POST /v1/identity/check` - Run behavioral tests

### Memory System
- `POST /v1/memory/reload` - Reload context corpus
- `GET /v1/memory/info` - Get memory statistics

### System
- `GET /health` - Health check
- `GET /stats` - System statistics
- `GET /v1/models` - List available models

## 🔒 Security Features

### Authentication
```bash
# Set API key for request authentication
AIVA_API_KEY=your_secure_key_here

# All endpoints require: Authorization: Bearer your_key
```

### Rate Limiting
```bash
# Configure per-IP rate limits
AIVA_RATE_LIMIT_PER_MIN=120

# Automatic request throttling
```

### Request Logging
```bash
# Enable detailed request logging
AIVA_LOG_REQUESTS=true

# Control log detail level
AIVA_LOG_MAX_BODY=2000
```

## 🧪 Testing & Verification

### Health Check
```bash
curl http://localhost:8000/health
# Should return: {"status": "healthy", "model_loaded": true, ...}
```

### Identity Verification
```bash
# Install identity pack
curl -X POST http://localhost:8000/v1/identity/install \
  -H "Authorization: Bearer your_key"

# Run behavioral tests
curl -X POST http://localhost:8000/v1/identity/check \
  -H "Authorization: Bearer your_key"
```

### Tool Testing
```bash
# Test tool execution
curl -X POST http://localhost:8000/v1/tools/call \
  -H "Authorization: Bearer your_key" \
  -H "Content-Type: application/json" \
  -d '{"tool": "list_dir", "arguments": {"path": "."}}'
```

## 🚀 Integration Examples

### Python Client
```python
import aiohttp
import asyncio

async def chat_with_aiva(message: str) -> str:
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": "mixtral-8x7b-instruct",
            "messages": [{"role": "user", "content": message}],
            "max_tokens": 512
        }
        headers = {"Authorization": "Bearer your_api_key"}

        async with session.post(
            "http://localhost:8000/v1/chat/completions",
            json=payload,
            headers=headers
        ) as resp:
            result = await resp.json()
            return result["choices"][0]["message"]["content"]

# Usage
response = asyncio.run(chat_with_aiva("Hello AIVA!"))
print(response)
```

### Cursor Extension Integration
The AIVA Cursor extension automatically connects to your local node:
```typescript
// In extension settings
"aiva.server.url": "http://localhost:8000",
"aiva.server.apiPath": "/v1/chat/completions"
```

## 📊 Performance Tuning

### GPU Optimization
```bash
# Adjust GPU memory usage
GPU_MEMORY_UTILIZATION=0.8

# Multi-GPU setup
TENSOR_PARALLEL_SIZE=2

# Model precision
DTYPE=float16  # or auto, int8, etc.
```

### Memory Management
```bash
# Context limits
MAX_CONTEXT_CHARS=4000
MAX_CONTEXT_PASSAGES=6

# Tool size limits
WRITE_MAX_BYTES=262144  # 256KB
WEB_MAX_BYTES=524288    # 512KB
```

## 🐛 Troubleshooting

### Common Issues

**Model Loading Fails**
```bash
# Check model path
ls -la $MODEL_PATH

# Verify HuggingFace access
export HF_TOKEN=your_token

# Clear cache
rm -rf ~/.cache/huggingface
```

**High Memory Usage**
```bash
# Reduce GPU memory utilization
export GPU_MEMORY_UTILIZATION=0.7

# Use smaller context window
export MAX_MODEL_LEN=2048
```

**Slow Inference**
```bash
# Enable Flash Attention
# (Check vLLM documentation)

# Use tensor parallelism
export TENSOR_PARALLEL_SIZE=2
```

**Tool Errors**
```bash
# Check file permissions
ls -la $CONTEXT_DIR

# Verify tool parameters
curl http://localhost:8000/v1/tools
```

## 🔄 Development

### Running Tests
```bash
# Unit tests
python -m pytest tests/ -v

# Integration tests
python tests/integration_test.py

# Load testing
python tests/load_test.py
```

### Adding New Tools
```python
# In src/tools.py
TOOL_SPECS["my_tool"] = {
    "description": "My custom tool",
    "parameters": {"type": "object", "properties": {"param": {"type": "string"}}}
}

async def execute_tool(tool: str, args: Dict[str, Any]) -> str:
    if tool == "my_tool":
        # Implement tool logic
        return f"Result: {args['param']}"
```

### Custom Identity
```python
# Modify src/identity.py
AIVA_SYSTEM_PROMPT = "Your custom personality..."

# Add custom behavioral tests
AIVA_BEHAV_TESTS = """{
  "version": 1,
  "tests": [
    {"name":"custom_test","prompt":"Test prompt","must_include":["expected"],"must_avoid":["avoided"]}
  ]
}"""
```

## 🚀 Scaling & Production

### Docker Deployment
```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3-pip
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

EXPOSE 8000
CMD ["python", "-m", "src.inference_server"]
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aiva-node
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: aiva
        image: aiva-local:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/models/Mixtral-8x7B-Instruct-v0.1"
        volumeMounts:
        - name: model-storage
          mountPath: /models
        resources:
          limits:
            nvidia.com/gpu: 1
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## 📄 License

MIT License - Full control over your AI assistant.

## 🙏 Acknowledgments

Built with ❤️ using modern AI infrastructure and a commitment to user privacy and control.

---

**Ready to experience AI sovereignty? Your local AIVA node awaits! 🚀**
