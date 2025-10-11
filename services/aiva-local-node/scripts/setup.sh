#!/bin/bash

# AIVA Local Node Setup Script
# This script sets up a complete local AIVA node environment

set -e

echo "🚀 Setting up AIVA Local Node..."

# Check prerequisites
command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3 is required but not installed. Aborting."; exit 1; }
command -v pip >/dev/null 2>&1 || { echo "❌ pip is required but not installed. Aborting."; exit 1; }
command -v git >/dev/null 2>&1 || { echo "❌ git is required but not installed. Aborting."; exit 1; }

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 11) else 1)'; then
    echo "✅ Python $PYTHON_VERSION is compatible"
else
    echo "⚠️  Python $PYTHON_VERSION detected. Python 3.11+ is recommended for optimal performance."
fi

# Create project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "📁 Working directory: $PROJECT_DIR"

# Create necessary directories
echo "📂 Creating directories..."
mkdir -p models data/chroma data/backups logs workspace

# Set up Python virtual environment
echo "🐍 Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Download models (optional - user can choose)
echo ""
echo "🤖 Model Setup Options:"
echo "1. Download Mixtral 8x7B (recommended for general tasks)"
echo "2. Download CodeLlama 13B (recommended for coding)"
echo "3. Download both models"
echo "4. Skip model download (you can download later)"
echo ""
read -p "Choose an option (1-4): " model_choice

case $model_choice in
    1)
        echo "📥 Downloading Mixtral 8x7B..."
        cd models
        git lfs install
        git clone https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
        cd ..
        ;;
    2)
        echo "📥 Downloading CodeLlama 13B..."
        cd models
        git lfs install
        git clone https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf
        cd ..
        ;;
    3)
        echo "📥 Downloading both models..."
        cd models
        git lfs install
        git clone https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
        git clone https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf
        cd ..
        ;;
    *)
        echo "⏭️  Skipping model download. You can download models later with:"
        echo "   cd models && git lfs install"
        echo "   git clone https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1"
        ;;
esac

# Create configuration files
echo "⚙️  Creating configuration files..."

# Environment configuration
cat > .env << EOF
# AIVA Local Node Configuration

# Inference Server
INFERENCE_HOST=0.0.0.0
INFERENCE_PORT=8000
MODEL_PATH=models/Mixtral-8x7B-Instruct-v0.1
MODEL_NAME=mixtral-8x7b-instruct
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=4096
DTYPE=auto

# vLLM Settings
TRUST_REMOTE_CODE=false

# AIVA Agent
AIVA_INFERENCE_URL=http://localhost:8000
AIVA_MODEL_NAME=mixtral-8x7b-instruct
AIVA_MAX_TOKENS=1024
AIVA_TEMPERATURE=0.7
AIVA_ENABLE_MEMORY=true
AIVA_ENABLE_TOOLS=true

# Development
NODE_ENV=development
LOG_LEVEL=INFO
EOF

# Create systemd service files (Linux only)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🔧 Creating systemd services..."

    # Inference server service
    cat > services/aiva-inference.service << EOF
[Unit]
Description=AIVA Local Inference Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/venv/bin
ExecStart=$PROJECT_DIR/venv/bin/python src/inference_server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    echo "📝 Systemd service created: services/aiva-inference.service"
    echo "   To install: sudo cp services/aiva-inference.service /etc/systemd/system/"
    echo "   To start: sudo systemctl start aiva-inference"
    echo "   To enable: sudo systemctl enable aiva-inference"
fi

# Create launch scripts
echo "📜 Creating launch scripts..."

# Start script
cat > start.sh << 'EOF'
#!/bin/bash
# AIVA Local Node Start Script

echo "🚀 Starting AIVA Local Node..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Start inference server
echo "🧠 Starting inference server..."
python src/inference_server.py &
INFERENCE_PID=$!

# Wait a moment for server to start
sleep 5

# Test inference server
echo "🧪 Testing inference server..."
curl -s http://localhost:8000/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Inference server is running on http://localhost:8000"
else
    echo "❌ Inference server failed to start"
    kill $INFERENCE_PID 2>/dev/null || true
    exit 1
fi

echo ""
echo "🎉 AIVA Local Node is running!"
echo "   Inference API: http://localhost:8000"
echo "   Health check: curl http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop..."

# Wait for interrupt
trap "echo '🛑 Stopping AIVA Local Node...'; kill $INFERENCE_PID 2>/dev/null || true; exit 0" INT
wait $INFERENCE_PID
EOF

chmod +x start.sh

# Test script
cat > test.sh << 'EOF'
#!/bin/bash
# AIVA Local Node Test Script

echo "🧪 Testing AIVA Local Node..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Test inference server
echo "Testing inference server health..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Inference server is healthy"
else
    echo "❌ Inference server is not running"
    echo "   Start it with: ./start.sh"
    exit 1
fi

# Test basic completion
echo "Testing basic completion..."
RESPONSE=$(curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "mixtral-8x7b-instruct",
        "prompt": "Hello, how are you?",
        "max_tokens": 50
    }')

if echo "$RESPONSE" | jq -e '.choices[0].text' > /dev/null 2>&1; then
    echo "✅ Basic completion works"
    echo "   Response preview: $(echo "$RESPONSE" | jq -r '.choices[0].text' | head -c 100)..."
else
    echo "❌ Basic completion failed"
    echo "   Response: $RESPONSE"
fi

# Test chat completion
echo "Testing chat completion..."
CHAT_RESPONSE=$(curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "mixtral-8x7b-instruct",
        "messages": [{"role": "user", "content": "Say hello in French"}],
        "max_tokens": 50
    }')

if echo "$CHAT_RESPONSE" | jq -e '.choices[0].message.content' > /dev/null 2>&1; then
    echo "✅ Chat completion works"
    echo "   Response: $(echo "$CHAT_RESPONSE" | jq -r '.choices[0].message.content')"
else
    echo "❌ Chat completion failed"
    echo "   Response: $CHAT_RESPONSE"
fi

# Test Python agent
echo "Testing AIVA agent..."
python -c "
import asyncio
from src.aiva_agent import create_aiva_agent

async def test():
    async with await create_aiva_agent() as agent:
        messages = [{'role': 'user', 'content': 'Hello, what can you do?'}]
        async for response in agent.chat(messages):
            if 'error' not in response:
                print('✅ Agent response:', response.get('response', '')[:100] + '...')
            else:
                print('❌ Agent error:', response['error'])
            break

asyncio.run(test())
"

echo ""
echo "🎉 All tests completed!"
EOF

chmod +x test.sh

# Create README
echo "📖 Creating documentation..."
cat > README.md << EOF
# AIVA Local Node

A complete local AI assistant powered by open-source models, with full control over your data and privacy.

## Quick Start

1. **Setup**: \`./scripts/setup.sh\`
2. **Start**: \`./start.sh\`
3. **Test**: \`./test.sh\`

## Architecture

- **Inference Server**: vLLM-powered API server
- **Vector Memory**: ChromaDB for conversation history and knowledge
- **Tool System**: Function calling for external integrations
- **Agent Core**: Orchestrates all components

## API Endpoints

- \`GET /health\` - Health check
- \`POST /v1/completions\` - Text completions
- \`POST /v1/chat/completions\` - Chat completions

## Configuration

Edit \`.env\` to configure:
- Model paths and settings
- Server ports and hosts
- Memory and tool options

## Models

Place your models in the \`models/\` directory:
- Mixtral 8x7B: General purpose reasoning
- CodeLlama 13B: Specialized for coding
- Custom fine-tuned models

## Development

Run individual components:
- Inference: \`python src/inference_server.py\`
- Agent: \`python src/aiva_agent.py\`
- Tests: \`python -m pytest tests/\`

## Integration

Use with Cursor IDE, custom applications, or any OpenAI-compatible client by pointing to:
- Base URL: \`http://localhost:8000/v1\`
- Model: \`mixtral-8x7b-instruct\`

## Troubleshooting

1. **CUDA Issues**: Check GPU compatibility and drivers
2. **Memory Issues**: Reduce \`GPU_MEMORY_UTILIZATION\` in .env
3. **Model Loading**: Ensure models are properly downloaded
4. **Port Conflicts**: Change ports in .env if needed

## Security

- All data stays local
- No external API calls unless configured
- Configurable access controls
- Encrypted storage options

## License

MIT License - Full control over your AI assistant.
EOF

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Review and edit .env configuration"
echo "   2. Download models if you skipped that step"
echo "   3. Run: ./start.sh"
echo "   4. Test: ./test.sh"
echo ""
echo "📚 Documentation: README.md"
echo "⚙️  Configuration: .env"
echo "🚀 Launch script: start.sh"
echo ""
echo "Happy chatting with your local AIVA! 🤖"
