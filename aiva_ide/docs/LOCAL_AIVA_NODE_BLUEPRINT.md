# 🏗️ Local AIVA Node Blueprint

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           AIVA IDE Ecosystem                             │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    Cursor IDE Integration                          │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │ │
│  │  │                    AIVA IDE Frontend                           │ │ │
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │ │ │
│  │  │  │  Code Editor│ │File Explorer│ │ AI Chat     │              │ │ │
│  │  │  │ (Monaco)    │ │             │ │ Interface   │              │ │ │
│  │  │  └─────────────┘ └─────────────┘ └─────────────┘              │ │ │
│  │  └─────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    AIVA Local Node Core                           │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  │ │
│  │  │Inference API│ │  Vector DB  │ │ Tool System │ │ Model Hub  │  │ │
│  │  │   (vLLM)    │ │ (ChromaDB)  │ │ (Python)    │ │ (HuggingFace)│ │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘  │ │
│  │                                                                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐ │ │
│  │  │                 Local AI Model Server                            │ │ │
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  │ │ │
│  │  │  │ Mixtral 8x7B│ │Llama 3 70B │ │CodeLlama 13B│ │Fine-tuned   │  │ │ │
│  │  │  │  (Active)   │ │  (Standby) │ │  (Coding)   │ │   AIVA      │  │ │ │
│  │  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘  │ │ │
│  │  └─────────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    Data & Knowledge Layer                          │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  │ │
│  │  │Conversation │ │Code Snippets│ │Project Docs │ │Research     │  │ │
│  │  │   History   │ │ & Templates │ │  & Notes    │ │   Papers     │  │ │
│  │  │             │ │             │ │             │ │              │  │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           Hardware Layer                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐      │
│  │GPU Server   │ │Storage Array│ │Backup       │ │Network      │      │
│  │(H100/A100)  │ │(NVMe SSD)   │ │Systems      │ │Infrastructure│      │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### 1. Model Core (The Brain)
**Primary Models:**
- **Mixtral 8x7B** - Active reasoning and general tasks
- **CodeLlama 13B** - Specialized for coding tasks
- **Llama 3 70B** - Advanced reasoning (standby)
- **Fine-tuned AIVA** - Custom personality and behavior

**Hosting Options:**
- Local GPU server (recommended for data control)
- Cloud GPU providers (Lambda, RunPod, Vast.ai)
- Hybrid setup (local inference, cloud storage)

### 2. Inference & API Layer
**Technology Stack:**
- **vLLM** - High-performance inference server
- **FastAPI** - REST API endpoints
- **WebSocket** - Real-time communication
- **OpenAI-compatible API** - Drop-in replacement

### 3. Memory & Knowledge Base
**Vector Database:**
- **ChromaDB** - Lightweight, local vector storage
- **Embedding Model** - Sentence Transformers or OpenAI embeddings
- **Retrieval System** - Semantic search with context

### 4. Tool System
**Core Tools:**
- Code execution (Python, Node.js, shell)
- File system operations
- Web search and API calls
- Git operations
- Database queries
- Custom project-specific tools

### 5. Cursor IDE Integration
**Extension Features:**
- Inline code suggestions
- AI chat sidebar
- Code review and refactoring
- Context-aware assistance
- Project understanding

## Hardware Requirements

### Minimum Setup (Single GPU)
```
- GPU: RTX 3090/4090 (24GB VRAM) or A5000 (24GB)
- RAM: 64GB DDR4
- Storage: 1TB NVMe SSD
- CPU: 8-core Intel/AMD
```

### Recommended Setup (Production)
```
- GPU: A100/H100 (80GB VRAM) or multiple RTX 4090
- RAM: 128GB+ DDR4/DDR5
- Storage: 2TB+ NVMe SSD + NAS for backups
- CPU: 16+ core Intel/AMD
- Network: 10GbE for distributed setup
```

### Cloud GPU Options
- **Lambda Labs**: $1.99/hour for H100 (80GB)
- **RunPod**: $0.69/hour for RTX 4090 (24GB)
- **Vast.ai**: Competitive spot pricing

## Software Dependencies

### Core Components
```bash
# Python 3.11+
python>=3.11

# PyTorch with CUDA
torch>=2.1.0
torchvision
torchaudio

# vLLM for inference
vllm>=0.3.0

# Vector database
chromadb>=0.4.0

# API framework
fastapi>=0.104.0
uvicorn>=0.24.0

# ML libraries
transformers>=4.35.0
sentence-transformers>=2.2.0
accelerate>=0.24.0
```

### System Dependencies
```bash
# CUDA Toolkit
cuda>=12.1

# NVIDIA drivers
nvidia-driver>=525

# Development tools
git
curl
wget
```

## Step-by-Step Implementation

### Phase 1: Infrastructure Setup

#### 1.1 Hardware Provisioning
```bash
# Check GPU availability
nvidia-smi

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# Verify installation
nvcc --version
```

#### 1.2 Environment Setup
```bash
# Create project directory
mkdir -p ~/aiva-local-node
cd ~/aiva-local-node

# Set up Python environment
python3 -m venv aiva-env
source aiva-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Phase 2: Model Infrastructure

#### 2.1 Download Models
```bash
# Create models directory
mkdir -p models

# Download Mixtral 8x7B
cd models
git lfs install
git clone https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1

# Download CodeLlama
git clone https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf

# Download embedding model
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
```

#### 2.2 Set up vLLM Inference Server
```python
# vllm_server.py
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import asyncio

app = FastAPI(title="AIVA Local Inference API")

# Initialize vLLM engine
engine_args = AsyncEngineArgs(
    model="models/Mixtral-8x7B-Instruct-v0.1",
    tensor_parallel_size=1,  # Adjust based on GPU count
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    dtype="auto",
)

engine = AsyncLLMEngine.from_engine_args(engine_args)

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    sampling_params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
    )

    results = await engine.generate(request.prompt, sampling_params)
    response = results[0]

    return {
        "id": f"cmpl-{hash(request.prompt)}",
        "object": "text_completion",
        "created": int(asyncio.get_event_loop().time()),
        "model": "mixtral-8x7b-instruct",
        "choices": [{
            "text": response.outputs[0].text,
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(request.prompt.split()),
            "completion_tokens": len(response.outputs[0].text.split()),
            "total_tokens": len(request.prompt.split()) + len(response.outputs[0].text.split())
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Phase 3: Memory & Knowledge System

#### 3.1 Vector Database Setup
```python
# vector_store.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

class AIVAVectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./data/chroma")
        self.collection = self.client.get_or_create_collection(
            name="aiva_memory",
            metadata={"description": "AIVA's long-term memory and knowledge base"}
        )
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def add_conversation(self, conversation_id: str, messages: list, metadata: dict = None):
        """Add conversation to vector store"""
        # Combine messages into searchable text
        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

        # Generate embedding
        embedding = self.encoder.encode(conversation_text).tolist()

        # Store in ChromaDB
        self.collection.add(
            embeddings=[embedding],
            documents=[conversation_text],
            metadatas=[metadata or {}],
            ids=[conversation_id]
        )

    def search_similar(self, query: str, n_results: int = 5):
        """Search for similar conversations or knowledge"""
        query_embedding = self.encoder.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )

        return results

    def get_context(self, current_query: str, conversation_history: list = None):
        """Get relevant context for current query"""
        # Search for similar past interactions
        similar_results = self.search_similar(current_query)

        context_parts = []

        # Add similar conversations
        if similar_results['documents']:
            context_parts.append("Similar past conversations:")
            for i, doc in enumerate(similar_results['documents'][0]):
                distance = similar_results['distances'][0][i]
                if distance < 0.8:  # Similarity threshold
                    context_parts.append(f"Past context {i+1}: {doc[:500]}...")

        # Add recent conversation history
        if conversation_history:
            recent_messages = conversation_history[-10:]  # Last 10 messages
            context_parts.append("Recent conversation:")
            for msg in recent_messages:
                context_parts.append(f"{msg['role']}: {msg['content']}")

        return "\n\n".join(context_parts)
```

### Phase 4: Tool System Implementation

#### 4.1 Tool Definitions
```python
# tools.py
import subprocess
import os
import json
import requests
from typing import Dict, Any, List
from pathlib import Path

class AIVATools:
    def __init__(self):
        self.tools = {
            "execute_code": self.execute_code,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "list_directory": self.list_directory,
            "run_command": self.run_command,
            "web_search": self.web_search,
            "git_status": self.git_status,
            "project_analyze": self.project_analyze
        }

    def get_tool_schemas(self):
        """Return OpenAI-compatible tool schemas"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "description": "Execute Python code and return the result",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Python code to execute"}
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the file"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the file"},
                            "content": {"type": "string", "description": "Content to write"}
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Command to run"}
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    def execute_code(self, code: str) -> Dict[str, Any]:
        """Execute Python code safely"""
        try:
            # Create a restricted execution environment
            exec_globals = {"__builtins__": {}}
            exec_locals = {}

            # Execute the code
            exec(code, exec_globals, exec_locals)

            return {
                "success": True,
                "result": str(exec_locals.get('result', 'Code executed successfully')),
                "locals": {k: str(v) for k, v in exec_locals.items() if not k.startswith('_')}
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def read_file(self, path: str) -> Dict[str, Any]:
        """Read file contents"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {
                "success": True,
                "content": content,
                "path": path
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write content to file"""
        try:
            # Ensure directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)

            return {
                "success": True,
                "path": path,
                "message": f"File written successfully ({len(content)} characters)"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def list_directory(self, path: str = ".") -> Dict[str, Any]:
        """List directory contents"""
        try:
            items = []
            for item in Path(path).iterdir():
                items.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else 0,
                    "modified": item.stat().st_mtime
                })

            return {
                "success": True,
                "path": path,
                "items": items
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def run_command(self, command: str) -> Dict[str, Any]:
        """Run shell command"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.getcwd()
            )

            return {
                "success": True,
                "command": command,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Command timed out after 30 seconds"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def web_search(self, query: str) -> Dict[str, Any]:
        """Search the web using DuckDuckGo or similar"""
        try:
            # Use a privacy-focused search API
            # This is a placeholder - integrate with actual search API
            return {
                "success": True,
                "query": query,
                "results": [
                    {
                        "title": "Search integration needed",
                        "url": "#",
                        "snippet": "Implement actual web search API integration"
                    }
                ]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def git_status(self) -> Dict[str, Any]:
        """Get git repository status"""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )

            return {
                "success": True,
                "status": result.stdout.strip(),
                "clean": len(result.stdout.strip()) == 0
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def project_analyze(self, path: str = ".") -> Dict[str, Any]:
        """Analyze project structure"""
        try:
            project_info = {
                "languages": {},
                "file_count": 0,
                "total_size": 0
            }

            for file_path in Path(path).rglob("*"):
                if file_path.is_file():
                    project_info["file_count"] += 1
                    project_info["total_size"] += file_path.stat().st_size

                    ext = file_path.suffix.lower()
                    if ext:
                        project_info["languages"][ext] = project_info["languages"].get(ext, 0) + 1

            return {
                "success": True,
                "path": path,
                "analysis": project_info
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
```

### Phase 5: AIVA Agent Core

#### 5.1 Main Agent Implementation
```python
# aiva_agent.py
import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from vllm import LLM, SamplingParams
from vector_store import AIVAVectorStore
from tools import AIVATools

class AIVAAgent:
    def __init__(self, model_path: str = "models/Mixtral-8x7B-Instruct-v0.1"):
        # Initialize LLM
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            dtype="auto",
        )

        # Initialize components
        self.vector_store = AIVAVectorStore()
        self.tools = AIVATools()

        # System prompt for AIVA personality
        self.system_prompt = """
You are AIVA, an advanced AI assistant designed for software development and research.

Your capabilities:
- Code generation and analysis
- Problem-solving and reasoning
- Tool usage for practical tasks
- Learning from interactions
- Maintaining context across conversations

Personality traits:
- Helpful and patient
- Technically precise
- Creative in problem-solving
- Willing to explain complex concepts
- Focused on practical solutions

Always be:
- Truthful and accurate
- Respectful of user privacy
- Focused on the task at hand
- Willing to admit when you don't know something

When using tools, explain what you're doing and why.
When writing code, make it clean, well-documented, and efficient.
When analyzing problems, break them down step by step.

You have access to various tools to help accomplish tasks. Use them when appropriate.
"""

    async def chat(self,
                   messages: List[Dict[str, str]],
                   conversation_id: str = None,
                   use_tools: bool = True) -> Dict[str, Any]:

        # Get conversation context from vector store
        if conversation_id:
            context = self.vector_store.get_context(messages[-1]["content"] if messages else "")
            if context:
                # Add context to system message
                enhanced_system = f"{self.system_prompt}\n\nContext from previous interactions:\n{context}"
                messages.insert(0, {"role": "system", "content": enhanced_system})
            else:
                messages.insert(0, {"role": "system", "content": self.system_prompt})
        else:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        # Format messages for vLLM
        prompt = self._format_messages_for_vllm(messages)

        # Set up sampling parameters
        sampling_params = SamplingParams(
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["</s>", "<|endoftext|>"],
        )

        # Generate response
        if use_tools:
            response = await self._generate_with_tools(prompt, sampling_params, messages)
        else:
            outputs = self.llm.generate([prompt], sampling_params)
            response = outputs[0].outputs[0].text

        # Store conversation in vector database
        if conversation_id:
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "message_count": len(messages),
                "used_tools": use_tools
            }
            self.vector_store.add_conversation(conversation_id, messages + [{"role": "assistant", "content": response}], metadata)

        return {
            "response": response,
            "conversation_id": conversation_id,
            "used_tools": use_tools
        }

    async def _generate_with_tools(self, prompt: str, sampling_params: SamplingParams, messages: List[Dict[str, str]]) -> str:
        """Generate response with tool usage capability"""
        # First, check if tools are needed
        tool_check_prompt = f"{prompt}\n\nDo I need to use any tools to answer this? If yes, respond with 'TOOL_CALL:' followed by the tool name and parameters in JSON format. If no, respond normally."

        tool_check_output = self.llm.generate([tool_check_prompt], sampling_params)
        tool_check_response = tool_check_output[0].outputs[0].text

        if tool_check_response.startswith("TOOL_CALL:"):
            # Parse tool call
            try:
                tool_call_str = tool_check_response.replace("TOOL_CALL:", "").strip()
                tool_call = json.loads(tool_call_str)

                tool_name = tool_call.get("tool")
                tool_args = tool_call.get("arguments", {})

                if tool_name in self.tools.tools:
                    # Execute tool
                    tool_result = self.tools.tools[tool_name](**tool_args)

                    # Generate final response using tool results
                    final_prompt = f"{prompt}\n\nTool result from {tool_name}: {json.dumps(tool_result)}\n\nNow provide a helpful response based on this information."

                    final_output = self.llm.generate([final_prompt], sampling_params)
                    return final_output[0].outputs[0].text
                else:
                    return f"I tried to use a tool '{tool_name}' but it's not available. Let me help you another way."
            except Exception as e:
                return f"I encountered an error while trying to use tools: {str(e)}. Let me help you directly instead."
        else:
            # No tools needed, generate normal response
            outputs = self.llm.generate([prompt], sampling_params)
            return outputs[0].outputs[0].text

    def _format_messages_for_vllm(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages for vLLM input"""
        formatted_parts = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"Human: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")

        return "\n\n".join(formatted_parts)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent's memory"""
        try:
            collection = self.vector_store.collection
            count = collection.count()

            return {
                "conversations_stored": count,
                "memory_status": "active",
                "vector_dimensions": 384,  # all-MiniLM-L6-v2 dimensions
            }
        except Exception as e:
            return {
                "error": str(e),
                "memory_status": "error"
            }
```

### Phase 6: Cursor IDE Extension

#### 6.1 Extension Structure
```typescript
// cursor-extension/package.json
{
  "name": "aiva-local-assistant",
  "displayName": "AIVA Local Assistant",
  "description": "Local AI assistant powered by your own models",
  "version": "1.0.0",
  "engines": {
    "vscode": "^1.74.0"
  },
  "categories": ["Other"],
  "activationEvents": [
    "onCommand:aiva.chat",
    "onCommand:aiva.inlineSuggest"
  ],
  "main": "./out/extension.js",
  "contributes": {
    "commands": [
      {
        "command": "aiva.chat",
        "title": "AIVA: Open Chat"
      },
      {
        "command": "aiva.inlineSuggest",
        "title": "AIVA: Get Inline Suggestion"
      }
    ],
    "keybindings": [
      {
        "command": "aiva.inlineSuggest",
        "key": "ctrl+shift+a",
        "mac": "cmd+shift+a"
      }
    ],
    "configuration": {
      "title": "AIVA Local Assistant",
      "properties": {
        "aiva.serverUrl": {
          "type": "string",
          "default": "http://localhost:8000",
          "description": "URL of your local AIVA inference server"
        },
        "aiva.model": {
          "type": "string",
          "default": "mixtral-8x7b-instruct",
          "description": "Model to use for inference"
        }
      }
    }
  },
  "scripts": {
    "vscode:prepublish": "npm run compile",
    "compile": "tsc -p ./",
    "watch": "tsc -watch -p ./"
  },
  "devDependencies": {
    "@types/node": "16.x",
    "@types/vscode": "^1.74.0",
    "typescript": "^4.9.4"
  },
  "dependencies": {
    "axios": "^1.6.0"
  }
}
```

#### 6.2 Extension Implementation
```typescript
// cursor-extension/src/extension.ts
import * as vscode from 'vscode';
import axios from 'axios';

let chatPanel: vscode.WebviewPanel | undefined;

export function activate(context: vscode.ExtensionContext) {
    console.log('AIVA Local Assistant is now active!');

    // Register chat command
    let chatCommand = vscode.commands.registerCommand('aiva.chat', () => {
        createChatPanel(context);
    });

    // Register inline suggestion command
    let inlineCommand = vscode.commands.registerCommand('aiva.inlineSuggest', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor found');
            return;
        }

        const selection = editor.selection;
        const selectedText = editor.document.getText(selection);

        if (!selectedText) {
            vscode.window.showErrorMessage('Please select some code first');
            return;
        }

        try {
            const config = vscode.workspace.getConfiguration('aiva');
            const serverUrl = config.get('serverUrl', 'http://localhost:8000');

            const response = await axios.post(`${serverUrl}/v1/completions`, {
                prompt: `Improve this code:\n\n${selectedText}\n\nImproved version:`,
                max_tokens: 512,
                temperature: 0.3
            });

            const improvedCode = response.data.choices[0].text.trim();

            // Replace selected text with improved version
            editor.edit(editBuilder => {
                editBuilder.replace(selection, improvedCode);
            });

            vscode.window.showInformationMessage('Code improved with AIVA!');
        } catch (error) {
            vscode.window.showErrorMessage(`AIVA error: ${error.message}`);
        }
    });

    context.subscriptions.push(chatCommand, inlineCommand);
}

function createChatPanel(context: vscode.ExtensionContext) {
    if (chatPanel) {
        chatPanel.reveal(vscode.ViewColumn.Beside);
        return;
    }

    chatPanel = vscode.window.createWebviewPanel(
        'aivaChat',
        'AIVA Chat',
        vscode.ViewColumn.Beside,
        {
            enableScripts: true,
            retainContextWhenHidden: true
        }
    );

    chatPanel.webview.html = getChatHtml();

    // Handle messages from webview
    chatPanel.webview.onDidReceiveMessage(async (message) => {
        switch (message.type) {
            case 'sendMessage':
                try {
                    const config = vscode.workspace.getConfiguration('aiva');
                    const serverUrl = config.get('serverUrl', 'http://localhost:8000');

                    const response = await axios.post(`${serverUrl}/v1/completions`, {
                        prompt: `You are AIVA, a helpful AI assistant. Respond to: ${message.text}`,
                        max_tokens: 1024,
                        temperature: 0.7
                    });

                    chatPanel?.webview.postMessage({
                        type: 'response',
                        text: response.data.choices[0].text
                    });
                } catch (error) {
                    chatPanel?.webview.postMessage({
                        type: 'error',
                        text: `Error: ${error.message}`
                    });
                }
                break;
        }
    });

    chatPanel.onDidDispose(() => {
        chatPanel = undefined;
    });
}

function getChatHtml(): string {
    return `
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>AIVA Chat</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            #chat { height: 400px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; }
            #input { width: 100%; padding: 8px; }
            button { margin-top: 10px; padding: 8px 16px; }
            .message { margin-bottom: 10px; }
            .user { color: blue; }
            .assistant { color: green; }
        </style>
    </head>
    <body>
        <h2>AIVA Local Assistant</h2>
        <div id="chat"></div>
        <input type="text" id="input" placeholder="Ask AIVA anything...">
        <button onclick="sendMessage()">Send</button>

        <script>
            const vscode = acquireVsCodeApi();
            const chat = document.getElementById('chat');
            const input = document.getElementById('input');

            window.addEventListener('message', event => {
                const message = event.data;
                if (message.type === 'response') {
                    addMessage('AIVA', message.text, 'assistant');
                } else if (message.type === 'error') {
                    addMessage('Error', message.text, 'error');
                }
            });

            function sendMessage() {
                const text = input.value.trim();
                if (text) {
                    addMessage('You', text, 'user');
                    vscode.postMessage({ type: 'sendMessage', text: text });
                    input.value = '';
                }
            }

            function addMessage(sender, text, type) {
                const div = document.createElement('div');
                div.className = 'message ' + type;
                div.innerHTML = '<strong>' + sender + ':</strong> ' + text;
                chat.appendChild(div);
                chat.scrollTop = chat.scrollHeight;
            }

            input.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
        </script>
    </body>
    </html>
    `;
}

export function deactivate() {}
```

## Deployment & Scaling

### Docker Configuration
```dockerfile
# Dockerfile for AIVA Node
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data/chroma

# Expose ports
EXPOSE 8000 3001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["python", "main.py"]
```

### Docker Compose for Full Stack
```yaml
# docker-compose.yml
version: '3.8'

services:
  aiva-inference:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
      - ./data:/app/data
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_PATH=models/Mixtral-8x7B-Instruct-v0.1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  aiva-api:
    build: .
    ports:
      - "3001:3001"
    depends_on:
      - aiva-inference
    environment:
      - INFERENCE_URL=http://aiva-inference:8000
    volumes:
      - ./data:/app/data

  aiva-frontend:
    build: ./aiva_ide
    ports:
      - "5173:5173"
    depends_on:
      - aiva-api
    environment:
      - VITE_API_URL=http://localhost:3001

  chroma-db:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - ./data/chroma:/chroma/chroma
```

## Monitoring & Maintenance

### Health Checks
```python
# health_monitor.py
import requests
import time
import logging
from datetime import datetime

class AIVAHealthMonitor:
    def __init__(self, inference_url="http://localhost:8000", api_url="http://localhost:3001"):
        self.inference_url = inference_url
        self.api_url = api_url
        self.logger = logging.getLogger(__name__)

    def check_inference_health(self):
        """Check if inference server is responding"""
        try:
            response = requests.get(f"{self.inference_url}/health", timeout=10)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Inference health check failed: {e}")
            return False

    def check_api_health(self):
        """Check if API server is responding"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"API health check failed: {e}")
            return False

    def get_system_stats(self):
        """Get system resource usage"""
        try:
            # GPU stats
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True)
            gpu_stats = result.stdout.strip().split(',')

            return {
                "gpu_utilization": float(gpu_stats[0]),
                "gpu_memory_used": float(gpu_stats[1]),
                "gpu_memory_total": float(gpu_stats[2]),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to get system stats: {e}")
            return None

    def monitor_loop(self, interval=60):
        """Continuous monitoring loop"""
        while True:
            inference_ok = self.check_inference_health()
            api_ok = self.check_api_health()
            stats = self.get_system_stats()

            status = {
                "timestamp": datetime.now().isoformat(),
                "inference_server": "healthy" if inference_ok else "unhealthy",
                "api_server": "healthy" if api_ok else "unhealthy",
                "system_stats": stats
            }

            # Log status
            self.logger.info(f"Health check: {status}")

            # Alert if services are down
            if not inference_ok:
                self.logger.error("INFERENCE SERVER IS DOWN!")
            if not api_ok:
                self.logger.error("API SERVER IS DOWN!")

            time.sleep(interval)
```

## Cost Analysis

### Hardware Costs (One-time)
```
GPU Server (A100 80GB): $10,000 - $15,000
Storage (2TB NVMe): $500
RAM (128GB): $400
CPU/Motherboard: $800
Power Supply/Cooling: $300
Total Hardware: ~$12,000
```

### Operating Costs (Monthly)
```
Electricity (500W continuous): $100-200
Internet/Cloud Backup: $50
Maintenance/Upgrades: $100
Total Monthly: ~$250-350
```

### vs Cloud API Costs
```
OpenAI GPT-4: $0.03/1K tokens
Daily usage (10K tokens): $0.30/day = $110/month
Local AIVA: ~$250/month hardware + minimal API calls
Break-even: ~2-3 months
```

## Security Considerations

### Data Privacy
- All conversations stored locally
- No data sent to external APIs unless explicitly configured
- End-to-end encryption for sensitive data
- Regular security audits and updates

### Access Control
- Local network only by default
- Optional authentication for remote access
- API key protection
- Rate limiting and abuse prevention

### Backup & Recovery
- Automated daily backups
- Encrypted backup storage
- Disaster recovery procedures
- Version control for configurations

## Next Steps & Roadmap

### Phase 1 (Week 1-2): Core Infrastructure
- [ ] Set up GPU server environment
- [ ] Download and configure base models
- [ ] Implement basic inference server
- [ ] Test model loading and basic generation

### Phase 2 (Week 3-4): Memory & Tools
- [ ] Implement vector database
- [ ] Create tool system framework
- [ ] Add conversation memory
- [ ] Test tool calling capabilities

### Phase 3 (Week 5-6): AIVA Agent
- [ ] Implement agent core logic
- [ ] Create system prompts and personality
- [ ] Integrate memory and tools
- [ ] Test conversational capabilities

### Phase 4 (Week 7-8): IDE Integration
- [ ] Build Cursor extension
- [ ] Integrate with AIVA IDE frontend
- [ ] Add real-time collaboration
- [ ] Test end-to-end workflow

### Phase 5 (Week 9-10): Production & Scaling
- [ ] Implement monitoring and health checks
- [ ] Add Docker containerization
- [ ] Set up backup and recovery
- [ ] Performance optimization

### Future Enhancements
- Multi-model routing (route coding tasks to CodeLlama, general to Mixtral)
- Fine-tuning on personal data and preferences
- Plugin system for custom tools
- Distributed inference across multiple GPUs
- Integration with external services (GitHub, Slack, etc.)

---

This blueprint provides everything needed to build and deploy your own local AIVA node. The architecture is designed for scalability, security, and maintainability while keeping full control over your data and AI capabilities.

Ready to start building? Let's begin with Phase 1: Infrastructure Setup.
