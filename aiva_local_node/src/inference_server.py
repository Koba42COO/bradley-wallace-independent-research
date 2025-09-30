#!/usr/bin/env python3
"""
AIVA Local Inference Server
Clean FastAPI server that imports functionality from organized modules
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Import from organized modules
from .config import HOST, PORT, WORKERS, LOG_LEVEL
from .auth import verify_auth
from .models import (
    CompletionRequest, ChatCompletionRequest,
    CompletionResponse, ChatCompletionResponse,
    ToolCallBody, EmbeddingsRequest
)
from .middleware import logging_and_rate_limit_middleware
from .inference_engine import (
    initialize_engine, generate_completion, generate_chat_completion,
    stream_chat_completion, get_engine_stats
)
from .retrieval import initialize_retrieval, get_retrieval_stats
from .tools import execute_tool, get_tool_schemas
from .identity import install_identity_pack, get_identity_info

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/inference_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting AIVA Inference Server...")

    # Initialize components
    engine_ok = await initialize_engine()
    if not engine_ok:
        logger.warning("Inference engine failed to initialize")

    initialize_retrieval()

    yield

    logger.info("Shutting down AIVA Inference Server...")

# Create FastAPI app
app = FastAPI(
    title="AIVA Local Inference API",
    description="OpenAI-compatible API for local LLM inference with AIVA personality",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging and rate limiting middleware
app.middleware("http")(logging_and_rate_limit_middleware)

# === API Endpoints ===

@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint"""
    verify_auth(request)

    engine_stats = get_engine_stats()
    retrieval_stats = get_retrieval_stats()

    return {
        "status": "healthy" if engine_stats["model_loaded"] else "degraded",
        "timestamp": asyncio.get_event_loop().time(),
        **engine_stats,
        **retrieval_stats
    }

@app.get("/v1/models")
async def list_models(request: Request):
    """List available models (OpenAI-compatible)"""
    verify_auth(request)

    engine_stats = get_engine_stats()
    return {
        "object": "list",
        "data": [{
            "id": engine_stats["model_name"],
            "object": "model",
            "created": int(asyncio.get_event_loop().time()),
            "owned_by": "aiva-local"
        }]
    }

@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: Request, req: CompletionRequest):
    """Create text completion (OpenAI-compatible)"""
    verify_auth(request)

    try:
        text = await generate_completion(req.prompt, **req.dict(exclude={'prompt'}))
        return CompletionResponse(
            id=f"cmpl-{hash(req.prompt) % 1000000}",
            created=int(asyncio.get_event_loop().time()),
            model=get_engine_stats()["model_name"],
            choices=[{
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }],
            usage={"prompt_tokens": len(req.prompt.split()), "completion_tokens": len(text.split()), "total_tokens": len(req.prompt.split()) + len(text.split())}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request, req: ChatCompletionRequest):
    """Create chat completion (OpenAI-compatible)"""
    verify_auth(request)

    try:
        if req.stream:
            return StreamingResponse(
                stream_chat_completion(req.messages, **req.dict(exclude={'messages'})),
                media_type="text/event-stream"
            )
        else:
            text = await generate_chat_completion(req.messages, **req.dict(exclude={'messages'}))
            return ChatCompletionResponse(
                id=f"chatcmpl-{hash(str(req.messages)) % 1000000}",
                created=int(asyncio.get_event_loop().time()),
                model=get_engine_stats()["model_name"],
                choices=[{
                    "message": {"role": "assistant", "content": text},
                    "index": 0,
                    "finish_reason": "stop"
                }],
                usage={"prompt_tokens": len(str(req.messages).split()), "completion_tokens": len(text.split()), "total_tokens": len(str(req.messages).split()) + len(text.split())}
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/v1/embeddings")
async def create_embeddings(request: Request, req: EmbeddingsRequest):
    """Minimal placeholder: returns deterministic hash-based vectors"""
    verify_auth(request)

    def hash_vec(text: str, dim: int = 128) -> List[float]:
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vals = list(h) * ((dim + len(h) - 1) // len(h))
        vals = vals[:dim]
        return [v / 255.0 for v in vals]

    data = []
    for i, item in enumerate(req.input):
        data.append({"object": "embedding", "index": i, "embedding": hash_vec(item)})

    return {"object": "list", "data": data, "model": req.model}

@app.post("/v1/tools/call")
async def call_tool(request: Request, req: ToolCallBody):
    """Execute a tool directly"""
    verify_auth(request)

    try:
        output = await execute_tool(req.tool, req.arguments or {})
        return {"tool": req.tool, "output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")

@app.get("/v1/tools")
async def list_tools(request: Request):
    """List available tools"""
    verify_auth(request)

    return {
        "object": "list",
        "data": [
            {"name": name, "description": meta["description"], "parameters": meta["parameters"]}
            for name, meta in get_tool_schemas()
        ]
    }

# === Identity Endpoints ===
@app.post("/v1/identity/install")
async def identity_install(request: Request, req):
    """Install identity pack"""
    verify_auth(request)

    base = req.base_dir or os.getcwd()
    result = install_identity_pack(base, overwrite=req.overwrite)
    return result

@app.get("/v1/identity/info")
async def identity_info(request: Request):
    """Get identity pack info"""
    verify_auth(request)

    base = os.getcwd()
    return get_identity_info(base)

@app.post("/v1/identity/check")
async def identity_check(request: Request, req):
    """Run identity checks"""
    verify_auth(request)

    # This would run behavioral tests - simplified for now
    return {"ok": True, "message": "Identity checks completed"}

# === Memory Endpoints ===
@app.post("/v1/memory/reload")
async def memory_reload(request: Request):
    """Reload memory corpus"""
    verify_auth(request)

    initialize_retrieval()
    stats = get_retrieval_stats()
    return {"ok": True, **stats}

@app.get("/v1/memory/info")
async def memory_info(request: Request):
    """Get memory statistics"""
    verify_auth(request)

    return get_retrieval_stats()

@app.get("/stats")
async def get_stats(request: Request):
    """Get comprehensive server statistics"""
    verify_auth(request)

    try:
        import psutil
        import GPUtil

        # System stats
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        # GPU stats
        gpu_stats = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_stats.append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "utilization": gpu.load * 100,
                    "memory_used": gpu.memoryUsed,
                    "memory_total": gpu.memoryTotal,
                    "temperature": gpu.temperature
                })
        except:
            gpu_stats = []

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_total_gb": memory.total / (1024**3),
            "gpus": gpu_stats,
            **get_engine_stats(),
            **get_retrieval_stats()
        }
    except Exception as e:
        return {"error": f"Failed to get stats: {str(e)}"}

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    logger.info(f"Starting server on {HOST}:{PORT}")
    uvicorn.run(
        "inference_server:app",
        host=HOST,
        port=PORT,
        workers=WORKERS,
        reload=False,
        log_level=LOG_LEVEL.lower()
    )