#!/usr/bin/env python3
"""
AIVA Data Models
Pydantic models for API requests and responses
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# === Chat Models ===
class ChatMessage(BaseModel):
    """Chat message format"""
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")

# === Completion Models ===
class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request"""
    model: str = Field(..., description="Model name to use")
    prompt: str = Field(..., description="Input prompt")
    max_tokens: int = Field(512, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Top-p sampling parameter")
    frequency_penalty: float = Field(0.0, description="Frequency penalty")
    presence_penalty: float = Field(0.0, description="Presence penalty")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Stream response")
    echo: bool = Field(False, description="Echo the prompt in response")

class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str = Field(..., description="Model name to use")
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    max_tokens: int = Field(512, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Top-p sampling parameter")
    frequency_penalty: float = Field(0.0, description="Frequency penalty")
    presence_penalty: float = Field(0.0, description="Presence penalty")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Stream response")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="Available tools")
    json_mode: bool = Field(False, description="Hint the model to return strict JSON only (no prose)")

# === Response Models ===
class CompletionResponse(BaseModel):
    """Completion response"""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class ChatCompletionResponse(BaseModel):
    """Chat completion response"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

# === Tool Models ===
class ToolCallBody(BaseModel):
    """Tool call request body"""
    tool: str
    arguments: Dict[str, Any] = {}

# === Identity Models ===
class IdentityInstallBody(BaseModel):
    """Identity pack installation request"""
    base_dir: Optional[str] = None
    overwrite: bool = False

class IdentityCheckBody(BaseModel):
    """Identity check request"""
    base_dir: Optional[str] = None

# === Embeddings Models ===
class EmbeddingsRequest(BaseModel):
    """Embeddings request"""
    model: str
    input: List[str]