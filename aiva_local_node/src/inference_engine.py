#!/usr/bin/env python3
"""
AIVA Inference Engine
Model loading and inference management using vLLM
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from vllm import SamplingParams, AsyncLLMEngine, AsyncEngineArgs
from .config import (
    MODEL_PATH, TENSOR_PARALLEL_SIZE, GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN, DTYPE, TRUST_REMOTE_CODE
)
from .models import ChatMessage
from .retrieval import retrieve_context

logger = logging.getLogger(__name__)

# Global variables
llm_engine: Optional[AsyncLLMEngine] = None
model_name: str = "mixtral-8x7b-instruct"

async def initialize_engine() -> bool:
    """
    Initialize the vLLM inference engine

    Returns:
        True if initialization successful
    """
    global llm_engine, model_name

    try:
        logger.info(f"Loading model: {MODEL_PATH}")

        engine_args = AsyncEngineArgs(
            model=MODEL_PATH,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            dtype=DTYPE,
            trust_remote_code=TRUST_REMOTE_CODE,
        )

        llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("Model loaded successfully")

        return True

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

def format_chat_messages(messages: List[ChatMessage], json_mode: bool = False) -> str:
    """Format chat messages into a single prompt with Aiva system prompt and optional retrieval context."""
    from .config import AIVA_SYSTEM_PROMPT

    formatted_parts = []

    # Always start with Aiva's system prompt
    formatted_parts.append(f"System: {AIVA_SYSTEM_PROMPT}")

    if json_mode:
        formatted_parts.append("System: Reply with VALID JSON only. No prose, no backticks, no commentary. If a list is natural, return a JSON array; otherwise a JSON object.")

    # Find latest user message to query retrieval
    last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
    ctx = retrieve_context(last_user)
    if ctx:
        formatted_parts.append("System: The following CONTEXT is non-authoritative memory snippets. Use if relevant.\n" + ctx)

    # Then include the actual conversation
    for msg in messages:
        role = msg.role
        content = msg.content

        # User-supplied system messages still included after Aiva prompt
        if role == "system":
            formatted_parts.append(f"System: {content}")
        elif role == "user":
            formatted_parts.append(f"Human: {content}")
        elif role == "assistant":
            formatted_parts.append(f"Assistant: {content}")

    return "\n\n".join(formatted_parts)

async def generate_completion(prompt: str, **kwargs) -> str:
    """
    Generate text completion

    Args:
        prompt: Input prompt
        **kwargs: Additional parameters

    Returns:
        Generated text
    """
    if not llm_engine:
        raise RuntimeError("Inference engine not initialized")

    sampling_params = SamplingParams(
        max_tokens=kwargs.get('max_tokens', 512),
        temperature=kwargs.get('temperature', 0.7),
        top_p=kwargs.get('top_p', 0.9),
        frequency_penalty=kwargs.get('frequency_penalty', 0.0),
        presence_penalty=kwargs.get('presence_penalty', 0.0),
        stop=kwargs.get('stop', []),
    )

    results = await llm_engine.generate(prompt, sampling_params)
    return results[0].outputs[0].text if results and results[0].outputs else ""

async def generate_chat_completion(messages: List[ChatMessage], **kwargs) -> str:
    """
    Generate chat completion

    Args:
        messages: Chat messages
        **kwargs: Additional parameters

    Returns:
        Generated response
    """
    json_mode = kwargs.get('json_mode', False)
    prompt = format_chat_messages(messages, json_mode=json_mode)

    # Handle tool calls if present
    if kwargs.get('tools'):
        prompt += "\n\nAvailable tools (respond exactly with TOOL_CALL: {\"tool\": ..., \"arguments\": {...}}):\n"
        for tool in kwargs['tools']:
            fn = tool["function"]
            name = fn["name"]
            desc = fn.get("description", "")
            schema_str = str(fn.get("parameters", {}))
            prompt += f"- {name}: {desc}\n  schema: {schema_str}\n"

    return await generate_completion(prompt, **kwargs)

async def generate_response_async(prompt: str) -> str:
    """
    Generate a simple response for identity testing

    Args:
        prompt: Input prompt

    Returns:
        Generated response
    """
    messages = [ChatMessage(role="user", content=prompt)]
    return await generate_chat_completion(messages)

def extract_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse a TOOL_CALL JSON from a string segment. Returns dict or None."""
    import json
    marker = "TOOL_CALL:"
    idx = text.find(marker)
    if idx == -1:
        return None
    payload = text[idx + len(marker):].strip()
    try:
        return json.loads(payload)
    except Exception:
        return None

async def stream_chat_completion(messages: List[ChatMessage], **kwargs):
    """
    Stream chat completion responses

    Args:
        messages: Chat messages
        **kwargs: Additional parameters

    Yields:
        Response chunks
    """
    if not llm_engine:
        yield {"error": "Inference engine not initialized"}
        return

    json_mode = kwargs.get('json_mode', False)
    prompt = format_chat_messages(messages, json_mode=json_mode)

    # Advertise built-in tools (no live tool execution until marker appears)
    if kwargs.get('tools'):
        prompt += "\n\nAvailable tools (respond exactly with TOOL_CALL: {\\\"tool\\\": ..., \\\"arguments\\\": {...}}):\n"
        for tool in kwargs['tools']:
            fn = tool["function"]
            name = fn["name"]
            desc = fn.get("description", "")
            schema_str = str(fn.get("parameters", {}))
            prompt += f"- {name}: {desc}\n  schema: {schema_str}\n"

    sampling_params = SamplingParams(
        max_tokens=kwargs.get('max_tokens', 1024),
        temperature=kwargs.get('temperature', 0.7),
        top_p=kwargs.get('top_p', 0.9),
        frequency_penalty=kwargs.get('frequency_penalty', 0.0),
        presence_penalty=kwargs.get('presence_penalty', 0.0),
        stop=kwargs.get('stop', []),
    )

    req_id = f"chatcmpl-{hash(prompt) % 1000000}"
    created = int(datetime.now().timestamp())

    try:
        gen = llm_engine.generate(prompt, sampling_params, request_id=req_id)
        acc = ""
        tool_executed = False

        async for result in gen:
            if not result.outputs:
                continue
            piece = result.outputs[0].text
            if not piece:
                continue
            acc += piece

            # Check if a complete TOOL_CALL JSON is present
            if not tool_executed:
                tc = extract_tool_call(acc)
                if tc is not None and isinstance(tc, dict):
                    tool_executed = True
                    # Execute tool and stream result
                    from .tools import execute_tool
                    tool_name = tc.get("tool")
                    tool_args = tc.get("arguments", {})
                    tool_output = await execute_tool(tool_name, tool_args)

                    # Stream tool output as the assistant content
                    out_delta = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {"content": tool_output}, "finish_reason": "stop"}]
                    }
                    yield out_delta
                    yield {"data": "[DONE]"}
                    return

            # If no tool call yet, stream the raw piece
            delta = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]
            }
            yield delta

        # Normal end of stream
        yield {"data": "[DONE]"}

    except Exception as e:
        logger.error(f"Chat streaming failed: {e}")
        yield {"error": f"Streaming failed: {str(e)}"}

def get_engine_stats() -> Dict[str, Any]:
    """Get inference engine statistics"""
    return {
        "model_loaded": llm_engine is not None,
        "model_name": model_name,
        "model_path": MODEL_PATH,
    }