#!/usr/bin/env python3
"""
AIVA Agent Core
Main agent that orchestrates model inference, memory, and tool usage
"""

import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass

import aiohttp
from pydantic import BaseModel, Field

from vector_store import AIVAVectorStore
from tool_system import AIVAToolSystem, ToolResult

logger = logging.getLogger(__name__)

class ChatMessage(BaseModel):
    """Chat message format"""
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")

class AIVAAgentConfig(BaseModel):
    """Configuration for AIVA agent"""
    inference_url: str = Field(default="http://localhost:8000", description="Inference server URL")
    model_name: str = Field(default="mixtral-8x7b-instruct", description="Model name")
    max_tokens: int = Field(default=1024, description="Maximum tokens per response")
    temperature: float = Field(default=0.7, description="Response temperature")
    enable_memory: bool = Field(default=True, description="Enable conversation memory")
    enable_tools: bool = Field(default=True, description="Enable tool usage")
    system_prompt: str = Field(default="", description="Custom system prompt")

@dataclass
class ConversationContext:
    """Context for a conversation"""
    conversation_id: str
    messages: List[Dict[str, str]]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

class AIVAAgent:
    """
    Main AIVA agent that combines inference, memory, and tool capabilities
    """

    def __init__(self, config: AIVAAgentConfig):
        """
        Initialize the AIVA agent

        Args:
            config: Agent configuration
        """
        self.config = config

        # Initialize components
        self.vector_store = AIVAVectorStore() if config.enable_memory else None
        self.tool_system = AIVAToolSystem() if config.enable_tools else None

        # HTTP client for inference server
        self.session: Optional[aiohttp.ClientSession] = None

        # Default system prompt
        self.default_system_prompt = """
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
"""

        if config.system_prompt:
            self.system_prompt = config.system_prompt
        else:
            self.system_prompt = self.default_system_prompt

        # Conversation tracking
        self.active_conversations: Dict[str, ConversationContext] = {}

        logger.info("AIVA Agent initialized")

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def chat(self,
                   messages: List[Dict[str, str]],
                   conversation_id: Optional[str] = None,
                   stream: bool = False) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main chat method - handles conversation with full AIVA capabilities

        Args:
            messages: Chat messages
            conversation_id: Optional conversation ID for memory
            stream: Whether to stream the response

        Yields:
            Response chunks or final result
        """
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(messages)) % 10000}"

        # Get or create conversation context
        context = self._get_conversation_context(conversation_id, messages)

        # Enhance messages with system prompt and context
        enhanced_messages = self._enhance_messages_with_context(messages, context)

        # Add tool schemas if tools are enabled
        tools = None
        if self.tool_system:
            tools = self.tool_system.get_tool_schemas()

        # Prepare request payload
        payload = {
            "model": self.config.model_name,
            "messages": enhanced_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": stream
        }

        if tools:
            payload["tools"] = tools

        try:
            # Make request to inference server
            async with self.session.post(
                f"{self.config.inference_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:

                if response.status != 200:
                    error_text = await response.text()
                    yield {
                        "error": f"Inference server error: {response.status} - {error_text}",
                        "conversation_id": conversation_id
                    }
                    return

                if stream:
                    # Handle streaming response
                    async for chunk in self._process_stream_response(response, conversation_id):
                        yield chunk
                else:
                    # Handle regular response
                    result = await response.json()
                    processed_result = await self._process_response(result, context)
                    yield processed_result

        except Exception as e:
            logger.error(f"Chat request failed: {e}")
            yield {
                "error": f"Chat request failed: {str(e)}",
                "conversation_id": conversation_id
            }

    async def _process_stream_response(self, response, conversation_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Process streaming response from inference server"""
        buffer = ""

        async for chunk in response.content.iter_any():
            chunk_str = chunk.decode('utf-8')
            buffer += chunk_str

            # Process complete lines
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()

                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix

                    if data == '[DONE]':
                        # End of stream
                        yield {
                            "done": True,
                            "conversation_id": conversation_id
                        }
                        return

                    try:
                        chunk_data = json.loads(data)
                        yield {
                            "chunk": chunk_data,
                            "conversation_id": conversation_id
                        }
                    except json.JSONDecodeError:
                        continue

    async def _process_response(self, response_data: Dict[str, Any], context: ConversationContext) -> Dict[str, Any]:
        """Process the final response from inference server"""
        try:
            choice = response_data["choices"][0]
            message = choice["message"]
            content = message.get("content", "")

            # Check for tool calls
            tool_calls = message.get("tool_calls", [])
            tool_results = []

            if tool_calls and self.tool_system:
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get("function", {}).get("name")
                        tool_args_str = tool_call.get("function", {}).get("arguments", "{}")

                        try:
                            tool_args = json.loads(tool_args_str)
                            result = self.tool_system.execute_tool(tool_name, **tool_args)
                            tool_results.append({
                                "tool_call": tool_call,
                                "result": result.dict() if hasattr(result, 'dict') else result
                            })
                        except Exception as e:
                            logger.error(f"Tool execution failed: {e}")
                            tool_results.append({
                                "tool_call": tool_call,
                                "error": str(e)
                            })

            # Update conversation context
            context.messages.append({"role": "assistant", "content": content})
            context.updated_at = datetime.now()

            # Store in vector memory if enabled
            if self.vector_store:
                self.vector_store.add_conversation(
                    context.conversation_id,
                    context.messages,
                    context.metadata
                )

            return {
                "response": content,
                "tool_results": tool_results,
                "usage": response_data.get("usage", {}),
                "conversation_id": context.conversation_id,
                "model": response_data.get("model", self.config.model_name)
            }

        except Exception as e:
            logger.error(f"Response processing failed: {e}")
            return {
                "error": f"Response processing failed: {str(e)}",
                "conversation_id": context.conversation_id
            }

    def _get_conversation_context(self, conversation_id: str, messages: List[Dict[str, str]]) -> ConversationContext:
        """Get or create conversation context"""
        if conversation_id in self.active_conversations:
            context = self.active_conversations[conversation_id]
            # Add new messages to context
            context.messages.extend(messages)
            context.updated_at = datetime.now()
            return context
        else:
            # Create new context
            context = ConversationContext(
                conversation_id=conversation_id,
                messages=messages.copy(),
                metadata={
                    "model": self.config.model_name,
                    "created_with_tools": self.config.enable_tools,
                    "memory_enabled": self.config.enable_memory
                },
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            self.active_conversations[conversation_id] = context
            return context

    def _enhance_messages_with_context(self, messages: List[Dict[str, str]], context: ConversationContext) -> List[Dict[str, str]]:
        """Enhance messages with system prompt and memory context"""
        enhanced = []

        # Add system prompt
        enhanced.append({
            "role": "system",
            "content": self.system_prompt
        })

        # Add memory context if available
        if self.vector_store and len(context.messages) > 1:
            memory_context = self.vector_store.get_conversation_context(
                messages[-1]["content"] if messages else "",
                context.messages[:-1]  # Exclude current message
            )

            if memory_context:
                enhanced.append({
                    "role": "system",
                    "content": f"Relevant context from previous conversations:\n\n{memory_context}"
                })

        # Add conversation messages
        enhanced.extend(context.messages)

        return enhanced

    async def add_knowledge(self, title: str, content: str, **metadata) -> bool:
        """Add knowledge to the vector store"""
        if not self.vector_store:
            return False

        try:
            self.vector_store.add_knowledge(title, content, **metadata)
            return True
        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}")
            return False

    def search_knowledge(self, query: str, **kwargs) -> Dict[str, Any]:
        """Search knowledge base"""
        if not self.vector_store:
            return {"error": "Vector store not enabled"}

        return self.vector_store.search_similar(query, **kwargs)

    def get_conversation_history(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get conversation history"""
        return self.active_conversations.get(conversation_id)

    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all active conversations"""
        return [
            {
                "conversation_id": ctx.conversation_id,
                "message_count": len(ctx.messages),
                "created_at": ctx.created_at.isoformat(),
                "updated_at": ctx.updated_at.isoformat(),
                "metadata": ctx.metadata
            }
            for ctx in self.active_conversations.values()
        ]

    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear a conversation from memory"""
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        stats = {
            "active_conversations": len(self.active_conversations),
            "config": self.config.dict(),
            "memory_enabled": self.config.enable_memory,
            "tools_enabled": self.config.enable_tools,
        }

        if self.vector_store:
            stats["vector_store_stats"] = self.vector_store.get_stats()

        if self.tool_system:
            stats["available_tools"] = list(self.tool_system.tools.keys())

        return stats

# Convenience functions
async def create_aiva_agent(inference_url: str = "http://localhost:8000",
                           model_name: str = "mixtral-8x7b-instruct",
                           **kwargs) -> AIVAAgent:
    """
    Create and return a configured AIVA agent

    Args:
        inference_url: URL of the inference server
        model_name: Model name to use
        **kwargs: Additional configuration options

    Returns:
        Configured AIVAAgent instance
    """
    config = AIVAAgentConfig(
        inference_url=inference_url,
        model_name=model_name,
        **kwargs
    )

    agent = AIVAAgent(config)
    return agent

async def test_aiva_agent():
    """Test function for the AIVA agent"""
    async with await create_aiva_agent() as agent:
        print("Testing AIVA Agent...")

        # Test basic chat
        messages = [{"role": "user", "content": "Hello, who are you?"}]

        print("Testing basic chat...")
        async for response in agent.chat(messages):
            if "error" in response:
                print(f"Error: {response['error']}")
            else:
                print(f"Response: {response.get('response', 'No response')}")
            break

        # Test tool usage
        if agent.tool_system:
            print("Testing tool system...")
            result = agent.tool_system.execute_tool("execute_python", code="print(42)")
            print(f"Tool result: {result.success}")

        # Test memory
        if agent.vector_store:
            print("Testing vector memory...")
            agent.vector_store.add_knowledge(
                "Test Knowledge",
                "This is a test knowledge entry for AIVA.",
                tags=["test"]
            )
            results = agent.vector_store.search_similar("test knowledge")
            print(f"Memory search results: {len(results.get('results', []))}")

        print("AIVA Agent test completed!")

if __name__ == "__main__":
    asyncio.run(test_aiva_agent())
