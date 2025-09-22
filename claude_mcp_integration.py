"""
Claude MCP (Model Context Protocol) Integration
Enables Claude to use chAIos - Chiral Harmonic Aligned Intelligence Optimisation System tools
"""

import asyncio
import json
from typing import Dict, Any, List
import httpx

class ConsciousnessMCPServer:
    """MCP Server for Claude integration with prime aligned compute tools"""
    
    def __init__(self, consciousness_api_url="http://localhost:8000", auth_token="mcp_token"):
        self.api_url = consciousness_api_url
        self.auth_token = auth_token
        self.client = httpx.AsyncClient()
        self.tools_catalog = None
    
    async def initialize(self):
        """Initialize MCP server and load tools catalog"""
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        response = await self.client.get(f"{self.api_url}/plugin/catalog", headers=headers)
        self.tools_catalog = response.json()
        return self.tools_catalog
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all available prime aligned compute tools for Claude"""
        if not self.tools_catalog:
            await self.initialize()
        
        mcp_tools = []
        for tool in self.tools_catalog['tools']:
            mcp_tool = {
                "name": tool['name'],
                "description": tool['description'],
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        param_name: {
                            "type": param_info.get("type", "string"),
                            "description": f"Parameter for {param_name}"
                        }
                        for param_name, param_info in tool['parameters'].items()
                    },
                    "required": [
                        param_name for param_name, param_info in tool['parameters'].items() 
                        if param_info.get("required", False)
                    ]
                }
            }
            mcp_tools.append(mcp_tool)
        
        return mcp_tools
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a prime aligned compute tool for Claude"""
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json"
        }
        
        request_data = {
            "tool_name": tool_name,
            "parameters": arguments,
            "llm_source": "claude",
            "context": {"mcp_integration": True}
        }
        
        response = await self.client.post(
            f"{self.api_url}/plugin/execute", 
            headers=headers, 
            json=request_data
        )
        
        result = response.json()
        
        if result['success']:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result['result'], indent=2)
                    }
                ],
                "isError": False
            }
        else:
            return {
                "content": [
                    {
                        "type": "text", 
                        "text": f"Error executing {tool_name}: {result['error']}"
                    }
                ],
                "isError": True
            }

# MCP Protocol Handler
class MCPProtocolHandler:
    def __init__(self):
        self.consciousness_server = ConsciousnessMCPServer()
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP protocol requests from Claude"""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        try:
            if method == "tools/list":
                tools = await self.consciousness_server.list_tools()
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"tools": tools}
                }
            
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                result = await self.consciousness_server.call_tool(tool_name, arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": result
                }
            
            elif method == "initialize":
                await self.consciousness_server.initialize()
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {"listChanged": True}
                        },
                        "serverInfo": {
                            "name": "prime aligned compute-platform-mcp",
                            "version": "1.0.0"
                        }
                    }
                }
            
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }
        
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }

# Example Claude integration
async def claude_integration_demo():
    """Demo how Claude would use prime aligned compute tools via MCP"""
    handler = MCPProtocolHandler()
    
    # Initialize
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    }
    init_response = await handler.handle_request(init_request)
    print("🔌 MCP Server initialized for Claude")
    print(f"📡 Protocol Version: {init_response['result']['protocolVersion']}")
    
    # List tools
    list_request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list"
    }
    list_response = await handler.handle_request(list_request)
    tools = list_response['result']['tools']
    print(f"\n🛠️  Available Tools: {len(tools)}")
    for tool in tools[:3]:  # Show first 3
        print(f"  • {tool['name']}: {tool['description']}")
    
    # Execute Wallace Transform
    execute_request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "wallace_transform_advanced",
            "arguments": {
                "sample_data": "Analyze prime aligned compute patterns in neural networks",
                "enhancement_level": 0.88
            }
        }
    }
    execute_response = await handler.handle_request(execute_request)
    print(f"\n🧠 Wallace Transform Result:")
    print(f"✅ Success: {not execute_response['result']['isError']}")
    if not execute_response['result']['isError']:
        content = execute_response['result']['content'][0]['text']
        print(f"📊 Output: {content[:200]}...")

if __name__ == "__main__":
    asyncio.run(claude_integration_demo())
