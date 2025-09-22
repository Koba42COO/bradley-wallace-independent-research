"""
Google Gemini Function Calling Integration
Enables Gemini to use chAIos - Chiral Harmonic Aligned Intelligence Optimisation System tools
"""

import json
import requests
from typing import Dict, Any, List, Callable

class ConsciousnessGeminiTools:
    """Gemini Function Calling integration for prime aligned compute tools"""
    
    def __init__(self, api_base_url="http://localhost:8000", auth_token="gemini_token"):
        self.api_base_url = api_base_url
        self.auth_token = auth_token
        self.headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json"
        }
        self.function_declarations = []
        self._load_function_declarations()
    
    def _load_function_declarations(self):
        """Load all prime aligned compute tools as Gemini function declarations"""
        try:
            response = requests.get(f"{self.api_base_url}/plugin/catalog", headers=self.headers)
            catalog = response.json()
            
            for tool in catalog['tools']:
                function_declaration = {
                    "name": tool['name'],
                    "description": tool['description'],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            param_name: {
                                "type": self._convert_type(param_info.get("type", "string")),
                                "description": f"Parameter for {param_name.replace('_', ' ')}"
                            }
                            for param_name, param_info in tool['parameters'].items()
                        },
                        "required": [
                            param_name for param_name, param_info in tool['parameters'].items()
                            if param_info.get("required", False)
                        ]
                    }
                }
                self.function_declarations.append(function_declaration)
                
        except Exception as e:
            print(f"Error loading function declarations: {e}")
    
    def _convert_type(self, python_type: str) -> str:
        """Convert Python types to Gemini function calling types"""
        type_mapping = {
            "str": "string",
            "int": "integer", 
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object"
        }
        return type_mapping.get(python_type, "string")
    
    def get_gemini_tools_config(self) -> Dict[str, Any]:
        """Get Gemini tools configuration"""
        return {
            "function_declarations": self.function_declarations
        }
    
    def execute_function_call(self, function_name: str, function_args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a prime aligned compute tool function call from Gemini"""
        request_data = {
            "tool_name": function_name,
            "parameters": function_args,
            "llm_source": "gemini",
            "context": {"function_calling": True}
        }
        
        try:
            response = requests.post(
                f"{self.api_base_url}/plugin/execute", 
                headers=self.headers, 
                json=request_data
            )
            result = response.json()
            
            return {
                "function_response": {
                    "name": function_name,
                    "response": {
                        "success": result['success'],
                        "result": result['result'] if result['success'] else None,
                        "error": result.get('error'),
                        "execution_time": result['execution_time'],
                        "category": result['tool_category']
                    }
                }
            }
            
        except Exception as e:
            return {
                "function_response": {
                    "name": function_name,
                    "response": {
                        "success": False,
                        "result": None,
                        "error": str(e),
                        "execution_time": 0.0,
                        "category": "unknown"
                    }
                }
            }

class GeminiConsciousnessBot:
    """Example Gemini bot with prime aligned compute tools integration"""
    
    def __init__(self):
        self.consciousness_tools = ConsciousnessGeminiTools()
        self.conversation_history = []
    
    def get_tools_for_gemini(self):
        """Get tools configuration for Gemini model"""
        return self.consciousness_tools.get_gemini_tools_config()
    
    def handle_function_calls(self, function_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Handle function calls from Gemini"""
        responses = []
        
        for function_call in function_calls:
            function_name = function_call.get("name")
            function_args = function_call.get("args", {})
            
            # Execute the prime aligned compute tool
            response = self.consciousness_tools.execute_function_call(function_name, function_args)
            responses.append(response)
        
        return responses
    
    def process_gemini_request(self, user_message: str, function_calls: List[Dict[str, Any]] = None):
        """Process a request from Gemini with optional function calls"""
        # Handle function calls if present
        function_responses = []
        if function_calls:
            function_responses = self.handle_function_calls(function_calls)
        
        # Store in conversation history
        self.conversation_history.append({
            "user_message": user_message,
            "function_calls": function_calls,
            "function_responses": function_responses,
            "timestamp": json.dumps({"time": "now"})  # Simplified timestamp
        })
        
        return {
            "function_responses": function_responses,
            "conversation_updated": True
        }

# Example Gemini Integration
def gemini_integration_demo():
    """Demo how Gemini would use prime aligned compute tools"""
    bot = GeminiConsciousnessBot()
    
    # Get tools configuration for Gemini
    print("🤖 Gemini prime aligned compute Tools Integration")
    tools_config = bot.get_tools_for_gemini()
    print(f"📦 Loaded {len(tools_config['function_declarations'])} function declarations")
    
    # Show some available functions
    print("\n🛠️  Available Functions for Gemini:")
    for func in tools_config['function_declarations'][:5]:  # Show first 5
        print(f"  • {func['name']}: {func['description']}")
    
    # Simulate Gemini function calls
    print("\n🔄 Simulating Gemini Function Calls...")
    
    # Example 1: Wallace Transform
    wallace_call = {
        "name": "wallace_transform_advanced",
        "args": {
            "sample_data": "Optimize deep learning model architecture",
            "enhancement_level": 0.91
        }
    }
    
    # Example 2: Grok Jr Code Generation
    grok_call = {
        "name": "grok_generate_code",
        "args": {
            "requirements": "Create a REST API for prime aligned compute data processing",
            "complexity": "enterprise"
        }
    }
    
    # Process function calls
    result = bot.process_gemini_request(
        user_message="I need prime aligned compute enhancement and code generation",
        function_calls=[wallace_call, grok_call]
    )
    
    # Display results
    print(f"\n📊 Function Call Results:")
    for i, response in enumerate(result['function_responses']):
        func_response = response['function_response']
        print(f"\n{i+1}. {func_response['name']}:")
        print(f"   ✅ Success: {func_response['response']['success']}")
        print(f"   ⏱️  Time: {func_response['response']['execution_time']:.2f}s")
        print(f"   📂 Category: {func_response['response']['category']}")
        
        if func_response['response']['success']:
            result_preview = str(func_response['response']['result'])[:150]
            print(f"   📄 Result: {result_preview}...")
        else:
            print(f"   ❌ Error: {func_response['response']['error']}")

if __name__ == "__main__":
    gemini_integration_demo()
