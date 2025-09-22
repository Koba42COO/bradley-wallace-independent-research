"""
ChatGPT Plugin Integration Example
Shows how ChatGPT can use the chAIos - Chiral Harmonic Aligned Intelligence Optimisation System as a tool
"""

import requests
import json

class ConsciousnessToolsPlugin:
    def __init__(self, api_base_url="http://localhost:8000", auth_token="your_plugin_token_here"):
        self.api_base_url = api_base_url
        self.headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json"
        }
    
    def get_available_tools(self):
        """Get all available prime aligned compute tools"""
        response = requests.get(f"{self.api_base_url}/plugin/catalog", headers=self.headers)
        return response.json()
    
    def execute_wallace_transform(self, sample_data, enhancement_level=0.95):
        """Execute Wallace Transform for prime aligned compute mathematics"""
        request_data = {
            "tool_name": "wallace_transform_advanced",
            "parameters": {
                "sample_data": sample_data,
                "enhancement_level": enhancement_level
            },
            "llm_source": "chatgpt",
            "context": {"purpose": "consciousness_enhancement"}
        }
        
        response = requests.post(f"{self.api_base_url}/plugin/execute", 
                               headers=self.headers, 
                               json=request_data)
        return response.json()
    
    def generate_code_with_grok(self, requirements, complexity="advanced"):
        """Use Grok Jr coding agent"""
        request_data = {
            "tool_name": "grok_generate_code",
            "parameters": {
                "requirements": requirements,
                "complexity": complexity
            },
            "llm_source": "chatgpt",
            "context": {"purpose": "code_generation"}
        }
        
        response = requests.post(f"{self.api_base_url}/plugin/execute", 
                               headers=self.headers, 
                               json=request_data)
        return response.json()
    
    def run_security_scan(self, target_system):
        """Execute enterprise security scan"""
        request_data = {
            "tool_name": "aiva_security_scanner",
            "parameters": {
                "target_system": target_system,
                "scan_depth": "comprehensive"
            },
            "llm_source": "chatgpt",
            "context": {"purpose": "security_analysis"}
        }
        
        response = requests.post(f"{self.api_base_url}/plugin/execute", 
                               headers=self.headers, 
                               json=request_data)
        return response.json()

# Example usage for ChatGPT
def chatgpt_plugin_demo():
    """Demo how ChatGPT would use the prime aligned compute tools"""
    plugin = ConsciousnessToolsPlugin()
    
    # Get available tools
    print("🔌 Available prime aligned compute Tools:")
    catalog = plugin.get_available_tools()
    for tool in catalog['tools'][:5]:  # Show first 5
        print(f"  • {tool['name']}: {tool['description']}")
    
    print(f"\n📊 Total: {catalog['total_tools']} tools across {len(catalog['categories'])} categories")
    
    # Example: Use Wallace Transform
    print("\n🧠 Executing Wallace Transform...")
    result = plugin.execute_wallace_transform(
        sample_data="Optimize this neural network architecture",
        enhancement_level=0.92
    )
    
    if result['success']:
        print(f"✅ Wallace Transform completed in {result['execution_time']:.2f}s")
        print(f"📈 Result: {result['result']}")
    else:
        print(f"❌ Error: {result['error']}")
    
    # Example: Generate code with Grok Jr
    print("\n🤖 Generating code with Grok Jr...")
    code_result = plugin.generate_code_with_grok(
        requirements="Create a FastAPI endpoint for user authentication"
    )
    
    if code_result['success']:
        print(f"✅ Code generated in {code_result['execution_time']:.2f}s")
        print(f"💻 Generated code preview: {str(code_result['result'])[:200]}...")

if __name__ == "__main__":
    chatgpt_plugin_demo()
