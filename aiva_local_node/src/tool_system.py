#!/usr/bin/env python3
"""
AIVA Tool System
Provides function calling capabilities for external tools and APIs
"""

import os
import sys
import json
import subprocess
import logging
import tempfile
import shutil
from typing import Dict, Any, List, Optional, Callable, Tuple
from pathlib import Path
from datetime import datetime
import asyncio
import re

import requests
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

class ToolResult(BaseModel):
    """Result from tool execution"""
    success: bool
    result: Any = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ToolDefinition(BaseModel):
    """Tool definition with schema"""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., ToolResult]

class AIVAToolSystem:
    """
    Tool system for AIVA that provides function calling capabilities
    """

    def __init__(self, working_directory: str = "./workspace"):
        """
        Initialize the tool system

        Args:
            working_directory: Default directory for file operations
        """
        self.working_directory = Path(working_directory)
        self.working_directory.mkdir(parents=True, exist_ok=True)

        # Tool registry
        self.tools: Dict[str, ToolDefinition] = {}

        # Register built-in tools
        self._register_built_in_tools()

        logger.info(f"Tool system initialized with {len(self.tools)} tools")

    def _register_built_in_tools(self):
        """Register all built-in tools"""

        # Code execution tools
        self.register_tool(
            name="execute_python",
            description="Execute Python code and return the result",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Execution timeout in seconds",
                        "default": 30
                    }
                },
                "required": ["code"]
            },
            handler=self._execute_python
        )

        self.register_tool(
            name="execute_shell",
            description="Execute a shell command and return the result",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Execution timeout in seconds",
                        "default": 30
                    },
                    "working_directory": {
                        "type": "string",
                        "description": "Working directory for command execution"
                    }
                },
                "required": ["command"]
            },
            handler=self._execute_shell
        )

        # File system tools
        self.register_tool(
            name="read_file",
            description="Read the contents of a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding",
                        "default": "utf-8"
                    }
                },
                "required": ["path"]
            },
            handler=self._read_file
        )

        self.register_tool(
            name="write_file",
            description="Write content to a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding",
                        "default": "utf-8"
                    }
                },
                "required": ["path", "content"]
            },
            handler=self._write_file
        )

        self.register_tool(
            name="list_directory",
            description="List contents of a directory",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the directory to list",
                        "default": "."
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to list recursively",
                        "default": false
                    }
                }
            },
            handler=self._list_directory
        )

        self.register_tool(
            name="create_directory",
            description="Create a new directory",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path of the directory to create"
                    }
                },
                "required": ["path"]
            },
            handler=self._create_directory
        )

        # Git tools
        self.register_tool(
            name="git_status",
            description="Get git repository status",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to git repository",
                        "default": "."
                    }
                }
            },
            handler=self._git_status
        )

        self.register_tool(
            name="git_commit",
            description="Commit changes to git repository",
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Commit message"
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to git repository",
                        "default": "."
                    }
                },
                "required": ["message"]
            },
            handler=self._git_commit
        )

        # Web tools
        self.register_tool(
            name="web_search",
            description="Search the web for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 5
                    }
                },
                "required": ["query"]
            },
            handler=self._web_search
        )

        self.register_tool(
            name="http_request",
            description="Make an HTTP request",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to request"
                    },
                    "method": {
                        "type": "string",
                        "description": "HTTP method",
                        "enum": ["GET", "POST", "PUT", "DELETE"],
                        "default": "GET"
                    },
                    "headers": {
                        "type": "object",
                        "description": "Request headers"
                    },
                    "data": {
                        "type": "string",
                        "description": "Request body data"
                    }
                },
                "required": ["url"]
            },
            handler=self._http_request
        )

        # Analysis tools
        self.register_tool(
            name="analyze_code",
            description="Analyze code for issues and improvements",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Code to analyze"
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language"
                    }
                },
                "required": ["code"]
            },
            handler=self._analyze_code
        )

        self.register_tool(
            name="analyze_project",
            description="Analyze project structure and provide insights",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to project directory",
                        "default": "."
                    }
                }
            },
            handler=self._analyze_project
        )

    def register_tool(self,
                     name: str,
                     description: str,
                     parameters: Dict[str, Any],
                     handler: Callable[..., ToolResult]) -> None:
        """
        Register a new tool

        Args:
            name: Tool name
            description: Tool description
            parameters: JSON schema for tool parameters
            handler: Function that handles tool execution
        """
        tool_def = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler
        )

        self.tools[name] = tool_def
        logger.info(f"Registered tool: {name}")

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Get OpenAI-compatible tool schemas

        Returns:
            List of tool schemas
        """
        schemas = []
        for tool in self.tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            })

        return schemas

    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Execute a tool with given parameters

        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool parameters

        Returns:
            Tool execution result
        """
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found"
            )

        tool = self.tools[tool_name]

        try:
            # Execute the tool
            result = tool.handler(**kwargs)
            logger.info(f"Executed tool: {tool_name}")
            return result

        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            return ToolResult(
                success=False,
                error=f"Tool execution failed: {str(e)}"
            )

    def execute_tools_from_response(self, response_text: str) -> List[ToolResult]:
        """
        Parse and execute tools from model response

        Args:
            response_text: Model response that may contain tool calls

        Returns:
            List of tool execution results
        """
        results = []

        # Look for tool call patterns
        tool_call_pattern = r'TOOL_CALL:\s*(\{.*?\})'
        matches = re.findall(tool_call_pattern, response_text, re.DOTALL)

        for match in matches:
            try:
                tool_call = json.loads(match.strip())
                tool_name = tool_call.get("tool")
                arguments = tool_call.get("arguments", {})

                if tool_name:
                    result = self.execute_tool(tool_name, **arguments)
                    results.append(result)

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool call: {e}")
                continue

        return results

    # Built-in tool implementations

    def _execute_python(self, code: str, timeout: int = 30) -> ToolResult:
        """Execute Python code safely"""
        try:
            # Create a temporary file for execution
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            # Execute the code
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_directory
            )

            # Clean up temp file
            os.unlink(temp_file)

            return ToolResult(
                success=result.returncode == 0,
                result={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                },
                metadata={"execution_time": "completed"}
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Code execution timed out after {timeout} seconds"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Code execution failed: {str(e)}"
            )

    def _execute_shell(self, command: str, timeout: int = 30, working_directory: str = None) -> ToolResult:
        """Execute shell command"""
        try:
            cwd = Path(working_directory) if working_directory else self.working_directory

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd
            )

            return ToolResult(
                success=result.returncode == 0,
                result={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "command": command
                },
                metadata={"working_directory": str(cwd)}
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Command timed out after {timeout} seconds"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Command execution failed: {str(e)}"
            )

    def _read_file(self, path: str, encoding: str = "utf-8") -> ToolResult:
        """Read file contents"""
        try:
            file_path = self.working_directory / path
            file_path = file_path.resolve()  # Prevent directory traversal

            # Security check - ensure file is within working directory
            if not str(file_path).startswith(str(self.working_directory.resolve())):
                return ToolResult(
                    success=False,
                    error="Access denied: File outside working directory"
                )

            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()

            return ToolResult(
                success=True,
                result={
                    "content": content,
                    "path": str(file_path),
                    "size": len(content)
                }
            )

        except FileNotFoundError:
            return ToolResult(
                success=False,
                error=f"File not found: {path}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to read file: {str(e)}"
            )

    def _write_file(self, path: str, content: str, encoding: str = "utf-8") -> ToolResult:
        """Write content to file"""
        try:
            file_path = self.working_directory / path
            file_path = file_path.resolve()

            # Security check
            if not str(file_path).startswith(str(self.working_directory.resolve())):
                return ToolResult(
                    success=False,
                    error="Access denied: Cannot write outside working directory"
                )

            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)

            return ToolResult(
                success=True,
                result={
                    "path": str(file_path),
                    "size": len(content),
                    "message": "File written successfully"
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to write file: {str(e)}"
            )

    def _list_directory(self, path: str = ".", recursive: bool = False) -> ToolResult:
        """List directory contents"""
        try:
            dir_path = self.working_directory / path
            dir_path = dir_path.resolve()

            # Security check
            if not str(dir_path).startswith(str(self.working_directory.resolve())):
                return ToolResult(
                    success=False,
                    error="Access denied: Directory outside working directory"
                )

            items = []
            if recursive:
                for item_path in dir_path.rglob("*"):
                    if item_path.is_file():
                        items.append({
                            "name": str(item_path.relative_to(dir_path)),
                            "type": "file",
                            "size": item_path.stat().st_size,
                            "modified": item_path.stat().st_mtime
                        })
            else:
                for item_path in dir_path.iterdir():
                    items.append({
                        "name": item_path.name,
                        "type": "directory" if item_path.is_dir() else "file",
                        "size": item_path.stat().st_size if item_path.is_file() else 0,
                        "modified": item_path.stat().st_mtime
                    })

            return ToolResult(
                success=True,
                result={
                    "path": str(dir_path),
                    "items": items,
                    "count": len(items)
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to list directory: {str(e)}"
            )

    def _create_directory(self, path: str) -> ToolResult:
        """Create a new directory"""
        try:
            dir_path = self.working_directory / path
            dir_path = dir_path.resolve()

            # Security check
            if not str(dir_path).startswith(str(self.working_directory.resolve())):
                return ToolResult(
                    success=False,
                    error="Access denied: Cannot create directory outside working directory"
                )

            dir_path.mkdir(parents=True, exist_ok=True)

            return ToolResult(
                success=True,
                result={
                    "path": str(dir_path),
                    "message": "Directory created successfully"
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to create directory: {str(e)}"
            )

    def _git_status(self, path: str = ".") -> ToolResult:
        """Get git repository status"""
        try:
            repo_path = self.working_directory / path

            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=repo_path
            )

            # Get branch info
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                cwd=repo_path
            )

            return ToolResult(
                success=True,
                result={
                    "status": result.stdout.strip(),
                    "branch": branch_result.stdout.strip(),
                    "clean": len(result.stdout.strip()) == 0
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Git operation failed: {str(e)}"
            )

    def _git_commit(self, message: str, path: str = ".") -> ToolResult:
        """Commit changes to git repository"""
        try:
            repo_path = self.working_directory / path

            # Stage all changes
            subprocess.run(
                ["git", "add", "."],
                cwd=repo_path,
                check=True
            )

            # Commit
            result = subprocess.run(
                ["git", "commit", "-m", message],
                capture_output=True,
                text=True,
                cwd=repo_path
            )

            return ToolResult(
                success=result.returncode == 0,
                result={
                    "message": message,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Git commit failed: {str(e)}"
            )

    def _web_search(self, query: str, max_results: int = 5) -> ToolResult:
        """Search the web (placeholder implementation)"""
        try:
            # This is a placeholder - in a real implementation you'd use
            # a search API like Google Custom Search, DuckDuckGo, etc.

            # For now, simulate a search result
            results = [
                {
                    "title": f"Search result for: {query}",
                    "url": f"https://example.com/search?q={query.replace(' ', '+')}",
                    "snippet": f"This is a simulated search result for the query: {query}"
                }
            ]

            return ToolResult(
                success=True,
                result={
                    "query": query,
                    "results": results[:max_results]
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Web search failed: {str(e)}"
            )

    def _http_request(self, url: str, method: str = "GET", headers: Dict = None, data: str = None) -> ToolResult:
        """Make an HTTP request"""
        try:
            headers = headers or {}
            headers.setdefault("User-Agent", "AIVA-Tool-System/1.0")

            response = requests.request(
                method=method.upper(),
                url=url,
                headers=headers,
                data=data,
                timeout=30
            )

            return ToolResult(
                success=True,
                result={
                    "url": url,
                    "method": method,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "content": response.text[:10000],  # Limit content size
                    "content_length": len(response.text)
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"HTTP request failed: {str(e)}"
            )

    def _analyze_code(self, code: str, language: str = "python") -> ToolResult:
        """Analyze code for issues and improvements"""
        try:
            analysis = {
                "language": language,
                "lines": len(code.split('\n')),
                "characters": len(code),
                "issues": [],
                "suggestions": []
            }

            # Basic analysis
            if language.lower() == "python":
                # Check for common issues
                if "print(" in code and "f\"" not in code:
                    analysis["suggestions"].append("Consider using f-strings for string formatting")

                if "import os" in code and "import sys" in code:
                    analysis["issues"].append("Consider organizing imports alphabetically")

            analysis["complexity"] = "low" if analysis["lines"] < 50 else "medium" if analysis["lines"] < 200 else "high"

            return ToolResult(
                success=True,
                result=analysis
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Code analysis failed: {str(e)}"
            )

    def _analyze_project(self, path: str = ".") -> ToolResult:
        """Analyze project structure"""
        try:
            project_path = self.working_directory / path

            analysis = {
                "path": str(project_path),
                "languages": {},
                "file_count": 0,
                "total_size": 0,
                "structure": {}
            }

            # Analyze files
            for file_path in project_path.rglob("*"):
                if file_path.is_file():
                    analysis["file_count"] += 1
                    analysis["total_size"] += file_path.stat().st_size

                    # Detect language by extension
                    ext = file_path.suffix.lower()
                    if ext:
                        analysis["languages"][ext] = analysis["languages"].get(ext, 0) + 1

            return ToolResult(
                success=True,
                result=analysis
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Project analysis failed: {str(e)}"
            )

# Convenience functions
def create_tool_system(working_directory: str = "./workspace") -> AIVAToolSystem:
    """
    Create and return a new AIVA tool system instance

    Args:
        working_directory: Working directory for file operations

    Returns:
        AIVAToolSystem instance
    """
    return AIVAToolSystem(working_directory=working_directory)

def test_tool_system():
    """Test function for the tool system"""
    tool_system = create_tool_system()

    print(f"Available tools: {list(tool_system.tools.keys())}")

    # Test file operations
    result = tool_system.execute_tool("create_directory", path="test_dir")
    print(f"Create directory: {result.success}")

    result = tool_system.execute_tool("write_file", path="test_dir/test.txt", content="Hello, World!")
    print(f"Write file: {result.success}")

    result = tool_system.execute_tool("read_file", path="test_dir/test.txt")
    print(f"Read file: {result.success}, content: {result.result}")

    result = tool_system.execute_tool("list_directory", path="test_dir")
    print(f"List directory: {result.success}, items: {len(result.result['items']) if result.result else 0}")

    # Test code execution
    result = tool_system.execute_tool("execute_python", code="print('Hello from tool system!')")
    print(f"Execute code: {result.success}")

    print("Tool system test completed successfully!")

if __name__ == "__main__":
    test_tool_system()
