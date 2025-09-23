#!/usr/bin/env python3
"""
SquashPlot API Server - Production-Ready Backend
===============================================

FastAPI-based server providing REST API endpoints for SquashPlot operations.
Built following Replit template architecture with Andy's CLI improvements integrated.

Features:
- Real-time server monitoring (Andy's check_server.py logic)
- CLI command execution and templates
- Compression operations and validation
- WebSocket support for live updates
- Professional error handling and logging
"""

import asyncio
import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

import psutil
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import FileResponse

# SquashPlot Core Imports - Andy's Graceful Fallback Approach
try:
    from squashplot import SquashPlotCompressor
    SQUASHPLOT_AVAILABLE = True
    print("✅ SquashPlot core compression engine loaded")
except (ImportError, NameError) as e:
    SQUASHPLOT_AVAILABLE = False
    print(f"⚠️ SquashPlot compression engine not available: {e}")
    print("🔄 Running in demo mode with CLI integration")

# Andy's check_server utility
try:
    from check_server import check_server
    CHECK_SERVER_AVAILABLE = True
    print("✅ Andy's check_server utility loaded")
except ImportError:
    CHECK_SERVER_AVAILABLE = False
    print("⚠️ check_server utility not available - using fallback")

# Configuration
class Config:
    TITLE = "SquashPlot API Server"
    VERSION = "2.0.0"
    DESCRIPTION = "Professional Chia Plot Compression API with Andy's CLI Integration"

    # Server settings
    HOST = "0.0.0.0"
    PORT = int(os.getenv("PORT", "8080"))

    # Replit-specific optimizations
    REPLIT_MODE = os.getenv("REPLIT", False)

    # Andy's CLI command templates
    CLI_COMMANDS = {
        "web": "python main.py --web",
        "cli": "python main.py --cli",
        "demo": "python main.py --demo",
        "check_server": "python check_server.py",
        "basic_plotting": "python squashplot.py -t /tmp/plot1 -d /plots -f YOUR_FARMER_KEY -p YOUR_POOL_KEY",
        "dual_temp": "python squashplot.py -t /tmp/plot1 -2 /tmp/plot2 -d /plots -f YOUR_FARMER_KEY -p YOUR_POOL_KEY -n 2",
        "with_compression": "python squashplot.py --compress 3 -t /tmp/plot1 -d /plots -f YOUR_FARMER_KEY -p YOUR_POOL_KEY"
    }

# Initialize FastAPI app
app = FastAPI(
    title=Config.TITLE,
    version=Config.VERSION,
    description=Config.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for Replit deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replit handles CORS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# WebSocket manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                self.disconnect(connection)

manager = ConnectionManager()

# Pydantic models
class CompressionRequest(BaseModel):
    input_file: str
    output_file: str
    level: int = 3
    algorithm: str = "auto"

class CLICommandRequest(BaseModel):
    command: str
    timeout: int = 30

class ServerStatus(BaseModel):
    status: str
    uptime: float
    timestamp: str
    version: str = Config.VERSION

# Routes

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main SquashPlot web interface"""
    try:
        return FileResponse("squashplot_web_interface.html", media_type="text/html")
    except FileNotFoundError:
        return HTMLResponse("""
        <h1>SquashPlot API Server</h1>
        <p>Web interface not found. API docs available at <a href="/docs">/docs</a></p>
        """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": Config.VERSION,
        "squashplot_available": SQUASHPLOT_AVAILABLE
    }

@app.get("/status", response_model=ServerStatus)
async def get_status():
    """Get comprehensive server status (Andy's check_server.py logic)"""
    start_time = time.time() - psutil.boot_time()

    # System information
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    # Check SquashPlot availability
    status = "online" if SQUASHPLOT_AVAILABLE else "limited"

    return ServerStatus(
        status=status,
        uptime=start_time,
        timestamp=datetime.now().isoformat(),
        version=Config.VERSION
    )

@app.get("/system-info")
async def system_info():
    """Detailed system information"""
    return {
        "cpu": {
            "cores": psutil.cpu_count(),
            "usage_percent": psutil.cpu_percent(interval=0.1)
        },
        "memory": {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
            "available_gb": round(psutil.virtual_memory().available / (1024**3), 2)
        },
        "disk": {
            "total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
            "used_gb": round(psutil.disk_usage('/').used / (1024**3), 2),
            "free_gb": round(psutil.disk_usage('/').free / (1024**3), 2)
        },
        "replit_mode": Config.REPLIT_MODE,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/cli-commands")
async def get_cli_commands():
    """Get available CLI commands (Andy's templates)"""
    return {
        "commands": Config.CLI_COMMANDS,
        "note": "These are command templates. Execute them in your local terminal for full functionality.",
        "web_execution": "Available through web interface for command preview and templates"
    }

@app.post("/cli/execute")
async def execute_cli_command(request: CLICommandRequest):
    """Execute CLI command (simulation - actual execution requires terminal)"""
    command = request.command.strip()

    # Security check - only allow safe SquashPlot commands
    allowed_commands = [
        "python main.py",
        "python check_server.py",
        "python squashplot.py"
    ]

    if not any(cmd in command for cmd in allowed_commands):
        raise HTTPException(status_code=400, detail="Command not allowed for web execution")

    # Simulate command execution (in real deployment, this would execute safely)
    return {
        "command": command,
        "status": "simulated",
        "output": f"Command would execute: {command}\n\nNote: For full functionality, run this command in your local terminal.",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/compress")
async def compress_file(request: CompressionRequest):
    """Compress a file using SquashPlot"""
    if not SQUASHPLOT_AVAILABLE:
        raise HTTPException(status_code=503, detail="SquashPlot compression engine not available")

    try:
        # Initialize compressor
        compressor = SquashPlotCompressor()

        # Perform compression
        result = compressor.compress_plot(
            request.input_file,
            request.output_file,
            compression_level=request.level
        )

        return {
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compression failed: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                # Handle WebSocket messages
                response = {"type": "echo", "data": message, "timestamp": datetime.now().isoformat()}
                await manager.send_personal_message(json.dumps(response), websocket)
            except json.JSONDecodeError:
                await manager.send_personal_message(json.dumps({"error": "Invalid JSON"}), websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(json.dumps({
            "type": "disconnect",
            "message": "Client disconnected",
            "timestamp": datetime.now().isoformat()
        }))

@app.get("/api/status")
async def api_status():
    """Simple API status for frontend checks"""
    return {"status": "online", "version": Config.VERSION, "timestamp": datetime.now().isoformat()}

# Replit-specific optimizations
if Config.REPLIT_MODE:
    print("🔧 Running in Replit mode - optimizing for Replit environment")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    print("🚀 SquashPlot API Server starting up...")
    print(f"📡 Server will be available at: https://your-replit-url.replit.dev")
    print(f"🔗 Local access at: http://localhost:{Config.PORT}")
    print("📖 API documentation at: /docs")
    # Broadcast startup message
    await manager.broadcast(json.dumps({
        "type": "startup",
        "message": "SquashPlot API Server started",
        "version": Config.VERSION,
        "timestamp": datetime.now().isoformat()
    }))

if __name__ == "__main__":
    print("🧠 Starting SquashPlot API Server...")
    print(f"📡 Port: {Config.PORT}")
    print(f"🌐 Replit Mode: {Config.REPLIT_MODE}")

    uvicorn.run(
        "squashplot_api_server:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=True,
        log_level="info"
    )
