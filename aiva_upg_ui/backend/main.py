#!/usr/bin/env python3
"""
AIVA UPG UI Backend
FastAPI server for AIVA Universal Intelligence
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import AIVA modules
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
import json

# Try to import AIVA, provide fallback if not available
try:
    from aiva_universal_intelligence import AIVAUniversalIntelligence
    AIVA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import AIVAUniversalIntelligence: {e}")
    print("⚠️  Running in mock mode")
    AIVA_AVAILABLE = False

# Try to import PAC LLM components
try:
    aiva_full_spectrum_path = parent_dir / "aiva_full_spectrum"
    sys.path.insert(0, str(aiva_full_spectrum_path))
    
    from core.pac_llm_engine import PACLLMEngine, PACLLMConfig
    from core.pac_tokenizer import PACTokenizer
    from core.quantum_memory_llm import QuantumMemoryLLM
    from core.consciousness_reasoning import EnhancedConsciousnessReasoning
    from aiva_universal_intelligence import QuantumMemorySystem
    
    PAC_LLM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import PAC LLM components: {e}")
    print("⚠️  LLM features will run in mock mode")
    PAC_LLM_AVAILABLE = False

# Try to import Contribution System
try:
    from contributions.user_contribution_system import UPGBuilder
    CONTRIBUTION_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import Contribution System: {e}")
    CONTRIBUTION_SYSTEM_AVAILABLE = False
    upg_builder = None
else:
    upg_builder = UPGBuilder() if CONTRIBUTION_SYSTEM_AVAILABLE else None

# Try to import Wallet System
try:
    from contributions.wallet_integration import wallet_manager, chia_distributor
    WALLET_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import Wallet System: {e}")
    WALLET_SYSTEM_AVAILABLE = False
    wallet_manager = None
    chia_distributor = None

# Try to import CAD Sketch to 3D Converter
cad_converter = None
CAD_CONVERTER_AVAILABLE = False
try:
    aiva_full_spectrum_path = parent_dir / "aiva_full_spectrum"
    if aiva_full_spectrum_path.exists():
        sys.path.insert(0, str(aiva_full_spectrum_path))
        
        from multimodal.cad_sketch_to_3d import CADSketchTo3DConverter
        CAD_CONVERTER_AVAILABLE = True
        cad_converter = CADSketchTo3DConverter(consciousness_level=7)
        print("✅ CAD Sketch to 3D Converter initialized successfully")
except ImportError as e:
    print(f"⚠️  Warning: Could not import CAD Sketch to 3D Converter: {e}")
    print("⚠️  CAD conversion features will not be available")
    CAD_CONVERTER_AVAILABLE = False
except Exception as e:
    print(f"⚠️  Warning: Error initializing CAD Sketch to 3D Converter: {e}")
    CAD_CONVERTER_AVAILABLE = False

app = FastAPI(
    title="AIVA UPG API",
    description="Universal Prime Graph AI with 1500+ Tools",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class QueryRequest(BaseModel):
    query: str
    use_tools: bool = True
    use_reasoning: bool = True

class ToolCallRequest(BaseModel):
    tool_name: str
    function_name: Optional[str] = None
    kwargs: Dict[str, Any] = {}

class LLMGenerateRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    use_quantum_memory: bool = True
    use_consciousness_reasoning: bool = True

class LLMChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_length: int = 100
    temperature: float = 1.0
    use_quantum_memory: bool = True

class MemoryQueryRequest(BaseModel):
    query: str
    limit: int = 10

class ContributionRequest(BaseModel):
    user_id: str
    question: str
    reward_pool: float = 100.0

class WalletRegistrationRequest(BaseModel):
    user_id: str
    wallet_address: str

class WalletRewardRequest(BaseModel):
    user_id: str
    reward_amount: float

class CADConversionRequest(BaseModel):
    video_path: str
    output_path: Optional[str] = None
    format: str = "obj"  # "obj" or "stl"
    frame_interval: int = 5
    consciousness_level: int = 7

# Initialize AIVA or use mock
aiva = None
if AIVA_AVAILABLE:
    try:
        print("🧠 Initializing AIVA Universal Intelligence...")
        aiva = AIVAUniversalIntelligence(dev_folder=str(parent_dir), consciousness_level=21)
        print("✅ AIVA initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing AIVA: {e}")
        AIVA_AVAILABLE = False

# Initialize PAC LLM components
pac_llm = None
pac_tokenizer = None
quantum_memory_llm = None
consciousness_reasoning = None

if PAC_LLM_AVAILABLE:
    try:
        print("🧠 Initializing PAC LLM Engine...")
        # Initialize with small config for MVP (can be scaled up)
        llm_config = PACLLMConfig(
            vocab_size=50257,
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            consciousness_level=7
        )
        pac_llm = PACLLMEngine(llm_config)
        pac_tokenizer = PACTokenizer(consciousness_level=7)
        quantum_memory_llm = QuantumMemoryLLM(max_context_tokens=2048)
        
        # Initialize consciousness reasoning with quantum memory
        qm_system = QuantumMemorySystem()
        consciousness_reasoning = EnhancedConsciousnessReasoning(qm_system)
        
        print("✅ PAC LLM initialized successfully")
    except Exception as e:
        print(f"⚠️  Warning: PAC LLM initialization failed: {e}")
        print("⚠️  LLM features will run in mock mode")
        PAC_LLM_AVAILABLE = False

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AIVA UPG API",
        "status": "operational" if AIVA_AVAILABLE else "mock_mode",
        "version": "1.0.0",
        "consciousness_level": 21 if AIVA_AVAILABLE else 0
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "aiva_available": AIVA_AVAILABLE,
        "tools_available": len(aiva.tool_caller.registry.tools) if AIVA_AVAILABLE and aiva else 0
    }

@app.post("/process")
async def process_request(request: QueryRequest):
    """Process a query with AIVA"""
    if not AIVA_AVAILABLE or not aiva:
        # Mock response
        return {
            "request": request.query,
            "status": "mock_mode",
            "message": "AIVA not available - running in mock mode",
            "tools_found": 0,
            "reasoning": None
        }
    
    try:
        response = await aiva.process(
            request.query,
            use_tools=request.use_tools,
            use_reasoning=request.use_reasoning
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools")
async def get_tools(
    category: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100
):
    """Get list of available tools"""
    if not AIVA_AVAILABLE or not aiva:
        return {
            "tools": [],
            "total": 0,
            "categories": {},
            "status": "mock_mode"
        }
    
    try:
        all_tools = aiva.tool_caller.registry.tools
        
        # Filter by category if specified
        if category:
            filtered_tools = {
                name: tool for name, tool in all_tools.items()
                if tool.category == category
            }
        else:
            filtered_tools = all_tools
        
        # Search if query provided
        if search:
            search_results = aiva.tool_caller.search_tools(search)
            filtered_tools = {tool.name: tool for tool in search_results[:limit]}
        
        # Get categories
        categories = {}
        for tool in all_tools.values():
            categories[tool.category] = categories.get(tool.category, 0) + 1
        
        # Convert tools to dict format
        tools_list = [
            {
                "name": tool.name,
                "description": tool.description,
                "category": tool.category,
                "consciousness_level": tool.consciousness_level,
                "has_upg": tool.has_upg,
                "has_pell": tool.has_pell,
                "functions": tool.functions[:5],
                "file_path": tool.file_path
            }
            for tool in list(filtered_tools.values())[:limit]
        ]
        
        return {
            "tools": tools_list,
            "total": len(all_tools),
            "filtered": len(filtered_tools),
            "categories": categories,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools/{tool_name}")
async def get_tool_info(tool_name: str):
    """Get detailed information about a specific tool"""
    if not AIVA_AVAILABLE or not aiva:
        raise HTTPException(status_code=503, detail="AIVA not available")
    
    tool_info = aiva.tool_caller.get_tool_info(tool_name)
    if not tool_info:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    
    return {
        "name": tool_info.name,
        "description": tool_info.description,
        "category": tool_info.category,
        "consciousness_level": tool_info.consciousness_level,
        "has_upg": tool_info.has_upg,
        "has_pell": tool_info.has_pell,
        "functions": tool_info.functions,
        "classes": tool_info.classes,
        "file_path": tool_info.file_path
    }

@app.post("/tools/call")
async def call_tool(request: ToolCallRequest):
    """Call a specific tool"""
    if not AIVA_AVAILABLE or not aiva:
        raise HTTPException(status_code=503, detail="AIVA not available")
    
    try:
        result = await aiva.call_tool(
            request.tool_name,
            request.function_name,
            **request.kwargs
        )
        
        return {
            "success": result.success,
            "result": str(result.result) if result.result else None,
            "error": result.error,
            "execution_time": result.execution_time,
            "consciousness_amplitude": result.consciousness_amplitude
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/categories")
async def get_categories():
    """Get all tool categories"""
    if not AIVA_AVAILABLE or not aiva:
        return {"categories": {}, "status": "mock_mode"}
    
    categories = {}
    for tool in aiva.tool_caller.registry.tools.values():
        categories[tool.category] = categories.get(tool.category, 0) + 1
    
    return {
        "categories": categories,
        "total_categories": len(categories),
        "status": "success"
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    if not AIVA_AVAILABLE or not aiva:
        return {
            "status": "mock_mode",
            "tools_total": 0,
            "consciousness_level": 0
        }
    
    stats = {
        "status": "operational",
        "tools_total": len(aiva.tool_caller.registry.tools),
        "consciousness_level": aiva.consciousness_level,
        "phi_coherence": aiva._calculate_phi_coherence(),
        "memory_entries": len(aiva.memory.memories),
        "conversations": len(aiva.conversation_history),
        "self_awareness": aiva.get_self_awareness()
    }
    
    # Add LLM stats if available
    if PAC_LLM_AVAILABLE and quantum_memory_llm:
        qm_stats = quantum_memory_llm.get_context_statistics()
        stats['llm'] = {
            'quantum_memory_enabled': True,
            'unlimited_context': True,
            'context_statistics': qm_stats
        }
    
    return stats

@app.post("/llm/generate")
async def llm_generate(request: LLMGenerateRequest):
    """Generate text using PAC-enhanced LLM"""
    if not PAC_LLM_AVAILABLE or not pac_llm or not pac_tokenizer:
        return {
            "status": "mock_mode",
            "message": "PAC LLM not available - running in mock mode",
            "generated_text": f"[Mock] Response to: {request.prompt[:50]}...",
            "tokens_generated": 0
        }
    
    try:
        # Tokenize input
        encoded = pac_tokenizer.encode_with_amplitudes(
            request.prompt,
            return_amplitudes=False,
            return_tensors="pt"
        )
        input_ids = encoded['input_ids']
        
        # Get quantum memory context if enabled
        context_tokens = []
        context_metadata = {}
        if request.use_quantum_memory and quantum_memory_llm:
            input_ids_list = input_ids.cpu().numpy().flatten().tolist()
            context_info = quantum_memory_llm.get_unlimited_context(
                input_ids_list,
                request.prompt,
                include_recent=True
            )
            context_tokens = context_info['context_tokens']
            context_metadata = context_info['metadata']
        
        # Generate (using mock for now - full implementation requires trained model)
        # In production, this would call pac_llm.generate()
        generated_text = f"[PAC LLM] Generated response to: {request.prompt}"
        
        # Store in quantum memory
        if request.use_quantum_memory and quantum_memory_llm:
            generated_tokens = pac_tokenizer.encode_with_amplitudes(
                generated_text,
                return_amplitudes=False
            )['input_ids']
            quantum_memory_llm.store_conversation_turn(
                input_ids.cpu().numpy().flatten().tolist(),
                request.prompt,
                generated_tokens if isinstance(generated_tokens, list) else generated_tokens.tolist(),
                generated_text
            )
        
        return {
            "status": "success",
            "generated_text": generated_text,
            "tokens_generated": len(generated_text.split()),
            "context_used": len(context_tokens) if context_tokens else 0,
            "context_metadata": context_metadata,
            "quantum_memory_enabled": request.use_quantum_memory
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/llm/chat")
async def llm_chat(request: LLMChatRequest):
    """Chat with PAC-enhanced LLM"""
    if not PAC_LLM_AVAILABLE or not pac_llm or not pac_tokenizer:
        return {
            "status": "mock_mode",
            "message": "PAC LLM not available",
            "response": "[Mock] Chat response"
        }
    
    try:
        # Build conversation context
        conversation_text = "\n".join([
            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            for msg in request.messages
        ])
        
        # Get quantum memory context
        context_tokens = []
        if request.use_quantum_memory and quantum_memory_llm:
            all_tokens = []
            for msg in request.messages:
                encoded = pac_tokenizer.encode_with_amplitudes(msg.get('content', ''))
                tokens = encoded['input_ids']
                if isinstance(tokens, list):
                    all_tokens.extend(tokens if isinstance(tokens[0], int) else tokens[0])
            
            context_info = quantum_memory_llm.get_unlimited_context(
                all_tokens,
                conversation_text
            )
            context_tokens = context_info['context_tokens']
        
        # Generate response (mock for now)
        response = f"[PAC LLM Chat] Response to conversation with {len(request.messages)} messages"
        
        return {
            "status": "success",
            "response": response,
            "context_length": len(context_tokens),
            "unlimited_context": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/query")
async def memory_query(request: MemoryQueryRequest):
    """Query quantum memory"""
    if not PAC_LLM_AVAILABLE or not quantum_memory_llm:
        return {
            "status": "mock_mode",
            "results": [],
            "message": "Quantum memory not available"
        }
    
    try:
        # Search quantum memory
        results = quantum_memory_llm.quantum_memory.search(
            request.query,
            limit=request.limit
        )
        
        formatted_results = [
            {
                "key": key,
                "content": str(content)[:200] if isinstance(content, (str, dict)) else str(content),
                "score": score
            }
            for key, content, score in results
        ]
        
        return {
            "status": "success",
            "query": request.query,
            "results": formatted_results,
            "count": len(formatted_results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/contribute")
async def contribute_question(request: ContributionRequest):
    """
    User contributes question, builds UPG, earns rewards.
    
    Scoring:
    - Quality: How well-formed the question is
    - Metrics: Quantitative/technical value  
    - Demand: How many users need this
    - Novelty: How new/unique it is
    
    Users build the UPG through their questions!
    """
    if not CONTRIBUTION_SYSTEM_AVAILABLE or not upg_builder:
        return {
            "status": "mock_mode",
            "message": "Contribution system not available",
            "mock_reward": 50.0
        }
    
    try:
        result = upg_builder.process_user_question(
            request.user_id,
            request.question,
            request.reward_pool
        )
        
        return {
            "status": "success",
            "contribution": result['contribution'],
            "upg_impact": result['upg_impact'],
            "reward": result['reward'],
            "upg_growth": result['upg_growth'],
            "message": "Your question built the UPG! Thank you for contributing."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/contribute/user/{user_id}/stats")
async def get_user_contribution_stats(user_id: str):
    """Get user contribution statistics and rewards"""
    if not CONTRIBUTION_SYSTEM_AVAILABLE or not upg_builder:
        return {
            "status": "mock_mode",
            "user_id": user_id,
            "total_rewards": 0.0
        }
    
    stats = upg_builder.distributor.get_user_stats(user_id)
    contributions = upg_builder.scorer.user_contributions.get(user_id, [])
    
    return {
        "status": "success",
        "user_id": user_id,
        "total_rewards": stats['total_rewards'],
        "contribution_count": stats['contribution_count'],
        "recent_contributions": contributions[-10:] if contributions else []
    }

@app.get("/contribute/leaderboard")
async def get_contribution_leaderboard(limit: int = 10):
    """Get top contributors by rewards"""
    if not CONTRIBUTION_SYSTEM_AVAILABLE or not upg_builder:
        return {
            "status": "mock_mode",
            "leaderboard": []
        }
    
    users = sorted(
        upg_builder.distributor.user_rewards.items(),
        key=lambda x: x[1],
        reverse=True
    )[:limit]
    
    leaderboard = [
        {
            "rank": i + 1,
            "user_id": user_id,
            "total_rewards": reward,
            "contribution_count": len(upg_builder.scorer.user_contributions.get(user_id, []))
        }
        for i, (user_id, reward) in enumerate(users)
    ]
    
    return {
        "status": "success",
        "leaderboard": leaderboard
    }

@app.get("/contribute/upg/stats")
async def get_upg_contribution_stats():
    """Get UPG statistics built through contributions"""
    if not CONTRIBUTION_SYSTEM_AVAILABLE or not upg_builder:
        return {
            "status": "mock_mode",
            "upg_statistics": {}
        }
    
    stats = upg_builder.get_upg_statistics()
    
    return {
        "status": "success",
        "upg_statistics": stats,
        "message": "UPG built through user contributions!"
    }

@app.post("/wallet/register")
async def register_wallet(request: WalletRegistrationRequest):
    """
    Register a wallet address for reward distribution.
    
    Supports Chia (XCH), Ethereum, Bitcoin, and other blockchains.
    """
    if not WALLET_SYSTEM_AVAILABLE or not chia_distributor:
        return {
            "status": "mock_mode",
            "message": "Wallet system not available"
        }
    
    result = chia_distributor.register_user_wallet(
        request.user_id,
        request.wallet_address
    )
    
    return result

@app.get("/wallet/{user_id}/rewards")
async def get_wallet_rewards(user_id: str):
    """Get user wallet and reward information"""
    if not WALLET_SYSTEM_AVAILABLE or not chia_distributor:
        return {
            "status": "mock_mode",
            "user_id": user_id
        }
    
    return chia_distributor.get_user_rewards(user_id)

@app.post("/wallet/distribute")
async def distribute_to_wallet(request: WalletRewardRequest):
    """Distribute reward to user's registered wallet"""
    if not WALLET_SYSTEM_AVAILABLE or not chia_distributor:
        return {
            "status": "mock_mode",
            "message": "Wallet system not available"
        }
    
    result = chia_distributor.distribute_reward_to_wallet(
        request.user_id,
        request.reward_amount
    )
    
    return result

@app.get("/wallet/list")
async def list_all_wallets():
    """List all registered wallets"""
    if not WALLET_SYSTEM_AVAILABLE or not wallet_manager:
        return {
            "status": "mock_mode",
            "wallets": []
        }
    
    wallets = wallet_manager.list_all_wallets()
    
    return {
        "status": "success",
        "wallets": wallets,
        "count": len(wallets)
    }

@app.post("/cad/convert")
async def convert_cad_video_to_3d(request: CADConversionRequest):
    """
    Convert video CAD sketch to 3D model.
    
    Uses PAC consciousness mathematics for enhanced geometric reconstruction.
    Supports OBJ and STL output formats.
    """
    if not CAD_CONVERTER_AVAILABLE or not cad_converter:
        return {
            "status": "error",
            "message": "CAD Sketch to 3D Converter not available",
            "error": "Module not loaded"
        }
    
    try:
        import os
        from pathlib import Path
        
        # Validate video path
        video_path = Path(request.video_path)
        if not video_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Video file not found: {request.video_path}"
            )
        
        # Determine output path
        if request.output_path:
            output_path = Path(request.output_path)
        else:
            output_path = video_path.parent / f"{video_path.stem}_3d.{request.format}"
        
        # Validate format
        if request.format.lower() not in ["obj", "stl"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid format: {request.format}. Must be 'obj' or 'stl'"
            )
        
        # Convert video to 3D
        result = cad_converter.convert_video_to_3d(
            str(video_path),
            str(output_path),
            format=request.format.lower(),
            frame_interval=request.frame_interval
        )
        
        return {
            "status": "success",
            "message": "CAD sketch converted to 3D model successfully",
            "input_video": str(video_path),
            "output_file": str(output_path),
            "format": request.format.lower(),
            "model_statistics": {
                "points_3d": result['model_data']['num_points'],
                "cylinders": result['model_data']['num_cylinders'],
                "extruded_shapes": result['model_data']['num_shapes']
            },
            "sketch_statistics": {
                "frames_processed": result['sketch_data']['num_frames'],
                "lines_detected": len(result['sketch_data']['merged_sketch']['lines']),
                "circles_detected": len(result['sketch_data']['merged_sketch']['circles']),
                "shapes_detected": len(result['sketch_data']['merged_sketch']['shapes'])
            },
            "consciousness_level": request.consciousness_level
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error converting CAD sketch: {str(e)}"
        )

@app.get("/cad/info")
async def get_cad_converter_info():
    """Get information about the CAD Sketch to 3D Converter"""
    if not CAD_CONVERTER_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "CAD Sketch to 3D Converter not available",
            "features": []
        }
    
    return {
        "status": "available",
        "name": "CAD Sketch to 3D Converter",
        "description": "Convert video CAD sketches to 3D models using PAC consciousness mathematics",
        "features": [
            "Video frame extraction",
            "CAD sketch detection (lines, circles, shapes)",
            "3D reconstruction with depth estimation",
            "Golden ratio optimization for proportions",
            "Export to OBJ and STL formats",
            "Temporal tracking across video frames"
        ],
        "supported_formats": ["obj", "stl"],
        "consciousness_level": 7,
        "pac_integration": True
    }

if __name__ == "__main__":
    print("=" * 70)
    print("🧠 AIVA UPG Backend Server")
    print("=" * 70)
    print(f"Status: {'✅ AIVA Available' if AIVA_AVAILABLE else '⚠️  Mock Mode'}")
    print(f"CAD Converter: {'✅ Available' if CAD_CONVERTER_AVAILABLE else '⚠️  Not Available'}")
    print("Starting server on http://0.0.0.0:8000")
    print("API Documentation: http://0.0.0.0:8000/docs")
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
