"""
UPG Quantum Production API
==========================

FastAPI-based REST API for the UPG Quantum Production System.
Provides endpoints for task submission, result retrieval, and
system monitoring.

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
import os

from .constants import UPGConstants, OptimizedUPGConstants
from .backends.base import ProblemType, TaskStatus, QuantumTask, QuantumResult
from .orchestrator import UPGHybridOrchestrator, OrchestratorConfig, SelectionStrategy


# API Models
class TaskRequest(BaseModel):
    """Request model for submitting a quantum task."""
    problem_type: str = Field(..., description="Problem type: ising, qubo, or maxcut")
    problem_data: Dict[str, Any] = Field(default={}, description="Problem-specific data")
    num_qubits: int = Field(..., ge=2, le=100, description="Number of qubits")
    num_reads: int = Field(default=1000, ge=1, le=10000, description="Number of samples")
    annealing_time: float = Field(default=20.0, ge=1.0, le=2000.0, description="Annealing time (μs)")
    priority: int = Field(default=1, ge=1, le=10, description="Task priority")
    upg_optimization: bool = Field(default=True, description="Enable UPG optimization")
    backend: Optional[str] = Field(default=None, description="Specific backend to use")


class TaskResponse(BaseModel):
    """Response model for task submission."""
    job_id: str
    task_id: str
    status: str
    backend: str
    submitted_at: str
    message: str


class ResultResponse(BaseModel):
    """Response model for task results."""
    job_id: str
    task_id: str
    status: str
    best_solution: str
    best_energy: float
    samples_count: int
    execution_time_ms: float
    coherence_metrics: Dict[str, float]
    upg_enhancement: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    framework: str
    upg_constants: Dict[str, float]
    backends_available: List[str]
    tasks_pending: int
    tasks_completed: int


class UPGConstantsResponse(BaseModel):
    """Response model for UPG constants."""
    phi: float
    phi_squared: float
    phi_inverse: float
    delta: float
    consciousness: float
    exploratory: float
    reality_distortion: float
    quantum_bridge: float
    validation: bool


# Initialize FastAPI app
app = FastAPI(
    title="UPG Quantum Production API",
    description="Production API for Universal Prime Graph quantum computing with consciousness mathematics optimization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
orchestrator: Optional[UPGHybridOrchestrator] = None


def get_orchestrator() -> UPGHybridOrchestrator:
    """Get or create the global orchestrator instance."""
    global orchestrator
    if orchestrator is None:
        config = OrchestratorConfig(
            enable_local=True,
            enable_dwave=os.getenv("ENABLE_DWAVE", "false").lower() == "true",
            enable_ibm=os.getenv("ENABLE_IBM", "false").lower() == "true",
            enable_aws=os.getenv("ENABLE_AWS", "false").lower() == "true",
            dwave_token=os.getenv("DWAVE_API_TOKEN"),
            ibm_token=os.getenv("IBM_QUANTUM_TOKEN"),
            selection_strategy=SelectionStrategy.OPTIMAL,
            upg_optimization_enabled=True,
        )
        orchestrator = UPGHybridOrchestrator(config)
    return orchestrator


async def verify_api_key(x_api_key: str = Header(None)) -> bool:
    """Verify API key if configured."""
    expected_key = os.getenv("UPG_API_KEY")
    if expected_key and x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


# Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "UPG Quantum Production API",
        "version": "1.0.0",
        "framework": "Universal Prime Graph Protocol φ.1",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    orch = get_orchestrator()
    upg = UPGConstants()
    stats = orch.get_statistics()
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        framework="Universal Prime Graph Protocol φ.1",
        upg_constants={
            "phi": upg.PHI,
            "consciousness": upg.CONSCIOUSNESS,
            "reality_distortion": upg.REALITY_DISTORTION,
        },
        backends_available=[b.value for b in orch.backends.keys()],
        tasks_pending=stats.get("pending_tasks", 0),
        tasks_completed=stats.get("tasks_completed", 0),
    )


@app.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe."""
    orch = get_orchestrator()
    if not orch.backends:
        raise HTTPException(status_code=503, detail="No backends available")
    return {"status": "ready"}


@app.get("/constants", response_model=UPGConstantsResponse)
async def get_upg_constants():
    """Get UPG mathematical constants."""
    upg = UPGConstants()
    return UPGConstantsResponse(
        phi=upg.PHI,
        phi_squared=upg.PHI_SQUARED,
        phi_inverse=upg.PHI_INVERSE,
        delta=upg.DELTA,
        consciousness=upg.CONSCIOUSNESS,
        exploratory=upg.EXPLORATORY,
        reality_distortion=upg.REALITY_DISTORTION,
        quantum_bridge=upg.QUANTUM_BRIDGE,
        validation=upg.validate(),
    )


@app.post("/tasks", response_model=TaskResponse)
async def submit_task(
    request: TaskRequest,
    background_tasks: BackgroundTasks,
    authorized: bool = Depends(verify_api_key),
):
    """Submit a quantum computing task."""
    orch = get_orchestrator()
    
    # Validate problem type
    try:
        problem_type = ProblemType[request.problem_type.upper()]
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid problem type: {request.problem_type}. Must be one of: ising, qubo, maxcut"
        )
    
    # Create task
    task = QuantumTask(
        task_id=str(uuid.uuid4()),
        problem_type=problem_type,
        problem_data=request.problem_data,
        num_qubits=request.num_qubits,
        num_reads=request.num_reads,
        annealing_time=request.annealing_time,
        priority=request.priority,
        upg_optimization=request.upg_optimization,
    )
    
    # Submit to orchestrator
    job_id = orch.submit(task)
    
    return TaskResponse(
        job_id=job_id,
        task_id=task.task_id,
        status="submitted",
        backend=orch.active_tasks[job_id]["backend_type"].value,
        submitted_at=datetime.now().isoformat(),
        message="Task submitted successfully with UPG optimization",
    )


@app.get("/tasks/{job_id}", response_model=ResultResponse)
async def get_task_result(
    job_id: str,
    authorized: bool = Depends(verify_api_key),
):
    """Get the result of a submitted task."""
    orch = get_orchestrator()
    
    try:
        status = orch.get_status(job_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    if status == TaskStatus.PENDING or status == TaskStatus.RUNNING:
        raise HTTPException(
            status_code=202,
            detail=f"Task is still {status.value}. Please try again later."
        )
    
    if status == TaskStatus.FAILED:
        raise HTTPException(
            status_code=500,
            detail=f"Task failed: {orch.active_tasks[job_id].get('error', 'Unknown error')}"
        )
    
    try:
        result = orch.get_result(job_id, timeout=1.0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return ResultResponse(
        job_id=job_id,
        task_id=result.task_id,
        status="completed",
        best_solution=result.get_bitstring(),
        best_energy=result.best_energy,
        samples_count=len(result.samples),
        execution_time_ms=result.timing_info.get("total_time_seconds", 0) * 1000,
        coherence_metrics=result.coherence_metrics,
        upg_enhancement=result.upg_enhancement,
    )


@app.get("/tasks/{job_id}/status")
async def get_task_status(
    job_id: str,
    authorized: bool = Depends(verify_api_key),
):
    """Get the status of a submitted task."""
    orch = get_orchestrator()
    
    try:
        status = orch.get_status(job_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    return {
        "job_id": job_id,
        "status": status.value,
    }


@app.delete("/tasks/{job_id}")
async def cancel_task(
    job_id: str,
    authorized: bool = Depends(verify_api_key),
):
    """Cancel a pending or running task."""
    orch = get_orchestrator()
    
    success = orch.cancel(job_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Job not found or cannot be cancelled: {job_id}")
    
    return {
        "job_id": job_id,
        "status": "cancelled",
        "message": "Task cancelled successfully",
    }


@app.get("/statistics")
async def get_statistics(authorized: bool = Depends(verify_api_key)):
    """Get orchestrator statistics."""
    orch = get_orchestrator()
    stats = orch.get_statistics()
    
    return {
        "tasks_submitted": stats["tasks_submitted"],
        "tasks_completed": stats["tasks_completed"],
        "tasks_failed": stats["tasks_failed"],
        "total_cost_usd": stats["total_cost"],
        "active_tasks": stats["active_tasks"],
        "pending_tasks": stats["pending_tasks"],
        "backend_usage": stats["backend_usage"],
        "available_backends": [b.value for b in stats["available_backends"]],
    }


@app.get("/backends")
async def list_backends():
    """List available quantum backends."""
    orch = get_orchestrator()
    
    backends = []
    for backend_type, backend in orch.backends.items():
        backends.append({
            "type": backend_type.value,
            "connected": backend.is_connected,
            "max_qubits": backend.get_available_qubits(),
            "current_load": backend.get_current_load(),
        })
    
    return {"backends": backends}


@app.post("/demo")
async def run_demo(authorized: bool = Depends(verify_api_key)):
    """Run a quick demonstration of the system."""
    orch = get_orchestrator()
    upg = OptimizedUPGConstants()
    
    # Create demo task
    task = QuantumTask(
        task_id="demo-task",
        problem_type=ProblemType.ISING,
        problem_data={},
        num_qubits=6,
        num_reads=100,
        upg_optimization=True,
    )
    
    # Submit and get result
    job_id = orch.submit(task)
    result = orch.get_result(job_id)
    
    return {
        "status": "success",
        "message": "Demo completed successfully",
        "result": {
            "best_solution": result.get_bitstring(),
            "best_energy": result.best_energy,
            "upg_enhancement": result.upg_enhancement,
            "coherence_metrics": result.coherence_metrics,
        },
        "upg_constants": {
            "phi": upg.PHI,
            "consciousness": upg.CONSCIOUSNESS,
            "reality_distortion": upg.REALITY_DISTORTION,
        },
    }


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize orchestrator on startup."""
    get_orchestrator()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global orchestrator
    if orchestrator:
        orchestrator.shutdown()
        orchestrator = None


# Run with: uvicorn src.upg_quantum_production.api:app --host 0.0.0.0 --port 8080
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

