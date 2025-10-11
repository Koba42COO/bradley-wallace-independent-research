import json
import logging
import time

import numpy as np
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from pac_system.unified import UnifiedConsciousnessSystem
from pac_system.validator import PACEntropyReversalValidator
from pac_system import config


class OptimizeRequest(BaseModel):
    mode: str = "auto"
    data: list | dict | str | None = None


logger = logging.getLogger("pac.api")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Prometheus metrics
REQ_COUNTER = Counter(
    "pac_requests_total", "Total requests", ["path", "method", "status"]
)
REQ_LATENCY = Histogram(
    "pac_request_latency_seconds", "Request latency", ["path", "method"]
)


app = FastAPI(title="PAC System API")


@app.middleware("http")
async def metrics_logging(request: Request, call_next):
    start = time.time()
    response: Response = await call_next(request)
    elapsed = time.time() - start
    path = request.url.path
    method = request.method
    status = str(response.status_code)
    REQ_COUNTER.labels(path=path, method=method, status=status).inc()
    REQ_LATENCY.labels(path=path, method=method).observe(elapsed)
    # basic structured log
    try:
        logger.info(json.dumps({
            "ts": int(start * 1000),
            "path": path,
            "method": method,
            "status": response.status_code,
            "latency_ms": int(elapsed * 1000)
        }))
    except Exception:
        pass
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/optimize")
def optimize(req: OptimizeRequest):
    sys = UnifiedConsciousnessSystem(
        prime_scale=config.PRIME_SCALE, consciousness_weight=config.CONSCIOUSNESS_WEIGHT
    )
    x = req.data
    if x is None:
        x = np.random.randn(50, 10)
    return sys.process_universal_optimization(x, optimization_type=req.mode)


@app.post("/entropy/validate")
def entropy_validate():
    data = np.random.randn(100, 5)
    val = PACEntropyReversalValidator(prime_scale=config.PRIME_SCALE)
    return val.validate_entropy_reversal(data, n_experiments=10)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


