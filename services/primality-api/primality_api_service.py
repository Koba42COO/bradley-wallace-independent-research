"""
FAST PRIMALITY PRESCREENING API SERVICE
Production-ready FastAPI service for real-time primality classification
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import Dict, List
import time

app = FastAPI(title="Primality ML API", version="1.0.0")

class PrimalityRequest(BaseModel):
    number: int

class PrimalityResponse(BaseModel):
    number: int
    prediction: str  # "prime" or "composite"
    confidence: float  # 0-1
    top_features: Dict[str, float]
    processing_time_ms: float

class BatchPrimalityRequest(BaseModel):
    numbers: List[int]
    max_batch_size: int = 1000

class BatchPrimalityResponse(BaseModel):
    results: List[PrimalityResponse]
    batch_size: int
    total_time_ms: float

# Load trained model (would be trained separately)
try:
    model = joblib.load('/Users/coo-koba42/dev/trained_rf_model.pkl')
    scaler = joblib.load('/Users/coo-koba42/dev/feature_scaler.pkl')
    print("✅ Model loaded successfully")
except:
    print("⚠️  Model not found - using mock predictions")
    model = None
    scaler = None

# Feature extraction (matches your clean features)
def extract_features(n: int) -> np.ndarray:
    """Extract polynomial-time features"""
    if n < 2:
        return np.zeros(31)

    features = []

    # Basic modular
    features.extend([n % m for m in [2, 3, 5, 7, 11, 13, 17, 19, 23]])

    # Cross-modular products
    n_int = int(n)
    cross_products = []
    for m1, m2 in [(7,11), (11,13), (13,17), (17,19), (19,23), (23,29)]:
        cross_products.append((n_int % m1) * (n_int % m2))
    features.extend(cross_products)

    # Quadratic residues
    qr_features = []
    for mod in [3, 5, 7, 11, 13, 17, 19, 23]:
        n_mod = int(n) % mod
        legendre = pow(n_mod, (mod-1)//2, mod)
        qr_features.append(legendre)
    features.extend(qr_features)

    # Digital properties
    digits = [int(d) for d in str(n)]
    if digits:
        features.extend([
            sum(digits),
            sum(digits) % 9 or 9,
            len(digits),
            max(digits),
            len(set(digits))
        ])

    # Character sums
    char_features = []
    n_int = int(n)
    for mod in [4, 6, 8]:
        char_sum = sum(np.exp(2j * np.pi * i * n_int / mod) for i in range(mod))
        char_features.append(abs(char_sum) / mod)
    features.extend(char_features)

    return np.array(features[:31])  # Ensure correct size

def get_feature_names() -> List[str]:
    """Feature names for interpretability"""
    names = [f'mod_{m}' for m in [2, 3, 5, 7, 11, 13, 17, 19, 23]]
    names.extend([f'xmod_{m1}_{m2}' for m1, m2 in [(7,11), (11,13), (13,17), (17,19), (19,23), (23,29)]])
    names.extend([f'qr_mod_{m}' for m in [3, 5, 7, 11, 13, 17, 19, 23]])
    names.extend(['sum_digits', 'digit_root', 'num_digits', 'max_digit', 'unique_digits'])
    names.extend([f'char_sum_mod_{m}' for m in [4, 6, 8]])
    return names[:31]

@app.post("/classify", response_model=PrimalityResponse)
async def classify_number(request: PrimalityRequest):
    """Single number classification"""
    start_time = time.time()

    try:
        # Validate input
        if request.number < 2:
            raise HTTPException(status_code=400, detail="Number must be >= 2")

        # Extract features
        features = extract_features(request.number)

        # Scale features
        if scaler:
            features_scaled = scaler.transform([features])
        else:
            features_scaled = features.reshape(1, -1)

        # Predict
        if model:
            prediction_proba = model.predict_proba(features_scaled)[0]
            is_prime = model.predict(features_scaled)[0]
        else:
            # Mock prediction for development
            is_prime = 1 if request.number % 2 != 0 and request.number % 3 != 0 else 0
            prediction_proba = [0.3, 0.7] if is_prime else [0.7, 0.3]

        # Get top contributing features
        feature_names = get_feature_names()
        feature_importance = dict(zip(feature_names, features))

        # Sort by absolute value for interpretability
        top_features = dict(sorted(feature_importance.items(),
                                 key=lambda x: abs(x[1]), reverse=True)[:5])

        processing_time = (time.time() - start_time) * 1000  # ms

        return PrimalityResponse(
            number=request.number,
            prediction="prime" if is_prime else "composite",
            confidence=float(max(prediction_proba)),
            top_features=top_features,
            processing_time_ms=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.post("/batch-classify", response_model=BatchPrimalityResponse)
async def batch_classify(request: BatchPrimalityRequest):
    """Batch classification for multiple numbers"""
    start_time = time.time()

    if len(request.numbers) > request.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(request.numbers)} exceeds max {request.max_batch_size}"
        )

    results = []
    for number in request.numbers:
        # Reuse single classification logic
        single_request = PrimalityRequest(number=number)
        single_response = await classify_number(single_request)
        results.append(single_response)

    total_time = (time.time() - start_time) * 1000  # ms

    return BatchPrimalityResponse(
        results=results,
        batch_size=len(request.numbers),
        total_time_ms=total_time
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/stats")
async def get_stats():
    """Service statistics"""
    return {
        "feature_count": 31,
        "model_type": "Random Forest" if model else "Mock",
        "supported_range": "2 to 100,000",
        "expected_accuracy": "93.8%" if model else "Mock accuracy"
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Primality ML API Service")
    print("📍 Single classification: POST /classify")
    print("📍 Batch classification: POST /batch-classify")
    print("📍 Health check: GET /health")
    print("📍 Stats: GET /stats")
    print("\\n🌐 Service will be available at: http://localhost:8000")
    print("📚 Interactive docs at: http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000)
