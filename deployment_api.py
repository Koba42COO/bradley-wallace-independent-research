#!/usr/bin/env python3
"""
PRIMALITY TESTING API - DEPLOYMENT PROTOTYPE

A Flask-based REST API for machine learning primality testing.
Provides both clean ML and hybrid ML approaches for primality prediction.

Usage:
    python deployment_api.py

Test:
    curl http://localhost:5000/health
    curl http://localhost:5000/predict/15013
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"numbers": [15013, 15017], "model": "clean_ml"}'
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import logging
from datetime import datetime
import traceback

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models
models = {}
model_files = {
    'clean_ml': ('clean_ml_model.pkl', 'clean_scaler.pkl'),
    'hybrid_ml': ('enhanced_rf_model.pkl', 'enhanced_scaler.pkl')
}

logger.info("Loading ML models...")
for model_name, (model_file, scaler_file) in model_files.items():
    try:
        models[model_name] = {
            'model': joblib.load(model_file),
            'scaler': joblib.load(scaler_file),
            'loaded': True
        }
        logger.info(f"✅ {model_name} model loaded successfully")
    except Exception as e:
        logger.error(f"⚠️  {model_name} model not available: {e}")
        models[model_name] = {'loaded': False}

# Feature functions
def clean_features(n):
    """Clean ML features (pure mathematical)"""
    n_int = int(n)
    features = [
        n_int % 2, n_int % 3, n_int % 5, n_int % 7, n_int % 11,
        n_int % 13, n_int % 17, n_int % 19, n_int % 23
    ]
    features.extend([
        (n_int % 7) * (n_int % 11),
        (n_int % 11) * (n_int % 13),
        (n_int % 13) * (n_int % 17)
    ])
    for mod in [3, 5, 7]:
        n_mod = n_int % mod
        features.append(pow(n_mod, (mod-1)//2, mod))

    digits = [int(d) for d in str(n_int)]
    features.extend([
        sum(digits),
        sum(digits) % 9 or 9,
        len(digits),
        max(digits),
        len(set(digits))
    ])

    for mod in [4, 6, 8]:
        char_sum = sum(np.exp(2j * np.pi * i * (n_int % mod) / mod) for i in range(mod))
        features.append(abs(char_sum) / mod)

    return [0.0 if not np.isfinite(f) else f for f in features]

def hybrid_features(n):
    """Hybrid ML features (mathematical + divisibility checks)"""
    features = clean_features(n)
    n_int = int(n)

    check_primes = [13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in check_primes:
        features.append(n_int % p)
        features.append(1 if n_int % p == 0 else 0)

    return features

feature_functions = {
    'clean_ml': clean_features,
    'hybrid_ml': hybrid_features
}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        available_models = [name for name, data in models.items() if data['loaded']]
        response = {
            'status': 'healthy' if available_models else 'degraded',
            'service': 'primality_testing_api',
            'version': '1.0.0',
            'models_available': available_models,
            'timestamp': datetime.utcnow().isoformat()
        }

        logger.info(f"Health check requested - Status: {response['status']}, Models: {len(available_models)}")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/predict/<int:number>', methods=['GET'])
def predict_single(number):
    """Predict primality for a single number"""
    if number < 2:
        return jsonify({
            'number': number,
            'prediction': 'composite',
            'confidence': 1.0,
            'method': 'trivial',
            'model': 'none'
        })

    # Use clean_ml as default, fallback to hybrid_ml if available
    model_name = 'clean_ml' if models['clean_ml']['loaded'] else 'hybrid_ml'

    if not models[model_name]['loaded']:
        return jsonify({'error': 'No models available'}), 500

    try:
        features = feature_functions[model_name](number)
        features_scaled = models[model_name]['scaler'].transform([features])
        prediction = models[model_name]['model'].predict(features_scaled)[0]
        probabilities = models[model_name]['model'].predict_proba(features_scaled)[0]

        result = {
            'number': number,
            'prediction': 'prime' if prediction == 1 else 'composite',
            'confidence': float(probabilities[1] if prediction == 1 else probabilities[0]),
            'model': model_name,
            'accuracy_rating': '95.73%' if model_name == 'clean_ml' else '98.13%',
            'timestamp': datetime.utcnow().isoformat()
        }

        logger.info(f"Single prediction: {number} -> {result['prediction']} (confidence: {result['confidence']:.3f})")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Single prediction failed for {number}: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'number': number,
            'model': model_name,
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/predict', methods=['POST'])
def predict_batch():
    """Predict primality for multiple numbers"""
    try:
        data = request.get_json()

        if not data or 'numbers' not in data:
            return jsonify({'error': 'Missing numbers array'}), 400

        numbers = data['numbers']
        model_name = data.get('model', 'clean_ml')

        if model_name not in models or not models[model_name]['loaded']:
            available_models = [name for name, data in models.items() if data['loaded']]
            return jsonify({
                'error': f'Model {model_name} not available',
                'available_models': available_models
            }), 400

        if not isinstance(numbers, list) or len(numbers) > 100:
            return jsonify({'error': 'Numbers must be array with max 100 elements'}), 400

        results = []

        for number in numbers:
            if not isinstance(number, int):
                results.append({
                    'number': number,
                    'error': 'Not an integer'
                })
                continue

            if number < 2:
                results.append({
                    'number': number,
                    'prediction': 'composite',
                    'confidence': 1.0,
                    'method': 'trivial'
                })
                continue

            try:
                features = feature_functions[model_name](number)
                features_scaled = models[model_name]['scaler'].transform([features])
                prediction = models[model_name]['model'].predict(features_scaled)[0]
                probabilities = models[model_name]['model'].predict_proba(features_scaled)[0]

                results.append({
                    'number': number,
                    'prediction': 'prime' if prediction == 1 else 'composite',
                    'confidence': float(probabilities[1] if prediction == 1 else probabilities[0])
                })

            except Exception as e:
                results.append({
                    'number': number,
                    'error': str(e)
                })

        return jsonify({
            'model': model_name,
            'results': results,
            'batch_size': len(results),
            'processed': len([r for r in results if 'prediction' in r]),
            'errors': len([r for r in results if 'error' in r]),
            'timestamp': str(np.datetime64('now'))
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/info', methods=['GET'])
def get_info():
    """Get detailed API information"""
    info = {
        'service': 'primality_testing_api',
        'version': '1.0.0',
        'description': 'Machine learning-based primality testing service using mathematical features',
        'models': {},
        'performance': {},
        'endpoints': {
            'GET /health': 'Health check and available models',
            'GET /predict/<number>': 'Predict primality for single number',
            'POST /predict': 'Batch prediction (JSON: {"numbers": [...], "model": "..."})',
            'GET /info': 'Detailed API information'
        },
        'limitations': [
            'Numbers must be integers ≥ 2',
            'Batch size limited to 100 numbers',
            'Models trained on numbers up to 20,000',
            'Probabilistic results (not deterministic like AKS/ECPP)'
        ]
    }

    for model_name, model_data in models.items():
        if model_data['loaded']:
            info['models'][model_name] = {
                'status': 'available',
                'accuracy': '95.73%' if model_name == 'clean_ml' else '98.13%',
                'complexity': 'O(log n)' if model_name == 'clean_ml' else 'O(k) k=20',
                'description': 'Pure mathematical features, polynomial time' if model_name == 'clean_ml' else 'Mathematical + divisibility checks',
                'features': 31 if model_name == 'clean_ml' else 71,
                'use_case': 'Research and general screening' if model_name == 'clean_ml' else 'High-reliability applications'
            }
            info['performance'][model_name] = {
                'inference_time': '~0.2-0.5ms per number',
                'memory_usage': '~50-100MB',
                'confidence_scores': 'Available for uncertainty quantification'
            }
        else:
            info['models'][model_name] = {'status': 'not_available'}

    return jsonify(info)

@app.route('/', methods=['GET'])
def home():
    """Home page with usage instructions"""
    return """
    <h1>Primality Testing API</h1>
    <p>Machine learning-based primality prediction service</p>

    <h2>Available Models:</h2>
    <ul>
        <li><b>clean_ml</b>: 95.73% accuracy, pure mathematical features</li>
        <li><b>hybrid_ml</b>: 98.13% accuracy, mathematical + divisibility checks</li>
    </ul>

    <h2>Endpoints:</h2>
    <ul>
        <li><code>GET /health</code> - Health check</li>
        <li><code>GET /predict/&lt;number&gt;</code> - Single prediction</li>
        <li><code>POST /predict</code> - Batch prediction</li>
        <li><code>GET /info</code> - API details</li>
    </ul>

    <h2>Examples:</h2>
    <pre>
curl http://localhost:5000/predict/15013
curl -X POST http://localhost:5000/predict \\
     -H "Content-Type: application/json" \\
     -d '{"numbers": [15013, 15017], "model": "clean_ml"}'
    </pre>
    """

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))  # Use 5001 as default to avoid conflicts

    print("\n🚀 STARTING PRIMALITY TESTING API")
    print("=" * 40)
    print("Service: ML-based primality prediction")
    print("Models:", [name for name, data in models.items() if data['loaded']])
    print(f"URL: http://localhost:{port}")
    print("\nPress Ctrl+C to stop")
    print("=" * 40)

    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
