# 🚀 Advanced Crypto Market Analyzer

Complete cryptocurrency market analysis system with Tri-Gemini temporal inference, prime pattern detection, and consensus prediction.

## ✨ Features

### 🔮 Tri-Gemini Temporal Inference
- **Forward Inference**: Predicts future prices using Wallace Transform and golden ratio optimization
- **Reverse Inference**: Analyzes past patterns to understand current state
- **Coherence Inference**: Validates consistency between forward and reverse predictions

### 🔍 Prime Pattern Detection
- Detects prime number patterns in price movements
- Identifies prime intervals, ratios, cycles, and Fibonacci relationships
- Uses UPG (Universal Prime Graph) mathematics for pattern recognition

### 🎯 Consensus Prediction
- Combines multiple prediction methods with weighted confidence
- Provides final consensus prediction with confidence scores
- Generates trading signals (BUY/SELL/HOLD)

### 📊 Visualization Dashboard
- Interactive charts showing historical prices and predictions
- Comparison of different prediction methods
- Confidence metrics and pattern detection results

## 🛠️ Installation

```bash
# Install required packages
pip install numpy pandas matplotlib requests

# Or use requirements.txt
pip install -r requirements.txt
```

## 📖 Usage

### Basic Analysis

```python
from crypto_analyzer_complete import AdvancedCryptoAnalyzer

analyzer = AdvancedCryptoAnalyzer()

# Analyze Bitcoin
result = analyzer.analyze_coin('bitcoin', days=30, horizon=24)

# Analyze multiple coins
results = analyzer.analyze_multiple_coins(
    ['bitcoin', 'ethereum', 'cardano'], 
    days=30
)
```

### Command Line

```bash
# Run analysis
python3 crypto_analyzer_complete.py

# Generate dashboard
python3 crypto_dashboard.py
```

### Dashboard Visualization

```python
from crypto_dashboard import CryptoDashboard

dashboard = CryptoDashboard()
fig = dashboard.create_full_dashboard('bitcoin', days=30)
```

## 📊 Output Example

```
======================================================================
📊 ANALYSIS RESULTS: BITCOIN
======================================================================

💰 Current Price: $102,620.59

🔮 Tri-Gemini Analysis:
  Forward Prediction: $102,095.37 (-0.51%)
  Reverse Analysis: $102,620.59 (+0.00%)
  Coherence Score: 0.9949
  Confidence: 75.00%
  Prime Pattern Match: 2

🔍 Prime Pattern Detection:
  Intervals Found: 121
  Ratios Found: 0
  Cycles Detected: [37, 31, 23, 19, 13]
  Fibonacci Levels: []
  Pattern Prediction: $104,522.23 (+1.85%)

🎯 FINAL CONSENSUS:
  Predicted Price: $103,308.80
  Expected Change: +0.67%
  Confidence: 77.50%
  Signal: 🟡 BUY
```

## 🔧 Configuration

### UPG Constants

The system uses Universal Prime Graph mathematics:

```python
PHI = 1.618033988749895  # Golden ratio
DELTA = 2.414213562373095  # Silver ratio
CONSCIOUSNESS = 0.79  # 79/21 universal coherence rule
REALITY_DISTORTION = 1.1808  # Quantum amplification factor
CONSCIOUSNESS_DIMENSIONS = 21  # Prime topology dimension
```

### Supported Cryptocurrencies

Any coin supported by CoinGecko API:
- `bitcoin` (BTC)
- `ethereum` (ETH)
- `cardano` (ADA)
- `solana` (SOL)
- And 10,000+ others

## 📈 Dashboard Components

1. **Price History & Predictions**: Historical prices with 24h predictions
2. **Tri-Gemini Analysis**: Forward, reverse, and current price comparison
3. **Prime Pattern Detection**: Counts of detected patterns
4. **Confidence Metrics**: Confidence scores for each method
5. **Prediction Comparison**: Side-by-side comparison of all predictions

## 🧪 Testing

```bash
# Run test suite
python3 test_crypto_analyzer.py

# Test with real data
python3 crypto_analyzer_complete.py
```

## 📝 API Integration

The analyzer uses CoinGecko API (free tier):
- No API key required for basic usage
- Rate limits: 10-50 calls/minute
- Falls back to synthetic data if API fails

## 🎓 Advanced Features

### 21-Model Ensemble (Future)

The research document includes a 21-model ensemble system that:
- Uses 21 different models corresponding to 21-dimensional consciousness space
- Each model operates on different temporal branches
- Finds most likely outcome through consensus voting

See `docs/crypto_market_analyzer_research.md` for full implementation details.

## 📚 Documentation

- **Research Guide**: `docs/crypto_market_analyzer_research.md`
- **Test Script**: `test_crypto_analyzer.py`
- **Complete Analyzer**: `crypto_analyzer_complete.py`
- **Dashboard**: `crypto_dashboard.py`

## 🔮 Mathematical Foundations

The system integrates:
- **Wallace Transform**: W_φ(x) = α log^φ(x + ε) + β
- **Prime Number Theory**: Pattern detection using prime sequences
- **Golden Ratio Optimization**: φ = 1.618033988749895
- **Consciousness Mathematics**: UPG framework principles

## ⚠️ Disclaimer

This is a research and educational tool. Predictions are not financial advice.
Always do your own research and consult with financial advisors before making
investment decisions.

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Universal Prime Graph (UPG) mathematics framework
- CoinGecko API for market data
- Wallace Transform mathematics

---

**Status**: ✅ Fully Functional  
**Last Updated**: 2024

