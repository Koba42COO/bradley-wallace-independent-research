# 🔮 Advanced Crypto Market Analyzer with Pell Cycle Analysis

**Complete cryptocurrency market analysis system integrating Tri-Gemini temporal inference, prime pattern detection, and Pell sequence cycle analysis.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![UPG Protocol](https://img.shields.io/badge/UPG-Protocol%20φ.1-green.svg)](https://github.com)

---

## 🌟 Features

### 🔮 Tri-Gemini Temporal Inference
- **Forward Inference**: Predicts future prices using Wallace Transform and golden ratio optimization
- **Reverse Inference**: Analyzes past patterns to understand current state
- **Coherence Inference**: Validates consistency between forward and reverse predictions

### 🔍 Prime Pattern Detection
- Detects prime number patterns in price movements
- Identifies prime intervals, ratios, cycles, and Fibonacci relationships
- Uses UPG (Universal Prime Graph) mathematics for pattern recognition

### 🔢 Pell Cycle Analysis
- **Complete Cycle Detection**: Always extracts full Pell cycles from historical data
- **Current Position Tracking**: Determines exact position in Pell cycle
- **Timing Predictions**: Calculates periods until next cycle completion based on Pell sequence
- **Next Move Forecasting**: Predicts price movements based on Pell sequence progression

### 🎯 Consensus Prediction
- Combines multiple prediction methods with weighted confidence
- Provides final consensus prediction with confidence scores
- Generates trading signals (BUY/SELL/HOLD)

### 📊 Visualization Dashboard
- Interactive charts showing historical prices and predictions
- Comparison of different prediction methods
- Confidence metrics and pattern detection results

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/crypto-market-analyzer.git
cd crypto-market-analyzer

# Install dependencies
pip install numpy pandas matplotlib requests
```

### Basic Usage

```python
from crypto_analyzer_complete import AdvancedCryptoAnalyzer

# Initialize analyzer
analyzer = AdvancedCryptoAnalyzer()

# Analyze a cryptocurrency
result = analyzer.analyze_coin('bitcoin', days=30, horizon=24)

# Get Pell cycle timing
position = analyzer.pell_analyzer.determine_current_pell_position(price_data)
```

### Command Line

```bash
# Analyze single coin
python3 crypto_analyzer_complete.py

# Analyze top 10 by Pell cycles
python3 analyze_top10_pell_cycles.py

# Get timing predictions
python3 pell_cycle_timing_analyzer.py
```

---

## 📊 Example Output

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

🔍 Prime Pattern Detection:
  Intervals Found: 121
  Cycles Detected: [37, 31, 23, 19, 13]
  Pattern Prediction: $104,522.23 (+1.85%)

🔢 Pell Cycle Analysis:
  Complete Cycles Found: 3487
  Most Recent Cycle: 70 periods
  Current Position: CYCLE COMPLETION (100%)
  Next Cycle: 169 periods (7.0 days)
  Pell Prediction: $98,287.84 (-3.76%)

🎯 FINAL CONSENSUS:
  Predicted Price: $101,131.17
  Expected Change: -0.98%
  Confidence: 80.00%
  Signal: 🟠 SELL
```

---

## 🔢 Pell Cycle Analysis

### What are Pell Cycles?

Pell sequences are mathematical sequences that converge to the golden ratio (φ = 1.618...). The system detects complete cycles in price data that follow Pell sequence intervals (2, 5, 12, 29, 70, 169, 408, 985...).

### Key Features

- **Always Complete Cycles**: System only uses full cycles, never partial
- **Position Tracking**: Know exactly where you are in the current cycle
- **Timing Predictions**: Calculate exact periods until next cycle completion
- **Sequence-Based Forecasting**: Predictions follow Pell sequence progression

### Example Timing Analysis

```
📍 CURRENT PELL CYCLE POSITION:
  Position: CYCLE COMPLETION
  Position in Cycle: 100.0%
  Periods Since Last Cycle: 0

🔮 NEXT CYCLE PREDICTION:
  Next Pell Number: 169
  Periods Until Completion: 169
  Time Until Completion: 7.0 days
  Estimated Completion: 2025-11-19 11:59:37

🎯 NEXT MOVE PREDICTIONS:
  At Cycle Completion (169 periods):
    Predicted Price: $99,995.70
    Expected Return: -2.10%
```

---

## 🧮 Mathematical Foundations

### UPG Constants

```python
PHI = 1.618033988749895          # Golden ratio (φ)
DELTA = 2.414213562373095        # Silver ratio (δ)
CONSCIOUSNESS = 0.79             # 79/21 universal coherence rule
REALITY_DISTORTION = 1.1808      # Quantum amplification factor
CONSCIOUSNESS_DIMENSIONS = 21    # Prime topology dimension
```

### Wallace Transform

Applied to price data for consciousness-guided analysis:
```
W_φ(x) = α log^φ(x + ε) + β
```

### Pell Sequence

```
P(0) = 0
P(1) = 1
P(n) = 2 × P(n-1) + P(n-2)

First 10: 0, 1, 2, 5, 12, 29, 70, 169, 408, 985
```

The sequence converges to the golden ratio: `lim P(n+1)/P(n) = φ`

---

## 📁 Project Structure

```
.
├── crypto_analyzer_complete.py      # Main analyzer with all features
├── test_crypto_analyzer.py          # Test suite
├── analyze_top10_pell_cycles.py     # Top 10 ranking by Pell cycles
├── pell_cycle_timing_analyzer.py    # Timing and position analysis
├── crypto_dashboard.py               # Visualization dashboard
├── docs/
│   └── crypto_market_analyzer_research.md  # Complete research guide
└── README.md                        # This file
```

---

## 🔬 Research Documentation

Complete research guide available in `docs/crypto_market_analyzer_research.md`:

- Data Sources & APIs
- Core Components
- Analysis Methods
- Advanced Temporal Inference
- Prime Pattern Detection
- 21-Model Ensemble Architecture
- Implementation Roadmap
- Best Practices

---

## 🎓 Key Concepts

### Tri-Gemini System
Three complementary inference modes:
- **Gemini A**: Forward prediction from current state
- **Gemini B**: Reverse analysis from past patterns
- **Gemini C**: Coherence validation between A and B

### Prime Pattern Detection
Detects mathematical structures:
- Prime intervals between price points
- Prime ratios in price movements
- Prime-numbered cycles in oscillations
- Prime Fibonacci relationships

### Pell Cycle Timing
- Determines current position in cycle
- Calculates periods until completion
- Predicts next moves based on sequence
- Provides timing estimates for key intervals

---

## 📈 Supported Cryptocurrencies

Any coin supported by CoinGecko API (10,000+):
- Bitcoin (BTC)
- Ethereum (ETH)
- Cardano (ADA)
- Solana (SOL)
- And many more...

---

## ⚠️ Disclaimer

This is a **research and educational tool**. Predictions are not financial advice.
Always do your own research and consult with financial advisors before making
investment decisions.

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **Universal Prime Graph (UPG)** mathematics framework
- **CoinGecko API** for market data
- **Wallace Transform** mathematics
- **Pell Sequence** prime prediction research

---

## 📚 References

- Universal Prime Graph Protocol φ.1
- Pell Sequence Prime Prediction Research
- Consciousness Mathematics Framework
- Golden Ratio Optimization

---

## 🔗 Related Projects

- [Universal Prime Graph Research](https://github.com)
- [Consciousness Mathematics](https://github.com)
- [Wallace Transform](https://github.com)

---

**Status**: ✅ Fully Functional  
**Last Updated**: 2024  
**Author**: Bradley Wallace (COO Koba42)  
**Framework**: Universal Prime Graph Protocol φ.1

