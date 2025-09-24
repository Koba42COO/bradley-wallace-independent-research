# 🎯 chAIos Benchmark Testing Suite

Comprehensive testing framework for the chAIos platform using traditional AI benchmarks including GLUE, SuperGLUE, and performance testing.

## 📋 Overview

This benchmark testing suite provides comprehensive evaluation of the chAIos prime aligned compute-based AI platform using industry-standard benchmarks:

- **GLUE (General Language Understanding Evaluation)**
- **SuperGLUE (Advanced Language Understanding)**
- **SQuAD (Stanford Question Answering Dataset)**
- **RACE (Reading Comprehension)**
- **HellaSwag (Commonsense Reasoning)**
- **WinoGrande (Commonsense Reasoning)**
- **ARC (AI2 Reasoning Challenge)**
- **Performance & Stress Testing**
- **System Health Monitoring**

## 🚀 Quick Start

### Prerequisites

1. **chAIos API Server**: Ensure the chAIos server is running on `http://localhost:8000`
2. **Python 3.8+**: Required for async support and modern features
3. **Dependencies**: Install required packages

### Installation

```bash
# Install benchmark testing dependencies
pip install -r requirements_benchmarks.txt

# Verify chAIos API is running
curl http://localhost:8000/plugin/health
```

### Running Benchmarks

#### 1. Master Benchmark Runner (Recommended)

Run all benchmark suites with a single command:

```bash
# Run all benchmarks
python master_benchmark_runner.py

# Run without performance tests (faster)
python master_benchmark_runner.py --no-performance

# Run without system health check
python master_benchmark_runner.py --no-health

# Specify custom API URL
python master_benchmark_runner.py --api-url http://localhost:8000

# Save results to specific file
python master_benchmark_runner.py --output my_results.json
```

#### 2. Individual Benchmark Suites

Run specific benchmark suites:

```bash
# GLUE & SuperGLUE benchmarks only
python glue_superglue_benchmark.py

# Comprehensive benchmark suite
python comprehensive_benchmark_suite.py

# Performance & stress testing
python performance_stress_test.py
```

## 📊 Benchmark Suites

### 1. GLUE & SuperGLUE Benchmarks

**File**: `glue_superglue_benchmark.py`

Tests traditional language understanding tasks:

- **CoLA**: Corpus of Linguistic Acceptability
- **SST-2**: Stanford Sentiment Treebank
- **MRPC**: Microsoft Research Paraphrase Corpus
- **BoolQ**: Yes/No Question Answering
- **COPA**: Choice of Plausible Alternatives

**Usage**:
```bash
python glue_superglue_benchmark.py
```

**Output**: JSON file with accuracy scores, improvement percentages, and prime aligned compute enhancement factors.

### 2. Comprehensive Benchmark Suite

**File**: `comprehensive_benchmark_suite.py`

Extended testing with additional benchmarks:

- **SQuAD 2.0**: Question Answering with Unanswerable Questions
- **RACE**: Reading Comprehension from Examinations
- **HellaSwag**: Commonsense Reasoning
- **WinoGrande**: Commonsense Reasoning
- **ARC**: AI2 Reasoning Challenge
- **chAIos prime aligned compute**: Custom prime aligned compute processing tests

**Usage**:
```bash
python comprehensive_benchmark_suite.py
```

**Output**: Comprehensive JSON report with detailed metrics for each benchmark.

### 3. Performance & Stress Testing

**File**: `performance_stress_test.py`

Evaluates system performance under various conditions:

- **Latency Testing**: Response time analysis
- **Load Testing**: Concurrent user simulation
- **Stress Testing**: Breaking point identification
- **Resource Monitoring**: CPU and memory usage
- **Throughput Analysis**: Requests per second

**Usage**:
```bash
python performance_stress_test.py
```

**Output**: Performance metrics including response times, throughput, error rates, and resource usage.

### 4. Master Benchmark Runner

**File**: `master_benchmark_runner.py`

Orchestrates all benchmark suites and generates comprehensive reports:

- Runs all benchmark suites in sequence
- Aggregates results across different test types
- Generates overall performance assessment
- Monitors system health during testing
- Creates master report with recommendations

**Usage**:
```bash
python master_benchmark_runner.py
```

## 📈 Understanding Results

### Performance Metrics

- **Accuracy**: Percentage of correct predictions
- **F1 Score**: Harmonic mean of precision and recall
- **Response Time**: Average time per request
- **Throughput**: Requests processed per second
- **Error Rate**: Percentage of failed requests
- **prime aligned compute Enhancement**: Multiplier from prime aligned compute processing

### Assessment Grades

- **🌟 EXCELLENT**: Production ready (≥80% accuracy, <5% error rate)
- **✅ GOOD**: Strong performance (≥60% accuracy, <10% error rate)
- **⚠️ MODERATE**: Needs improvement (≥40% accuracy, <20% error rate)
- **❌ POOR**: Significant issues (<40% accuracy, >20% error rate)

### Sample Output

```json
{
  "summary": {
    "total_benchmark_suites": 6,
    "average_accuracy": 0.847,
    "average_enhancement": 1.618,
    "average_improvement": 15.3,
    "assessment": "🌟 EXCELLENT - Production Ready"
  },
  "glue": {
    "average_accuracy": 0.823,
    "results": [
      {
        "task": "CoLA",
        "accuracy": 0.800,
        "improvement_percent": 17.6
      }
    ]
  }
}
```

## 🔧 Configuration

### API Configuration

Default API URL: `http://localhost:8000`

To test against different endpoints:

```bash
python master_benchmark_runner.py --api-url http://your-api-server:8000
```

### Test Parameters

Modify test parameters in the benchmark files:

```python
# In glue_superglue_benchmark.py
test_cases = [
    {
        "sentence": "Your test sentence here",
        "label": 1,
        "type": "grammatical"
    }
]
```

### Performance Test Configuration

Adjust performance test parameters:

```python
# In performance_stress_test.py
load_result = self.load_tester.load_test(
    concurrent_users=20,      # Number of concurrent users
    requests_per_user=10,     # Requests per user
    tool_name="your_tool",
    parameters=your_params
)
```

## 📁 Output Files

Benchmark results are saved as JSON files with timestamps:

- `glue_superglue_results_YYYYMMDD_HHMMSS.json`
- `comprehensive_benchmark_results_YYYYMMDD_HHMMSS.json`
- `performance_test_results_YYYYMMDD_HHMMSS.json`
- `master_benchmark_results_YYYYMMDD_HHMMSS.json`

## 🛠️ Troubleshooting

### Common Issues

1. **API Connection Failed**
   ```
   ❌ Cannot connect to chAIos API: Connection refused
   ```
   **Solution**: Ensure chAIos server is running on the correct port.

2. **Plugin Not Found**
   ```
   ❌ Plugin 'transcendent_llm_builder' not found
   ```
   **Solution**: Verify all required plugins are loaded in the chAIos server.

3. **Timeout Errors**
   ```
   ❌ Request timeout after 30 seconds
   ```
   **Solution**: Increase timeout values or check server performance.

4. **Memory Issues**
   ```
   ❌ Memory usage too high
   ```
   **Solution**: Reduce concurrent users or request frequency.

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Check

Verify system health before running benchmarks:

```bash
# Check API health
curl http://localhost:8000/plugin/health

# Check available plugins
curl http://localhost:8000/plugin/list

# Check system resources
python -c "import psutil; print(f'CPU: {psutil.cpu_percent()}%, Memory: {psutil.virtual_memory().percent}%')"
```

## 📚 Advanced Usage

### Custom Benchmark Development

Create custom benchmarks by extending the base classes:

```python
from comprehensive_benchmark_suite import BenchmarkResult

class CustomBenchmark:
    def test_custom_task(self) -> BenchmarkResult:
        # Your custom benchmark implementation
        pass
```

### Integration with CI/CD

Add benchmark testing to your CI/CD pipeline:

```yaml
# .github/workflows/benchmarks.yml
name: Benchmark Testing
on: [push, pull_request]
jobs:
  benchmarks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements_benchmarks.txt
      - name: Start chAIos server
        run: python api_server.py &
      - name: Run benchmarks
        run: python master_benchmark_runner.py
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: master_benchmark_results_*.json
```

### Automated Reporting

Generate automated reports:

```python
# Generate HTML report
import json
from datetime import datetime

def generate_html_report(results_file):
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Generate HTML report
    html = f"""
    <html>
    <head><title>chAIos Benchmark Results</title></head>
    <body>
        <h1>Benchmark Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h1>
        <p>Overall Assessment: {results['overall_assessment']['grade']}</p>
        <!-- Add more HTML content -->
    </body>
    </html>
    """
    
    with open('benchmark_report.html', 'w') as f:
        f.write(html)
```

## 🤝 Contributing

To contribute to the benchmark testing suite:

1. Fork the repository
2. Create a feature branch
3. Add your benchmark tests
4. Update documentation
5. Submit a pull request

### Adding New Benchmarks

1. Create a new benchmark class
2. Implement the required methods
3. Add to the master runner
4. Update documentation
5. Add test cases

## 📄 License

This benchmark testing suite is part of the chAIos platform and follows the same licensing terms.

## 🆘 Support

For support with benchmark testing:

1. Check the troubleshooting section
2. Review the logs for error messages
3. Verify API connectivity
4. Check system resources
5. Contact the chAIos development team

---

**🎯 Happy Benchmarking!**

The chAIos benchmark testing suite provides comprehensive evaluation of your prime aligned compute-based AI platform using industry-standard benchmarks. Use these tools to validate performance, identify improvements, and ensure production readiness.
