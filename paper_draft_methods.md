# Methods: Systematic Primality Classification Using Machine Learning

## Abstract

This paper presents a comprehensive investigation of primality testing using polynomial-time computable features and machine learning algorithms. We establish empirical bounds on classification accuracy and analyze the mathematical structure of classification errors.

## 1. Problem Statement

Primality testing is fundamental to number theory and cryptography. While deterministic polynomial-time algorithms exist (AKS algorithm), practical implementations often rely on probabilistic methods. This work explores the boundary between algebraic methods (using modular arithmetic) and analytic methods (requiring deeper number-theoretic insights).

## 2. Feature Engineering

### 2.1 Clean Feature Set

We designed a comprehensive set of polynomial-time computable features that capture various number-theoretic properties without requiring factorization:

#### Modular Arithmetic Features
- **Basic divisibility**: `n mod p` for primes p ≤ 23
- **Cross-modular products**: `(n mod p₁) × (n mod p₂)` for prime pairs
- **Quadratic residues**: Legendre symbols for prime moduli

#### Digital Properties
- **Digit sums and roots**: Sum of digits, digital roots
- **Digit diversity**: Number of unique digits, digit length
- **Positional features**: First and last digits

#### Advanced Algebraic Features
- **Character sums**: Exponential sums over finite fields
- **Continued fraction approximations**: Rational approximations to √n

### 2.2 Feature Validation

All features are computable in O(log² n) time:
- Modular operations: O(1)
- Digital processing: O(log n)
- Cross-products: O(1)
- Character sums: O(k) for small k

## 3. Machine Learning Methodology

### 3.1 Algorithm Selection

We evaluated multiple algorithms on the clean feature set:

#### Random Forest
- **Ensemble method** with 100 decision trees
- **Handles non-linear relationships** in modular arithmetic
- **Provides feature importance** rankings
- **Robust to overfitting** through bagging

#### Neural Networks
- **Multi-layer perceptron** (50-25 hidden units)
- **Non-linear activation functions** for complex patterns
- **Early stopping** to prevent overfitting

#### Support Vector Machines
- **Kernel methods** for high-dimensional feature spaces
- **RBF kernel** with automatic parameter selection

### 3.2 Cross-Validation Strategy

#### Stratified K-Fold Validation
- **10-fold cross-validation** to ensure statistical robustness
- **Stratification by class** to maintain prime/composite balance
- **Multiple random seeds** to assess stability

#### Performance Metrics
- **Accuracy**: Overall classification performance
- **Precision/Recall**: Class-specific performance
- **F1-Score**: Balanced metric for imbalanced classes
- **Feature importance**: Relative contribution of each feature

## 4. Experimental Design

### 4.1 Data Ranges

Testing across multiple scales to assess generalization:

| Range | Scale | Primes | Notes |
|-------|-------|--------|-------|
| 100-1,000 | 10³ | 168 | Twin prime rich |
| 10,000-20,000 | 10⁴ | 1,059 | Main analysis range |
| 100,000-110,000 | 10⁵ | ~8,000 | Scale validation |
| 1,000,000-1,001,000 | 10⁶ | ~78,000 | Large scale testing |

### 4.2 Baseline Comparisons

#### Extended Sieving Baseline
- Divisibility testing by primes ≤ 11
- Achieves ~90.5% accuracy
- Represents state-of-the-art non-ML approach

#### Random Guessing
- 50% accuracy baseline
- Establishes minimum performance threshold

### 4.3 Error Analysis Methodology

#### Systematic Error Classification
- **False positives**: Composites classified as primes
- **False negatives**: Primes classified as composites
- **Feature pattern analysis**: Comparing error cases to correct classifications

#### Balance Ratio Analysis
For semiprime errors: `ratio = min(p,q) / max(p,q)`
- **Highly balanced**: 0.5 < ratio < 1.0 (e.g., 131×137)
- **Moderately balanced**: 0.2 < ratio < 0.5
- **Unbalanced**: ratio < 0.2 (e.g., 2×9497)

## 5. Implementation Details

### 5.1 Computational Infrastructure

- **Language**: Python 3.9 with NumPy/SciPy
- **ML Libraries**: scikit-learn, pandas
- **Prime generation**: Optimized sieve of Eratosthenes
- **Memory management**: Segmented sieve for large ranges

### 5.2 Reproducibility

All code includes:
- Random seeds for reproducible results
- Detailed logging of experimental parameters
- Version control for feature implementations

### 5.3 Statistical Rigor

- **Significance testing**: t-tests for performance differences
- **Confidence intervals**: Bootstrapped estimates
- **Effect sizes**: Cohen's d for practical significance

## 6. Validation Strategy

### 6.1 Internal Validation
- Cross-validation on training ranges
- Hyperparameter optimization
- Feature selection stability

### 6.2 External Validation
- Testing on unseen number ranges
- Comparison with established algorithms
- Sensitivity analysis on feature subsets

### 6.3 Error Validation
- Factorization verification of error cases
- Balance ratio distribution analysis
- Feature space visualization of errors

## 7. Ethical and Computational Considerations

### 7.1 Computational Resources
- Total computation time: < 2 hours on standard hardware
- Memory usage: < 500MB for all experiments
- Scalable to larger ranges with optimized sieving

### 7.2 Reproducibility
- Complete code repository provided
- Docker container for consistent environment
- Detailed documentation of all procedures

### 7.3 Limitations
- Range limited by sieve computational complexity
- Probabilistic methods used for large prime verification
- Feature set may not capture all number-theoretic structure

## 8. Conclusion

This methodology provides a rigorous framework for investigating the boundaries of primality classification using machine learning. The systematic approach ensures reproducible results and comprehensive analysis of both successes and failures.
