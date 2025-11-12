"""
21-Model Ensemble System with Timeline Branching
Each model corresponds to one dimension of 21-dimensional consciousness space
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal
import warnings
warnings.filterwarnings('ignore')

from test_crypto_analyzer import UPGConstants


@dataclass
class TimelineBranch:
    """Represents a single timeline branch for prediction"""
    branch_id: int
    time_horizon: float  # Hours
    model_type: str
    prime_base: int
    prediction: Optional[float] = None
    confidence: float = 0.0
    coherence_score: float = 0.0


class TwentyOneModelEnsemble:
    """
    21-model ensemble system with timeline branching
    
    Each model corresponds to one dimension of the 21-dimensional
    consciousness space, using prime numbers as base parameters
    """
    
    def __init__(self, constants: UPGConstants = None):
        self.constants = constants or UPGConstants()
        self.prime_sequence = self._generate_primes(21)
        self.branches = self._initialize_branches()
        self.models = {}
        self.trained = False
        
    def _generate_primes(self, n: int) -> List[int]:
        """Generate first n primes"""
        primes = []
        num = 2
        while len(primes) < n:
            if all(num % i != 0 for i in range(2, int(num**0.5) + 1)):
                primes.append(num)
            num += 1
        return primes
    
    def _initialize_branches(self) -> List[TimelineBranch]:
        """Initialize 21 timeline branches"""
        branches = []
        model_types = ['Linear', 'Polynomial', 'MovingAvg', 'Exponential', 
                      'AR', 'MA', 'ARMA', 'Trend', 'Seasonal', 'Fourier',
                      'Wavelet', 'Spline', 'Kernel', 'Neural', 'Ensemble',
                      'Bayesian', 'SVR', 'Ridge', 'Lasso', 'Elastic', 'Random']
        
        for i in range(21):
            prime = self.prime_sequence[i]
            # Time horizons based on prime sequence (hours)
            time_horizon = prime * 2  # 4, 6, 10, 14, 22, 26, 34, 38, 46, 58...
            model_type = model_types[i % len(model_types)]
            
            branch = TimelineBranch(
                branch_id=i,
                time_horizon=time_horizon,
                model_type=model_type,
                prime_base=prime
            )
            branches.append(branch)
        
        return branches
    
    def _create_simple_model(self, model_type: str, prime_base: int, data: pd.Series):
        """Create and train a simple model based on type"""
        try:
            if model_type in ['Linear', 'Trend']:
                # Simple linear regression
                x = np.arange(len(data))
                y = data.values
                coeffs = np.polyfit(x, y, 1)
                return {'type': 'linear', 'coeffs': coeffs}
            
            elif model_type == 'MovingAvg':
                # Moving average
                window = min(prime_base, len(data) // 2)
                return {'type': 'moving_avg', 'window': window}
            
            elif model_type == 'Exponential':
                # Exponential smoothing
                alpha = 1.0 / prime_base if prime_base > 0 else 0.1
                return {'type': 'exponential', 'alpha': alpha}
            
            elif model_type == 'Polynomial':
                # Polynomial fit
                degree = min(prime_base % 5 + 1, 5)
                x = np.arange(len(data))
                y = data.values
                coeffs = np.polyfit(x, y, degree)
                return {'type': 'polynomial', 'coeffs': coeffs, 'degree': degree}
            
            elif model_type == 'AR':
                # Autoregressive
                order = min(prime_base % 10 + 1, len(data) // 2)
                return {'type': 'ar', 'order': order}
            
            elif model_type == 'Fourier':
                # Fourier transform based
                return {'type': 'fourier', 'components': prime_base}
            
            else:
                # Default: linear
                x = np.arange(len(data))
                y = data.values
                coeffs = np.polyfit(x, y, 1)
                return {'type': 'linear', 'coeffs': coeffs}
        except Exception as e:
            # Fallback to simple linear
            x = np.arange(len(data))
            y = data.values
            coeffs = np.polyfit(x, y, 1)
            return {'type': 'linear', 'coeffs': coeffs}
    
    def _predict_simple_model(self, model: Dict, data: pd.Series, horizon: int) -> float:
        """Make prediction using simple model"""
        try:
            if model['type'] == 'linear':
                coeffs = model['coeffs']
                x = len(data) + horizon
                return coeffs[0] * x + coeffs[1]
            
            elif model['type'] == 'moving_avg':
                window = model['window']
                return data.tail(window).mean()
            
            elif model['type'] == 'exponential':
                alpha = model['alpha']
                last_value = data.iloc[-1]
                # Simple exponential smoothing
                return last_value * (1 + alpha * horizon)
            
            elif model['type'] == 'polynomial':
                coeffs = model['coeffs']
                x = len(data) + horizon
                return np.polyval(coeffs, x)
            
            elif model['type'] == 'ar':
                order = model['order']
                # Simple AR prediction
                recent = data.tail(order).values
                weights = np.array([0.5 ** i for i in range(len(recent))])
                weights = weights / weights.sum()
                return np.dot(recent, weights)
            
            elif model['type'] == 'fourier':
                # Simple trend continuation
                return data.iloc[-1] * (1 + 0.01 * horizon)
            
            else:
                # Default: last value
                return data.iloc[-1]
        except Exception:
            return data.iloc[-1]
    
    def train_models(self, training_data: pd.Series):
        """Train all 21 models on different timeline branches"""
        if len(training_data) < 10:
            return
        
        for branch in self.branches:
            try:
                # Prepare data for this branch's time horizon
                branch_data = self._prepare_branch_data(training_data, branch)
                
                if len(branch_data) < 3:
                    continue
                
                # Create and train model
                model = self._create_simple_model(
                    branch.model_type, 
                    branch.prime_base, 
                    branch_data
                )
                
                self.models[branch.branch_id] = {
                    'model': model,
                    'branch_data': branch_data
                }
            except Exception as e:
                # Skip this branch if training fails
                continue
        
        self.trained = True
    
    def _prepare_branch_data(self, data: pd.Series, branch: TimelineBranch) -> pd.Series:
        """Prepare data for specific branch's time horizon"""
        try:
            # Apply Wallace Transform with branch-specific parameters
            alpha = float(self.constants.PHI) * (branch.prime_base / 7.0)
            beta = branch.prime_base / 10.0
            epsilon = 1e-15
            
            # Transform data
            transformed = data.copy()
            transformed = transformed.apply(
                lambda x: alpha * np.log(x + epsilon) ** float(self.constants.PHI) + beta
            )
            
            # Downsample if needed (simulate prime-based interval)
            if len(transformed) > branch.prime_base * 10:
                step = max(1, len(transformed) // (branch.prime_base * 10))
                transformed = transformed.iloc[::step]
            
            return transformed
        except Exception:
            return data
    
    def predict_all_branches(self, current_data: pd.Series) -> Dict[int, float]:
        """Generate predictions from all 21 branches"""
        predictions = {}
        
        for branch in self.branches:
            if branch.branch_id not in self.models:
                # Use simple prediction if not trained
                branch.prediction = current_data.iloc[-1]
                branch.confidence = 0.5
                predictions[branch.branch_id] = float(branch.prediction)
                continue
            
            try:
                model_info = self.models[branch.branch_id]
                model = model_info['model']
                branch_data = model_info.get('branch_data', current_data)
                
                # Predict
                horizon = int(branch.time_horizon)
                prediction = self._predict_simple_model(model, branch_data, horizon)
                
                predictions[branch.branch_id] = float(prediction)
                branch.prediction = float(prediction)
                
                # Calculate confidence
                branch.confidence = self._calculate_branch_confidence(
                    branch, current_data
                )
            except Exception:
                # Fallback
                branch.prediction = current_data.iloc[-1]
                branch.confidence = 0.5
                predictions[branch.branch_id] = float(branch.prediction)
        
        return predictions
    
    def _calculate_branch_confidence(self, branch: TimelineBranch, 
                                     data: pd.Series) -> float:
        """Calculate confidence for a branch based on prime pattern matching"""
        try:
            # Check if branch's prime appears in price patterns
            price_changes = data.diff().dropna()
            if len(price_changes) == 0:
                return 0.5
            
            normalized = (price_changes - price_changes.mean()) / (price_changes.std() + 1e-10)
            
            # Count occurrences of prime-based patterns
            prime_pattern_count = 0
            for val in normalized:
                if abs(val - (branch.prime_base % 10)) < 0.5:
                    prime_pattern_count += 1
            
            # Confidence based on pattern frequency
            confidence = min(0.95, 0.5 + (prime_pattern_count / len(normalized)) * 0.45)
            
            # Apply consciousness weighting
            confidence *= float(self.constants.CONSCIOUSNESS)
            
            return float(confidence)
        except Exception:
            return 0.5
    
    def find_most_likely(self, predictions: Dict[int, float], 
                         current_price: float) -> Dict[str, any]:
        """
        Find the most likely outcome from 21-model ensemble
        
        Uses consensus voting with prime-weighted confidence
        """
        if not predictions:
            return {
                'consensus_prediction': current_price,
                'confidence': 0.5,
                'coherence': 0.5,
                'most_confident_branch': 0,
                'branch_predictions': {},
                'branch_confidences': {},
                'price_change_pct': 0.0
            }
        
        # Calculate weighted predictions
        weighted_sum = 0.0
        total_weight = 0.0
        
        for branch_id, prediction in predictions.items():
            branch = self.branches[branch_id]
            weight = branch.confidence * (branch.prime_base / 7.0)  # Normalize to prime 7
            weighted_sum += prediction * weight
            total_weight += weight
        
        # Consensus prediction
        consensus_prediction = weighted_sum / total_weight if total_weight > 0 else current_price
        
        # Calculate coherence across branches
        prediction_std = np.std(list(predictions.values()))
        coherence = 1.0 / (1.0 + prediction_std / (current_price + 1e-10))
        
        # Find most confident branch
        most_confident_branch = max(self.branches, key=lambda b: b.confidence)
        
        # Calculate overall confidence
        avg_confidence = np.mean([b.confidence for b in self.branches if b.confidence > 0])
        overall_confidence = avg_confidence * coherence if avg_confidence > 0 else 0.5
        
        return {
            'consensus_prediction': float(consensus_prediction),
            'confidence': float(overall_confidence),
            'coherence': float(coherence),
            'most_confident_branch': most_confident_branch.branch_id,
            'branch_predictions': {b.branch_id: b.prediction for b in self.branches if b.prediction is not None},
            'branch_confidences': {b.branch_id: b.confidence for b in self.branches},
            'price_change_pct': float((consensus_prediction - current_price) / current_price * 100)
        }
    
    def analyze_timeline_branches(self, current_data: pd.Series) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Analyze all timeline branches and return comprehensive results
        """
        # Train if not already trained
        if not self.trained:
            self.train_models(current_data)
        
        # Get predictions from all branches
        predictions = self.predict_all_branches(current_data)
        
        # Find most likely outcome
        result = self.find_most_likely(predictions, current_data.iloc[-1])
        
        # Create summary DataFrame
        branch_data = []
        for branch in self.branches:
            branch_data.append({
                'branch_id': branch.branch_id,
                'prime_base': branch.prime_base,
                'time_horizon_hours': branch.time_horizon,
                'model_type': branch.model_type,
                'prediction': branch.prediction if branch.prediction is not None else current_data.iloc[-1],
                'confidence': branch.confidence,
                'price_change_pct': ((branch.prediction - current_data.iloc[-1]) / 
                                   current_data.iloc[-1] * 100) if branch.prediction is not None else 0
            })
        
        df = pd.DataFrame(branch_data)
        df = df.sort_values('confidence', ascending=False)
        
        return df, result

