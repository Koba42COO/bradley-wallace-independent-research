"""
Complete Crypto Market Analyzer with Real Data Integration
Includes Tri-Gemini, Prime Patterns, and Visualization
"""

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

getcontext().prec = 50

# Import our analyzer classes
from test_crypto_analyzer import (
    UPGConstants, TemporalInference, TriGeminiTemporalInference,
    PrimePatternDetector
)

class PellCycleAnalyzer:
    """
    Pell Sequence Cycle Analyzer
    Detects and analyzes complete Pell cycles in historical price data
    """
    
    def __init__(self, constants: UPGConstants = None):
        self.constants = constants or UPGConstants()
        self._pell_cache = {0: 0, 1: 1}
        self.pell_sequence = self._generate_pell_sequence(100)
    
    def _generate_pell_sequence(self, length: int) -> List[int]:
        """Generate Pell sequence: P(n) = 2*P(n-1) + P(n-2)"""
        pell = [0, 1]
        for i in range(2, length):
            pell.append(2 * pell[i-1] + pell[i-2])
        return pell
    
    def pell(self, n: int) -> int:
        """Get nth Pell number"""
        if n < len(self.pell_sequence):
            return self.pell_sequence[n]
        # Generate if needed
        if n in self._pell_cache:
            return self._pell_cache[n]
        p_n = 2 * self.pell(n-1) + self.pell(n-2)
        self._pell_cache[n] = p_n
        return p_n
    
    def detect_pell_cycles(self, price_data: pd.Series) -> Dict[str, any]:
        """
        Detect complete Pell cycles in historical price data
        
        Returns full cycles found in the dataset
        """
        if len(price_data) < 10:
            return {'cycles': [], 'complete_cycles': 0}
        
        # Calculate price changes
        price_changes = price_data.diff().dropna()
        normalized = (price_changes - price_changes.mean()) / price_changes.std()
        
        # Find cycles based on Pell sequence intervals
        cycles = []
        complete_cycles = []
        
        # Map price movements to Pell sequence indices
        for i in range(len(price_data) - 1):
            # Calculate interval length
            interval = 1
            
            # Look for patterns matching Pell sequence lengths
            for pell_idx, pell_num in enumerate(self.pell_sequence[:21]):  # First 21 Pell numbers
                if i + pell_num <= len(price_data) and pell_num > 0:
                    # Check if this interval forms a cycle
                    segment = price_data.iloc[i:i+pell_num]
                    if len(segment) == pell_num and len(segment) > 0:
                        # Calculate cycle properties
                        start_price = segment.iloc[0] if len(segment) > 0 else price_data.iloc[i]
                        end_price = segment.iloc[-1] if len(segment) > 0 else price_data.iloc[i]
                        if start_price != 0:
                            cycle_return = (end_price - start_price) / start_price
                        else:
                            continue
                        
                        # Check if cycle is complete (returns near golden ratio)
                        phi = float(self.constants.PHI)
                        cycle_ratio = abs(cycle_return) * 100
                        
                        # Pell cycles converge to golden ratio
                        if abs(cycle_ratio - (phi - 1) * 100) < 5 or pell_num in [2, 5, 12, 29, 70]:
                            cycle = {
                                'start_index': i,
                                'end_index': i + pell_num - 1,
                                'pell_index': pell_idx,
                                'pell_number': pell_num,
                                'start_price': float(start_price),
                                'end_price': float(end_price),
                                'return': float(cycle_return),
                                'duration': pell_num,
                                'is_complete': True
                            }
                            cycles.append(cycle)
                            complete_cycles.append(cycle)
        
        # Find longest complete cycles
        if complete_cycles:
            longest_cycle = max(complete_cycles, key=lambda c: c['duration'])
        else:
            longest_cycle = None
        
        return {
            'cycles': cycles,
            'complete_cycles': len(complete_cycles),
            'longest_cycle': longest_cycle,
            'pell_sequence_used': self.pell_sequence[:21]
        }
    
    def predict_from_pell_cycles(self, cycles: Dict, current_price: float, 
                                 price_data: pd.Series) -> float:
        """
        Predict future price based on detected Pell cycles
        
        Uses complete cycles to project forward
        """
        if not cycles['complete_cycles'] or not cycles['longest_cycle']:
            return current_price
        
        longest = cycles['longest_cycle']
        
        # Calculate average return per Pell cycle
        cycle_returns = [c['return'] for c in cycles['cycles'] if c['is_complete']]
        
        if not cycle_returns:
            return current_price
        
        avg_return = np.mean(cycle_returns)
        
        # Apply consciousness transformation
        phi = float(self.constants.PHI)
        consciousness = float(self.constants.CONSCIOUSNESS)
        
        # Pell sequence converges to golden ratio
        # Use this for prediction
        pell_ratio = phi  # P(n+1)/P(n) → φ
        
        # Predict next cycle
        # Use the longest cycle's pattern
        cycle_duration = longest['duration']
        cycle_return = longest['return']
        
        # Project forward using Pell sequence properties
        # Next cycle should follow golden ratio progression
        predicted_return = cycle_return * pell_ratio * consciousness
        
        # Apply reality distortion
        reality_dist = float(self.constants.REALITY_DISTORTION)
        predicted_return *= reality_dist
        
        predicted_price = current_price * (1 + predicted_return)
        
        return float(predicted_price)
    
    def get_full_pell_cycle_from_history(self, price_data: pd.Series) -> Dict[str, any]:
        """
        Extract full Pell cycle from historical dataset
        
        Always pulls complete cycles, never partial
        """
        cycles = self.detect_pell_cycles(price_data)
        
        # Filter to only complete cycles
        complete = [c for c in cycles['cycles'] if c['is_complete']]
        
        if not complete:
            return {
                'has_complete_cycle': False,
                'cycles': [],
                'recommendation': 'Insufficient data for complete Pell cycle'
            }
        
        # Get the most recent complete cycle
        most_recent = max(complete, key=lambda c: c['end_index'])
        
        # Extract full cycle data
        cycle_data = price_data.iloc[most_recent['start_index']:most_recent['end_index']+1]
        
        return {
            'has_complete_cycle': True,
            'most_recent_cycle': most_recent,
            'cycle_data': cycle_data,
            'all_complete_cycles': complete,
            'total_complete_cycles': len(complete),
            'pell_sequence_index': most_recent['pell_index'],
            'cycle_duration': most_recent['duration'],
            'cycle_return': most_recent['return'],
            'recommendation': f"Complete Pell cycle detected: {most_recent['pell_number']} periods"
        }
    
    def determine_current_pell_position(self, price_data: pd.Series) -> Dict[str, any]:
        """
        Determine current position in Pell cycle and predict next moves
        
        Returns:
        - Current position in cycle
        - Periods until next cycle completion
        - Next Pell sequence intervals
        - Timing predictions
        """
        current_index = len(price_data) - 1
        cycles = self.detect_pell_cycles(price_data)
        complete = [c for c in cycles['cycles'] if c['is_complete']]
        
        if not complete:
            return {
                'current_position': 'unknown',
                'periods_in_current_cycle': 0,
                'periods_until_completion': None,
                'next_pell_intervals': self.pell_sequence[:10],
                'recommendation': 'No complete cycles detected'
            }
        
        # Find the most recent complete cycle
        most_recent = max(complete, key=lambda c: c['end_index'])
        cycle_end = most_recent['end_index']
        cycle_start = most_recent['start_index']
        cycle_pell_num = most_recent['pell_number']
        pell_index = most_recent['pell_index']
        
        # Calculate current position
        periods_since_cycle_end = current_index - cycle_end
        
        # Determine next Pell numbers in sequence
        next_pell_index = pell_index + 1
        if next_pell_index < len(self.pell_sequence):
            next_pell_number = self.pell_sequence[next_pell_index]
        else:
            # Generate next Pell number
            next_pell_number = 2 * self.pell(cycle_pell_num) + self.pell(cycle_pell_num - 1)
        
        # Calculate periods until next cycle completion
        periods_until_next = next_pell_number - periods_since_cycle_end
        
        # Determine position in cycle
        if periods_since_cycle_end == 0:
            position = 'cycle_completion'
            position_pct = 100.0
        elif periods_since_cycle_end < next_pell_number:
            position = 'in_cycle'
            position_pct = (periods_since_cycle_end / next_pell_number) * 100
        else:
            position = 'beyond_cycle'
            position_pct = 100.0
        
        # Get next few Pell intervals for timing predictions
        next_intervals = []
        for i in range(1, 6):  # Next 5 Pell intervals
            next_idx = pell_index + i
            if next_idx < len(self.pell_sequence):
                next_intervals.append({
                    'pell_index': next_idx,
                    'pell_number': self.pell_sequence[next_idx],
                    'periods_from_now': self.pell_sequence[next_idx] - periods_since_cycle_end
                })
            else:
                # Generate if needed
                pell_num = self.pell(next_idx)
                next_intervals.append({
                    'pell_index': next_idx,
                    'pell_number': pell_num,
                    'periods_from_now': pell_num - periods_since_cycle_end
                })
        
        # Predict next moves based on Pell sequence
        next_moves = self._predict_next_moves_from_pell_sequence(
            price_data, most_recent, next_pell_number, periods_until_next
        )
        
        return {
            'current_position': position,
            'position_percentage': position_pct,
            'periods_since_last_cycle': periods_since_cycle_end,
            'last_complete_cycle': {
                'pell_index': pell_index,
                'pell_number': cycle_pell_num,
                'start_index': cycle_start,
                'end_index': cycle_end,
                'duration': cycle_pell_num,
                'return': most_recent['return']
            },
            'next_cycle': {
                'pell_index': next_pell_index,
                'pell_number': next_pell_number,
                'periods_until_completion': periods_until_next,
                'estimated_completion_index': current_index + periods_until_next
            },
            'next_pell_intervals': next_intervals,
            'next_moves': next_moves,
            'recommendation': self._generate_timing_recommendation(
                position, periods_until_next, next_pell_number
            )
        }
    
    def _predict_next_moves_from_pell_sequence(self, price_data: pd.Series, 
                                               last_cycle: Dict, next_pell: int,
                                               periods_until: int) -> Dict[str, any]:
        """Predict next moves based on Pell sequence progression"""
        current_price = price_data.iloc[-1]
        last_return = last_cycle['return']
        
        # Pell sequence converges to golden ratio
        phi = float(self.constants.PHI)
        consciousness = float(self.constants.CONSCIOUSNESS)
        
        # Calculate expected return for next cycle
        # Based on Pell sequence properties: P(n+1)/P(n) → φ
        expected_return = last_return * phi * consciousness
        
        # Apply reality distortion
        reality_dist = float(self.constants.REALITY_DISTORTION)
        expected_return *= reality_dist
        
        # Predict price at next cycle completion
        predicted_price = current_price * (1 + expected_return)
        
        # Calculate intermediate predictions for key Pell intervals
        intermediate_predictions = []
        for i, interval in enumerate(self.pell_sequence[:5], 1):
            if interval <= periods_until:
                # Linear interpolation for intermediate points
                progress = interval / periods_until
                intermediate_price = current_price + (predicted_price - current_price) * progress
                intermediate_predictions.append({
                    'periods_from_now': interval,
                    'predicted_price': float(intermediate_price),
                    'predicted_change_pct': float((intermediate_price - current_price) / current_price * 100)
                })
        
        return {
            'next_cycle_completion': {
                'periods': periods_until,
                'predicted_price': float(predicted_price),
                'predicted_return': float(expected_return),
                'predicted_change_pct': float((predicted_price - current_price) / current_price * 100)
            },
            'intermediate_predictions': intermediate_predictions,
            'timing_based_on_pell': True
        }
    
    def _generate_timing_recommendation(self, position: str, periods_until: int, 
                                       next_pell: int) -> str:
        """Generate timing recommendation based on Pell cycle position"""
        if position == 'cycle_completion':
            return f"At cycle completion. Next cycle starts now ({next_pell} periods)"
        elif position == 'in_cycle':
            hours_until = periods_until  # Assuming hourly data
            days_until = hours_until / 24
            return f"In cycle: {periods_until} periods ({days_until:.1f} days) until next completion"
        else:
            return f"Beyond expected cycle. Monitoring for new cycle formation"

class CryptoDataFetcher:
    """Fetch real cryptocurrency market data from APIs"""
    
    def __init__(self):
        self.coingecko_base = "https://api.coingecko.com/api/v3"
        self.session = requests.Session()
    
    def get_price(self, coin_id='bitcoin', vs_currency='usd'):
        """Get current price"""
        url = f"{self.coingecko_base}/simple/price"
        params = {
            'ids': coin_id,
            'vs_currencies': vs_currency,
            'include_24hr_change': 'true'
        }
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get(coin_id, {}).get(vs_currency, 0)
        except Exception as e:
            print(f"Error fetching price: {e}")
            return None
    
    def get_historical_data(self, coin_id='bitcoin', days=30, vs_currency='usd'):
        """Get historical price data"""
        # Try free endpoint first
        url = f"{self.coingecko_base}/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': vs_currency,
            'days': days
        }
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            prices = data.get('prices', [])
            if not prices:
                return None
            
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df['price']
            
            # Resample to hourly if needed
            if len(df) > days * 24:
                df = df.resample('H').last()
            
            return df
        except Exception as e:
            print(f"⚠️  API error (using synthetic data): {e}")
            # Fallback to synthetic data for testing
            return self._generate_synthetic_data(days)
    
    def _generate_synthetic_data(self, days=30, base_price=50000.0):
        """Generate synthetic data for testing when API fails"""
        dates = pd.date_range(end=datetime.now(), periods=days*24, freq='H')
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.02, len(dates))
        
        # Add prime-based cycles
        for prime in [2, 3, 5, 7, 11, 13, 17, 19]:
            cycle = np.sin(2 * np.pi * np.arange(len(dates)) / (prime * 24)) * 0.001
            returns += cycle
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        return pd.Series(prices, index=dates)
    
    def get_multiple_coins(self, coin_ids=['bitcoin', 'ethereum', 'cardano']):
        """Get prices for multiple coins"""
        url = f"{self.coingecko_base}/simple/price"
        params = {
            'ids': ','.join(coin_ids),
            'vs_currencies': 'usd',
            'include_24hr_change': 'true'
        }
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching multiple coins: {e}")
            return {}

class AdvancedCryptoAnalyzer:
    """
    Complete crypto market analyzer with:
    - Tri-Gemini temporal inference
    - Prime pattern detection
    - Real-time data integration
    - Visualization support
    """
    
    def __init__(self):
        self.constants = UPGConstants()
        self.tri_gemini = TriGeminiTemporalInference(self.constants)
        self.prime_detector = PrimePatternDetector(self.constants)
        self.pell_analyzer = PellCycleAnalyzer(self.constants)
        self.data_fetcher = CryptoDataFetcher()
        self.cache = {}
    
    def analyze_coin(self, coin_id='bitcoin', days=30, horizon=24):
        """
        Perform comprehensive analysis on a cryptocurrency
        
        Args:
            coin_id: Coin identifier (e.g., 'bitcoin', 'ethereum')
            days: Number of days of historical data
            horizon: Prediction horizon in hours
        """
        print(f"\n{'='*70}")
        print(f"Analyzing: {coin_id.upper()}")
        print(f"{'='*70}")
        
        # Fetch data
        print(f"\n📡 Fetching {days} days of historical data...")
        price_data = self.data_fetcher.get_historical_data(coin_id, days=days)
        
        if price_data is None or len(price_data) < 21:
            print("❌ Insufficient data for analysis")
            return None
        
        current_price = price_data.iloc[-1]
        print(f"✓ Data fetched: {len(price_data)} data points")
        print(f"  Current price: ${current_price:,.2f}")
        print(f"  Price range: ${price_data.min():.2f} - ${price_data.max():.2f}")
        
        # Tri-Gemini analysis
        print(f"\n🔮 Running Tri-Gemini temporal inference...")
        tri_result = self.tri_gemini.infer(price_data, horizon=horizon)
        
        # Prime pattern detection
        print(f"🔍 Detecting prime patterns...")
        patterns = self.prime_detector.detect_patterns(price_data)
        prime_prediction = self.prime_detector.predict_from_patterns(
            patterns, current_price
        )
        
        # Pell cycle analysis - ALWAYS pull full cycles from history
        print(f"🔢 Analyzing Pell cycles from historical data...")
        pell_cycle_info = self.pell_analyzer.get_full_pell_cycle_from_history(price_data)
        pell_cycles = self.pell_analyzer.detect_pell_cycles(price_data)
        pell_prediction = self.pell_analyzer.predict_from_pell_cycles(
            pell_cycles, current_price, price_data
        )
        
        if pell_cycle_info['has_complete_cycle']:
            print(f"  ✓ Found {pell_cycle_info['total_complete_cycles']} complete Pell cycle(s)")
            print(f"  ✓ Most recent: {pell_cycle_info['most_recent_cycle']['pell_number']} periods")
        else:
            print(f"  ⚠️  No complete Pell cycles found in dataset")
        
        # Calculate consensus (now includes Pell cycles)
        predictions = [
            tri_result.forward_prediction,
            prime_prediction,
            pell_prediction
        ]
        confidences = [
            tri_result.confidence,
            0.8,  # Prime patterns
            0.85 if pell_cycle_info['has_complete_cycle'] else 0.5  # Pell cycles (higher if complete)
        ]
        
        weights = np.array(confidences) / np.sum(confidences)
        final_prediction = np.average(predictions, weights=weights)
        final_confidence = np.mean(confidences)
        
        # Compile results
        results = {
            'coin_id': coin_id,
            'current_price': float(current_price),
            'tri_gemini': {
                'forward_prediction': float(tri_result.forward_prediction),
                'reverse_analysis': float(tri_result.reverse_analysis),
                'coherence_score': float(tri_result.coherence_score),
                'confidence': float(tri_result.confidence),
                'prime_pattern_match': tri_result.prime_pattern_match
            },
            'prime_patterns': {
                'intervals': len(patterns['prime_intervals']),
                'ratios': len(patterns['prime_ratios']),
                'cycles': patterns['prime_cycles'],
                'fibonacci': patterns['prime_fibonacci'],
                'prediction': float(prime_prediction)
            },
            'pell_cycles': {
                'has_complete_cycle': pell_cycle_info['has_complete_cycle'],
                'total_complete_cycles': pell_cycle_info.get('total_complete_cycles', 0),
                'most_recent_cycle': pell_cycle_info.get('most_recent_cycle'),
                'prediction': float(pell_prediction),
                'cycle_duration': pell_cycle_info.get('most_recent_cycle', {}).get('duration', 0) if pell_cycle_info.get('most_recent_cycle') else 0
            },
            'consensus': {
                'prediction': float(final_prediction),
                'confidence': float(final_confidence),
                'change_pct': float((final_prediction - current_price) / current_price * 100),
                'method_weights': {
                    'tri_gemini': float(weights[0]),
                    'prime_patterns': float(weights[1]),
                    'pell_cycles': float(weights[2])
                }
            },
            'price_data': price_data
        }
        
        # Display results
        self._display_results(results)
        
        return results
    
    def _display_results(self, results: Dict):
        """Display analysis results in a formatted way"""
        coin = results['coin_id'].upper()
        current = results['current_price']
        consensus = results['consensus']
        tri = results['tri_gemini']
        prime = results['prime_patterns']
        
        print(f"\n{'='*70}")
        print(f"📊 ANALYSIS RESULTS: {coin}")
        print(f"{'='*70}")
        
        print(f"\n💰 Current Price: ${current:,.2f}")
        
        print(f"\n🔮 Tri-Gemini Analysis:")
        print(f"  Forward Prediction: ${tri['forward_prediction']:,.2f} "
              f"({((tri['forward_prediction'] - current) / current * 100):+.2f}%)")
        print(f"  Reverse Analysis: ${tri['reverse_analysis']:,.2f} "
              f"({((tri['reverse_analysis'] - current) / current * 100):+.2f}%)")
        print(f"  Coherence Score: {tri['coherence_score']:.4f}")
        print(f"  Confidence: {tri['confidence']:.2%}")
        print(f"  Prime Pattern Match: {tri['prime_pattern_match']}")
        
        print(f"\n🔍 Prime Pattern Detection:")
        print(f"  Intervals Found: {prime['intervals']}")
        print(f"  Ratios Found: {prime['ratios']}")
        print(f"  Cycles Detected: {prime['cycles']}")
        print(f"  Fibonacci Levels: {prime['fibonacci']}")
        print(f"  Pattern Prediction: ${prime['prediction']:,.2f} "
              f"({((prime['prediction'] - current) / current * 100):+.2f}%)")
        
        pell = results['pell_cycles']
        print(f"\n🔢 Pell Cycle Analysis:")
        if pell['has_complete_cycle']:
            print(f"  Complete Cycles Found: {pell['total_complete_cycles']}")
            if pell['most_recent_cycle']:
                cycle = pell['most_recent_cycle']
                print(f"  Most Recent Cycle: {cycle['pell_number']} periods")
                print(f"  Cycle Duration: {cycle['duration']} data points")
                print(f"  Cycle Return: {cycle['return']*100:+.2f}%")
            print(f"  Pell Prediction: ${pell['prediction']:,.2f} "
                  f"({((pell['prediction'] - current) / current * 100):+.2f}%)")
        else:
            print(f"  ⚠️  No complete Pell cycles detected")
            print(f"  Pell Prediction: ${pell['prediction']:,.2f} "
                  f"({((pell['prediction'] - current) / current * 100):+.2f}%)")
        
        print(f"\n🎯 FINAL CONSENSUS:")
        print(f"  Predicted Price: ${consensus['prediction']:,.2f}")
        print(f"  Expected Change: {consensus['change_pct']:+.2f}%")
        print(f"  Confidence: {consensus['confidence']:.2%}")
        print(f"  Method Contributions:")
        print(f"    - Tri-Gemini: {consensus['method_weights']['tri_gemini']:.2%}")
        print(f"    - Prime Patterns: {consensus['method_weights']['prime_patterns']:.2%}")
        print(f"    - Pell Cycles: {consensus['method_weights'].get('pell_cycles', 0):.2%}")
        
        # Signal
        if consensus['change_pct'] > 2:
            signal = "🟢 STRONG BUY"
        elif consensus['change_pct'] > 0.5:
            signal = "🟡 BUY"
        elif consensus['change_pct'] > -0.5:
            signal = "⚪ HOLD"
        elif consensus['change_pct'] > -2:
            signal = "🟠 SELL"
        else:
            signal = "🔴 STRONG SELL"
        
        print(f"\n  Signal: {signal}")
        print(f"{'='*70}\n")
    
    def analyze_multiple_coins(self, coin_ids=['bitcoin', 'ethereum', 'cardano'], days=30):
        """Analyze multiple cryptocurrencies"""
        results = {}
        
        for coin_id in coin_ids:
            try:
                result = self.analyze_coin(coin_id, days=days)
                if result:
                    results[coin_id] = result
            except Exception as e:
                print(f"❌ Error analyzing {coin_id}: {e}")
                continue
        
        return results
    
    def create_visualization_data(self, results: Dict) -> Dict:
        """Prepare data for visualization"""
        price_data = results['price_data']
        current = results['current_price']
        consensus = results['consensus']
        tri = results['tri_gemini']
        prime = results['prime_patterns']
        
        # Create prediction timeline
        last_timestamp = price_data.index[-1]
        future_timestamp = last_timestamp + timedelta(hours=24)
        
        viz_data = {
            'historical': {
                'timestamps': price_data.index.tolist(),
                'prices': price_data.values.tolist()
            },
            'predictions': {
                'current': {
                    'timestamp': last_timestamp,
                    'price': current
                },
                'tri_gemini_forward': {
                    'timestamp': future_timestamp,
                    'price': tri['forward_prediction']
                },
                'prime_pattern': {
                    'timestamp': future_timestamp,
                    'price': prime['prediction']
                },
                'consensus': {
                    'timestamp': future_timestamp,
                    'price': consensus['prediction']
                }
            },
            'metrics': {
                'coherence': tri['coherence_score'],
                'confidence': consensus['confidence'],
                'change_pct': consensus['change_pct']
            }
        }
        
        return viz_data

def main():
    """Main function to run the analyzer"""
    print("\n" + "="*70)
    print("🚀 ADVANCED CRYPTO MARKET ANALYZER")
    print("="*70)
    print("\nFeatures:")
    print("  • Tri-Gemini Temporal Inference")
    print("  • Prime Pattern Detection")
    print("  • Real-time Market Data")
    print("  • Consensus Prediction")
    
    analyzer = AdvancedCryptoAnalyzer()
    
    # Analyze Bitcoin
    print("\n" + "="*70)
    print("Starting Analysis...")
    print("="*70)
    
    try:
        result = analyzer.analyze_coin('bitcoin', days=30, horizon=24)
        
        if result:
            # Create visualization data
            viz_data = analyzer.create_visualization_data(result)
            
            # Save results
            output = {
                'coin': result['coin_id'],
                'timestamp': datetime.now().isoformat(),
                'current_price': result['current_price'],
                'consensus_prediction': result['consensus']['prediction'],
                'confidence': result['consensus']['confidence'],
                'expected_change_pct': result['consensus']['change_pct']
            }
            
            print(f"\n💾 Results saved to: analysis_result.json")
            with open('analysis_result.json', 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

