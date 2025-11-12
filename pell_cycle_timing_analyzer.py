"""
Pell Cycle Timing Analyzer
Determines current position in Pell cycle and predicts next moves based on Pell sequence
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
from crypto_analyzer_complete import AdvancedCryptoAnalyzer
import json

class PellCycleTimingAnalyzer:
    """Analyze current Pell cycle position and timing for top cryptocurrencies"""
    
    def __init__(self):
        self.analyzer = AdvancedCryptoAnalyzer()
        
        # Top cryptocurrencies
        self.coins = [
            'bitcoin', 'ethereum', 'tether', 'binancecoin', 'solana',
            'usd-coin', 'xrp', 'cardano', 'dogecoin', 'avalanche-2'
        ]
    
    def analyze_coin_timing(self, coin_id: str, days: int = 30):
        """Analyze current Pell cycle position and timing for a coin"""
        print(f"\n{'='*70}")
        print(f"🔢 PELL CYCLE TIMING ANALYSIS: {coin_id.upper()}")
        print(f"{'='*70}")
        
        # Get price data
        price_data = self.analyzer.data_fetcher.get_historical_data(coin_id, days=days)
        
        if price_data is None or len(price_data) < 21:
            print("❌ Insufficient data")
            return None
        
        current_price = price_data.iloc[-1]
        current_time = price_data.index[-1]
        
        # Determine current Pell cycle position
        position_info = self.analyzer.pell_analyzer.determine_current_pell_position(price_data)
        
        # Get Pell cycle info
        pell_cycle_info = self.analyzer.pell_analyzer.get_full_pell_cycle_from_history(price_data)
        
        # Get predictions
        pell_cycles = self.analyzer.pell_analyzer.detect_pell_cycles(price_data)
        pell_prediction = self.analyzer.pell_analyzer.predict_from_pell_cycles(
            pell_cycles, current_price, price_data
        )
        
        result = {
            'coin_id': coin_id,
            'current_price': float(current_price),
            'current_time': current_time.isoformat(),
            'position_info': position_info,
            'pell_cycle_info': pell_cycle_info,
            'pell_prediction': float(pell_prediction)
        }
        
        # Display results
        self._display_timing_analysis(result)
        
        return result
    
    def _display_timing_analysis(self, result: Dict):
        """Display detailed timing analysis"""
        coin = result['coin_id'].upper()
        current_price = result['current_price']
        position = result['position_info']
        next_moves = position.get('next_moves', {})
        
        print(f"\n💰 Current Price: ${current_price:,.2f}")
        print(f"📅 Current Time: {result['current_time']}")
        
        print(f"\n📍 CURRENT PELL CYCLE POSITION:")
        print(f"  Position: {position['current_position'].upper().replace('_', ' ')}")
        print(f"  Position in Cycle: {position['position_percentage']:.1f}%")
        print(f"  Periods Since Last Cycle: {position['periods_since_last_cycle']}")
        
        last_cycle = position.get('last_complete_cycle', {})
        if last_cycle:
            print(f"\n📊 LAST COMPLETE CYCLE:")
            print(f"  Pell Number: {last_cycle.get('pell_number', 'N/A')}")
            print(f"  Duration: {last_cycle.get('duration', 'N/A')} periods")
            print(f"  Return: {last_cycle.get('return', 0)*100:+.2f}%")
        
        next_cycle = position.get('next_cycle', {})
        if next_cycle:
            print(f"\n🔮 NEXT CYCLE PREDICTION:")
            print(f"  Next Pell Number: {next_cycle.get('pell_number', 'N/A')}")
            print(f"  Periods Until Completion: {next_cycle.get('periods_until_completion', 'N/A')}")
            
            periods = next_cycle.get('periods_until_completion', 0)
            hours = periods
            days = hours / 24
            weeks = days / 7
            
            print(f"  Time Until Completion:")
            print(f"    • {periods} periods (hours)")
            print(f"    • {days:.1f} days")
            print(f"    • {weeks:.1f} weeks")
            
            # Estimate completion time
            if periods > 0:
                completion_time = datetime.now() + timedelta(hours=periods)
                print(f"  Estimated Completion: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if next_moves:
            next_completion = next_moves.get('next_cycle_completion', {})
            if next_completion:
                print(f"\n🎯 NEXT MOVE PREDICTIONS (Based on Pell Sequence):")
                print(f"  At Cycle Completion ({next_completion.get('periods', 'N/A')} periods):")
                print(f"    Predicted Price: ${next_completion.get('predicted_price', 0):,.2f}")
                print(f"    Expected Return: {next_completion.get('predicted_return', 0)*100:+.2f}%")
                print(f"    Expected Change: {next_completion.get('predicted_change_pct', 0):+.2f}%")
            
            intermediate = next_moves.get('intermediate_predictions', [])
            if intermediate:
                print(f"\n  Intermediate Predictions (Key Pell Intervals):")
                for pred in intermediate[:5]:  # Show first 5
                    periods = pred.get('periods_from_now', 0)
                    price = pred.get('predicted_price', 0)
                    change = pred.get('predicted_change_pct', 0)
                    days = periods / 24
                    print(f"    {periods} periods ({days:.1f} days): ${price:,.2f} ({change:+.2f}%)")
        
        # Next Pell intervals
        intervals = position.get('next_pell_intervals', [])
        if intervals:
            print(f"\n📈 NEXT PELL SEQUENCE INTERVALS:")
            for interval in intervals[:5]:
                pell_num = interval.get('pell_number', 0)
                periods = interval.get('periods_from_now', 0)
                days = periods / 24 if periods > 0 else 0
                print(f"  Pell #{interval.get('pell_index', 'N/A')}: {pell_num} periods ({days:.1f} days from now)")
        
        print(f"\n💡 Recommendation: {position.get('recommendation', 'N/A')}")
        print(f"{'='*70}\n")
    
    def analyze_all_coins_timing(self, days: int = 30):
        """Analyze timing for all coins"""
        print("\n" + "="*70)
        print("🔢 TOP 10 CRYPTOCURRENCIES - PELL CYCLE TIMING ANALYSIS")
        print("="*70)
        print("\nDetermining current position in Pell cycles and predicting next moves...")
        
        results = []
        
        for i, coin_id in enumerate(self.coins, 1):
            try:
                result = self.analyze_coin_timing(coin_id, days=days)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"❌ Error analyzing {coin_id}: {str(e)[:50]}")
                continue
        
        return results
    
    def create_timing_summary(self, results: List[Dict]):
        """Create summary table of timing predictions"""
        print("\n" + "="*70)
        print("📊 PELL CYCLE TIMING SUMMARY")
        print("="*70)
        
        print(f"\n{'Coin':<15} {'Position':<15} {'Next Cycle':<12} {'Days Until':<12} {'Predicted Price':<15} {'Change':<10}")
        print("-" * 70)
        
        for result in results:
            coin = result['coin_id'].upper()
            position = result['position_info']
            next_cycle = position.get('next_cycle', {})
            next_moves = position.get('next_moves', {})
            
            pos = position.get('current_position', 'unknown').replace('_', ' ').title()
            next_pell = next_cycle.get('pell_number', 'N/A')
            periods = next_cycle.get('periods_until_completion', 0)
            days = periods / 24 if periods > 0 else 0
            
            next_completion = next_moves.get('next_cycle_completion', {})
            pred_price = next_completion.get('predicted_price', 0)
            change = next_completion.get('predicted_change_pct', 0)
            
            print(f"{coin:<15} {pos:<15} {next_pell:<12} {days:<12.1f} ${pred_price:<14,.2f} {change:+.2f}%")
        
        print("\n" + "="*70)
    
    def save_timing_results(self, results: List[Dict], filename='pell_cycle_timing_results.json'):
        """Save timing analysis results"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'total_coins': len(results),
            'results': results
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {filename}")

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("🚀 PELL CYCLE TIMING ANALYZER")
    print("="*70)
    print("\nThis analyzer:")
    print("  • Determines current position in Pell cycle")
    print("  • Calculates periods until next cycle completion")
    print("  • Predicts next moves based on Pell sequence")
    print("  • Provides timing estimates for key intervals")
    
    analyzer = PellCycleTimingAnalyzer()
    
    try:
        # Analyze all coins
        results = analyzer.analyze_all_coins_timing(days=30)
        
        if results:
            # Create summary
            analyzer.create_timing_summary(results)
            
            # Save results
            analyzer.save_timing_results(results)
            
            print("\n✅ TIMING ANALYSIS COMPLETE")
        else:
            print("\n❌ No results to analyze")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())

