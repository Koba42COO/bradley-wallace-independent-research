"""
Analyze Top 10 Cryptocurrencies Based on Pell Cycle Strength
Ranks coins by Pell cycle completeness, pattern strength, and prediction confidence
"""

import numpy as np
import pandas as pd
from datetime import datetime
from crypto_analyzer_complete import AdvancedCryptoAnalyzer, CryptoDataFetcher
import json

class Top10PellCycleAnalyzer:
    """Analyze and rank top cryptocurrencies by Pell cycle metrics"""
    
    def __init__(self):
        self.analyzer = AdvancedCryptoAnalyzer()
        self.data_fetcher = CryptoDataFetcher()
        
        # Top 10 cryptocurrencies by market cap
        self.top_coins = [
            'bitcoin', 'ethereum', 'tether', 'binancecoin', 'solana',
            'usd-coin', 'xrp', 'cardano', 'dogecoin', 'avalanche-2'
        ]
    
    def analyze_all_coins(self, days=30):
        """Analyze all top 10 coins and rank by Pell cycle strength"""
        print("\n" + "="*70)
        print("🔢 TOP 10 CRYPTOCURRENCIES - PELL CYCLE ANALYSIS")
        print("="*70)
        print(f"\nAnalyzing {len(self.top_coins)} cryptocurrencies...")
        print(f"Historical data: {days} days\n")
        
        results = []
        
        for i, coin_id in enumerate(self.top_coins, 1):
            print(f"[{i}/{len(self.top_coins)}] Analyzing {coin_id.upper()}...", end=" ")
            
            try:
                # Get price data
                price_data = self.analyzer.data_fetcher.get_historical_data(coin_id, days=days)
                
                if price_data is None or len(price_data) < 21:
                    print("❌ Insufficient data")
                    continue
                
                current_price = price_data.iloc[-1]
                
                # Pell cycle analysis
                pell_cycle_info = self.analyzer.pell_analyzer.get_full_pell_cycle_from_history(price_data)
                pell_cycles = self.analyzer.pell_analyzer.detect_pell_cycles(price_data)
                pell_prediction = self.analyzer.pell_analyzer.predict_from_pell_cycles(
                    pell_cycles, current_price, price_data
                )
                
                # Calculate Pell cycle score
                pell_score = self._calculate_pell_score(pell_cycle_info, pell_cycles, price_data)
                
                # Get current price info
                current_price_usd = self.analyzer.data_fetcher.get_price(coin_id)
                if current_price_usd is None:
                    current_price_usd = current_price
                
                result = {
                    'coin_id': coin_id,
                    'current_price': float(current_price_usd),
                    'pell_score': pell_score,
                    'complete_cycles': pell_cycle_info.get('total_complete_cycles', 0),
                    'most_recent_cycle_duration': pell_cycle_info.get('most_recent_cycle', {}).get('duration', 0),
                    'most_recent_cycle_pell': pell_cycle_info.get('most_recent_cycle', {}).get('pell_number', 0),
                    'cycle_return': pell_cycle_info.get('most_recent_cycle', {}).get('return', 0),
                    'pell_prediction': float(pell_prediction),
                    'predicted_change_pct': float((pell_prediction - current_price) / current_price * 100),
                    'has_complete_cycle': pell_cycle_info.get('has_complete_cycle', False),
                    'data_points': len(price_data)
                }
                
                results.append(result)
                
                status = "✓" if pell_cycle_info.get('has_complete_cycle') else "⚠️"
                cycles = pell_cycle_info.get('total_complete_cycles', 0)
                print(f"{status} {cycles} cycles, Score: {pell_score:.2f}")
                
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}")
                continue
        
        return results
    
    def _calculate_pell_score(self, pell_cycle_info, pell_cycles, price_data):
        """
        Calculate comprehensive Pell cycle strength score
        
        Factors:
        - Number of complete cycles
        - Cycle completeness ratio
        - Cycle duration (longer = better)
        - Pattern consistency
        - Golden ratio convergence
        """
        score = 0.0
        
        # Base score for having complete cycles
        if pell_cycle_info.get('has_complete_cycle'):
            score += 30.0
        
        # Number of complete cycles (more = better, capped at 40 points)
        complete_cycles = pell_cycle_info.get('total_complete_cycles', 0)
        score += min(40.0, complete_cycles * 0.5)
        
        # Most recent cycle quality
        most_recent = pell_cycle_info.get('most_recent_cycle')
        if most_recent:
            # Longer cycles are more significant
            duration = most_recent.get('duration', 0)
            score += min(15.0, duration * 0.2)
            
            # Pell number significance (higher Pell numbers = more significant)
            pell_num = most_recent.get('pell_number', 0)
            if pell_num >= 70:
                score += 10.0
            elif pell_num >= 29:
                score += 7.0
            elif pell_num >= 12:
                score += 5.0
            elif pell_num >= 5:
                score += 3.0
            
            # Cycle return consistency (closer to golden ratio = better)
            cycle_return = abs(most_recent.get('return', 0))
            phi_ratio = 0.618  # Golden ratio - 1
            if abs(cycle_return - phi_ratio) < 0.1:
                score += 10.0
            elif abs(cycle_return - phi_ratio) < 0.2:
                score += 5.0
        
        # Data quality (more data points = better)
        data_points = len(price_data)
        score += min(5.0, data_points / 100)
        
        return score
    
    def rank_by_pell_cycles(self, results):
        """Rank coins by Pell cycle score"""
        # Sort by Pell score (descending)
        ranked = sorted(results, key=lambda x: x['pell_score'], reverse=True)
        return ranked
    
    def display_top10_ranking(self, ranked_results):
        """Display formatted top 10 ranking"""
        print("\n" + "="*70)
        print("🏆 TOP 10 RANKING BY PELL CYCLE STRENGTH")
        print("="*70)
        
        print(f"\n{'Rank':<6} {'Coin':<15} {'Price':<12} {'Score':<8} {'Cycles':<8} {'Cycle':<8} {'Prediction':<12} {'Change':<10}")
        print("-" * 70)
        
        for i, result in enumerate(ranked_results[:10], 1):
            coin = result['coin_id'].upper()
            price = f"${result['current_price']:,.2f}"
            score = f"{result['pell_score']:.1f}"
            cycles = result['complete_cycles']
            cycle_dur = result['most_recent_cycle_pell']
            pred = f"${result['pell_prediction']:,.2f}"
            change = f"{result['predicted_change_pct']:+.2f}%"
            
            # Medal emoji for top 3
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            
            print(f"{medal} {i:<4} {coin:<15} {price:<12} {score:<8} {cycles:<8} {cycle_dur:<8} {pred:<12} {change:<10}")
        
        print("\n" + "="*70)
    
    def detailed_analysis(self, ranked_results):
        """Show detailed analysis for top 3"""
        print("\n" + "="*70)
        print("📊 DETAILED ANALYSIS: TOP 3 COINS")
        print("="*70)
        
        for i, result in enumerate(ranked_results[:3], 1):
            print(f"\n{'='*70}")
            print(f"#{i} - {result['coin_id'].upper()}")
            print(f"{'='*70}")
            
            print(f"\n💰 Current Price: ${result['current_price']:,.2f}")
            print(f"📈 Pell Cycle Score: {result['pell_score']:.2f}")
            
            print(f"\n🔢 Pell Cycle Metrics:")
            print(f"  Complete Cycles Found: {result['complete_cycles']}")
            print(f"  Most Recent Cycle: {result['most_recent_cycle_pell']} periods")
            print(f"  Cycle Duration: {result['most_recent_cycle_duration']} data points")
            print(f"  Cycle Return: {result['cycle_return']*100:+.2f}%")
            print(f"  Has Complete Cycle: {'✓ Yes' if result['has_complete_cycle'] else '✗ No'}")
            
            print(f"\n🔮 Pell-Based Prediction:")
            print(f"  Predicted Price: ${result['pell_prediction']:,.2f}")
            print(f"  Expected Change: {result['predicted_change_pct']:+.2f}%")
            
            # Signal
            change = result['predicted_change_pct']
            if change > 5:
                signal = "🟢 STRONG BUY"
            elif change > 2:
                signal = "🟡 BUY"
            elif change > -2:
                signal = "⚪ HOLD"
            elif change > -5:
                signal = "🟠 SELL"
            else:
                signal = "🔴 STRONG SELL"
            
            print(f"  Signal: {signal}")
    
    def save_results(self, ranked_results, filename='top10_pell_analysis.json'):
        """Save results to JSON file"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'total_coins_analyzed': len(ranked_results),
            'ranking': ranked_results
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("🚀 TOP 10 CRYPTOCURRENCY PELL CYCLE ANALYSIS")
    print("="*70)
    print("\nThis analysis ranks cryptocurrencies by:")
    print("  • Pell cycle completeness")
    print("  • Number of complete cycles detected")
    print("  • Cycle pattern strength")
    print("  • Golden ratio convergence")
    print("  • Prediction confidence")
    
    analyzer = Top10PellCycleAnalyzer()
    
    try:
        # Analyze all coins
        results = analyzer.analyze_all_coins(days=30)
        
        if not results:
            print("\n❌ No results to analyze")
            return 1
        
        # Rank by Pell cycles
        ranked = analyzer.rank_by_pell_cycles(results)
        
        # Display ranking
        analyzer.display_top10_ranking(ranked)
        
        # Detailed analysis
        analyzer.detailed_analysis(ranked)
        
        # Save results
        analyzer.save_results(ranked)
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nTop 3 by Pell Cycle Strength:")
        for i, result in enumerate(ranked[:3], 1):
            print(f"  {i}. {result['coin_id'].upper()} - Score: {result['pell_score']:.1f}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())

