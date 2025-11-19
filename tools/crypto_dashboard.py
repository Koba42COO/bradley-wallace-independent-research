"""
Interactive Dashboard for Crypto Market Analyzer
Uses matplotlib for visualization
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from crypto_analyzer_complete import AdvancedCryptoAnalyzer

class CryptoDashboard:
    """Create visualizations for crypto market analysis"""
    
    def __init__(self):
        self.analyzer = AdvancedCryptoAnalyzer()
        plt.style.use('dark_background')
    
    def create_full_dashboard(self, coin_id='bitcoin', days=30):
        """Create a complete dashboard with all visualizations"""
        # Run analysis
        result = self.analyzer.analyze_coin(coin_id, days=days)
        
        if not result:
            print("❌ No data to visualize")
            return None
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Price chart with predictions
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_price_predictions(ax1, result)
        
        # 2. Tri-Gemini analysis
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_tri_gemini(ax2, result)
        
        # 3. Prime patterns
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_prime_patterns(ax3, result)
        
        # 4. Confidence metrics
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_confidence(ax4, result)
        
        # 5. Prediction comparison
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_prediction_comparison(ax5, result)
        
        # Title
        fig.suptitle(f'{coin_id.upper()} Market Analysis Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save
        filename = f'{coin_id}_dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n💾 Dashboard saved: {filename}")
        
        return fig
    
    def _plot_price_predictions(self, ax, result):
        """Plot historical prices and predictions"""
        price_data = result['price_data']
        current = result['current_price']
        tri = result['tri_gemini']
        prime = result['prime_patterns']
        consensus = result['consensus']
        
        # Historical prices
        ax.plot(price_data.index, price_data.values, 
               label='Historical Price', color='cyan', linewidth=2, alpha=0.8)
        
        # Current price marker
        last_time = price_data.index[-1]
        ax.scatter([last_time], [current], color='yellow', 
                  s=100, zorder=5, label='Current Price')
        
        # Predictions
        future_time = last_time + timedelta(hours=24)
        
        ax.scatter([future_time], [tri['forward_prediction']], 
                  color='green', s=80, marker='^', 
                  label=f"Tri-Gemini: ${tri['forward_prediction']:,.0f}", zorder=5)
        
        ax.scatter([future_time], [prime['prediction']], 
                  color='orange', s=80, marker='^', 
                  label=f"Prime Pattern: ${prime['prediction']:,.0f}", zorder=5)
        
        ax.scatter([future_time], [consensus['prediction']], 
                  color='red', s=120, marker='*', 
                  label=f"Consensus: ${consensus['prediction']:,.0f}", zorder=5)
        
        # Prediction lines
        ax.plot([last_time, future_time], [current, tri['forward_prediction']], 
               'g--', alpha=0.5, linewidth=1)
        ax.plot([last_time, future_time], [current, prime['prediction']], 
               'orange', linestyle='--', alpha=0.5, linewidth=1)
        ax.plot([last_time, future_time], [current, consensus['prediction']], 
               'r--', alpha=0.7, linewidth=2)
        
        ax.set_title('Price History & Predictions (24h)', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_tri_gemini(self, ax, result):
        """Plot Tri-Gemini analysis metrics"""
        tri = result['tri_gemini']
        current = result['current_price']
        
        metrics = ['Forward', 'Reverse', 'Current']
        values = [
            tri['forward_prediction'],
            tri['reverse_analysis'],
            current
        ]
        colors = ['green', 'blue', 'yellow']
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='white')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${val:,.0f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_title('Tri-Gemini Analysis', fontweight='bold')
        ax.set_ylabel('Price (USD)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add coherence score
        ax.text(0.5, 0.95, f"Coherence: {tri['coherence_score']:.3f}",
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.3),
               fontsize=10, fontweight='bold')
    
    def _plot_prime_patterns(self, ax, result):
        """Plot prime pattern detection results"""
        prime = result['prime_patterns']
        
        categories = ['Intervals', 'Ratios', 'Cycles', 'Fibonacci']
        counts = [
            prime['intervals'],
            prime['ratios'],
            len(prime['cycles']),
            len(prime['fibonacci'])
        ]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
        bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='white')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_title('Prime Pattern Detection', fontweight='bold')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Show detected cycles
        if prime['cycles']:
            cycles_str = ', '.join(map(str, prime['cycles'][:5]))
            ax.text(0.5, 0.95, f"Cycles: {cycles_str}",
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3),
                   fontsize=8)
    
    def _plot_confidence(self, ax, result):
        """Plot confidence metrics"""
        tri = result['tri_gemini']
        consensus = result['consensus']
        
        metrics = ['Tri-Gemini\nConfidence', 'Consensus\nConfidence']
        values = [tri['confidence'], consensus['confidence']]
        colors = ['cyan', 'red']
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='white')
        
        # Add percentage labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1%}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_title('Confidence Metrics', fontweight='bold')
        ax.set_ylabel('Confidence')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_prediction_comparison(self, ax, result):
        """Compare different prediction methods"""
        current = result['current_price']
        tri = result['tri_gemini']
        prime = result['prime_patterns']
        consensus = result['consensus']
        
        methods = ['Current', 'Tri-Gemini', 'Prime\nPattern', 'Consensus']
        prices = [current, tri['forward_prediction'], 
                 prime['prediction'], consensus['prediction']]
        
        # Calculate changes
        changes = [(p - current) / current * 100 for p in prices]
        colors = ['yellow', 'green', 'orange', 'red']
        
        bars = ax.bar(methods, changes, color=colors, alpha=0.7, edgecolor='white')
        
        # Add percentage labels
        for bar, change in zip(bars, changes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., 
                   height + (0.1 if height >= 0 else -0.3),
                   f'{change:+.2f}%',
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=10, fontweight='bold')
        
        ax.axhline(y=0, color='white', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_title('Prediction Comparison (% Change)', fontweight='bold')
        ax.set_ylabel('Change (%)')
        ax.grid(True, alpha=0.3, axis='y')

def main():
    """Run the dashboard"""
    print("\n" + "="*70)
    print("📊 CRYPTO MARKET ANALYZER DASHBOARD")
    print("="*70)
    
    dashboard = CryptoDashboard()
    
    try:
        # Create dashboard for Bitcoin
        fig = dashboard.create_full_dashboard('bitcoin', days=30)
        
        if fig:
            print("\n✅ Dashboard created successfully!")
            print("\n💡 Tip: Close the plot window to continue")
            plt.show()
        else:
            print("\n❌ Failed to create dashboard")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

