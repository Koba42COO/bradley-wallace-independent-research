"""
Backtesting Framework for Futures Markets
Tests Pell cycle-based trading strategies on historical data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

from crypto_analyzer_complete import AdvancedCryptoAnalyzer, PellCycleAnalyzer
from test_crypto_analyzer import UPGConstants


@dataclass
class Trade:
    """Represents a single trade"""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_type: str  # 'long' or 'short'
    pell_cycle: int
    confidence: float
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None


@dataclass
class BacktestResult:
    """Backtesting results"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    total_pnl_pct: float
    win_rate: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float
    trades: List[Trade]
    equity_curve: pd.Series


class FuturesBacktester:
    """
    Backtesting framework for futures markets using Pell cycle analysis
    """
    
    def __init__(self, initial_capital: float = 10000.0, leverage: float = 1.0):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.constants = UPGConstants()
        self.analyzer = AdvancedCryptoAnalyzer()
        self.pell_analyzer = PellCycleAnalyzer(self.constants)
        
        # Trading parameters
        self.min_confidence = 0.70
        self.risk_per_trade = 0.02  # 2% risk per trade
        
    def backtest(self, price_data: pd.Series, 
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None) -> BacktestResult:
        """
        Run backtest on historical price data
        
        Strategy:
        1. Enter at Pell cycle completion points
        2. Exit at next cycle completion or stop-loss/take-profit
        3. Position sizing based on confidence
        """
        # Filter data by date range
        if start_date:
            price_data = price_data[price_data.index >= start_date]
        if end_date:
            price_data = price_data[price_data.index <= end_date]
        
        if len(price_data) < 100:
            raise ValueError("Insufficient data for backtesting")
        
        trades = []
        equity = [self.initial_capital]
        current_position = None
        max_equity = self.initial_capital
        max_drawdown = 0.0
        
        # Analyze data in rolling windows
        window_size = 500
        step_size = 50
        
        for i in range(window_size, len(price_data), step_size):
            window_data = price_data.iloc[i-window_size:i]
            current_price = price_data.iloc[i]
            current_time = price_data.index[i]
            
            # Detect Pell cycles
            cycles = self.pell_analyzer.detect_pell_cycles(window_data)
            complete_cycles = [c for c in cycles.get('cycles', []) if c.get('is_complete')]
            
            if not complete_cycles:
                continue
            
            # Get most recent cycle
            most_recent = max(complete_cycles, key=lambda c: c['end_index'])
            
            # Determine position in cycle
            position_info = self.pell_analyzer.determine_current_pell_position(window_data)
            
            # Check for entry signal (cycle completion)
            if position_info['current_position'] == 'cycle_completion':
                if current_position is None:  # No open position
                    # Calculate confidence
                    analysis = self.analyzer.analyze_coin_data(window_data, horizon=24)
                    confidence = analysis.get('consensus', {}).get('confidence', 0.5)
                    
                    if confidence >= self.min_confidence:
                        # Determine direction based on predicted return
                        predicted_return = analysis.get('consensus', {}).get('expected_change_pct', 0)
                        
                        if abs(predicted_return) > 1.0:  # Minimum 1% expected move
                            position_type = 'long' if predicted_return > 0 else 'short'
                            
                            # Calculate position size
                            position_size = self._calculate_position_size(
                                confidence, 
                                abs(predicted_return),
                                current_price
                            )
                            
                            # Create trade
                            trade = Trade(
                                entry_time=current_time,
                                exit_time=None,
                                entry_price=current_price,
                                exit_price=None,
                                position_type=position_type,
                                pell_cycle=most_recent.get('pell_number', 0),
                                confidence=confidence
                            )
                            
                            current_position = trade
                            trades.append(trade)
            
            # Check for exit signal
            elif current_position and position_info['current_position'] == 'cycle_completion':
                # Exit at next cycle completion
                current_position.exit_time = current_time
                current_position.exit_price = current_price
                
                # Calculate P&L
                if current_position.position_type == 'long':
                    pnl = (current_price - current_position.entry_price) * self.leverage
                else:
                    pnl = (current_position.entry_price - current_price) * self.leverage
                
                pnl_pct = (pnl / current_position.entry_price) * 100
                
                current_position.pnl = pnl
                current_position.pnl_pct = pnl_pct
                
                # Update equity
                new_equity = equity[-1] + pnl
                equity.append(new_equity)
                
                if new_equity > max_equity:
                    max_equity = new_equity
                
                drawdown = (max_equity - new_equity) / max_equity
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                
                current_position = None
            
            # Check stop-loss / take-profit for open positions
            elif current_position:
                if current_position.position_type == 'long':
                    price_change = (current_price - current_position.entry_price) / current_position.entry_price
                else:
                    price_change = (current_position.entry_price - current_price) / current_position.entry_price
                
                # Stop-loss: -5%
                if price_change < -0.05:
                    current_position.exit_time = current_time
                    current_position.exit_price = current_price
                    pnl = price_change * current_position.entry_price * self.leverage
                    current_position.pnl = pnl
                    current_position.pnl_pct = price_change * 100
                    equity.append(equity[-1] + pnl)
                    current_position = None
                
                # Take-profit: +10%
                elif price_change > 0.10:
                    current_position.exit_time = current_time
                    current_position.exit_price = current_price
                    pnl = price_change * current_position.entry_price * self.leverage
                    current_position.pnl = pnl
                    current_position.pnl_pct = price_change * 100
                    equity.append(equity[-1] + pnl)
                    current_position = None
        
        # Close any remaining positions
        if current_position:
            final_price = price_data.iloc[-1]
            current_position.exit_time = price_data.index[-1]
            current_position.exit_price = final_price
            
            if current_position.position_type == 'long':
                pnl = (final_price - current_position.entry_price) * self.leverage
            else:
                pnl = (current_position.entry_price - final_price) * self.leverage
            
            current_position.pnl = pnl
            current_position.pnl_pct = (pnl / current_position.entry_price) * 100
            equity.append(equity[-1] + pnl)
        
        # Calculate metrics
        completed_trades = [t for t in trades if t.exit_time is not None]
        winning_trades = [t for t in completed_trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in completed_trades if t.pnl and t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in completed_trades if t.pnl)
        total_pnl_pct = (equity[-1] - self.initial_capital) / self.initial_capital * 100
        
        win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Sharpe ratio
        returns = pd.Series(equity).pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0
        
        equity_curve = pd.Series(equity, index=price_data.index[:len(equity)])
        
        return BacktestResult(
            total_trades=len(completed_trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            trades=completed_trades,
            equity_curve=equity_curve
        )
    
    def _calculate_position_size(self, confidence: float, expected_return: float, 
                                 current_price: float) -> float:
        """Calculate position size based on confidence and risk"""
        # Base position size
        base_size = self.initial_capital * self.risk_per_trade
        
        # Adjust for confidence
        confidence_multiplier = confidence
        
        # Adjust for expected return
        return_multiplier = min(2.0, abs(expected_return) / 5.0)  # Cap at 2x
        
        position_size = base_size * confidence_multiplier * return_multiplier
        
        # Convert to number of units
        units = position_size / current_price
        
        return units
    
    def generate_report(self, result: BacktestResult) -> str:
        """Generate human-readable backtest report"""
        report = []
        report.append("=" * 70)
        report.append("📊 FUTURES BACKTESTING RESULTS")
        report.append("=" * 70)
        report.append(f"\n💰 Initial Capital: ${self.initial_capital:,.2f}")
        report.append(f"📈 Final Equity: ${self.initial_capital + result.total_pnl:,.2f}")
        report.append(f"💵 Total P&L: ${result.total_pnl:,.2f} ({result.total_pnl_pct:.2f}%)")
        report.append(f"\n📊 Trade Statistics:")
        report.append(f"   Total Trades: {result.total_trades}")
        report.append(f"   Winning Trades: {result.winning_trades}")
        report.append(f"   Losing Trades: {result.losing_trades}")
        report.append(f"   Win Rate: {result.win_rate*100:.2f}%")
        report.append(f"\n💹 Performance Metrics:")
        report.append(f"   Average Win: ${result.avg_win:,.2f}")
        report.append(f"   Average Loss: ${result.avg_loss:,.2f}")
        report.append(f"   Max Drawdown: {result.max_drawdown*100:.2f}%")
        report.append(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
        report.append("=" * 70)
        
        return "\n".join(report)


def main():
    """Example backtesting"""
    print("🚀 Starting Futures Backtesting")
    print("=" * 70)
    
    # Load or generate sample data
    # In practice, load from exchange API or CSV
    dates = pd.date_range(start='2024-01-01', end='2024-11-01', freq='1H')
    # Simulate price data (replace with real data)
    np.random.seed(42)
    prices = 10000 + np.cumsum(np.random.randn(len(dates)) * 100)
    price_data = pd.Series(prices, index=dates)
    
    # Create backtester
    backtester = FuturesBacktester(initial_capital=10000.0, leverage=1.0)
    
    # Run backtest
    try:
        result = backtester.backtest(price_data)
        
        # Print report
        print(backtester.generate_report(result))
        
        # Save results
        with open('backtest_results.json', 'w') as f:
            json.dump({
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'total_pnl': result.total_pnl,
                'total_pnl_pct': result.total_pnl_pct,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown
            }, f, indent=2)
        
        print("\n✅ Backtest complete! Results saved to backtest_results.json")
        
    except Exception as e:
        print(f"❌ Backtest error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

