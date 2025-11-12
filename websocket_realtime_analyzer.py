"""
Real-Time WebSocket Data Integration for Crypto Market Analyzer
Supports multiple exchanges and real-time analysis
"""

import asyncio
import json
import websockets
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Callable
from collections import deque
import time

from crypto_analyzer_complete import AdvancedCryptoAnalyzer
from twenty_one_model_ensemble import TwentyOneModelEnsemble
from test_crypto_analyzer import UPGConstants


class WebSocketRealtimeAnalyzer:
    """
    Real-time cryptocurrency market analyzer using WebSocket streams
    """
    
    def __init__(self, coin_id: str = 'bitcoin', exchange: str = 'binance'):
        self.coin_id = coin_id
        self.exchange = exchange
        self.constants = UPGConstants()
        self.analyzer = AdvancedCryptoAnalyzer()
        self.ensemble = TwentyOneModelEnsemble(self.constants)
        
        # Data buffers
        self.price_buffer = deque(maxlen=1000)  # Store last 1000 prices
        self.time_buffer = deque(maxlen=1000)
        self.volume_buffer = deque(maxlen=1000)
        
        # Analysis results
        self.latest_analysis = None
        self.callbacks = []
        
        # WebSocket connection
        self.ws = None
        self.running = False
        
    def add_callback(self, callback: Callable):
        """Add callback function to be called on new analysis"""
        self.callbacks.append(callback)
    
    def _notify_callbacks(self, analysis_result: Dict):
        """Notify all registered callbacks"""
        for callback in self.callbacks:
            try:
                callback(analysis_result)
            except Exception as e:
                print(f"Callback error: {e}")
    
    def _get_binance_ws_url(self, symbol: str = 'BTCUSDT') -> str:
        """Get Binance WebSocket URL"""
        return f"wss://stream.binance.com:9443/ws/{symbol.lower()}@ticker"
    
    def _get_coingecko_ws_url(self) -> str:
        """Get CoinGecko WebSocket URL (if available)"""
        # CoinGecko doesn't have public WebSocket, use REST API polling instead
        return None
    
    def _parse_binance_message(self, message: str) -> Optional[Dict]:
        """Parse Binance WebSocket message"""
        try:
            data = json.loads(message)
            return {
                'price': float(data.get('c', 0)),  # Last price
                'volume': float(data.get('v', 0)),  # 24h volume
                'timestamp': datetime.now(),
                'exchange': 'binance'
            }
        except Exception as e:
            print(f"Parse error: {e}")
            return None
    
    async def _connect_binance(self, symbol: str = 'BTCUSDT'):
        """Connect to Binance WebSocket"""
        url = self._get_binance_ws_url(symbol)
        
        try:
            async with websockets.connect(url) as websocket:
                self.ws = websocket
                print(f"✅ Connected to Binance WebSocket: {symbol}")
                
                async for message in websocket:
                    if not self.running:
                        break
                    
                    data = self._parse_binance_message(message)
                    if data:
                        await self._process_realtime_data(data)
        except Exception as e:
            print(f"WebSocket connection error: {e}")
            self.running = False
    
    async def _process_realtime_data(self, data: Dict):
        """Process incoming real-time data"""
        # Add to buffers
        self.price_buffer.append(data['price'])
        self.time_buffer.append(data['timestamp'])
        self.volume_buffer.append(data['volume'])
        
        # Perform analysis when we have enough data
        if len(self.price_buffer) >= 50:
            await self._perform_realtime_analysis()
    
    async def _perform_realtime_analysis(self):
        """Perform real-time analysis on buffered data"""
        try:
            # Convert buffers to pandas Series
            price_series = pd.Series(list(self.price_buffer), index=list(self.time_buffer))
            
            if len(price_series) < 50:
                return
            
            # Run analysis
            analysis = self.analyzer.analyze_coin_data(
                price_series,
                horizon=24
            )
            
            # Add ensemble analysis
            try:
                branch_df, ensemble_result = self.ensemble.analyze_timeline_branches(price_series)
                analysis['ensemble_21_models'] = {
                    'consensus_prediction': ensemble_result['consensus_prediction'],
                    'confidence': ensemble_result['confidence'],
                    'coherence': ensemble_result['coherence'],
                    'most_confident_branch': ensemble_result['most_confident_branch']
                }
                analysis['branch_analysis'] = branch_df.to_dict('records')
            except Exception as e:
                print(f"Ensemble analysis error: {e}")
            
            # Add real-time metadata
            analysis['realtime'] = {
                'timestamp': datetime.now().isoformat(),
                'data_points': len(price_series),
                'current_price': float(price_series.iloc[-1]),
                'exchange': self.exchange
            }
            
            self.latest_analysis = analysis
            
            # Notify callbacks
            self._notify_callbacks(analysis)
            
        except Exception as e:
            print(f"Analysis error: {e}")
    
    async def start(self, symbol: str = 'BTCUSDT'):
        """Start real-time analysis"""
        self.running = True
        
        if self.exchange == 'binance':
            await self._connect_binance(symbol)
        else:
            print(f"Exchange {self.exchange} not yet supported")
            self.running = False
    
    def stop(self):
        """Stop real-time analysis"""
        self.running = False
        if self.ws:
            # Close connection
            pass
    
    def get_latest_analysis(self) -> Optional[Dict]:
        """Get latest analysis results"""
        return self.latest_analysis
    
    def get_price_history(self, limit: int = 100) -> pd.Series:
        """Get recent price history"""
        if len(self.price_buffer) == 0:
            return pd.Series()
        
        prices = list(self.price_buffer)[-limit:]
        times = list(self.time_buffer)[-limit:]
        return pd.Series(prices, index=times)


class RealtimeAnalysisPrinter:
    """Callback class to print analysis results"""
    
    def __init__(self):
        self.update_count = 0
    
    def __call__(self, analysis: Dict):
        self.update_count += 1
        print(f"\n{'='*70}")
        print(f"📊 REAL-TIME ANALYSIS UPDATE #{self.update_count}")
        print(f"{'='*70}")
        
        if 'realtime' in analysis:
            rt = analysis['realtime']
            print(f"⏰ Time: {rt['timestamp']}")
            print(f"💰 Current Price: ${rt['current_price']:,.2f}")
            print(f"📈 Data Points: {rt['data_points']}")
        
        if 'tri_gemini' in analysis:
            tg = analysis['tri_gemini']
            print(f"\n🔮 Tri-Gemini Prediction: ${tg.get('forward_prediction', 0):,.2f}")
            print(f"   Confidence: {tg.get('confidence', 0)*100:.1f}%")
            print(f"   Coherence: {tg.get('coherence_score', 0):.4f}")
        
        if 'pell_cycle' in analysis:
            pc = analysis['pell_cycle']
            if pc.get('has_complete_cycle'):
                cycle = pc['most_recent_cycle']
                print(f"\n🔢 Pell Cycle: #{cycle.get('pell_number', 0)} periods")
                print(f"   Return: {cycle.get('return', 0)*100:.2f}%")
        
        if 'ensemble_21_models' in analysis:
            ens = analysis['ensemble_21_models']
            print(f"\n🎯 21-Model Ensemble:")
            print(f"   Consensus: ${ens.get('consensus_prediction', 0):,.2f}")
            print(f"   Confidence: {ens.get('confidence', 0)*100:.1f}%")
            print(f"   Coherence: {ens.get('coherence', 0):.4f}")
            print(f"   Best Branch: #{ens.get('most_confident_branch', 0)}")
        
        if 'consensus' in analysis:
            cons = analysis['consensus']
            print(f"\n✅ Final Consensus:")
            print(f"   Predicted Price: ${cons.get('predicted_price', 0):,.2f}")
            print(f"   Expected Change: {cons.get('expected_change_pct', 0):.2f}%")
            print(f"   Confidence: {cons.get('confidence', 0)*100:.1f}%")
            signal = cons.get('signal', 'HOLD')
            print(f"   Signal: {signal}")
        
        print(f"{'='*70}\n")


async def main():
    """Example usage"""
    print("🚀 Starting Real-Time Crypto Market Analyzer")
    print("=" * 70)
    
    # Create analyzer
    analyzer = WebSocketRealtimeAnalyzer(coin_id='bitcoin', exchange='binance')
    
    # Add callback
    printer = RealtimeAnalysisPrinter()
    analyzer.add_callback(printer)
    
    # Start analysis
    try:
        await analyzer.start(symbol='BTCUSDT')
    except KeyboardInterrupt:
        print("\n⏹️  Stopping analyzer...")
        analyzer.stop()


if __name__ == "__main__":
    # Run with: python -m asyncio websocket_realtime_analyzer.py
    asyncio.run(main())

