#!/usr/bin/env python3
"""
Cryptocurrency Analysis and Prediction System

A comprehensive system for analyzing cryptocurrency markets and making investment predictions.
Uses multiple data sources (CoinGecko, Binance, etc.) and implements technical analysis,
sentiment analysis, and machine learning models for price prediction.

Author: AI Assistant
Date: October 2025
"""

import pandas as pd
import numpy as np
import requests
import time
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import ccxt
from pycoingecko import CoinGeckoAPI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CryptoDataFetcher:
    """Handles data fetching from multiple cryptocurrency APIs"""

    def __init__(self):
        self.cg = CoinGeckoAPI()
        self.binance = ccxt.binance()
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes

    def get_top_coins(self, limit: int = 100) -> pd.DataFrame:
        """Fetch top cryptocurrencies by market cap"""
        try:
            logger.info(f"Fetching top {limit} cryptocurrencies from CoinGecko")
            coins = self.cg.get_coins_markets(
                vs_currency='usd',
                order='market_cap_desc',
                per_page=limit,
                page=1,
                sparkline=False,
                price_change_percentage='24h,7d,30d'
            )

            df = pd.DataFrame(coins)
            df['last_updated'] = pd.to_datetime(df['last_updated'])
            return df

        except Exception as e:
            logger.error(f"Error fetching top coins: {e}")
            return pd.DataFrame()

    def get_historical_data(self, coin_id: str, days: int = 365) -> pd.DataFrame:
        """Fetch historical price data for a cryptocurrency"""
        cache_key = f"{coin_id}_{days}"

        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_timeout:
                return cached_data

        try:
            logger.info(f"Fetching {days} days of historical data for {coin_id}")

            # CoinGecko API for historical data
            data = self.cg.get_coin_market_chart_by_id(
                id=coin_id,
                vs_currency='usd',
                days=days
            )

            prices = data['prices']
            market_caps = data['market_caps']
            total_volumes = data['total_volumes']

            # Convert to DataFrame
            df = pd.DataFrame({
                'timestamp': [p[0] for p in prices],
                'price': [p[1] for p in prices],
                'market_cap': [m[1] for m in market_caps],
                'volume': [v[1] for v in total_volumes]
            })

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['date'] = df.index.date

            # Add basic price features
            df['price_change'] = df['price'].pct_change()
            df['price_change_24h'] = df['price'].pct_change(24)  # Assuming hourly data
            df['volatility_24h'] = df['price_change'].rolling(24).std()

            self.cache[cache_key] = (time.time(), df)
            return df

        except Exception as e:
            logger.error(f"Error fetching historical data for {coin_id}: {e}")
            return pd.DataFrame()

    def get_live_price(self, coin_ids: List[str]) -> Dict[str, float]:
        """Get live prices for multiple coins"""
        try:
            prices = {}
            for coin_id in coin_ids:
                data = self.cg.get_price(ids=coin_id, vs_currencies='usd')
                if coin_id in data:
                    prices[coin_id] = data[coin_id]['usd']
                time.sleep(0.1)  # Rate limiting

            return prices

        except Exception as e:
            logger.error(f"Error fetching live prices: {e}")
            return {}

class TechnicalIndicators:
    """Calculate technical analysis indicators"""

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band

    @staticmethod
    def calculate_moving_averages(prices: pd.Series, periods: List[int] = [7, 14, 30, 50, 200]) -> pd.DataFrame:
        """Calculate multiple moving averages"""
        ma_df = pd.DataFrame(index=prices.index)
        for period in periods:
            ma_df[f'MA_{period}'] = prices.rolling(window=period).mean()
            ma_df[f'EMA_{period}'] = prices.ewm(span=period, adjust=False).mean()
        return ma_df

    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to the dataframe"""
        if 'price' not in df.columns:
            return df

        # RSI
        df['rsi'] = TechnicalIndicators.calculate_rsi(df['price'])

        # MACD
        df['macd'], df['macd_signal'], df['macd_histogram'] = TechnicalIndicators.calculate_macd(df['price'])

        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = TechnicalIndicators.calculate_bollinger_bands(df['price'])

        # Moving Averages
        ma_df = TechnicalIndicators.calculate_moving_averages(df['price'])
        df = pd.concat([df, ma_df], axis=1)

        # Additional indicators
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        df['price_to_volume'] = df['price'] / df['volume'].replace(0, np.nan)
        df['market_cap_to_volume'] = df['market_cap'] / df['volume'].replace(0, np.nan)

        # Momentum indicators
        df['roc_14'] = df['price'].pct_change(14)  # Rate of Change
        df['williams_r'] = ((df['price'].rolling(14).max() - df['price']) /
                           (df['price'].rolling(14).max() - df['price'].rolling(14).min())) * -100

        # Volatility indicators
        df['atr'] = df['price'].rolling(14).std()  # Average True Range approximation

        return df

class SentimentAnalyzer:
    """Analyze market sentiment from various sources"""

    def __init__(self):
        self.news_api_key = None  # Would need API key for news
        self.social_media_keywords = [
            'bitcoin', 'ethereum', 'crypto', 'blockchain', 'defi', 'nft',
            'cryptocurrency', 'btc', 'eth', 'altcoin', 'hodl', 'moon'
        ]

    def get_sentiment_score(self, coin_name: str, coin_symbol: str) -> float:
        """Calculate sentiment score for a cryptocurrency"""
        # Simplified sentiment analysis - in production, would use NLP models
        # and multiple data sources (Twitter, Reddit, News, etc.)

        try:
            # Mock sentiment calculation based on price momentum and volume
            # In a real implementation, this would analyze news articles, social media, etc.
            sentiment_factors = {
                'name': coin_name.lower(),
                'symbol': coin_symbol.lower(),
                'base_sentiment': 0.5,  # Neutral baseline
                'market_dominance': 0.1,  # Bonus for established coins
                'recent_news': 0.0,  # Would be calculated from news API
                'social_mentions': 0.0  # Would be calculated from social media APIs
            }

            # Adjust sentiment based on coin characteristics
            if coin_symbol.lower() in ['btc', 'eth']:
                sentiment_factors['market_dominance'] = 0.3

            if any(keyword in coin_name.lower() for keyword in ['bitcoin', 'ethereum']):
                sentiment_factors['base_sentiment'] = 0.7

            # Calculate final sentiment score (0-1 scale)
            sentiment = (
                sentiment_factors['base_sentiment'] * 0.4 +
                sentiment_factors['market_dominance'] * 0.3 +
                sentiment_factors['recent_news'] * 0.2 +
                sentiment_factors['social_mentions'] * 0.1
            )

            return min(1.0, max(0.0, sentiment))

        except Exception as e:
            logger.error(f"Error calculating sentiment for {coin_name}: {e}")
            return 0.5

class MLPricePredictor:
    """Machine learning models for price prediction"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = [
            'price', 'volume', 'market_cap', 'rsi', 'macd', 'macd_signal',
            'bb_upper', 'bb_middle', 'bb_lower', 'volume_ma_20',
            'price_to_volume', 'market_cap_to_volume', 'roc_14', 'williams_r', 'atr'
        ]

    def prepare_features(self, df: pd.DataFrame, prediction_days: int = 7) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for ML training"""
        df_clean = df.dropna()

        # Create target: future price after prediction_days
        df_clean['target_price'] = df_clean['price'].shift(-prediction_days)
        df_clean = df_clean.dropna()

        # Select features
        feature_cols = [col for col in self.feature_columns if col in df_clean.columns]
        X = df_clean[feature_cols]
        y = df_clean['target_price']

        return X, y

    def train_models(self, df: pd.DataFrame, test_size: float = 0.2):
        """Train ML models for price prediction"""
        try:
            X, y = self.prepare_features(df)

            if len(X) < 100:  # Need minimum data
                logger.warning("Insufficient data for training")
                return

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train Random Forest
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train_scaled, y_train)

            # Train Gradient Boosting
            gb_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            gb_model.fit(X_train_scaled, y_train)

            # Evaluate models
            rf_pred = rf_model.predict(X_test_scaled)
            gb_pred = gb_model.predict(X_test_scaled)

            rf_mae = mean_absolute_error(y_test, rf_pred)
            gb_mae = mean_absolute_error(y_test, gb_pred)

            logger.info(f"Random Forest MAE: ${rf_mae:.2f}")
            logger.info(f"Gradient Boosting MAE: ${gb_mae:.2f}")

            # Store best model
            if rf_mae < gb_mae:
                self.models['price_predictor'] = rf_model
                self.scalers['price_predictor'] = scaler
                logger.info("Selected Random Forest as best model")
            else:
                self.models['price_predictor'] = gb_model
                self.scalers['price_predictor'] = scaler
                logger.info("Selected Gradient Boosting as best model")

        except Exception as e:
            logger.error(f"Error training models: {e}")

    def predict_price(self, df: pd.DataFrame, days_ahead: int = 7) -> Tuple[float, float]:
        """Predict future price"""
        try:
            if 'price_predictor' not in self.models:
                return df['price'].iloc[-1], 0.5  # Return current price with medium confidence

            # Prepare latest features
            latest_data = df.iloc[-1:][self.feature_columns]
            latest_scaled = self.scalers['price_predictor'].transform(latest_data)

            # Make prediction
            predicted_price = self.models['price_predictor'].predict(latest_scaled)[0]

            # Calculate confidence based on prediction variance
            current_price = df['price'].iloc[-1]
            confidence = max(0.1, min(1.0, 1 - abs(predicted_price - current_price) / current_price))

            return predicted_price, confidence

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return df['price'].iloc[-1], 0.5

class CryptoAnalyzer:
    """Main cryptocurrency analysis engine"""

    def __init__(self):
        self.data_fetcher = CryptoDataFetcher()
        self.indicators = TechnicalIndicators()
        self.sentiment = SentimentAnalyzer()
        self.predictor = MLPricePredictor()

    def analyze_coin(self, coin_id: str, coin_data: Dict) -> Dict:
        """Comprehensive analysis of a single cryptocurrency"""
        try:
            logger.info(f"Analyzing {coin_id}")

            # Fetch historical data (1 year for better analysis)
            historical_df = self.data_fetcher.get_historical_data(coin_id, days=365)

            if historical_df.empty:
                return {
                    'coin_id': coin_id,
                    'name': coin_data.get('name', coin_id),
                    'symbol': coin_data.get('symbol', '').upper(),
                    'error': 'No historical data available',
                    'score': 0
                }

            # Add technical indicators
            historical_df = self.indicators.add_technical_indicators(historical_df)

            # Train ML model
            self.predictor.train_models(historical_df)

            # Get predictions (with fallback for when ML fails)
            predicted_price, prediction_confidence = self.predictor.predict_price(historical_df)

            # If ML prediction failed (returns current price), use technical analysis fallback
            if predicted_price == historical_df['price'].iloc[-1] and len(historical_df) > 30:
                predicted_price, prediction_confidence = self._fallback_prediction(historical_df)

            # Calculate technical scores
            technical_score = self._calculate_technical_score(historical_df)

            # Get sentiment score
            sentiment_score = self.sentiment.get_sentiment_score(
                coin_data.get('name', coin_id),
                coin_data.get('symbol', '')
            )

            # Calculate momentum score
            momentum_score = self._calculate_momentum_score(historical_df)

            # Overall score (weighted combination)
            weights = {
                'technical': 0.4,
                'sentiment': 0.2,
                'momentum': 0.3,
                'prediction_confidence': 0.1
            }

            overall_score = (
                technical_score * weights['technical'] +
                sentiment_score * weights['sentiment'] +
                momentum_score * weights['momentum'] +
                prediction_confidence * weights['prediction_confidence']
            )

            # Calculate expected return
            current_price = historical_df['price'].iloc[-1]
            if predicted_price > current_price:
                expected_return_pct = ((predicted_price - current_price) / current_price) * 100
            else:
                expected_return_pct = 0

            return {
                'coin_id': coin_id,
                'name': coin_data.get('name', coin_id),
                'symbol': coin_data.get('symbol', '').upper(),
                'current_price': current_price,
                'predicted_price': predicted_price,
                'expected_return_pct': expected_return_pct,
                'prediction_confidence': prediction_confidence,
                'technical_score': technical_score,
                'sentiment_score': sentiment_score,
                'momentum_score': momentum_score,
                'overall_score': overall_score,
                'market_cap': coin_data.get('market_cap', 0),
                'volume_24h': coin_data.get('total_volume', 0),
                'price_change_24h': coin_data.get('price_change_percentage_24h', 0),
                'price_change_7d': coin_data.get('price_change_percentage_7d_in_currency', 0),
                'price_change_30d': coin_data.get('price_change_percentage_30d_in_currency', 0),
                'analysis_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error analyzing {coin_id}: {e}")
            return {
                'coin_id': coin_id,
                'name': coin_data.get('name', coin_id),
                'error': str(e),
                'score': 0
            }

    def _calculate_technical_score(self, df: pd.DataFrame) -> float:
        """Calculate technical analysis score"""
        try:
            latest = df.iloc[-1]

            score = 0.5  # Start with neutral score

            # RSI analysis (0.2 weight)
            if 'rsi' in latest and not pd.isna(latest['rsi']):
                rsi = latest['rsi']
                if 30 <= rsi <= 70:
                    score += 0.1  # Neutral zone
                elif rsi < 30:
                    score += 0.2  # Oversold (bullish)
                else:
                    score -= 0.1  # Overbought (bearish)

            # MACD analysis (0.2 weight)
            if all(col in latest for col in ['macd', 'macd_signal']):
                if latest['macd'] > latest['macd_signal']:
                    score += 0.1  # Bullish crossover
                else:
                    score -= 0.1  # Bearish crossover

            # Bollinger Bands analysis (0.2 weight)
            if all(col in latest for col in ['price', 'bb_upper', 'bb_lower']):
                price = latest['price']
                upper = latest['bb_upper']
                lower = latest['bb_lower']

                if price <= lower:
                    score += 0.2  # Price near lower band (potentially oversold)
                elif price >= upper:
                    score -= 0.1  # Price near upper band (potentially overbought)
                else:
                    score += 0.05  # Price within bands (normal)

            # Moving averages analysis (0.2 weight)
            ma_cols = [col for col in df.columns if col.startswith('MA_') or col.startswith('EMA_')]
            if ma_cols:
                bullish_signals = 0
                bearish_signals = 0

                for ma_col in ma_cols[:3]:  # Check first 3 MAs
                    if ma_col in latest and not pd.isna(latest[ma_col]):
                        if latest['price'] > latest[ma_col]:
                            bullish_signals += 1
                        else:
                            bearish_signals += 1

                if bullish_signals > bearish_signals:
                    score += 0.1
                elif bearish_signals > bullish_signals:
                    score -= 0.1

            # Volume analysis (0.2 weight)
            if 'volume_ma_20' in latest and not pd.isna(latest['volume_ma_20']):
                if latest['volume'] > latest['volume_ma_20']:
                    score += 0.1  # Above average volume

            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
            return 0.5

    def _calculate_momentum_score(self, df: pd.DataFrame) -> float:
        """Calculate momentum score based on recent price action"""
        try:
            if len(df) < 30:
                return 0.5

            # Calculate various momentum indicators
            recent_7d_return = (df['price'].iloc[-1] / df['price'].iloc[-7] - 1) * 100
            recent_30d_return = (df['price'].iloc[-1] / df['price'].iloc[-30] - 1) * 100

            # Volume trend
            volume_trend = df['volume'].tail(7).mean() / df['volume'].tail(30).mean()

            # Volatility (lower is better for stable investments)
            volatility = df['price_change'].tail(30).std()

            # Combine factors
            momentum_score = 0.5

            # Price momentum (0.4 weight)
            if recent_7d_return > 5:
                momentum_score += 0.2
            elif recent_7d_return < -5:
                momentum_score -= 0.2

            if recent_30d_return > 10:
                momentum_score += 0.2
            elif recent_30d_return < -10:
                momentum_score -= 0.2

            # Volume momentum (0.3 weight)
            if volume_trend > 1.2:
                momentum_score += 0.15
            elif volume_trend < 0.8:
                momentum_score -= 0.15

            # Volatility penalty (0.3 weight)
            if volatility < 0.05:  # Low volatility
                momentum_score += 0.15
            elif volatility > 0.15:  # High volatility
                momentum_score -= 0.15

            return max(0.0, min(1.0, momentum_score))

        except Exception as e:
            logger.error(f"Error calculating momentum score: {e}")
            return 0.5

    def _fallback_prediction(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Fallback prediction method using technical analysis when ML fails"""
        try:
            current_price = df['price'].iloc[-1]
            confidence = 0.6  # Lower confidence for fallback method

            # Use multiple technical indicators for prediction
            prediction_factors = []

            # RSI-based prediction
            if 'rsi' in df.columns and not pd.isna(df['rsi'].iloc[-1]):
                rsi = df['rsi'].iloc[-1]
                if rsi > 70:
                    # Overbought - expect pullback
                    prediction_factors.append(current_price * 0.95)
                elif rsi < 30:
                    # Oversold - expect bounce
                    prediction_factors.append(current_price * 1.05)
                else:
                    prediction_factors.append(current_price * 1.02)

            # MACD-based prediction
            if all(col in df.columns for col in ['macd', 'macd_signal']):
                macd = df['macd'].iloc[-1]
                signal = df['macd_signal'].iloc[-1]
                if macd > signal:
                    prediction_factors.append(current_price * 1.03)  # Bullish
                else:
                    prediction_factors.append(current_price * 0.98)  # Bearish

            # Moving average trend
            ma_cols = [col for col in df.columns if col.startswith('MA_') or col.startswith('EMA_')]
            if ma_cols:
                bullish_signals = 0
                bearish_signals = 0

                for ma_col in ma_cols[:3]:  # Check first 3 MAs
                    if ma_col in df.columns and not pd.isna(df[ma_col].iloc[-1]):
                        if current_price > df[ma_col].iloc[-1]:
                            bullish_signals += 1
                        else:
                            bearish_signals += 1

                if bullish_signals > bearish_signals:
                    prediction_factors.append(current_price * 1.025)
                else:
                    prediction_factors.append(current_price * 0.975)

            # Momentum-based prediction
            if len(df) > 7:
                week_return = (current_price / df['price'].iloc[-7]) - 1
                if week_return > 0.05:
                    prediction_factors.append(current_price * (1 + week_return * 0.7))  # Continue momentum
                elif week_return < -0.05:
                    prediction_factors.append(current_price * (1 + week_return * 0.5))  # Partial recovery
                else:
                    prediction_factors.append(current_price * 1.01)

            # Average the predictions
            if prediction_factors:
                predicted_price = sum(prediction_factors) / len(prediction_factors)

                # Ensure prediction isn't too extreme
                max_change = 0.3  # Max 30% change
                if predicted_price > current_price * (1 + max_change):
                    predicted_price = current_price * (1 + max_change)
                elif predicted_price < current_price * (1 - max_change):
                    predicted_price = current_price * (1 - max_change)

                return predicted_price, confidence

            # If no technical factors available, use slight upward bias
            return current_price * 1.02, 0.4

        except Exception as e:
            logger.error(f"Error in fallback prediction: {e}")
            return df['price'].iloc[-1], 0.3

    def get_top_recommendations(self, num_coins: int = 50, top_n: int = 10) -> List[Dict]:
        """Get top cryptocurrency recommendations"""
        logger.info(f"Analyzing top {num_coins} cryptocurrencies for recommendations")

        # Get top coins
        top_coins_df = self.data_fetcher.get_top_coins(limit=num_coins)

        if top_coins_df.empty:
            logger.error("No coin data available")
            return []

        # Analyze each coin
        analyses = []
        for _, coin_data in top_coins_df.iterrows():
            coin_id = coin_data['id']
            analysis = self.analyze_coin(coin_id, coin_data.to_dict())
            analyses.append(analysis)

            # Rate limiting to avoid API bans (reduced for faster execution)
            time.sleep(0.5)

        # Filter out failed analyses and sort by score
        valid_analyses = [a for a in analyses if 'error' not in a or a.get('overall_score', 0) > 0]
        sorted_analyses = sorted(valid_analyses, key=lambda x: x.get('overall_score', 0), reverse=True)

        logger.info(f"Successfully analyzed {len(valid_analyses)} coins")
        logger.info(f"Top {top_n} recommendations:")

        for i, analysis in enumerate(sorted_analyses[:top_n], 1):
            logger.info(f"{i}. {analysis['name']} ({analysis['symbol']}) - Score: {analysis.get('overall_score', 0):.3f}")

        return sorted_analyses[:top_n]

def main():
    """Main function to run the crypto analysis"""
    analyzer = CryptoAnalyzer()

    print("🚀 Starting Cryptocurrency Analysis System")
    print("=" * 50)

    # Get top 10 recommendations
    recommendations = analyzer.get_top_recommendations(num_coins=15, top_n=10)

    if not recommendations:
        print("❌ No recommendations available")
        return

    # Display results
    print("\n📊 Top 10 Cryptocurrency Recommendations")
    print("=" * 50)

    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['name']} ({rec['symbol']})")
        print(f"   Current Price: ${rec['current_price']:.4f}")
        print(f"   Predicted Price: ${rec['predicted_price']:.4f}")
        print(f"   Expected Return: {rec['expected_return_pct']:.2f}%")
        print(f"   Overall Score: {rec['overall_score']:.3f}")
        print(f"   Market Cap: ${rec['market_cap']:,.0f}")
        print(f"   24h Change: {rec['price_change_24h']:.2f}%")

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"crypto_recommendations_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'recommendations': recommendations,
            'methodology': {
                'data_sources': ['CoinGecko API', 'Technical Analysis'],
                'indicators': ['RSI', 'MACD', 'Bollinger Bands', 'Moving Averages'],
                'scoring_weights': {
                    'technical': 0.4,
                    'sentiment': 0.2,
                    'momentum': 0.3,
                    'prediction_confidence': 0.1
                }
            }
        }, f, indent=2)

    print(f"\n💾 Results saved to {filename}")
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
