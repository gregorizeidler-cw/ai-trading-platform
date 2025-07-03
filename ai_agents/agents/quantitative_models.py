"""
Advanced Quantitative Models for Statistical Arbitrage and ML-Based Trading
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from dataclasses import dataclass
from loguru import logger
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent
from ..llm.openai_client import OpenAIClient


@dataclass
class QuantitativeSignal:
    """Quantitative trading signal"""
    symbol: str
    signal_type: str
    strength: float  # -1 to 1
    confidence: float  # 0 to 1
    expected_return: float
    risk_score: float
    holding_period: int
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class PairsTradingSignal:
    """Pairs trading signal"""
    symbol_1: str
    symbol_2: str
    spread: float
    z_score: float
    signal: str  # 'long_spread', 'short_spread', 'neutral'
    confidence: float
    half_life: float
    timestamp: datetime


class QuantitativeModels(BaseAgent):
    """Advanced quantitative models for systematic trading"""
    
    def __init__(self, openai_client: OpenAIClient):
        super().__init__(
            name="Quantitative Models",
            description="Advanced statistical arbitrage and ML-based trading models",
            openai_client=openai_client
        )
        self.models = {}
        self.scaler = StandardScaler()
        self.factor_loadings = {}
        self.cointegration_cache = {}
        
    async def statistical_arbitrage_analysis(self, symbols: List[str]) -> List[QuantitativeSignal]:
        """Advanced statistical arbitrage analysis"""
        try:
            signals = []
            data = await self._get_multi_symbol_data(symbols, 252)
            
            for symbol in symbols:
                if symbol not in data:
                    continue
                    
                symbol_data = data[symbol]
                
                # Calculate various signals
                mean_reversion = self._calculate_mean_reversion_signal(symbol_data)
                momentum = self._calculate_momentum_signal(symbol_data)
                volatility = self._calculate_volatility_signal(symbol_data)
                ml_signal = await self._calculate_ml_signal(symbol, symbol_data)
                
                # Combine signals
                combined = self._combine_signals([mean_reversion, momentum, volatility, ml_signal])
                
                signal = QuantitativeSignal(
                    symbol=symbol,
                    signal_type="statistical_arbitrage",
                    strength=combined['strength'],
                    confidence=combined['confidence'],
                    expected_return=combined['expected_return'],
                    risk_score=combined['risk_score'],
                    holding_period=combined['holding_period'],
                    timestamp=datetime.now(),
                    metadata={
                        'mean_reversion': mean_reversion,
                        'momentum': momentum,
                        'volatility': volatility,
                        'ml_prediction': ml_signal
                    }
                )
                
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in statistical arbitrage analysis: {e}")
            return []
    
    async def pairs_trading_analysis(
        self, 
        symbols: List[str], 
        lookback_days: int = 252
    ) -> List[PairsTradingSignal]:
        """Advanced pairs trading analysis with cointegration"""
        try:
            signals = []
            
            # Get data for all symbols
            data = await self._get_multi_symbol_data(symbols, lookback_days)
            
            # Find cointegrated pairs
            pairs = await self._find_cointegrated_pairs(data)
            
            for pair in pairs:
                symbol_1, symbol_2 = pair['symbols']
                
                # Calculate spread
                spread = self._calculate_spread(
                    data[symbol_1]['Close'], 
                    data[symbol_2]['Close'],
                    pair['hedge_ratio']
                )
                
                # Calculate z-score
                z_score = self._calculate_z_score(spread)
                
                # Generate signal
                signal_type = self._generate_pairs_signal(z_score)
                
                # Calculate half-life of mean reversion
                half_life = self._calculate_half_life(spread)
                
                # Calculate confidence
                confidence = self._calculate_pairs_confidence(pair, z_score)
                
                signal = PairsTradingSignal(
                    symbol_1=symbol_1,
                    symbol_2=symbol_2,
                    spread=spread.iloc[-1],
                    z_score=z_score.iloc[-1],
                    signal=signal_type,
                    confidence=confidence,
                    half_life=half_life,
                    timestamp=datetime.now()
                )
                
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in pairs trading analysis: {e}")
            return []
    
    async def factor_model_analysis(
        self, 
        symbols: List[str], 
        factors: List[str] = None
    ) -> Dict[str, Any]:
        """Multi-factor model analysis"""
        try:
            if factors is None:
                factors = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM']  # Market factors
            
            # Get data
            symbol_data = await self._get_multi_symbol_data(symbols, 252)
            factor_data = await self._get_multi_symbol_data(factors, 252)
            
            results = {}
            
            for symbol in symbols:
                if symbol not in symbol_data:
                    continue
                
                # Calculate returns
                returns = symbol_data[symbol]['Close'].pct_change().dropna()
                
                # Build factor matrix
                factor_returns = pd.DataFrame()
                for factor in factors:
                    if factor in factor_data:
                        factor_returns[factor] = factor_data[factor]['Close'].pct_change()
                
                # Align data
                aligned_data = pd.concat([returns, factor_returns], axis=1).dropna()
                
                if len(aligned_data) < 50:
                    continue
                
                y = aligned_data.iloc[:, 0]  # Symbol returns
                X = aligned_data.iloc[:, 1:]  # Factor returns
                
                # Fit factor model
                model = LinearRegression()
                model.fit(X, y)
                
                # Calculate metrics
                r_squared = model.score(X, y)
                predictions = model.predict(X)
                residuals = y - predictions
                
                # Factor loadings
                loadings = dict(zip(factors, model.coef_))
                
                # Calculate alpha (risk-adjusted return)
                alpha = model.intercept_ * 252  # Annualized
                
                # Tracking error
                tracking_error = residuals.std() * np.sqrt(252)
                
                # Information ratio
                info_ratio = alpha / tracking_error if tracking_error > 0 else 0
                
                results[symbol] = {
                    'alpha': alpha,
                    'r_squared': r_squared,
                    'factor_loadings': loadings,
                    'tracking_error': tracking_error,
                    'information_ratio': info_ratio,
                    'residual_volatility': residuals.std()
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in factor model analysis: {e}")
            return {}
    
    async def machine_learning_predictions(
        self, 
        symbol: str, 
        prediction_days: int = 5
    ) -> Dict[str, Any]:
        """Machine learning based price predictions"""
        try:
            # Get extended data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="2y", interval="1d")
            
            if len(data) < 100:
                return {"error": "Insufficient data"}
            
            # Feature engineering
            features = self._create_ml_features(data)
            
            # Prepare target variable (future returns)
            target = data['Close'].pct_change(prediction_days).shift(-prediction_days)
            
            # Align features and target
            ml_data = pd.concat([features, target], axis=1).dropna()
            
            if len(ml_data) < 50:
                return {"error": "Insufficient aligned data"}
            
            # Split data
            train_size = int(len(ml_data) * 0.8)
            train_data = ml_data.iloc[:train_size]
            test_data = ml_data.iloc[train_size:]
            
            X_train = train_data.iloc[:, :-1]
            y_train = train_data.iloc[:, -1]
            X_test = test_data.iloc[:, :-1]
            y_test = test_data.iloc[:, -1]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train models
            models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'linear_regression': LinearRegression()
            }
            
            predictions = {}
            
            for name, model in models.items():
                # Train
                model.fit(X_train_scaled, y_train)
                
                # Predict
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
                
                # Metrics
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)
                
                # Current prediction
                current_features = features.iloc[-1:].values
                current_scaled = self.scaler.transform(current_features)
                current_prediction = model.predict(current_scaled)[0]
                
                predictions[name] = {
                    'prediction': current_prediction,
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'feature_importance': getattr(model, 'feature_importances_', None)
                }
            
            # Ensemble prediction
            ensemble_pred = np.mean([pred['prediction'] for pred in predictions.values()])
            
            return {
                'symbol': symbol,
                'prediction_days': prediction_days,
                'ensemble_prediction': ensemble_pred,
                'individual_predictions': predictions,
                'confidence': self._calculate_ml_confidence(predictions),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in ML predictions: {e}")
            return {"error": str(e)}
    
    async def _get_multi_symbol_data(self, symbols: List[str], days: int) -> Dict[str, pd.DataFrame]:
        """Get data for multiple symbols"""
        data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=f"{days}d")
                
                if not hist.empty:
                    data[symbol] = hist
                    
            except Exception as e:
                logger.warning(f"Could not get data for {symbol}: {e}")
                continue
        
        return data
    
    def _calculate_mean_reversion_signal(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate mean reversion signal"""
        try:
            prices = data['Close']
            z_score = (prices.iloc[-1] - prices.mean()) / prices.std()
            hurst = self._calculate_hurst_exponent(prices)
            
            strength = -z_score / 2  # Normalize to [-1, 1]
            strength = max(min(strength, 1), -1)
            
            return {
                'strength': strength,
                'z_score': z_score,
                'hurst_exponent': hurst,
                'confidence': 1 - hurst if hurst < 0.5 else 0.1
            }
            
        except Exception as e:
            logger.error(f"Error calculating mean reversion signal: {e}")
            return {'strength': 0, 'z_score': 0, 'hurst_exponent': 0.5, 'confidence': 0}
    
    def _calculate_momentum_signal(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum signal"""
        try:
            prices = data['Close']
            mom_1m = prices.pct_change(21).iloc[-1]
            mom_3m = prices.pct_change(63).iloc[-1]
            mom_6m = prices.pct_change(126).iloc[-1]
            
            momentum = 0.5 * mom_1m + 0.3 * mom_3m + 0.2 * mom_6m
            strength = np.tanh(momentum * 10)
            
            return {
                'strength': strength,
                'momentum_1m': mom_1m,
                'momentum_3m': mom_3m,
                'momentum_6m': mom_6m,
                'confidence': min(abs(momentum) * 5, 1)
            }
            
        except Exception as e:
            logger.error(f"Error calculating momentum signal: {e}")
            return {'strength': 0, 'momentum_1m': 0, 'momentum_3m': 0, 'momentum_6m': 0, 'confidence': 0}
    
    def _calculate_volatility_signal(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility-based signal"""
        try:
            returns = data['Close'].pct_change().dropna()
            current_vol = returns.rolling(20).std().iloc[-1]
            historical_vol = returns.std()
            
            vol_ratio = current_vol / historical_vol
            strength = -np.tanh((vol_ratio - 1) * 2)
            
            return {
                'strength': strength,
                'volatility_ratio': vol_ratio,
                'current_volatility': current_vol,
                'historical_volatility': historical_vol,
                'confidence': min(abs(vol_ratio - 1), 1)
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility signal: {e}")
            return {'strength': 0, 'volatility_ratio': 1, 'current_volatility': 0, 'historical_volatility': 0, 'confidence': 0}
    
    async def _calculate_ml_signal(self, symbol: str, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate ML-based signal"""
        try:
            features = self._create_ml_features(data)
            
            if len(features) < 20:
                return {'strength': 0, 'prediction': 0, 'confidence': 0}
            
            returns = data['Close'].pct_change(5).shift(-5)
            ml_data = pd.concat([features, returns], axis=1).dropna()
            
            if len(ml_data) < 10:
                return {'strength': 0, 'prediction': 0, 'confidence': 0}
            
            correlations = ml_data.corr().iloc[-1, :-1]
            current_features = features.iloc[-1]
            signal = np.dot(correlations, current_features) / len(correlations)
            
            strength = np.tanh(signal * 10)
            
            return {
                'strength': strength,
                'prediction': signal,
                'confidence': min(abs(signal) * 5, 1)
            }
            
        except Exception as e:
            logger.error(f"Error calculating ML signal: {e}")
            return {'strength': 0, 'prediction': 0, 'confidence': 0}
    
    def _create_ml_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create ML features from price data"""
        try:
            features = pd.DataFrame(index=data.index)
            
            features['rsi'] = self._calculate_rsi(data['Close'])
            features['macd'] = self._calculate_macd(data['Close'])
            features['bb_position'] = self._calculate_bollinger_position(data['Close'])
            features['volume_sma_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
            features['volatility'] = data['Close'].pct_change().rolling(20).std()
            features['momentum_5'] = data['Close'].pct_change(5)
            features['momentum_20'] = data['Close'].pct_change(20)
            
            return features.dropna()
            
        except Exception as e:
            logger.error(f"Error creating ML features: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD"""
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        return exp1 - exp2
    
    def _calculate_bollinger_position(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        return (prices - sma) / (2 * std)
    
    def _calculate_hurst_exponent(self, prices: pd.Series) -> float:
        """Calculate Hurst exponent"""
        try:
            lags = range(2, 100)
            tau = [np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except:
            return 0.5
    
    def _combine_signals(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple signals"""
        try:
            weights = [0.3, 0.25, 0.2, 0.25]  # Mean reversion, momentum, volatility, ML
            
            combined_strength = sum(w * s['strength'] for w, s in zip(weights, signals))
            combined_confidence = sum(w * s['confidence'] for w, s in zip(weights, signals))
            
            expected_return = combined_strength * 0.05
            risk_score = 1 - combined_confidence
            holding_period = 20
            
            return {
                'strength': combined_strength,
                'confidence': combined_confidence,
                'expected_return': expected_return,
                'risk_score': risk_score,
                'holding_period': holding_period
            }
            
        except Exception as e:
            logger.error(f"Error combining signals: {e}")
            return {'strength': 0, 'confidence': 0, 'expected_return': 0, 'risk_score': 1, 'holding_period': 20}
    
    async def _find_cointegrated_pairs(self, data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Find cointegrated pairs"""
        pairs = []
        symbols = list(data.keys())
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol_1, symbol_2 = symbols[i], symbols[j]
                
                try:
                    # Get aligned price data
                    prices_1 = data[symbol_1]['Close']
                    prices_2 = data[symbol_2]['Close']
                    
                    aligned = pd.concat([prices_1, prices_2], axis=1).dropna()
                    
                    if len(aligned) < 50:
                        continue
                    
                    # Test for cointegration
                    y = aligned.iloc[:, 0]
                    x = aligned.iloc[:, 1]
                    
                    # OLS regression
                    model = LinearRegression()
                    model.fit(x.values.reshape(-1, 1), y)
                    
                    hedge_ratio = model.coef_[0]
                    residuals = y - model.predict(x.values.reshape(-1, 1))
                    
                    # ADF test on residuals (simplified)
                    adf_stat = self._adf_test(residuals)
                    
                    if adf_stat < -2.5:  # Simplified threshold
                        pairs.append({
                            'symbols': (symbol_1, symbol_2),
                            'hedge_ratio': hedge_ratio,
                            'adf_statistic': adf_stat,
                            'correlation': np.corrcoef(y, x)[0, 1]
                        })
                        
                except Exception as e:
                    continue
        
        return pairs
    
    def _adf_test(self, series: pd.Series) -> float:
        """Simplified ADF test"""
        try:
            # This is a very simplified version
            # In practice, use statsmodels.tsa.stattools.adfuller
            y = series.values
            y_lag = np.roll(y, 1)[1:]
            y_diff = np.diff(y)
            
            # Simple regression
            model = LinearRegression()
            model.fit(y_lag.reshape(-1, 1), y_diff)
            
            return model.coef_[0]  # Simplified statistic
            
        except:
            return 0
    
    def _calculate_spread(self, prices_1: pd.Series, prices_2: pd.Series, hedge_ratio: float) -> pd.Series:
        """Calculate spread for pairs trading"""
        return prices_1 - hedge_ratio * prices_2
    
    def _calculate_z_score(self, spread: pd.Series, window: int = 20) -> pd.Series:
        """Calculate z-score of spread"""
        return (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    
    def _generate_pairs_signal(self, z_score: pd.Series, threshold: float = 2.0) -> str:
        """Generate pairs trading signal"""
        current_z = z_score.iloc[-1]
        
        if current_z > threshold:
            return "short_spread"  # Short first asset, long second
        elif current_z < -threshold:
            return "long_spread"   # Long first asset, short second
        else:
            return "neutral"
    
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate half-life of mean reversion"""
        try:
            spread_lag = spread.shift(1)
            spread_diff = spread.diff()
            
            # Align data
            data = pd.concat([spread_diff, spread_lag], axis=1).dropna()
            
            if len(data) < 10:
                return 30.0  # Default
            
            # OLS regression
            model = LinearRegression()
            model.fit(data.iloc[:, 1].values.reshape(-1, 1), data.iloc[:, 0])
            
            lambda_coef = model.coef_[0]
            
            if lambda_coef >= 0:
                return 30.0  # Default if no mean reversion
            
            half_life = -np.log(2) / lambda_coef
            return max(min(half_life, 100), 1)  # Bound between 1 and 100 days
            
        except:
            return 30.0
    
    def _calculate_pairs_confidence(self, pair: Dict[str, Any], z_score: pd.Series) -> float:
        """Calculate confidence in pairs trading signal"""
        try:
            # Based on ADF statistic and current z-score
            adf_strength = min(abs(pair['adf_statistic']) / 5, 1)
            z_strength = min(abs(z_score.iloc[-1]) / 3, 1)
            
            return (adf_strength + z_strength) / 2
            
        except:
            return 0.5
    
    def _calculate_ml_confidence(self, predictions: Dict[str, Any]) -> float:
        """Calculate ML prediction confidence"""
        try:
            # Based on model agreement and error rates
            pred_values = [pred['prediction'] for pred in predictions.values()]
            
            if len(pred_values) < 2:
                return 0.5
            
            # Agreement between models
            agreement = 1 - (np.std(pred_values) / (np.mean(np.abs(pred_values)) + 1e-8))
            
            return max(min(agreement, 1), 0)
            
        except:
            return 0.5 