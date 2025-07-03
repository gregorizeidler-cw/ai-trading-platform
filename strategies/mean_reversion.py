"""
Mean Reversion Trading Strategy
Professional implementation with statistical analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from scipy import stats

@dataclass
class MeanReversionSignal:
    symbol: str
    signal: str  # BUY, SELL, HOLD
    z_score: float
    mean_price: float
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    timestamp: datetime

class MeanReversionStrategy:
    """
    Professional Mean Reversion Trading Strategy
    
    Features:
    - Statistical analysis (Z-score, Bollinger Bands)
    - Multiple mean reversion indicators
    - Dynamic position sizing
    - Risk management with stop losses
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.positions = {}
        self.signals_history = []
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            "lookback_period": 20,
            "z_score_threshold": 2.0,
            "bollinger_std": 2.0,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "min_volume_ratio": 1.2,
            "risk_per_trade": 0.015,  # 1.5% risk per trade
            "reward_risk_ratio": 1.5,
            "max_holding_period": 10,  # days
            "statistical_significance": 0.95
        }
    
    def analyze_mean_reversion(self, data: Dict[str, pd.DataFrame]) -> List[MeanReversionSignal]:
        """Analyze mean reversion opportunities"""
        signals = []
        
        for symbol, df in data.items():
            if len(df) < self.config["lookback_period"]:
                continue
            
            signal = self._generate_mean_reversion_signal(symbol, df)
            if signal:
                signals.append(signal)
        
        return signals
    
    def _generate_mean_reversion_signal(self, symbol: str, df: pd.DataFrame) -> Optional[MeanReversionSignal]:
        """Generate mean reversion signal"""
        
        # Calculate statistical indicators
        z_score = self._calculate_z_score(df)
        bollinger_signal = self._calculate_bollinger_signal(df)
        rsi_signal = self._calculate_rsi_signal(df)
        volume_confirmation = self._check_volume_confirmation(df)
        
        # Determine signal strength
        signal_strength = abs(z_score) / self.config["z_score_threshold"]
        
        # Generate signal
        if z_score > self.config["z_score_threshold"] and bollinger_signal == "SELL":
            signal_type = "SELL"  # Price too high, expect reversion
        elif z_score < -self.config["z_score_threshold"] and bollinger_signal == "BUY":
            signal_type = "BUY"   # Price too low, expect reversion
        else:
            return None
        
        # Calculate entry and exit levels
        current_price = df['close'].iloc[-1]
        mean_price = df['close'].rolling(self.config["lookback_period"]).mean().iloc[-1]
        std_price = df['close'].rolling(self.config["lookback_period"]).std().iloc[-1]
        
        if signal_type == "BUY":
            entry_price = current_price
            take_profit = mean_price  # Target mean reversion
            stop_loss = current_price - (2 * std_price)
        else:  # SELL
            entry_price = current_price
            take_profit = mean_price  # Target mean reversion
            stop_loss = current_price + (2 * std_price)
        
        # Calculate confidence
        confidence = self._calculate_confidence(z_score, rsi_signal, volume_confirmation)
        
        return MeanReversionSignal(
            symbol=symbol,
            signal=signal_type,
            z_score=z_score,
            mean_price=mean_price,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            timestamp=datetime.now()
        )
    
    def _calculate_z_score(self, df: pd.DataFrame) -> float:
        """Calculate Z-score for current price"""
        lookback = self.config["lookback_period"]
        current_price = df['close'].iloc[-1]
        mean_price = df['close'].rolling(lookback).mean().iloc[-1]
        std_price = df['close'].rolling(lookback).std().iloc[-1]
        
        if std_price == 0:
            return 0.0
        
        return (current_price - mean_price) / std_price
    
    def _calculate_bollinger_signal(self, df: pd.DataFrame) -> str:
        """Calculate Bollinger Bands signal"""
        lookback = self.config["lookback_period"]
        std_multiplier = self.config["bollinger_std"]
        
        sma = df['close'].rolling(lookback).mean()
        std = df['close'].rolling(lookback).std()
        
        upper_band = sma + (std * std_multiplier)
        lower_band = sma - (std * std_multiplier)
        
        current_price = df['close'].iloc[-1]
        
        if current_price > upper_band.iloc[-1]:
            return "SELL"  # Price above upper band
        elif current_price < lower_band.iloc[-1]:
            return "BUY"   # Price below lower band
        else:
            return "HOLD"
    
    def _calculate_rsi_signal(self, df: pd.DataFrame) -> str:
        """Calculate RSI signal"""
        rsi = self._calculate_rsi(df)
        
        if rsi is None:
            return "HOLD"
        
        if rsi > self.config["rsi_overbought"]:
            return "SELL"
        elif rsi < self.config["rsi_oversold"]:
            return "BUY"
        else:
            return "HOLD"
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Calculate RSI indicator"""
        if len(df) < period + 1:
            return None
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else None
    
    def _check_volume_confirmation(self, df: pd.DataFrame) -> bool:
        """Check if volume confirms the signal"""
        lookback = self.config["lookback_period"]
        
        recent_volume = df['volume'].iloc[-3:].mean()
        avg_volume = df['volume'].rolling(lookback).mean().iloc[-1]
        
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 0
        
        return volume_ratio > self.config["min_volume_ratio"]
    
    def _calculate_confidence(self, z_score: float, rsi_signal: str, volume_confirmation: bool) -> float:
        """Calculate signal confidence"""
        
        # Z-score strength (0-1)
        z_strength = min(1.0, abs(z_score) / (self.config["z_score_threshold"] * 2))
        
        # RSI confirmation (0-1)
        rsi_confirmation = 1.0 if rsi_signal in ["BUY", "SELL"] else 0.5
        
        # Volume confirmation (0-1)
        volume_score = 1.0 if volume_confirmation else 0.7
        
        # Combined confidence
        confidence = (z_strength * 0.5 + rsi_confirmation * 0.3 + volume_score * 0.2)
        
        return min(0.95, max(0.5, confidence))
    
    def calculate_position_size(self, signal: MeanReversionSignal, account_balance: float) -> float:
        """Calculate position size for mean reversion trade"""
        
        risk_amount = account_balance * self.config["risk_per_trade"]
        risk_per_share = abs(signal.entry_price - signal.stop_loss)
        
        if risk_per_share == 0:
            return 0
        
        position_size = risk_amount / risk_per_share
        
        # Adjust for confidence and Z-score strength
        z_score_adjustment = min(1.0, abs(signal.z_score) / self.config["z_score_threshold"])
        position_size *= signal.confidence * z_score_adjustment
        
        return position_size
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get mean reversion strategy performance"""
        
        if not self.signals_history:
            return {"total_signals": 0, "win_rate": 0, "avg_return": 0}
        
        total_signals = len(self.signals_history)
        profitable_signals = sum(1 for s in self.signals_history if s.get("pnl", 0) > 0)
        
        win_rate = profitable_signals / total_signals if total_signals > 0 else 0
        avg_return = np.mean([s.get("pnl", 0) for s in self.signals_history])
        
        # Mean reversion specific metrics
        avg_reversion_time = np.mean([s.get("holding_period", 0) for s in self.signals_history])
        successful_reversions = sum(1 for s in self.signals_history if s.get("hit_target", False))
        
        return {
            "total_signals": total_signals,
            "profitable_signals": profitable_signals,
            "win_rate": win_rate,
            "avg_return": avg_return,
            "avg_reversion_time": avg_reversion_time,
            "successful_reversions": successful_reversions,
            "reversion_rate": successful_reversions / total_signals if total_signals > 0 else 0,
            "sharpe_ratio": self._calculate_sharpe_ratio(),
            "max_drawdown": self._calculate_max_drawdown()
        }
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate strategy Sharpe ratio"""
        returns = [s.get("pnl", 0) for s in self.signals_history]
        
        if not returns or np.std(returns) == 0:
            return 0.0
        
        return np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        cumulative_returns = np.cumsum([s.get("pnl", 0) for s in self.signals_history])
        
        if not cumulative_returns:
            return 0.0
        
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        
        return np.min(drawdown) if len(drawdown) > 0 else 0.0 