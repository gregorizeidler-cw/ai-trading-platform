"""
Momentum Trading Strategy
Professional implementation with multiple timeframe analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class MomentumSignal:
    symbol: str
    signal: str  # BUY, SELL, HOLD
    strength: float  # 0-1
    timeframe: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    timestamp: datetime

class MomentumStrategy:
    """
    Professional Momentum Trading Strategy
    
    Features:
    - Multi-timeframe momentum analysis
    - Risk-adjusted position sizing
    - Dynamic stop losses
    - Momentum confirmation filters
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.positions = {}
        self.signals_history = []
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            "timeframes": ["1H", "4H", "1D"],
            "lookback_periods": {"1H": 20, "4H": 14, "1D": 10},
            "momentum_threshold": 0.02,  # 2% minimum momentum
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "volume_confirmation": True,
            "min_volume_ratio": 1.5,  # 1.5x average volume
            "risk_per_trade": 0.02,  # 2% risk per trade
            "reward_risk_ratio": 2.0,  # 2:1 reward to risk
            "max_positions": 5,
            "momentum_lookback": 20
        }
    
    def analyze_momentum(self, data: Dict[str, pd.DataFrame]) -> List[MomentumSignal]:
        """
        Analyze momentum across multiple timeframes
        """
        signals = []
        
        for symbol, df in data.items():
            if len(df) < max(self.config["lookback_periods"].values()):
                continue
                
            # Multi-timeframe momentum analysis
            momentum_scores = self._calculate_multi_timeframe_momentum(df)
            
            # Generate signals
            signal = self._generate_momentum_signal(symbol, df, momentum_scores)
            if signal:
                signals.append(signal)
        
        return signals
    
    def _calculate_multi_timeframe_momentum(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum across different timeframes"""
        momentum_scores = {}
        
        for timeframe in self.config["timeframes"]:
            lookback = self.config["lookback_periods"][timeframe]
            
            # Price momentum
            price_momentum = self._calculate_price_momentum(df, lookback)
            
            # Volume momentum
            volume_momentum = self._calculate_volume_momentum(df, lookback)
            
            # RSI momentum
            rsi_momentum = self._calculate_rsi_momentum(df, lookback)
            
            # MACD momentum
            macd_momentum = self._calculate_macd_momentum(df)
            
            # Combined momentum score
            momentum_scores[timeframe] = {
                "price": price_momentum,
                "volume": volume_momentum,
                "rsi": rsi_momentum,
                "macd": macd_momentum,
                "combined": np.mean([price_momentum, volume_momentum, rsi_momentum, macd_momentum])
            }
        
        return momentum_scores
    
    def _calculate_price_momentum(self, df: pd.DataFrame, lookback: int) -> float:
        """Calculate price-based momentum"""
        if len(df) < lookback:
            return 0.0
        
        current_price = df['close'].iloc[-1]
        past_price = df['close'].iloc[-lookback]
        
        momentum = (current_price - past_price) / past_price
        
        # Normalize to 0-1 scale
        return max(0, min(1, (momentum + 0.1) / 0.2))
    
    def _calculate_volume_momentum(self, df: pd.DataFrame, lookback: int) -> float:
        """Calculate volume-based momentum"""
        if len(df) < lookback:
            return 0.0
        
        recent_volume = df['volume'].iloc[-5:].mean()
        avg_volume = df['volume'].iloc[-lookback:].mean()
        
        if avg_volume == 0:
            return 0.0
        
        volume_ratio = recent_volume / avg_volume
        
        # Normalize to 0-1 scale
        return max(0, min(1, (volume_ratio - 1) / 2))
    
    def _calculate_rsi_momentum(self, df: pd.DataFrame, lookback: int) -> float:
        """Calculate RSI-based momentum"""
        rsi = self._calculate_rsi(df, lookback)
        
        if rsi is None:
            return 0.0
        
        # RSI momentum: higher when RSI is trending up and not overbought
        if rsi > 70:
            return 0.2  # Overbought
        elif rsi < 30:
            return 0.8  # Oversold - potential momentum reversal
        else:
            return (rsi - 30) / 40  # Normalize 30-70 to 0-1
    
    def _calculate_macd_momentum(self, df: pd.DataFrame) -> float:
        """Calculate MACD-based momentum"""
        macd_line, signal_line = self._calculate_macd(df)
        
        if macd_line is None or signal_line is None:
            return 0.0
        
        # MACD momentum: positive when MACD > Signal
        macd_diff = macd_line - signal_line
        
        # Normalize to 0-1 scale
        return max(0, min(1, (macd_diff + 1) / 2))
    
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
    
    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD indicator"""
        if len(df) < slow:
            return None, None
        
        exp1 = df['close'].ewm(span=fast).mean()
        exp2 = df['close'].ewm(span=slow).mean()
        
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        
        return macd_line.iloc[-1], signal_line.iloc[-1]
    
    def _generate_momentum_signal(self, symbol: str, df: pd.DataFrame, momentum_scores: Dict) -> Optional[MomentumSignal]:
        """Generate trading signal based on momentum analysis"""
        
        # Calculate overall momentum score
        timeframe_scores = [scores["combined"] for scores in momentum_scores.values()]
        overall_momentum = np.mean(timeframe_scores)
        
        # Determine signal direction
        if overall_momentum > 0.7:
            signal_type = "BUY"
            strength = overall_momentum
        elif overall_momentum < 0.3:
            signal_type = "SELL"
            strength = 1 - overall_momentum
        else:
            return None  # No clear signal
        
        # Calculate entry, stop loss, and take profit
        current_price = df['close'].iloc[-1]
        atr = self._calculate_atr(df)
        
        if signal_type == "BUY":
            entry_price = current_price
            stop_loss = current_price - (2 * atr)
            take_profit = current_price + (self.config["reward_risk_ratio"] * 2 * atr)
        else:  # SELL
            entry_price = current_price
            stop_loss = current_price + (2 * atr)
            take_profit = current_price - (self.config["reward_risk_ratio"] * 2 * atr)
        
        # Calculate confidence based on momentum alignment
        confidence = self._calculate_signal_confidence(momentum_scores, overall_momentum)
        
        return MomentumSignal(
            symbol=symbol,
            signal=signal_type,
            strength=strength,
            timeframe="MULTI",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            timestamp=datetime.now()
        )
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(df) < period:
            return df['close'].iloc[-1] * 0.02  # 2% fallback
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        
        return atr.iloc[-1] if not atr.empty else df['close'].iloc[-1] * 0.02
    
    def _calculate_signal_confidence(self, momentum_scores: Dict, overall_momentum: float) -> float:
        """Calculate confidence score for the signal"""
        
        # Check timeframe alignment
        timeframe_alignment = 0
        for timeframe, scores in momentum_scores.items():
            if scores["combined"] > 0.5:
                timeframe_alignment += 1
        
        alignment_score = timeframe_alignment / len(momentum_scores)
        
        # Check momentum strength
        strength_score = abs(overall_momentum - 0.5) * 2  # Distance from neutral
        
        # Combine scores
        confidence = (alignment_score * 0.6 + strength_score * 0.4)
        
        return min(0.95, max(0.5, confidence))
    
    def calculate_position_size(self, signal: MomentumSignal, account_balance: float) -> float:
        """Calculate position size based on risk management"""
        
        risk_amount = account_balance * self.config["risk_per_trade"]
        
        # Calculate risk per share
        risk_per_share = abs(signal.entry_price - signal.stop_loss)
        
        if risk_per_share == 0:
            return 0
        
        # Position size
        position_size = risk_amount / risk_per_share
        
        # Apply confidence adjustment
        position_size *= signal.confidence
        
        return position_size
    
    def update_stops(self, symbol: str, current_price: float) -> Dict[str, float]:
        """Update trailing stops for momentum trades"""
        
        if symbol not in self.positions:
            return {}
        
        position = self.positions[symbol]
        atr = position.get("atr", current_price * 0.02)
        
        updates = {}
        
        if position["side"] == "LONG":
            # Trailing stop for long position
            new_stop = current_price - (2 * atr)
            if new_stop > position["stop_loss"]:
                updates["stop_loss"] = new_stop
        else:
            # Trailing stop for short position
            new_stop = current_price + (2 * atr)
            if new_stop < position["stop_loss"]:
                updates["stop_loss"] = new_stop
        
        return updates
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get strategy performance metrics"""
        
        if not self.signals_history:
            return {"total_signals": 0, "win_rate": 0, "avg_return": 0}
        
        total_signals = len(self.signals_history)
        profitable_signals = sum(1 for s in self.signals_history if s.get("pnl", 0) > 0)
        
        win_rate = profitable_signals / total_signals if total_signals > 0 else 0
        avg_return = np.mean([s.get("pnl", 0) for s in self.signals_history])
        
        return {
            "total_signals": total_signals,
            "profitable_signals": profitable_signals,
            "win_rate": win_rate,
            "avg_return": avg_return,
            "sharpe_ratio": self._calculate_sharpe_ratio(),
            "max_drawdown": self._calculate_max_drawdown()
        }
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate strategy Sharpe ratio"""
        returns = [s.get("pnl", 0) for s in self.signals_history]
        
        if not returns or np.std(returns) == 0:
            return 0.0
        
        return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        cumulative_returns = np.cumsum([s.get("pnl", 0) for s in self.signals_history])
        
        if not cumulative_returns:
            return 0.0
        
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        
        return np.min(drawdown) if len(drawdown) > 0 else 0.0 