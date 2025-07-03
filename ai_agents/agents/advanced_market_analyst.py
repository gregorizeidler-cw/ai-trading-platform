"""
Advanced Market Analyst Agent - Professional Trading Analysis
Supports multiple timeframes, asset classes, and advanced technical indicators
"""

import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from .base_agent import BaseAgent
from ..llm.openai_client import OpenAIClient


class TimeFrame(Enum):
    """Trading timeframes"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"


class AssetClass(Enum):
    """Asset classes"""
    STOCKS = "stocks"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITIES = "commodities"
    BONDS = "bonds"
    OPTIONS = "options"
    FUTURES = "futures"


@dataclass
class TechnicalIndicators:
    """Technical indicators container"""
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    sma_20: float
    sma_50: float
    sma_200: float
    ema_12: float
    ema_26: float
    stoch_k: float
    stoch_d: float
    adx: float
    atr: float
    volume_sma: float
    obv: float
    williams_r: float
    cci: float


@dataclass
class MarketStructure:
    """Market structure analysis"""
    trend_direction: str
    trend_strength: float
    support_levels: List[float]
    resistance_levels: List[float]
    key_levels: List[float]
    market_phase: str
    volatility_regime: str
    volume_profile: Dict[str, float]


class AdvancedMarketAnalyst(BaseAgent):
    """Advanced Market Analyst with professional trading capabilities"""
    
    def __init__(self):
        super().__init__(
            name="Advanced Market Analyst",
            description="Professional multi-timeframe, multi-asset market analysis with advanced indicators"
        )
        
        # Analysis configuration
        self.supported_timeframes = [tf.value for tf in TimeFrame]
        self.supported_assets = [ac.value for ac in AssetClass]
        self.min_confidence_threshold = 0.65
        
        # Technical analysis parameters
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.trend_strength_threshold = 0.7
        self.volume_surge_threshold = 1.5
        
        # Market regime detection
        self.volatility_lookback = 20
        self.trend_lookback = 50
        
    async def analyze_multi_timeframe(
        self,
        symbol: str,
        asset_class: AssetClass,
        timeframes: List[TimeFrame],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform multi-timeframe analysis"""
        
        analysis_results = {}
        
        for timeframe in timeframes:
            tf_data = market_data.get(timeframe.value, {})
            if not tf_data:
                continue
                
            # Technical indicators calculation
            indicators = await self._calculate_advanced_indicators(tf_data, symbol)
            
            # Market structure analysis
            structure = await self._analyze_market_structure(tf_data, indicators)
            
            # Price action analysis
            price_action = await self._analyze_price_action(tf_data)
            
            # Volume analysis
            volume_analysis = await self._analyze_volume(tf_data)
            
            # AI-powered analysis
            ai_analysis = await self._get_ai_analysis(
                symbol, asset_class, timeframe, indicators, structure, tf_data
            )
            
            analysis_results[timeframe.value] = {
                "indicators": indicators.__dict__ if indicators else {},
                "market_structure": structure.__dict__ if structure else {},
                "price_action": price_action,
                "volume_analysis": volume_analysis,
                "ai_analysis": ai_analysis,
                "confidence_score": self._calculate_confidence_score(indicators, structure),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Multi-timeframe consensus
        consensus = await self._build_timeframe_consensus(analysis_results)
        
        return {
            "symbol": symbol,
            "asset_class": asset_class.value,
            "timeframe_analysis": analysis_results,
            "consensus": consensus,
            "overall_confidence": consensus.get("confidence", 0.0),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _calculate_advanced_indicators(
        self,
        data: Dict[str, Any],
        symbol: str
    ) -> Optional[TechnicalIndicators]:
        """Calculate comprehensive technical indicators"""
        
        try:
            prices = data.get("prices", [])
            volumes = data.get("volumes", [])
            
            if len(prices) < 200:  # Need sufficient data
                return None
            
            df = pd.DataFrame({
                'close': [p['close'] for p in prices],
                'high': [p['high'] for p in prices],
                'low': [p['low'] for p in prices],
                'open': [p['open'] for p in prices],
                'volume': volumes if volumes else [0] * len(prices)
            })
            
            # RSI
            rsi = self._calculate_rsi(df['close'])
            
            # MACD
            macd, macd_signal, macd_hist = self._calculate_macd(df['close'])
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'])
            
            # Moving Averages
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            sma_200 = df['close'].rolling(200).mean().iloc[-1]
            
            # EMAs
            ema_12 = df['close'].ewm(span=12).mean().iloc[-1]
            ema_26 = df['close'].ewm(span=26).mean().iloc[-1]
            
            # Stochastic
            stoch_k, stoch_d = self._calculate_stochastic(df)
            
            # ADX
            adx = self._calculate_adx(df)
            
            # ATR
            atr = self._calculate_atr(df)
            
            # Volume indicators
            volume_sma = df['volume'].rolling(20).mean().iloc[-1]
            obv = self._calculate_obv(df)
            
            # Williams %R
            williams_r = self._calculate_williams_r(df)
            
            # CCI
            cci = self._calculate_cci(df)
            
            return TechnicalIndicators(
                rsi=rsi,
                macd=macd,
                macd_signal=macd_signal,
                macd_histogram=macd_hist,
                bb_upper=bb_upper,
                bb_middle=bb_middle,
                bb_lower=bb_lower,
                sma_20=sma_20,
                sma_50=sma_50,
                sma_200=sma_200,
                ema_12=ema_12,
                ema_26=ema_26,
                stoch_k=stoch_k,
                stoch_d=stoch_d,
                adx=adx,
                atr=atr,
                volume_sma=volume_sma,
                obv=obv,
                williams_r=williams_r,
                cci=cci
            )
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            return None
    
    async def _analyze_market_structure(
        self,
        data: Dict[str, Any],
        indicators: TechnicalIndicators
    ) -> Optional[MarketStructure]:
        """Analyze market structure and key levels"""
        
        try:
            prices = data.get("prices", [])
            if not prices or not indicators:
                return None
            
            highs = [p['high'] for p in prices[-100:]]  # Last 100 bars
            lows = [p['low'] for p in prices[-100:]]
            closes = [p['close'] for p in prices[-100:]]
            
            # Trend analysis
            trend_direction = self._determine_trend_direction(indicators)
            trend_strength = self._calculate_trend_strength(indicators)
            
            # Support/Resistance levels
            support_levels = self._find_support_levels(lows)
            resistance_levels = self._find_resistance_levels(highs)
            
            # Key levels (psychological, pivot points)
            key_levels = self._identify_key_levels(prices)
            
            # Market phase
            market_phase = self._determine_market_phase(indicators, trend_strength)
            
            # Volatility regime
            volatility_regime = self._assess_volatility_regime(indicators.atr, closes)
            
            # Volume profile
            volume_profile = self._analyze_volume_profile(data)
            
            return MarketStructure(
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                key_levels=key_levels,
                market_phase=market_phase,
                volatility_regime=volatility_regime,
                volume_profile=volume_profile
            )
            
        except Exception as e:
            logger.error(f"Error analyzing market structure: {e}")
            return None
    
    async def _get_ai_analysis(
        self,
        symbol: str,
        asset_class: AssetClass,
        timeframe: TimeFrame,
        indicators: TechnicalIndicators,
        structure: MarketStructure,
        raw_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get AI-powered analysis from OpenAI"""
        
        try:
            # Prepare comprehensive prompt
            analysis_prompt = self._build_professional_analysis_prompt(
                symbol, asset_class, timeframe, indicators, structure, raw_data
            )
            
            # Call OpenAI
            response = await self.llm_client.analyze_market_data(
                market_data={
                    "symbol": symbol,
                    "asset_class": asset_class.value,
                    "timeframe": timeframe.value,
                    "indicators": indicators.__dict__ if indicators else {},
                    "structure": structure.__dict__ if structure else {},
                    "raw_data": raw_data
                },
                context=analysis_prompt
            )
            
            if "error" in response:
                return {"error": response["error"]}
            
            return json.loads(response["analysis"])
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            return {"error": str(e)}
    
    def _build_professional_analysis_prompt(
        self,
        symbol: str,
        asset_class: AssetClass,
        timeframe: TimeFrame,
        indicators: TechnicalIndicators,
        structure: MarketStructure,
        raw_data: Dict[str, Any]
    ) -> str:
        """Build comprehensive analysis prompt for professional trading"""
        
        return f"""
        PROFESSIONAL TRADING ANALYSIS REQUEST
        
        Asset: {symbol} ({asset_class.value.upper()})
        Timeframe: {timeframe.value}
        Analysis Timestamp: {datetime.utcnow().isoformat()}
        
        TECHNICAL INDICATORS:
        - RSI: {indicators.rsi:.2f}
        - MACD: {indicators.macd:.4f} | Signal: {indicators.macd_signal:.4f}
        - Bollinger Bands: {indicators.bb_lower:.2f} - {indicators.bb_middle:.2f} - {indicators.bb_upper:.2f}
        - Moving Averages: SMA20({indicators.sma_20:.2f}) SMA50({indicators.sma_50:.2f}) SMA200({indicators.sma_200:.2f})
        - ADX: {indicators.adx:.2f} | ATR: {indicators.atr:.4f}
        - Stochastic: K({indicators.stoch_k:.2f}) D({indicators.stoch_d:.2f})
        
        MARKET STRUCTURE:
        - Trend: {structure.trend_direction} (Strength: {structure.trend_strength:.2f})
        - Market Phase: {structure.market_phase}
        - Volatility Regime: {structure.volatility_regime}
        - Support Levels: {structure.support_levels}
        - Resistance Levels: {structure.resistance_levels}
        
        ANALYSIS REQUIREMENTS:
        1. Provide professional trading recommendation (BUY/SELL/HOLD)
        2. Include confidence score (0-1)
        3. Define entry price, stop loss, and take profit levels
        4. Specify position sizing recommendation
        5. Identify key risk factors
        6. Provide time horizon for the trade
        7. Include market context and rationale
        
        Respond in JSON format with professional trading analysis.
        """
    
    # Technical indicator calculation methods
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[float, float, float]:
        """Calculate MACD"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        return macd.iloc[-1], signal.iloc[-1], histogram.iloc[-1]
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper.iloc[-1], sma.iloc[-1], lower.iloc[-1]
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14) -> Tuple[float, float]:
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=3).mean()
        return k_percent.iloc[-1], d_percent.iloc[-1]
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX"""
        # Simplified ADX calculation
        high_diff = df['high'].diff()
        low_diff = df['low'].diff().abs()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        return atr.iloc[-1]
    
    def _calculate_obv(self, df: pd.DataFrame) -> float:
        """Calculate On-Balance Volume"""
        obv = (df['volume'] * df['close'].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)).cumsum()
        return obv.iloc[-1]
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Williams %R"""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        williams_r = -100 * ((high_max - df['close']) / (high_max - low_min))
        return williams_r.iloc[-1]
    
    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate Commodity Channel Index"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: pd.Series(x).mad())
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci.iloc[-1]
    
    # Market structure analysis methods
    def _determine_trend_direction(self, indicators: TechnicalIndicators) -> str:
        """Determine overall trend direction"""
        if indicators.sma_20 > indicators.sma_50 > indicators.sma_200:
            return "bullish"
        elif indicators.sma_20 < indicators.sma_50 < indicators.sma_200:
            return "bearish"
        else:
            return "sideways"
    
    def _calculate_trend_strength(self, indicators: TechnicalIndicators) -> float:
        """Calculate trend strength score"""
        adx_strength = min(indicators.adx / 50, 1.0)  # Normalize ADX
        ma_alignment = self._calculate_ma_alignment_score(indicators)
        return (adx_strength + ma_alignment) / 2
    
    def _calculate_ma_alignment_score(self, indicators: TechnicalIndicators) -> float:
        """Calculate moving average alignment score"""
        mas = [indicators.sma_20, indicators.sma_50, indicators.sma_200]
        
        # Check if MAs are aligned (ascending or descending)
        ascending = all(mas[i] <= mas[i+1] for i in range(len(mas)-1))
        descending = all(mas[i] >= mas[i+1] for i in range(len(mas)-1))
        
        if ascending or descending:
            return 1.0
        else:
            return 0.5
    
    def _find_support_levels(self, lows: List[float]) -> List[float]:
        """Find significant support levels"""
        # Simplified support level detection
        supports = []
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1] and lows[i] < lows[i-2] and lows[i] < lows[i+2]:
                supports.append(lows[i])
        
        # Return top 3 most significant levels
        return sorted(set(supports))[:3]
    
    def _find_resistance_levels(self, highs: List[float]) -> List[float]:
        """Find significant resistance levels"""
        # Simplified resistance level detection
        resistances = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1] and highs[i] > highs[i-2] and highs[i] > highs[i+2]:
                resistances.append(highs[i])
        
        # Return top 3 most significant levels
        return sorted(set(resistances), reverse=True)[:3]
    
    def _identify_key_levels(self, prices: List[Dict[str, float]]) -> List[float]:
        """Identify key psychological and pivot levels"""
        current_price = prices[-1]['close']
        
        # Psychological levels (round numbers)
        key_levels = []
        base = int(current_price / 10) * 10
        
        for level in [base - 20, base - 10, base, base + 10, base + 20]:
            if level > 0:
                key_levels.append(level)
        
        return key_levels
    
    def _determine_market_phase(self, indicators: TechnicalIndicators, trend_strength: float) -> str:
        """Determine current market phase"""
        if trend_strength > 0.7:
            if indicators.sma_20 > indicators.sma_50:
                return "strong_uptrend"
            else:
                return "strong_downtrend"
        elif trend_strength > 0.4:
            return "trending"
        else:
            return "ranging"
    
    def _assess_volatility_regime(self, atr: float, closes: List[float]) -> str:
        """Assess current volatility regime"""
        if not closes:
            return "unknown"
        
        current_price = closes[-1]
        volatility_pct = (atr / current_price) * 100
        
        if volatility_pct > 3:
            return "high_volatility"
        elif volatility_pct > 1.5:
            return "medium_volatility"
        else:
            return "low_volatility"
    
    def _analyze_volume_profile(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze volume profile"""
        volumes = data.get("volumes", [])
        if not volumes:
            return {"volume_trend": 0.0, "volume_strength": 0.0}
        
        recent_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else 0
        avg_volume = np.mean(volumes) if volumes else 0
        
        volume_trend = (recent_volume - avg_volume) / avg_volume if avg_volume > 0 else 0
        volume_strength = min(recent_volume / avg_volume, 3.0) if avg_volume > 0 else 1.0
        
        return {
            "volume_trend": volume_trend,
            "volume_strength": volume_strength
        }
    
    async def _analyze_price_action(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze price action patterns"""
        prices = data.get("prices", [])
        if len(prices) < 5:
            return {"patterns": [], "strength": 0.0}
        
        # Simplified pattern recognition
        patterns = []
        
        # Check for doji patterns
        recent_candles = prices[-5:]
        for candle in recent_candles:
            body_size = abs(candle['close'] - candle['open'])
            wick_size = candle['high'] - candle['low']
            
            if body_size < (wick_size * 0.1):
                patterns.append("doji")
        
        # Check for engulfing patterns
        if len(prices) >= 2:
            prev_candle = prices[-2]
            curr_candle = prices[-1]
            
            if (prev_candle['close'] < prev_candle['open'] and 
                curr_candle['close'] > curr_candle['open'] and
                curr_candle['close'] > prev_candle['open'] and
                curr_candle['open'] < prev_candle['close']):
                patterns.append("bullish_engulfing")
            
            elif (prev_candle['close'] > prev_candle['open'] and 
                  curr_candle['close'] < curr_candle['open'] and
                  curr_candle['close'] < prev_candle['open'] and
                  curr_candle['open'] > prev_candle['close']):
                patterns.append("bearish_engulfing")
        
        return {
            "patterns": patterns,
            "strength": len(patterns) / 5.0  # Normalize pattern strength
        }
    
    async def _analyze_volume(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volume characteristics"""
        volumes = data.get("volumes", [])
        prices = data.get("prices", [])
        
        if not volumes or not prices:
            return {"volume_trend": "neutral", "volume_confirmation": False}
        
        # Volume trend analysis
        recent_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else 0
        avg_volume = np.mean(volumes) if volumes else 0
        
        volume_trend = "increasing" if recent_volume > avg_volume * 1.2 else \
                      "decreasing" if recent_volume < avg_volume * 0.8 else "neutral"
        
        # Volume-price confirmation
        price_direction = "up" if prices[-1]['close'] > prices[-5]['close'] else "down"
        volume_confirmation = (volume_trend == "increasing" and price_direction == "up") or \
                             (volume_trend == "decreasing" and price_direction == "down")
        
        return {
            "volume_trend": volume_trend,
            "volume_confirmation": volume_confirmation,
            "relative_volume": recent_volume / avg_volume if avg_volume > 0 else 1.0
        }
    
    async def _build_timeframe_consensus(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Build consensus across multiple timeframes"""
        if not analysis_results:
            return {"consensus": "neutral", "confidence": 0.0}
        
        # Collect signals from all timeframes
        signals = []
        confidences = []
        
        for tf, result in analysis_results.items():
            ai_analysis = result.get("ai_analysis", {})
            if "recommendation" in ai_analysis:
                signals.append(ai_analysis["recommendation"])
                confidences.append(result.get("confidence_score", 0.0))
        
        if not signals:
            return {"consensus": "neutral", "confidence": 0.0}
        
        # Calculate consensus
        buy_signals = signals.count("BUY")
        sell_signals = signals.count("SELL")
        hold_signals = signals.count("HOLD")
        
        total_signals = len(signals)
        avg_confidence = np.mean(confidences)
        
        if buy_signals > sell_signals and buy_signals > hold_signals:
            consensus = "BUY"
            strength = buy_signals / total_signals
        elif sell_signals > buy_signals and sell_signals > hold_signals:
            consensus = "SELL"
            strength = sell_signals / total_signals
        else:
            consensus = "HOLD"
            strength = hold_signals / total_signals
        
        return {
            "consensus": consensus,
            "strength": strength,
            "confidence": avg_confidence * strength,
            "signal_distribution": {
                "buy": buy_signals,
                "sell": sell_signals,
                "hold": hold_signals
            }
        }
    
    def _calculate_confidence_score(
        self,
        indicators: Optional[TechnicalIndicators],
        structure: Optional[MarketStructure]
    ) -> float:
        """Calculate overall confidence score"""
        if not indicators or not structure:
            return 0.0
        
        # Base confidence
        confidence = 0.5
        
        # Trend strength contribution
        confidence += structure.trend_strength * 0.3
        
        # Technical indicator alignment
        if 30 < indicators.rsi < 70:  # Not extreme
            confidence += 0.1
        
        if abs(indicators.macd - indicators.macd_signal) > 0.001:  # Clear MACD signal
            confidence += 0.1
        
        # Volume confirmation
        if structure.volume_profile.get("volume_strength", 0) > 1.2:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Main analysis method"""
        symbol = data.get("symbol", "UNKNOWN")
        asset_class_str = data.get("asset_class", "stocks")
        timeframes_str = data.get("timeframes", ["1h", "4h", "1d"])
        
        try:
            asset_class = AssetClass(asset_class_str)
            timeframes = [TimeFrame(tf) for tf in timeframes_str]
            
            return await self.analyze_multi_timeframe(
                symbol=symbol,
                asset_class=asset_class,
                timeframes=timeframes,
                market_data=data.get("market_data", {})
            )
            
        except Exception as e:
            logger.error(f"Error in advanced market analysis: {e}")
            return {
                "error": str(e),
                "symbol": symbol,
                "confidence": 0.0,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate professional trading recommendations"""
        if "error" in analysis:
            return []
        
        recommendations = []
        consensus = analysis.get("consensus", {})
        
        if consensus.get("confidence", 0) > self.min_confidence_threshold:
            recommendation = {
                "symbol": analysis.get("symbol"),
                "action": consensus.get("consensus", "HOLD"),
                "confidence": consensus.get("confidence", 0.0),
                "timeframe_analysis": analysis.get("timeframe_analysis", {}),
                "reasoning": self._build_professional_reasoning(analysis),
                "risk_assessment": self._assess_trade_risk(analysis),
                "position_sizing": self._calculate_position_size(analysis),
                "trade_management": self._build_trade_management_plan(analysis)
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def _build_professional_reasoning(self, analysis: Dict[str, Any]) -> str:
        """Build professional reasoning for recommendations"""
        consensus = analysis.get("consensus", {})
        signal_dist = consensus.get("signal_distribution", {})
        
        reasoning = f"Multi-timeframe analysis shows {consensus.get('consensus', 'HOLD')} consensus "
        reasoning += f"with {consensus.get('confidence', 0):.1%} confidence. "
        reasoning += f"Signal distribution: {signal_dist.get('buy', 0)} BUY, "
        reasoning += f"{signal_dist.get('sell', 0)} SELL, {signal_dist.get('hold', 0)} HOLD across timeframes."
        
        return reasoning
    
    def _assess_trade_risk(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess trade risk factors"""
        # Simplified risk assessment
        return {
            "risk_level": "medium",
            "key_risks": ["market_volatility", "trend_reversal"],
            "risk_score": 0.5
        }
    
    def _calculate_position_size(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate recommended position size"""
        confidence = analysis.get("consensus", {}).get("confidence", 0.0)
        
        # Kelly criterion approximation
        base_size = 0.02  # 2% base position
        adjusted_size = base_size * confidence
        
        return {
            "recommended_size": adjusted_size,
            "max_size": 0.05,  # 5% maximum
            "sizing_method": "confidence_adjusted"
        }
    
    def _build_trade_management_plan(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive trade management plan"""
        return {
            "entry_strategy": "market_order",
            "exit_strategy": "stop_loss_take_profit",
            "monitoring_frequency": "hourly",
            "adjustment_triggers": ["trend_change", "volume_divergence"]
        }