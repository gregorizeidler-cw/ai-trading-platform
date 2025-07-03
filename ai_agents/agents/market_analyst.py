"""
Market Analyst Agent - Specialized in technical and fundamental analysis.
"""

import json
from typing import Dict, List, Any
from datetime import datetime
from loguru import logger
from .base_agent import BaseAgent


class MarketAnalystAgent(BaseAgent):
    """Agent specialized in market analysis using OpenAI GPT"""
    
    def __init__(self):
        super().__init__(
            name="Market Analyst",
            description="Specialized in technical and fundamental market analysis"
        )
        self.analysis_types = [
            "technical_analysis",
            "fundamental_analysis", 
            "trend_analysis",
            "support_resistance",
            "volume_analysis"
        ]
    
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive market analysis"""
        try:
            market_data = data.get("market_data", {})
            technical_indicators = data.get("technical_indicators", {})
            context = data.get("context", "")
            
            # Use OpenAI to analyze market data
            analysis_result = await self.llm_client.analyze_market_data(
                market_data=market_data,
                context=f"Technical Indicators: {technical_indicators}. Context: {context}"
            )
            
            if "error" in analysis_result:
                raise Exception(analysis_result["error"])
            
            # Parse the JSON response
            analysis_json = json.loads(analysis_result["analysis"])
            
            # Add our specialized analysis
            enhanced_analysis = {
                **analysis_json,
                "technical_strength": self._calculate_technical_strength(technical_indicators),
                "market_phase": self._determine_market_phase(market_data),
                "volatility_assessment": self._assess_volatility(market_data),
                "confidence_score": self._calculate_confidence_score(analysis_json, technical_indicators),
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "tokens_used": analysis_result.get("tokens_used", 0)
            }
            
            return enhanced_analysis
            
        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            return {
                "error": str(e),
                "confidence_score": 0.0,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading recommendations based on analysis"""
        recommendations = []
        
        if "error" in analysis:
            return recommendations
        
        try:
            symbol_analysis = analysis.get("symbol_analysis", {})
            
            for symbol, symbol_data in symbol_analysis.items():
                recommendation = {
                    "symbol": symbol,
                    "action": symbol_data.get("recommendation", "hold").upper(),
                    "confidence": symbol_data.get("confidence_score", 0.0),
                    "reasoning": self._build_reasoning(symbol_data, analysis),
                    "technical_score": self._calculate_technical_score(symbol_data),
                    "risk_level": self._assess_risk_level(symbol_data),
                    "time_horizon": self._determine_time_horizon(symbol_data),
                    "entry_conditions": self._get_entry_conditions(symbol_data),
                    "exit_conditions": self._get_exit_conditions(symbol_data)
                }
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _calculate_technical_strength(self, indicators: Dict[str, Any]) -> float:
        """Calculate overall technical strength score"""
        if not indicators:
            return 0.5
        
        strength_score = 0.0
        factors = 0
        
        # RSI analysis
        if "rsi" in indicators:
            rsi = indicators["rsi"]
            if rsi < 30:
                strength_score += 0.8  # Oversold - potential buy
            elif rsi > 70:
                strength_score += 0.2  # Overbought - potential sell
            else:
                strength_score += 0.5  # Neutral
            factors += 1
        
        # MACD analysis
        if "macd" in indicators and "macd_signal" in indicators:
            macd = indicators["macd"]
            signal = indicators["macd_signal"]
            if macd > signal:
                strength_score += 0.7  # Bullish
            else:
                strength_score += 0.3  # Bearish
            factors += 1
        
        # Moving averages
        if "sma_20" in indicators and "sma_50" in indicators:
            sma_20 = indicators["sma_20"]
            sma_50 = indicators["sma_50"]
            if sma_20 > sma_50:
                strength_score += 0.6  # Bullish
            else:
                strength_score += 0.4  # Bearish
            factors += 1
        
        return strength_score / factors if factors > 0 else 0.5
    
    def _determine_market_phase(self, market_data: Dict[str, Any]) -> str:
        """Determine current market phase"""
        if not market_data:
            return "unknown"
        
        # Simple phase determination based on price action
        # In a real implementation, this would be more sophisticated
        recent_prices = market_data.get("recent_prices", [])
        if len(recent_prices) < 2:
            return "unknown"
        
        trend = "sideways"
        if recent_prices[-1] > recent_prices[0] * 1.05:
            trend = "uptrend"
        elif recent_prices[-1] < recent_prices[0] * 0.95:
            trend = "downtrend"
        
        return trend
    
    def _assess_volatility(self, market_data: Dict[str, Any]) -> str:
        """Assess market volatility"""
        volatility = market_data.get("volatility", 0.0)
        
        if volatility > 0.3:
            return "high"
        elif volatility > 0.15:
            return "medium"
        else:
            return "low"
    
    def _calculate_confidence_score(self, analysis: Dict[str, Any], indicators: Dict[str, Any]) -> float:
        """Calculate overall confidence score"""
        base_confidence = 0.5
        
        # Adjust based on data quality
        if indicators:
            base_confidence += 0.2
        
        # Adjust based on market sentiment clarity
        sentiment = analysis.get("overall_market_sentiment", "neutral")
        if sentiment in ["bullish", "bearish"]:
            base_confidence += 0.2
        
        # Adjust based on number of symbols analyzed
        symbol_count = len(analysis.get("symbol_analysis", {}))
        if symbol_count > 0:
            base_confidence += min(0.1 * symbol_count, 0.3)
        
        return min(base_confidence, 1.0)
    
    def _build_reasoning(self, symbol_data: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Build reasoning for recommendation"""
        reasoning_parts = []
        
        trend = symbol_data.get("trend", "unknown")
        strength = symbol_data.get("strength", "unknown")
        
        reasoning_parts.append(f"Technical trend: {trend} with {strength} strength")
        
        if "support_levels" in symbol_data:
            reasoning_parts.append(f"Support levels: {symbol_data['support_levels']}")
        
        if "resistance_levels" in symbol_data:
            reasoning_parts.append(f"Resistance levels: {symbol_data['resistance_levels']}")
        
        market_sentiment = analysis.get("overall_market_sentiment", "neutral")
        reasoning_parts.append(f"Overall market sentiment: {market_sentiment}")
        
        return ". ".join(reasoning_parts)
    
    def _calculate_technical_score(self, symbol_data: Dict[str, Any]) -> float:
        """Calculate technical analysis score"""
        return symbol_data.get("confidence_score", 0.5)
    
    def _assess_risk_level(self, symbol_data: Dict[str, Any]) -> str:
        """Assess risk level for the symbol"""
        confidence = symbol_data.get("confidence_score", 0.5)
        
        if confidence > 0.8:
            return "low"
        elif confidence > 0.6:
            return "medium"
        else:
            return "high"
    
    def _determine_time_horizon(self, symbol_data: Dict[str, Any]) -> str:
        """Determine appropriate time horizon"""
        strength = symbol_data.get("strength", "moderate")
        
        if strength == "strong":
            return "medium"  # 1-3 months
        elif strength == "moderate":
            return "short"   # 1-4 weeks
        else:
            return "short"   # 1-2 weeks
    
    def _get_entry_conditions(self, symbol_data: Dict[str, Any]) -> List[str]:
        """Get entry conditions"""
        conditions = []
        
        if "support_levels" in symbol_data:
            conditions.append(f"Price above support: {symbol_data['support_levels']}")
        
        recommendation = symbol_data.get("recommendation", "hold")
        if recommendation == "buy":
            conditions.append("Bullish technical indicators aligned")
        elif recommendation == "sell":
            conditions.append("Bearish technical indicators aligned")
        
        return conditions
    
    def _get_exit_conditions(self, symbol_data: Dict[str, Any]) -> List[str]:
        """Get exit conditions"""
        conditions = []
        
        if "resistance_levels" in symbol_data:
            conditions.append(f"Take profit near resistance: {symbol_data['resistance_levels']}")
        
        if "support_levels" in symbol_data:
            conditions.append(f"Stop loss below support: {symbol_data['support_levels']}")
        
        return conditions 