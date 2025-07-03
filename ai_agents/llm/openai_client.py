"""
OpenAI client for LLM integration in trading agents.
"""

import asyncio
from typing import Dict, List, Any, Optional
from openai import AsyncOpenAI
from loguru import logger
from config.settings import config


class OpenAIClient:
    """OpenAI client for trading agent LLM operations"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = AsyncOpenAI(
            api_key=api_key or config.api.openai_api_key
        )
        self.model = "gpt-4-turbo-preview"
        self.max_tokens = 4096
        self.temperature = 0.1  # Low temperature for consistent financial analysis
        
    async def analyze_market_data(
        self,
        market_data: Dict[str, Any],
        context: str = ""
    ) -> Dict[str, Any]:
        """Analyze market data using GPT"""
        try:
            prompt = self._build_market_analysis_prompt(market_data, context)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert financial analyst specializing in market data analysis and trading signals."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            analysis = response.choices[0].message.content
            logger.info(f"Market analysis completed for {len(market_data)} symbols")
            
            return {
                "analysis": analysis,
                "model_used": self.model,
                "tokens_used": response.usage.total_tokens
            }
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return {"error": str(e)}
    
    async def generate_trading_signals(
        self,
        technical_indicators: Dict[str, Any],
        market_sentiment: Dict[str, Any],
        risk_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate trading signals based on multiple data sources"""
        try:
            prompt = self._build_trading_signals_prompt(
                technical_indicators,
                market_sentiment,
                risk_parameters
            )
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional trading signal generator. Provide clear BUY/SELL/HOLD signals with confidence scores and reasoning."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            signals = response.choices[0].message.content
            logger.info("Trading signals generated successfully")
            
            return {
                "signals": signals,
                "model_used": self.model,
                "tokens_used": response.usage.total_tokens
            }
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return {"error": str(e)}
    
    async def analyze_news_sentiment(
        self,
        news_data: List[Dict[str, Any]],
        symbols: List[str]
    ) -> Dict[str, Any]:
        """Analyze news sentiment impact on specific symbols"""
        try:
            prompt = self._build_news_sentiment_prompt(news_data, symbols)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial news analyst specializing in sentiment analysis and market impact assessment."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            sentiment_analysis = response.choices[0].message.content
            logger.info(f"News sentiment analysis completed for {len(news_data)} articles")
            
            return {
                "sentiment_analysis": sentiment_analysis,
                "model_used": self.model,
                "tokens_used": response.usage.total_tokens
            }
            
        except Exception as e:
            logger.error(f"Error in news sentiment analysis: {e}")
            return {"error": str(e)}
    
    async def assess_portfolio_risk(
        self,
        portfolio_data: Dict[str, Any],
        market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess portfolio risk and suggest adjustments"""
        try:
            prompt = self._build_risk_assessment_prompt(portfolio_data, market_conditions)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a risk management expert specializing in portfolio optimization and risk assessment."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            risk_assessment = response.choices[0].message.content
            logger.info("Portfolio risk assessment completed")
            
            return {
                "risk_assessment": risk_assessment,
                "model_used": self.model,
                "tokens_used": response.usage.total_tokens
            }
            
        except Exception as e:
            logger.error(f"Error in portfolio risk assessment: {e}")
            return {"error": str(e)}
    
    def _build_market_analysis_prompt(
        self,
        market_data: Dict[str, Any],
        context: str
    ) -> str:
        """Build prompt for market data analysis"""
        return f"""
        Analyze the following market data and provide insights:
        
        Market Data:
        {market_data}
        
        Additional Context:
        {context}
        
        Please provide a JSON response with the following structure:
        {{
            "overall_market_sentiment": "bullish/bearish/neutral",
            "key_insights": ["insight1", "insight2", "insight3"],
            "symbol_analysis": {{
                "SYMBOL": {{
                    "trend": "up/down/sideways",
                    "strength": "strong/weak/moderate",
                    "support_levels": [price1, price2],
                    "resistance_levels": [price1, price2],
                    "recommendation": "buy/sell/hold",
                    "confidence_score": 0.85
                }}
            }},
            "market_outlook": "short_term_outlook",
            "risk_factors": ["factor1", "factor2"]
        }}
        """
    
    def _build_trading_signals_prompt(
        self,
        technical_indicators: Dict[str, Any],
        market_sentiment: Dict[str, Any],
        risk_parameters: Dict[str, Any]
    ) -> str:
        """Build prompt for trading signal generation"""
        return f"""
        Generate trading signals based on the following data:
        
        Technical Indicators:
        {technical_indicators}
        
        Market Sentiment:
        {market_sentiment}
        
        Risk Parameters:
        {risk_parameters}
        
        Please provide a JSON response with trading signals:
        {{
            "signals": [
                {{
                    "symbol": "AAPL",
                    "action": "BUY/SELL/HOLD",
                    "confidence": 0.85,
                    "entry_price": 150.00,
                    "stop_loss": 145.00,
                    "take_profit": 160.00,
                    "position_size": 0.05,
                    "reasoning": "Technical and fundamental analysis reasoning",
                    "time_horizon": "short/medium/long"
                }}
            ],
            "market_conditions": "current market assessment",
            "overall_strategy": "aggressive/conservative/balanced"
        }}
        """
    
    def _build_news_sentiment_prompt(
        self,
        news_data: List[Dict[str, Any]],
        symbols: List[str]
    ) -> str:
        """Build prompt for news sentiment analysis"""
        return f"""
        Analyze the sentiment of the following news articles and their impact on these symbols: {symbols}
        
        News Data:
        {news_data}
        
        Please provide a JSON response with sentiment analysis:
        {{
            "overall_sentiment": "positive/negative/neutral",
            "sentiment_score": 0.75,
            "symbol_impact": {{
                "SYMBOL": {{
                    "sentiment": "positive/negative/neutral",
                    "impact_score": 0.85,
                    "key_themes": ["theme1", "theme2"],
                    "potential_price_impact": "up/down/neutral",
                    "time_sensitivity": "immediate/short_term/long_term"
                }}
            }},
            "key_events": ["event1", "event2"],
            "market_moving_news": true/false
        }}
        """
    
    def _build_risk_assessment_prompt(
        self,
        portfolio_data: Dict[str, Any],
        market_conditions: Dict[str, Any]
    ) -> str:
        """Build prompt for portfolio risk assessment"""
        return f"""
        Assess the risk profile of the following portfolio given current market conditions:
        
        Portfolio Data:
        {portfolio_data}
        
        Market Conditions:
        {market_conditions}
        
        Please provide a JSON response with risk assessment:
        {{
            "overall_risk_level": "low/medium/high",
            "risk_score": 0.65,
            "diversification_score": 0.80,
            "concentration_risks": ["risk1", "risk2"],
            "recommended_adjustments": [
                {{
                    "action": "reduce/increase/rebalance",
                    "asset": "symbol or asset class",
                    "percentage": 0.10,
                    "reasoning": "explanation"
                }}
            ],
            "stress_test_results": {{
                "market_crash_scenario": -0.25,
                "sector_rotation_impact": -0.15,
                "interest_rate_sensitivity": -0.10
            }},
            "hedging_suggestions": ["suggestion1", "suggestion2"]
        }}
        """ 