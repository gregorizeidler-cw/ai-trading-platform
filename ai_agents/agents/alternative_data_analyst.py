"""
Alternative Data Analyst Agent
Incorporates sentiment analysis, social media, and economic indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import yfinance as yf
from dataclasses import dataclass
from loguru import logger
import requests
import json

from .base_agent import BaseAgent
from ..llm.openai_client import OpenAIClient


@dataclass
class AlternativeDataSignal:
    """Alternative data signal"""
    data_type: str
    signal_strength: float  # -1 to 1
    confidence: float  # 0 to 1
    source: str
    timestamp: datetime
    metadata: Dict[str, Any]


class AlternativeDataAnalyst(BaseAgent):
    """Analyzes alternative data sources for trading signals"""
    
    def __init__(self, openai_client: OpenAIClient):
        super().__init__(
            name="Alternative Data Analyst",
            description="Analyzes sentiment, social media, and economic indicators for trading signals",
            openai_client=openai_client
        )
        
        self.sentiment_cache = {}
        self.economic_indicators = {}
        
    async def analyze_market_sentiment(self, symbol: str) -> AlternativeDataSignal:
        """Analyze market sentiment from multiple sources"""
        try:
            # Get news sentiment
            news_sentiment = await self._analyze_news_sentiment(symbol)
            
            # Get social media sentiment (simulated)
            social_sentiment = await self._analyze_social_sentiment(symbol)
            
            # Get options sentiment
            options_sentiment = await self._analyze_options_sentiment(symbol)
            
            # Combine sentiments
            combined_sentiment = (
                0.4 * news_sentiment['sentiment'] +
                0.3 * social_sentiment['sentiment'] +
                0.3 * options_sentiment['sentiment']
            )
            
            # Calculate confidence
            confidence = np.mean([
                news_sentiment['confidence'],
                social_sentiment['confidence'],
                options_sentiment['confidence']
            ])
            
            return AlternativeDataSignal(
                data_type="market_sentiment",
                signal_strength=combined_sentiment,
                confidence=confidence,
                source="news_social_options",
                timestamp=datetime.now(),
                metadata={
                    'news_sentiment': news_sentiment,
                    'social_sentiment': social_sentiment,
                    'options_sentiment': options_sentiment,
                    'symbol': symbol
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {e}")
            return AlternativeDataSignal(
                data_type="market_sentiment",
                signal_strength=0.0,
                confidence=0.0,
                source="error",
                timestamp=datetime.now(),
                metadata={'error': str(e)}
            )
    
    async def _analyze_news_sentiment(self, symbol: str) -> Dict[str, float]:
        """Analyze news sentiment for a symbol"""
        try:
            # Get recent news
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                return {'sentiment': 0.0, 'confidence': 0.0, 'article_count': 0}
            
            # Analyze sentiment of headlines using AI
            headlines = [article.get('title', '') for article in news[:10]]
            
            sentiment_scores = []
            for headline in headlines:
                if headline:
                    score = await self._analyze_text_sentiment(headline)
                    sentiment_scores.append(score)
            
            if not sentiment_scores:
                return {'sentiment': 0.0, 'confidence': 0.0, 'article_count': 0}
            
            avg_sentiment = np.mean(sentiment_scores)
            confidence = min(len(sentiment_scores) / 10, 1.0)  # More articles = higher confidence
            
            return {
                'sentiment': avg_sentiment,
                'confidence': confidence,
                'article_count': len(sentiment_scores)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            return {'sentiment': 0.0, 'confidence': 0.0, 'article_count': 0}
    
    async def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using AI"""
        try:
            prompt = f"""
            Analyze the sentiment of this financial news headline on a scale from -1 (very negative) to +1 (very positive):
            
            "{text}"
            
            Consider:
            - Impact on stock price
            - Market implications
            - Investor sentiment
            
            Respond with only a number between -1 and 1.
            """
            
            response = await self.openai_client.generate_completion(
                prompt=prompt,
                max_tokens=10,
                temperature=0.1
            )
            
            # Parse the response
            try:
                sentiment = float(response.strip())
                return max(min(sentiment, 1.0), -1.0)
            except:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {e}")
            return 0.0
    
    async def _analyze_social_sentiment(self, symbol: str) -> Dict[str, float]:
        """Analyze social media sentiment (simulated)"""
        try:
            # This would integrate with Twitter API, Reddit API, etc.
            # For now, we'll simulate based on recent price action
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            
            if hist.empty:
                return {'sentiment': 0.0, 'confidence': 0.0, 'mentions': 0}
            
            # Simple proxy: recent price performance
            recent_return = hist['Close'].pct_change(3).iloc[-1]
            
            # Convert to sentiment score
            sentiment = np.tanh(recent_return * 10)  # Normalize to [-1, 1]
            
            # Simulate confidence and mentions
            confidence = 0.6  # Moderate confidence for simulated data
            mentions = np.random.randint(50, 500)  # Simulated mention count
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'mentions': mentions
            }
            
        except Exception as e:
            logger.error(f"Error analyzing social sentiment: {e}")
            return {'sentiment': 0.0, 'confidence': 0.0, 'mentions': 0}
    
    async def _analyze_options_sentiment(self, symbol: str) -> Dict[str, float]:
        """Analyze options sentiment (put/call ratio)"""
        try:
            # Get options data
            ticker = yf.Ticker(symbol)
            
            # Get options chain
            options_dates = ticker.options
            
            if not options_dates:
                return {'sentiment': 0.0, 'confidence': 0.0, 'put_call_ratio': 1.0}
            
            # Get first available options date
            first_date = options_dates[0]
            options_chain = ticker.option_chain(first_date)
            
            calls = options_chain.calls
            puts = options_chain.puts
            
            if calls.empty or puts.empty:
                return {'sentiment': 0.0, 'confidence': 0.0, 'put_call_ratio': 1.0}
            
            # Calculate put/call ratio by volume
            call_volume = calls['volume'].fillna(0).sum()
            put_volume = puts['volume'].fillna(0).sum()
            
            if call_volume == 0:
                put_call_ratio = 2.0  # High put activity
            else:
                put_call_ratio = put_volume / call_volume
            
            # Convert to sentiment (low P/C ratio = bullish)
            # Typical P/C ratio is around 0.7-1.0
            if put_call_ratio < 0.7:
                sentiment = 0.5  # Bullish
            elif put_call_ratio > 1.3:
                sentiment = -0.5  # Bearish
            else:
                sentiment = 0.0  # Neutral
            
            confidence = min((call_volume + put_volume) / 1000, 1.0)
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'put_call_ratio': put_call_ratio
            }
            
        except Exception as e:
            logger.error(f"Error analyzing options sentiment: {e}")
            return {'sentiment': 0.0, 'confidence': 0.0, 'put_call_ratio': 1.0}
    
    async def analyze_economic_indicators(self) -> Dict[str, Any]:
        """Analyze key economic indicators"""
        try:
            # Get economic data (using proxy indicators from market data)
            indicators = await self._get_economic_proxy_data()
            
            # Analyze each indicator
            analysis = {}
            
            for indicator, data in indicators.items():
                if not data.empty:
                    # Calculate trend
                    recent_change = data['Close'].pct_change(20).iloc[-1]  # 20-day change
                    
                    # Normalize to signal strength
                    signal_strength = np.tanh(recent_change * 5)
                    
                    # Calculate confidence based on data quality
                    confidence = min(len(data) / 100, 1.0)
                    
                    analysis[indicator] = {
                        'signal_strength': signal_strength,
                        'recent_change': recent_change,
                        'confidence': confidence,
                        'current_level': data['Close'].iloc[-1],
                        'trend': 'positive' if recent_change > 0.01 else 'negative' if recent_change < -0.01 else 'neutral'
                    }
            
            # Generate overall economic sentiment
            if analysis:
                overall_sentiment = np.mean([a['signal_strength'] for a in analysis.values()])
                overall_confidence = np.mean([a['confidence'] for a in analysis.values()])
            else:
                overall_sentiment = 0.0
                overall_confidence = 0.0
            
            return {
                'overall_sentiment': overall_sentiment,
                'overall_confidence': overall_confidence,
                'indicators': analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing economic indicators: {e}")
            return {'overall_sentiment': 0.0, 'overall_confidence': 0.0, 'indicators': {}}
    
    async def _get_economic_proxy_data(self) -> Dict[str, pd.DataFrame]:
        """Get economic proxy data from market instruments"""
        try:
            # Economic proxy symbols
            proxies = {
                'yield_curve': '^TNX',  # 10-year Treasury
                'dollar_strength': 'DX-Y.NYB',  # Dollar Index
                'inflation_expectations': 'TIPS',  # TIPS ETF
                'credit_risk': 'HYG',  # High Yield Bond ETF
                'volatility': '^VIX',  # VIX
                'commodities': 'DJP'  # Commodity ETF
            }
            
            data = {}
            
            for indicator, symbol in proxies.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="6mo")
                    
                    if not hist.empty:
                        data[indicator] = hist
                        
                except Exception as e:
                    logger.warning(f"Could not get data for {symbol}: {e}")
                    continue
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting economic proxy data: {e}")
            return {}
    
    async def analyze_sector_rotation(self) -> Dict[str, Any]:
        """Analyze sector rotation patterns"""
        try:
            # Sector ETFs
            sectors = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financials': 'XLF',
                'Energy': 'XLE',
                'Industrials': 'XLI',
                'Consumer Discretionary': 'XLY',
                'Consumer Staples': 'XLP',
                'Utilities': 'XLU',
                'Real Estate': 'XLRE',
                'Materials': 'XLB',
                'Communication': 'XLC'
            }
            
            # Get sector data
            sector_data = {}
            for sector, symbol in sectors.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="3mo")
                    
                    if not hist.empty:
                        sector_data[sector] = hist
                        
                except Exception as e:
                    continue
            
            # Calculate relative performance
            sector_performance = {}
            
            for sector, data in sector_data.items():
                # Calculate various timeframe returns
                returns_1w = data['Close'].pct_change(5).iloc[-1]
                returns_1m = data['Close'].pct_change(21).iloc[-1]
                returns_3m = data['Close'].pct_change(63).iloc[-1]
                
                # Calculate momentum score
                momentum = 0.5 * returns_1w + 0.3 * returns_1m + 0.2 * returns_3m
                
                # Calculate volatility
                volatility = data['Close'].pct_change().rolling(20).std().iloc[-1]
                
                sector_performance[sector] = {
                    'returns_1w': returns_1w,
                    'returns_1m': returns_1m,
                    'returns_3m': returns_3m,
                    'momentum_score': momentum,
                    'volatility': volatility,
                    'risk_adjusted_return': momentum / volatility if volatility > 0 else 0
                }
            
            # Rank sectors
            sorted_sectors = sorted(
                sector_performance.items(),
                key=lambda x: x[1]['momentum_score'],
                reverse=True
            )
            
            # Identify rotation patterns
            top_sectors = [s[0] for s in sorted_sectors[:3]]
            bottom_sectors = [s[0] for s in sorted_sectors[-3:]]
            
            return {
                'sector_performance': sector_performance,
                'top_performing_sectors': top_sectors,
                'bottom_performing_sectors': bottom_sectors,
                'rotation_signal': 'risk_on' if 'Technology' in top_sectors else 'risk_off',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sector rotation: {e}")
            return {}
    
    async def generate_alternative_data_report(self, symbols: List[str]) -> Dict[str, Any]:
        """Generate comprehensive alternative data report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'symbols_analyzed': symbols,
                'sentiment_analysis': {},
                'economic_indicators': await self.analyze_economic_indicators(),
                'sector_rotation': await self.analyze_sector_rotation(),
                'overall_market_sentiment': 0.0,
                'confidence_score': 0.0
            }
            
            # Analyze sentiment for each symbol
            sentiment_scores = []
            confidences = []
            
            for symbol in symbols:
                sentiment_signal = await self.analyze_market_sentiment(symbol)
                report['sentiment_analysis'][symbol] = {
                    'signal_strength': sentiment_signal.signal_strength,
                    'confidence': sentiment_signal.confidence,
                    'metadata': sentiment_signal.metadata
                }
                
                sentiment_scores.append(sentiment_signal.signal_strength)
                confidences.append(sentiment_signal.confidence)
            
            # Calculate overall metrics
            if sentiment_scores:
                report['overall_market_sentiment'] = np.mean(sentiment_scores)
                report['confidence_score'] = np.mean(confidences)
            
            # Generate AI summary
            report['ai_summary'] = await self._generate_alternative_data_summary(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating alternative data report: {e}")
            return {'error': str(e)}
    
    async def _generate_alternative_data_summary(self, report: Dict[str, Any]) -> str:
        """Generate AI summary of alternative data analysis"""
        try:
            prompt = f"""
            Summarize the following alternative data analysis:
            
            Overall Market Sentiment: {report['overall_market_sentiment']:.2f}
            Confidence Score: {report['confidence_score']:.2f}
            
            Economic Indicators:
            - Overall Economic Sentiment: {report['economic_indicators'].get('overall_sentiment', 0):.2f}
            - Confidence: {report['economic_indicators'].get('overall_confidence', 0):.2f}
            
            Sector Rotation:
            - Top Sectors: {report['sector_rotation'].get('top_performing_sectors', [])}
            - Bottom Sectors: {report['sector_rotation'].get('bottom_performing_sectors', [])}
            - Rotation Signal: {report['sector_rotation'].get('rotation_signal', 'neutral')}
            
            Provide a concise summary (3-4 sentences) focusing on:
            1. Overall market sentiment
            2. Key economic themes
            3. Sector rotation implications
            4. Trading implications
            """
            
            response = await self.openai_client.generate_completion(
                prompt=prompt,
                max_tokens=200,
                temperature=0.3
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating alternative data summary: {e}")
            return "Alternative data analysis completed with mixed signals across sentiment and economic indicators." 