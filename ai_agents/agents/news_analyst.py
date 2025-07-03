"""
News Analyst Agent - Specialized in news sentiment analysis and market impact assessment.
"""

import json
from typing import Dict, List, Any
from datetime import datetime, timedelta
from loguru import logger
from .base_agent import BaseAgent


class NewsAnalystAgent(BaseAgent):
    """Agent specialized in news sentiment analysis using OpenAI GPT"""
    
    def __init__(self):
        super().__init__(
            name="News Analyst",
            description="Specialized in news sentiment analysis and market impact assessment"
        )
        self.sentiment_categories = [
            "positive", "negative", "neutral", "mixed"
        ]
        self.impact_levels = [
            "high", "medium", "low", "minimal"
        ]
        self.time_sensitivity = [
            "immediate", "short_term", "medium_term", "long_term"
        ]
    
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive news sentiment analysis"""
        try:
            news_data = data.get("news_data", [])
            symbols = data.get("symbols", [])
            market_context = data.get("market_context", {})
            
            if not news_data:
                return {
                    "overall_sentiment": "neutral",
                    "sentiment_score": 0.5,
                    "symbol_impact": {},
                    "confidence_score": 0.3,
                    "analysis_timestamp": datetime.utcnow().isoformat(),
                    "news_count": 0
                }
            
            # Use OpenAI to analyze news sentiment
            sentiment_analysis = await self.llm_client.analyze_news_sentiment(
                news_data=news_data,
                symbols=symbols
            )
            
            if "error" in sentiment_analysis:
                raise Exception(sentiment_analysis["error"])
            
            # Parse the JSON response
            sentiment_json = json.loads(sentiment_analysis["sentiment_analysis"])
            
            # Add our specialized analysis
            enhanced_analysis = {
                **sentiment_json,
                "news_classification": self._classify_news_types(news_data),
                "temporal_analysis": self._analyze_temporal_patterns(news_data),
                "market_correlation": self._analyze_market_correlation(sentiment_json, market_context),
                "breaking_news_impact": self._assess_breaking_news_impact(news_data),
                "sector_sentiment": self._analyze_sector_sentiment(news_data, symbols),
                "confidence_score": self._calculate_sentiment_confidence(sentiment_json, news_data),
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "tokens_used": sentiment_analysis.get("tokens_used", 0),
                "news_count": len(news_data)
            }
            
            return enhanced_analysis
            
        except Exception as e:
            logger.error(f"News sentiment analysis error: {e}")
            return {
                "error": str(e),
                "confidence_score": 0.0,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "news_count": len(data.get("news_data", []))
            }
    
    async def get_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading recommendations based on news sentiment"""
        recommendations = []
        
        if "error" in analysis:
            return recommendations
        
        try:
            symbol_impact = analysis.get("symbol_impact", {})
            overall_sentiment = analysis.get("overall_sentiment", "neutral")
            breaking_news = analysis.get("breaking_news_impact", {})
            
            # Generate symbol-specific recommendations
            for symbol, impact_data in symbol_impact.items():
                recommendation = self._generate_symbol_recommendation(
                    symbol, impact_data, overall_sentiment, breaking_news
                )
                if recommendation:
                    recommendations.append(recommendation)
            
            # Generate market-wide recommendations
            market_recommendation = self._generate_market_recommendation(
                analysis, overall_sentiment
            )
            if market_recommendation:
                recommendations.append(market_recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating news-based recommendations: {e}")
            return []
    
    def _classify_news_types(self, news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Classify news by type and importance"""
        classification = {
            "earnings": 0,
            "regulatory": 0,
            "product_launch": 0,
            "merger_acquisition": 0,
            "management_change": 0,
            "market_analysis": 0,
            "economic_data": 0,
            "other": 0
        }
        
        keywords = {
            "earnings": ["earnings", "revenue", "profit", "quarterly", "eps"],
            "regulatory": ["regulation", "sec", "fda", "approval", "compliance"],
            "product_launch": ["launch", "product", "release", "unveil"],
            "merger_acquisition": ["merger", "acquisition", "buyout", "takeover"],
            "management_change": ["ceo", "cfo", "resignation", "appointment"],
            "economic_data": ["gdp", "inflation", "unemployment", "fed", "interest"]
        }
        
        for news in news_data:
            headline = news.get("headline", "").lower()
            summary = news.get("summary", "").lower()
            text = f"{headline} {summary}"
            
            classified = False
            for category, words in keywords.items():
                if any(word in text for word in words):
                    classification[category] += 1
                    classified = True
                    break
            
            if not classified:
                classification["other"] += 1
        
        return classification
    
    def _analyze_temporal_patterns(self, news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in news"""
        now = datetime.utcnow()
        time_buckets = {
            "last_hour": 0,
            "last_4_hours": 0,
            "last_24_hours": 0,
            "last_week": 0,
            "older": 0
        }
        
        for news in news_data:
            timestamp_str = news.get("timestamp", "")
            try:
                if timestamp_str:
                    news_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    time_diff = now - news_time.replace(tzinfo=None)
                    
                    if time_diff <= timedelta(hours=1):
                        time_buckets["last_hour"] += 1
                    elif time_diff <= timedelta(hours=4):
                        time_buckets["last_4_hours"] += 1
                    elif time_diff <= timedelta(days=1):
                        time_buckets["last_24_hours"] += 1
                    elif time_diff <= timedelta(weeks=1):
                        time_buckets["last_week"] += 1
                    else:
                        time_buckets["older"] += 1
            except:
                time_buckets["older"] += 1
        
        # Calculate news velocity
        recent_news = time_buckets["last_hour"] + time_buckets["last_4_hours"]
        velocity = "high" if recent_news > 5 else "medium" if recent_news > 2 else "low"
        
        return {
            "time_distribution": time_buckets,
            "news_velocity": velocity,
            "breaking_news_count": time_buckets["last_hour"]
        }
    
    def _analyze_market_correlation(
        self, 
        sentiment_data: Dict[str, Any], 
        market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze correlation between news sentiment and market conditions"""
        market_sentiment = market_context.get("sentiment", "neutral")
        news_sentiment = sentiment_data.get("overall_sentiment", "neutral")
        
        # Simple correlation analysis
        correlation = "aligned" if market_sentiment == news_sentiment else "divergent"
        
        # Calculate sentiment strength
        sentiment_score = sentiment_data.get("sentiment_score", 0.5)
        strength = "strong" if abs(sentiment_score - 0.5) > 0.3 else "moderate"
        
        return {
            "market_news_correlation": correlation,
            "sentiment_strength": strength,
            "potential_catalyst": correlation == "divergent" and strength == "strong"
        }
    
    def _assess_breaking_news_impact(self, news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess impact of breaking news"""
        breaking_news = []
        high_impact_keywords = [
            "breaking", "urgent", "alert", "emergency", "crisis",
            "crash", "surge", "plunge", "soar", "tumble"
        ]
        
        for news in news_data:
            headline = news.get("headline", "").lower()
            
            # Check for breaking news indicators
            is_breaking = any(keyword in headline for keyword in high_impact_keywords)
            
            if is_breaking:
                breaking_news.append({
                    "headline": news.get("headline", ""),
                    "timestamp": news.get("timestamp", ""),
                    "symbols": news.get("symbols", []),
                    "impact_assessment": "high"
                })
        
        return {
            "breaking_news_count": len(breaking_news),
            "breaking_news_items": breaking_news[:5],  # Top 5
            "market_moving_potential": "high" if len(breaking_news) > 2 else "medium" if len(breaking_news) > 0 else "low"
        }
    
    def _analyze_sector_sentiment(
        self, 
        news_data: List[Dict[str, Any]], 
        symbols: List[str]
    ) -> Dict[str, Any]:
        """Analyze sentiment by sector"""
        # Simplified sector mapping
        sector_mapping = {
            "AAPL": "technology", "MSFT": "technology", "GOOGL": "technology",
            "TSLA": "automotive", "F": "automotive", "GM": "automotive",
            "JPM": "financial", "BAC": "financial", "WFC": "financial",
            "JNJ": "healthcare", "PFE": "healthcare", "MRNA": "healthcare"
        }
        
        sector_sentiment = {}
        
        for symbol in symbols:
            sector = sector_mapping.get(symbol, "other")
            if sector not in sector_sentiment:
                sector_sentiment[sector] = {
                    "positive_news": 0,
                    "negative_news": 0,
                    "neutral_news": 0,
                    "symbols": []
                }
            
            sector_sentiment[sector]["symbols"].append(symbol)
            
            # Analyze news for this symbol
            for news in news_data:
                if symbol in news.get("symbols", []):
                    sentiment = news.get("sentiment", "neutral")
                    if sentiment == "positive":
                        sector_sentiment[sector]["positive_news"] += 1
                    elif sentiment == "negative":
                        sector_sentiment[sector]["negative_news"] += 1
                    else:
                        sector_sentiment[sector]["neutral_news"] += 1
        
        # Calculate sector sentiment scores
        for sector, data in sector_sentiment.items():
            total = data["positive_news"] + data["negative_news"] + data["neutral_news"]
            if total > 0:
                score = (data["positive_news"] - data["negative_news"]) / total
                data["sentiment_score"] = score
                data["sentiment"] = "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral"
            else:
                data["sentiment_score"] = 0.0
                data["sentiment"] = "neutral"
        
        return sector_sentiment
    
    def _generate_symbol_recommendation(
        self,
        symbol: str,
        impact_data: Dict[str, Any],
        overall_sentiment: str,
        breaking_news: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate recommendation for specific symbol"""
        sentiment = impact_data.get("sentiment", "neutral")
        impact_score = impact_data.get("impact_score", 0.0)
        time_sensitivity = impact_data.get("time_sensitivity", "medium_term")
        
        # Determine action based on sentiment and impact
        if sentiment == "positive" and impact_score > 0.7:
            action = "BUY"
            confidence = min(0.8, impact_score)
        elif sentiment == "negative" and impact_score > 0.7:
            action = "SELL"
            confidence = min(0.8, impact_score)
        else:
            action = "HOLD"
            confidence = 0.5
        
        # Adjust for breaking news
        if symbol in str(breaking_news.get("breaking_news_items", [])):
            confidence *= 0.8  # Reduce confidence due to volatility
        
        return {
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "reasoning": f"News sentiment: {sentiment}, Impact score: {impact_score:.2f}, Time sensitivity: {time_sensitivity}",
            "news_based": True,
            "sentiment": sentiment,
            "impact_score": impact_score,
            "time_horizon": "short" if time_sensitivity == "immediate" else "medium",
            "risk_level": "high" if time_sensitivity == "immediate" else "medium"
        }
    
    def _generate_market_recommendation(
        self,
        analysis: Dict[str, Any],
        overall_sentiment: str
    ) -> Dict[str, Any]:
        """Generate market-wide recommendation"""
        breaking_news_impact = analysis.get("breaking_news_impact", {})
        market_moving_potential = breaking_news_impact.get("market_moving_potential", "low")
        
        if market_moving_potential == "high":
            return {
                "symbol": "MARKET",
                "action": "MONITOR",
                "confidence": 0.8,
                "reasoning": f"High market-moving news detected. Overall sentiment: {overall_sentiment}",
                "news_based": True,
                "sentiment": overall_sentiment,
                "time_horizon": "immediate",
                "risk_level": "high",
                "special_action": "increase_monitoring"
            }
        
        return None
    
    def _calculate_sentiment_confidence(
        self, 
        sentiment_json: Dict[str, Any], 
        news_data: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for sentiment analysis"""
        base_confidence = 0.6
        
        # Adjust based on news volume
        news_count = len(news_data)
        if news_count > 10:
            base_confidence += 0.2
        elif news_count > 5:
            base_confidence += 0.1
        
        # Adjust based on sentiment clarity
        sentiment_score = sentiment_json.get("sentiment_score", 0.5)
        if abs(sentiment_score - 0.5) > 0.3:
            base_confidence += 0.1
        
        # Adjust based on market moving news
        if sentiment_json.get("market_moving_news", False):
            base_confidence += 0.1
        
        return min(base_confidence, 1.0) 