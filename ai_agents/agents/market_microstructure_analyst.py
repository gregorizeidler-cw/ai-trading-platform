"""
Market Microstructure Analyst Agent
Analyzes order book dynamics, bid-ask spreads, market liquidity, and trading patterns
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from dataclasses import dataclass
from loguru import logger

from .base_agent import BaseAgent
from ..llm.openai_client import OpenAIClient


@dataclass
class MarketMicrostructureData:
    """Market microstructure analysis data"""
    symbol: str
    timestamp: datetime
    bid_ask_spread: float
    relative_spread: float
    market_impact: float
    liquidity_score: float
    order_flow_imbalance: float
    volatility_clustering: float
    price_discovery_efficiency: float
    trading_intensity: float
    market_depth: Dict[str, float]
    intraday_patterns: Dict[str, float]


class MarketMicrostructureAnalyst(BaseAgent):
    """Advanced market microstructure analyst using AI and quantitative methods"""
    
    def __init__(self, openai_client: OpenAIClient):
        super().__init__(
            name="Market Microstructure Analyst",
            description="Analyzes market microstructure, liquidity, and trading patterns",
            openai_client=openai_client
        )
        self.analysis_cache = {}
        self.liquidity_models = {}
        
    async def analyze_market_microstructure(
        self, 
        symbol: str, 
        period: str = "1d",
        interval: str = "1m"
    ) -> MarketMicrostructureData:
        """Comprehensive market microstructure analysis"""
        try:
            # Get high-frequency data
            data = await self._get_high_frequency_data(symbol, period, interval)
            
            # Calculate microstructure metrics
            bid_ask_spread = self._calculate_bid_ask_spread(data)
            relative_spread = self._calculate_relative_spread(data)
            market_impact = self._calculate_market_impact(data)
            liquidity_score = self._calculate_liquidity_score(data)
            order_flow_imbalance = self._calculate_order_flow_imbalance(data)
            volatility_clustering = self._calculate_volatility_clustering(data)
            price_discovery = self._calculate_price_discovery_efficiency(data)
            trading_intensity = self._calculate_trading_intensity(data)
            market_depth = self._analyze_market_depth(data)
            intraday_patterns = self._analyze_intraday_patterns(data)
            
            microstructure_data = MarketMicrostructureData(
                symbol=symbol,
                timestamp=datetime.now(),
                bid_ask_spread=bid_ask_spread,
                relative_spread=relative_spread,
                market_impact=market_impact,
                liquidity_score=liquidity_score,
                order_flow_imbalance=order_flow_imbalance,
                volatility_clustering=volatility_clustering,
                price_discovery_efficiency=price_discovery,
                trading_intensity=trading_intensity,
                market_depth=market_depth,
                intraday_patterns=intraday_patterns
            )
            
            # Generate AI analysis
            ai_analysis = await self._generate_ai_analysis(microstructure_data)
            
            return {
                "microstructure_data": microstructure_data,
                "ai_analysis": ai_analysis,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in market microstructure analysis: {e}")
            raise
    
    async def _get_high_frequency_data(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Get high-frequency market data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Calculate additional metrics
            data['Returns'] = data['Close'].pct_change()
            data['HL_Spread'] = (data['High'] - data['Low']) / data['Close']
            data['Volume_Price'] = data['Volume'] * data['Close']
            data['VWAP'] = (data['Volume_Price'].cumsum() / data['Volume'].cumsum())
            data['Price_Range'] = data['High'] - data['Low']
            data['True_Range'] = np.maximum(
                data['High'] - data['Low'],
                np.maximum(
                    abs(data['High'] - data['Close'].shift(1)),
                    abs(data['Low'] - data['Close'].shift(1))
                )
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting high-frequency data: {e}")
            raise
    
    def _calculate_bid_ask_spread(self, data: pd.DataFrame) -> float:
        """Calculate effective bid-ask spread"""
        try:
            # Approximation using high-low spread
            spreads = data['HL_Spread'].dropna()
            return float(spreads.mean()) if not spreads.empty else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating bid-ask spread: {e}")
            return 0.0
    
    def _calculate_relative_spread(self, data: pd.DataFrame) -> float:
        """Calculate relative bid-ask spread"""
        try:
            spread = self._calculate_bid_ask_spread(data)
            avg_price = data['Close'].mean()
            return spread / avg_price if avg_price > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating relative spread: {e}")
            return 0.0
    
    def _calculate_market_impact(self, data: pd.DataFrame) -> float:
        """Calculate market impact based on volume and price movements"""
        try:
            # Kyle's Lambda (price impact coefficient)
            returns = data['Returns'].dropna()
            volume = data['Volume'].dropna()
            
            if len(returns) < 2 or len(volume) < 2:
                return 0.0
            
            # Align series
            min_len = min(len(returns), len(volume))
            returns = returns.iloc[-min_len:]
            volume = volume.iloc[-min_len:]
            
            # Calculate correlation between volume and absolute returns
            abs_returns = abs(returns)
            correlation = np.corrcoef(volume, abs_returns)[0, 1]
            
            return float(correlation) if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating market impact: {e}")
            return 0.0
    
    def _calculate_liquidity_score(self, data: pd.DataFrame) -> float:
        """Calculate overall liquidity score"""
        try:
            # Amihud illiquidity measure (inverted for liquidity)
            returns = abs(data['Returns'].dropna())
            volume = data['Volume'].dropna()
            
            if len(returns) == 0 or len(volume) == 0:
                return 0.0
            
            # Align series
            min_len = min(len(returns), len(volume))
            returns = returns.iloc[-min_len:]
            volume = volume.iloc[-min_len:]
            
            # Avoid division by zero
            volume = volume.replace(0, np.nan)
            illiquidity = (returns / volume).dropna()
            
            if len(illiquidity) == 0:
                return 0.0
            
            # Convert to liquidity score (0-100)
            avg_illiquidity = illiquidity.mean()
            liquidity_score = 100 / (1 + avg_illiquidity * 1000000)  # Scale factor
            
            return float(min(max(liquidity_score, 0), 100))
            
        except Exception as e:
            logger.error(f"Error calculating liquidity score: {e}")
            return 0.0
    
    def _calculate_order_flow_imbalance(self, data: pd.DataFrame) -> float:
        """Calculate order flow imbalance"""
        try:
            # Approximation using volume and price changes
            price_changes = data['Close'].diff()
            volume = data['Volume']
            
            # Buy volume approximation (positive price change)
            buy_volume = volume.where(price_changes > 0, 0).sum()
            sell_volume = volume.where(price_changes < 0, 0).sum()
            
            total_volume = buy_volume + sell_volume
            
            if total_volume == 0:
                return 0.0
            
            imbalance = (buy_volume - sell_volume) / total_volume
            return float(imbalance)
            
        except Exception as e:
            logger.error(f"Error calculating order flow imbalance: {e}")
            return 0.0
    
    def _calculate_volatility_clustering(self, data: pd.DataFrame) -> float:
        """Calculate volatility clustering coefficient"""
        try:
            returns = data['Returns'].dropna()
            
            if len(returns) < 10:
                return 0.0
            
            # Calculate rolling volatility
            volatility = returns.rolling(window=5).std()
            
            # Calculate autocorrelation of squared returns
            squared_returns = returns ** 2
            autocorr = squared_returns.autocorr(lag=1)
            
            return float(autocorr) if not np.isnan(autocorr) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating volatility clustering: {e}")
            return 0.0
    
    def _calculate_price_discovery_efficiency(self, data: pd.DataFrame) -> float:
        """Calculate price discovery efficiency"""
        try:
            # Variance ratio test approximation
            returns = data['Returns'].dropna()
            
            if len(returns) < 20:
                return 0.0
            
            # Calculate variance ratios for different horizons
            var_1 = returns.var()
            var_5 = returns.rolling(window=5).sum().var() / 5
            
            if var_1 == 0:
                return 0.0
            
            variance_ratio = var_5 / var_1
            
            # Convert to efficiency score (closer to 1 = more efficient)
            efficiency = 1 / (1 + abs(variance_ratio - 1))
            
            return float(efficiency * 100)
            
        except Exception as e:
            logger.error(f"Error calculating price discovery efficiency: {e}")
            return 0.0
    
    def _calculate_trading_intensity(self, data: pd.DataFrame) -> float:
        """Calculate trading intensity"""
        try:
            volume = data['Volume'].dropna()
            
            if len(volume) == 0:
                return 0.0
            
            # Trading intensity as coefficient of variation of volume
            mean_volume = volume.mean()
            std_volume = volume.std()
            
            if mean_volume == 0:
                return 0.0
            
            intensity = std_volume / mean_volume
            return float(intensity)
            
        except Exception as e:
            logger.error(f"Error calculating trading intensity: {e}")
            return 0.0
    
    def _analyze_market_depth(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze market depth characteristics"""
        try:
            volume = data['Volume'].dropna()
            price_range = data['Price_Range'].dropna()
            
            if len(volume) == 0 or len(price_range) == 0:
                return {"depth_score": 0.0, "resilience": 0.0}
            
            # Market depth approximation
            avg_volume = volume.mean()
            avg_range = price_range.mean()
            
            # Depth score (higher volume, lower price impact)
            depth_score = avg_volume / (avg_range + 1e-8)
            
            # Resilience (recovery speed after shocks)
            returns = data['Returns'].dropna()
            if len(returns) > 1:
                resilience = 1 - abs(returns.autocorr(lag=1))
            else:
                resilience = 0.0
            
            return {
                "depth_score": float(depth_score),
                "resilience": float(resilience) if not np.isnan(resilience) else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market depth: {e}")
            return {"depth_score": 0.0, "resilience": 0.0}
    
    def _analyze_intraday_patterns(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze intraday trading patterns"""
        try:
            if data.empty:
                return {"opening_effect": 0.0, "closing_effect": 0.0, "lunch_effect": 0.0}
            
            # Add hour information
            data_with_hour = data.copy()
            data_with_hour['Hour'] = data_with_hour.index.hour
            
            # Calculate hourly statistics
            hourly_volume = data_with_hour.groupby('Hour')['Volume'].mean()
            hourly_volatility = data_with_hour.groupby('Hour')['Returns'].std()
            
            # Opening effect (first hour vs average)
            if 9 in hourly_volume.index:
                opening_effect = hourly_volume[9] / hourly_volume.mean()
            else:
                opening_effect = 1.0
            
            # Closing effect (last hour vs average)
            if 15 in hourly_volume.index:
                closing_effect = hourly_volume[15] / hourly_volume.mean()
            else:
                closing_effect = 1.0
            
            # Lunch effect (midday vs average)
            if 12 in hourly_volume.index:
                lunch_effect = hourly_volume[12] / hourly_volume.mean()
            else:
                lunch_effect = 1.0
            
            return {
                "opening_effect": float(opening_effect),
                "closing_effect": float(closing_effect),
                "lunch_effect": float(lunch_effect)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing intraday patterns: {e}")
            return {"opening_effect": 0.0, "closing_effect": 0.0, "lunch_effect": 0.0}
    
    async def _generate_ai_analysis(self, microstructure_data: MarketMicrostructureData) -> str:
        """Generate AI-powered market microstructure analysis"""
        try:
            prompt = f"""
            Analyze the following market microstructure data for {microstructure_data.symbol}:
            
            Liquidity Metrics:
            - Bid-Ask Spread: {microstructure_data.bid_ask_spread:.4f}
            - Relative Spread: {microstructure_data.relative_spread:.4f}
            - Liquidity Score: {microstructure_data.liquidity_score:.2f}/100
            - Market Depth Score: {microstructure_data.market_depth.get('depth_score', 0):.2f}
            - Market Resilience: {microstructure_data.market_depth.get('resilience', 0):.2f}
            
            Trading Dynamics:
            - Market Impact: {microstructure_data.market_impact:.4f}
            - Order Flow Imbalance: {microstructure_data.order_flow_imbalance:.4f}
            - Trading Intensity: {microstructure_data.trading_intensity:.4f}
            - Volatility Clustering: {microstructure_data.volatility_clustering:.4f}
            - Price Discovery Efficiency: {microstructure_data.price_discovery_efficiency:.2f}%
            
            Intraday Patterns:
            - Opening Effect: {microstructure_data.intraday_patterns.get('opening_effect', 0):.2f}x
            - Closing Effect: {microstructure_data.intraday_patterns.get('closing_effect', 0):.2f}x
            - Lunch Effect: {microstructure_data.intraday_patterns.get('lunch_effect', 0):.2f}x
            
            Provide a comprehensive analysis covering:
            1. Overall market quality assessment
            2. Liquidity conditions and trading costs
            3. Market efficiency and price discovery
            4. Optimal execution strategies
            5. Risk considerations for different order types
            6. Intraday timing recommendations
            
            Focus on actionable insights for institutional trading.
            """
            
            response = await self.openai_client.generate_completion(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.3
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating AI analysis: {e}")
            return f"Error generating analysis: {str(e)}"
    
    async def analyze_execution_quality(
        self, 
        symbol: str, 
        order_size: float, 
        order_type: str = "market"
    ) -> Dict[str, Any]:
        """Analyze execution quality for a specific order"""
        try:
            microstructure = await self.analyze_market_microstructure(symbol)
            
            # Calculate execution cost estimates
            spread_cost = microstructure["microstructure_data"].bid_ask_spread / 2
            impact_cost = microstructure["microstructure_data"].market_impact * order_size
            
            # Timing recommendations
            intraday = microstructure["microstructure_data"].intraday_patterns
            best_time = min(intraday.items(), key=lambda x: x[1])
            
            return {
                "symbol": symbol,
                "order_size": order_size,
                "estimated_spread_cost": spread_cost,
                "estimated_impact_cost": impact_cost,
                "total_estimated_cost": spread_cost + impact_cost,
                "liquidity_score": microstructure["microstructure_data"].liquidity_score,
                "recommended_execution_time": best_time[0],
                "execution_difficulty": "Low" if microstructure["microstructure_data"].liquidity_score > 70 else "High",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing execution quality: {e}")
            return {"error": str(e)} 