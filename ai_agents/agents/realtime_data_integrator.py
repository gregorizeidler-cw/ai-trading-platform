"""
Real-Time Data Integration Agent
Handles live market data feeds and real-time analysis using Yahoo Finance
"""

import pandas as pd
import numpy as np
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import yfinance as yf
from dataclasses import dataclass
from loguru import logger
import time

from .base_agent import BaseAgent
from ..llm.openai_client import OpenAIClient


@dataclass
class RealTimeData:
    """Real-time market data structure"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None


@dataclass
class MarketAlert:
    """Market alert structure"""
    symbol: str
    alert_type: str
    message: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    timestamp: datetime
    data: Dict[str, Any]


class RealTimeDataIntegrator(BaseAgent):
    """Real-time data integration and monitoring agent"""
    
    def __init__(self, openai_client: OpenAIClient):
        super().__init__(
            name="Real-Time Data Integrator",
            description="Handles live market data feeds and real-time analysis",
            openai_client=openai_client
        )
        
        self.watchlist = []
        self.price_cache = {}
        self.alerts = []
        self.alert_thresholds = {}
        self.is_monitoring = False
        
    async def add_to_watchlist(self, symbols: List[str]) -> Dict[str, Any]:
        """Add symbols to real-time monitoring watchlist"""
        try:
            new_symbols = []
            
            for symbol in symbols:
                if symbol not in self.watchlist:
                    # Validate symbol
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    if info and 'symbol' in info:
                        self.watchlist.append(symbol)
                        new_symbols.append(symbol)
                        
                        # Set default alert thresholds
                        self.alert_thresholds[symbol] = {
                            'price_change_percent': 5.0,  # 5% price change
                            'volume_spike': 2.0,  # 2x average volume
                            'volatility_spike': 2.0  # 2x average volatility
                        }
            
            return {
                'added_symbols': new_symbols,
                'total_watchlist_size': len(self.watchlist),
                'current_watchlist': self.watchlist,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error adding symbols to watchlist: {e}")
            return {'error': str(e)}
    
    async def get_realtime_data(self, symbols: List[str]) -> List[RealTimeData]:
        """Get real-time data for symbols"""
        try:
            realtime_data = []
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="2d", interval="1m")
                    
                    if hist.empty:
                        continue
                    
                    current_price = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-391] if len(hist) > 390 else hist['Close'].iloc[0]
                    
                    change = current_price - prev_close
                    change_percent = (change / prev_close) * 100 if prev_close > 0 else 0
                    current_volume = int(hist['Volume'].iloc[-1])
                    
                    rt_data = RealTimeData(
                        symbol=symbol,
                        price=current_price,
                        change=change,
                        change_percent=change_percent,
                        volume=current_volume,
                        timestamp=datetime.now()
                    )
                    
                    realtime_data.append(rt_data)
                    self.price_cache[symbol] = rt_data
                    
                except Exception as e:
                    logger.warning(f"Could not get real-time data for {symbol}: {e}")
                    continue
            
            return realtime_data
            
        except Exception as e:
            logger.error(f"Error getting real-time data: {e}")
            return []
    
    async def monitor_markets(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """Monitor markets for specified duration and generate alerts"""
        try:
            self.is_monitoring = True
            monitoring_results = {
                'start_time': datetime.now().isoformat(),
                'duration_minutes': duration_minutes,
                'alerts_generated': [],
                'price_updates': [],
                'summary': {}
            }
            
            end_time = datetime.now() + timedelta(minutes=duration_minutes)
            update_count = 0
            
            logger.info(f"Starting market monitoring for {duration_minutes} minutes")
            
            while datetime.now() < end_time and self.is_monitoring:
                try:
                    # Get real-time data
                    realtime_data = await self.get_realtime_data(self.watchlist)
                    
                    # Process each symbol
                    for data in realtime_data:
                        # Check for alerts
                        alerts = await self._check_alerts(data)
                        monitoring_results['alerts_generated'].extend(alerts)
                        
                        # Store price update
                        monitoring_results['price_updates'].append({
                            'symbol': data.symbol,
                            'price': data.price,
                            'change_percent': data.change_percent,
                            'volume': data.volume,
                            'timestamp': data.timestamp.isoformat()
                        })
                    
                    update_count += 1
                    
                    # Wait before next update (1 minute intervals)
                    await asyncio.sleep(60)
                    
                except Exception as e:
                    logger.error(f"Error during monitoring iteration: {e}")
                    continue
            
            # Generate summary
            monitoring_results['end_time'] = datetime.now().isoformat()
            monitoring_results['total_updates'] = update_count
            monitoring_results['summary'] = await self._generate_monitoring_summary(monitoring_results)
            
            self.is_monitoring = False
            
            return monitoring_results
            
        except Exception as e:
            logger.error(f"Error monitoring markets: {e}")
            self.is_monitoring = False
            return {'error': str(e)}
    
    async def _check_alerts(self, data: RealTimeData) -> List[MarketAlert]:
        """Check for alert conditions"""
        try:
            alerts = []
            thresholds = self.alert_thresholds.get(data.symbol, {})
            
            # Price change alert
            price_threshold = thresholds.get('price_change_percent', 5.0)
            if abs(data.change_percent) >= price_threshold:
                severity = 'high' if abs(data.change_percent) >= 10 else 'medium'
                
                alert = MarketAlert(
                    symbol=data.symbol,
                    alert_type='price_change',
                    message=f"{data.symbol} moved {data.change_percent:.2f}% to ${data.price:.2f}",
                    severity=severity,
                    timestamp=data.timestamp,
                    data={
                        'price': data.price,
                        'change_percent': data.change_percent,
                        'threshold': price_threshold
                    }
                )
                alerts.append(alert)
            
            # Volume alert
            if data.symbol in self.price_cache:
                # Get historical volume for comparison
                try:
                    ticker = yf.Ticker(data.symbol)
                    hist = ticker.history(period="30d")
                    
                    if not hist.empty:
                        avg_volume = hist['Volume'].mean()
                        volume_ratio = data.volume / avg_volume if avg_volume > 0 else 1
                        
                        volume_threshold = thresholds.get('volume_spike', 2.0)
                        if volume_ratio >= volume_threshold:
                            alert = MarketAlert(
                                symbol=data.symbol,
                                alert_type='volume_spike',
                                message=f"{data.symbol} volume spike: {volume_ratio:.1f}x average ({data.volume:,} vs {avg_volume:,.0f})",
                                severity='medium',
                                timestamp=data.timestamp,
                                data={
                                    'current_volume': data.volume,
                                    'average_volume': avg_volume,
                                    'volume_ratio': volume_ratio
                                }
                            )
                            alerts.append(alert)
                            
                except Exception as e:
                    logger.warning(f"Could not check volume for {data.symbol}: {e}")
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking alerts for {data.symbol}: {e}")
            return []
    
    async def _generate_monitoring_summary(self, monitoring_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-powered monitoring summary"""
        try:
            alerts = monitoring_results['alerts_generated']
            price_updates = monitoring_results['price_updates']
            
            # Calculate summary statistics
            total_alerts = len(alerts)
            alert_types = {}
            symbol_activity = {}
            
            for alert in alerts:
                alert_types[alert.alert_type] = alert_types.get(alert.alert_type, 0) + 1
                symbol_activity[alert.symbol] = symbol_activity.get(alert.symbol, 0) + 1
            
            # Most active symbols
            most_active = sorted(symbol_activity.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Price movements
            if price_updates:
                latest_updates = {}
                for update in price_updates:
                    symbol = update['symbol']
                    if symbol not in latest_updates or update['timestamp'] > latest_updates[symbol]['timestamp']:
                        latest_updates[symbol] = update
                
                biggest_movers = sorted(
                    latest_updates.values(),
                    key=lambda x: abs(x['change_percent']),
                    reverse=True
                )[:5]
            else:
                biggest_movers = []
            
            # Generate AI summary
            ai_summary = await self._generate_ai_monitoring_summary(
                total_alerts, alert_types, most_active, biggest_movers
            )
            
            return {
                'total_alerts': total_alerts,
                'alert_types': alert_types,
                'most_active_symbols': most_active,
                'biggest_movers': biggest_movers,
                'ai_summary': ai_summary
            }
            
        except Exception as e:
            logger.error(f"Error generating monitoring summary: {e}")
            return {'error': str(e)}
    
    async def _generate_ai_monitoring_summary(
        self,
        total_alerts: int,
        alert_types: Dict[str, int],
        most_active: List[tuple],
        biggest_movers: List[Dict[str, Any]]
    ) -> str:
        """Generate AI summary of monitoring session"""
        try:
            prompt = f"""
            Summarize the following market monitoring session:
            
            Alert Summary:
            - Total Alerts: {total_alerts}
            - Alert Types: {alert_types}
            
            Most Active Symbols: {[f"{symbol} ({count} alerts)" for symbol, count in most_active]}
            
            Biggest Price Movers: {[f"{mover['symbol']}: {mover['change_percent']:.2f}%" for mover in biggest_movers[:3]]}
            
            Provide a concise summary (3-4 sentences) covering:
            1. Overall market activity level
            2. Key themes or patterns observed
            3. Notable individual stock movements
            4. Risk considerations
            
            Focus on actionable insights for traders.
            """
            
            response = await self.openai_client.generate_completion(
                prompt=prompt,
                max_tokens=200,
                temperature=0.3
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating AI monitoring summary: {e}")
            return "Market monitoring session completed. Review alerts and price movements for detailed insights."
    
    async def get_market_overview(self) -> Dict[str, Any]:
        """Get comprehensive market overview"""
        try:
            indices = ['SPY', 'QQQ', 'IWM', '^VIX']
            index_data = await self.get_realtime_data(indices)
            
            return {
                'indices': [
                    {
                        'symbol': data.symbol,
                        'price': data.price,
                        'change_percent': data.change_percent,
                        'volume': data.volume
                    } for data in index_data
                ],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return {'error': str(e)}
    
    async def _calculate_market_sentiment(
        self, 
        index_data: List[RealTimeData], 
        watchlist_data: List[RealTimeData]
    ) -> Dict[str, Any]:
        """Calculate overall market sentiment"""
        try:
            sentiment_indicators = {}
            
            # Index sentiment
            if index_data:
                spy_change = next((d.change_percent for d in index_data if d.symbol == 'SPY'), 0)
                vix_change = next((d.change_percent for d in index_data if d.symbol == '^VIX'), 0)
                
                # Simple sentiment calculation
                if spy_change > 1 and vix_change < -5:
                    sentiment = 'bullish'
                elif spy_change < -1 and vix_change > 5:
                    sentiment = 'bearish'
                else:
                    sentiment = 'neutral'
                
                sentiment_indicators['index_sentiment'] = sentiment
                sentiment_indicators['spy_change'] = spy_change
                sentiment_indicators['vix_change'] = vix_change
            
            # Breadth indicators
            if watchlist_data:
                advancing = len([d for d in watchlist_data if d.change_percent > 0])
                declining = len([d for d in watchlist_data if d.change_percent < 0])
                
                advance_decline_ratio = advancing / declining if declining > 0 else float('inf')
                
                sentiment_indicators['advance_decline_ratio'] = advance_decline_ratio
                sentiment_indicators['advancing_stocks'] = advancing
                sentiment_indicators['declining_stocks'] = declining
            
            return sentiment_indicators
            
        except Exception as e:
            logger.error(f"Error calculating market sentiment: {e}")
            return {}
    
    def stop_monitoring(self):
        """Stop market monitoring"""
        self.is_monitoring = False
        logger.info("Market monitoring stopped")
    
    def set_alert_threshold(self, symbol: str, threshold_type: str, value: float):
        """Set custom alert threshold for a symbol"""
        if symbol not in self.alert_thresholds:
            self.alert_thresholds[symbol] = {}
        
        self.alert_thresholds[symbol][threshold_type] = value
        logger.info(f"Set {threshold_type} threshold for {symbol} to {value}")
    
    async def get_symbol_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get detailed real-time analysis for a specific symbol"""
        try:
            # Get real-time data
            rt_data = await self.get_realtime_data([symbol])
            
            if not rt_data:
                return {'error': f'No data available for {symbol}'}
            
            data = rt_data[0]
            
            # Get historical context
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="30d", interval="1d")
            
            if hist.empty:
                return {'error': f'No historical data available for {symbol}'}
            
            # Technical indicators
            sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
            volatility = hist['Close'].pct_change().std() * np.sqrt(252)
            rsi = self._calculate_rsi(hist['Close']).iloc[-1]
            
            # Position relative to recent range
            high_52w = hist['High'].max()
            low_52w = hist['Low'].min()
            position_in_range = (data.price - low_52w) / (high_52w - low_52w) if high_52w > low_52w else 0.5
            
            return {
                'symbol': symbol,
                'current_data': {
                    'price': data.price,
                    'change': data.change,
                    'change_percent': data.change_percent,
                    'volume': data.volume,
                    'high': data.high,
                    'low': data.low
                },
                'technical_analysis': {
                    'sma_20': sma_20,
                    'price_vs_sma20': ((data.price - sma_20) / sma_20 * 100) if sma_20 > 0 else 0,
                    'volatility_annualized': volatility,
                    'rsi': rsi,
                    'position_in_52w_range': position_in_range * 100
                },
                'alerts': [alert for alert in self.alerts if alert.symbol == symbol],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing symbol {symbol}: {e}")
            return {'error': str(e)}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs)) 