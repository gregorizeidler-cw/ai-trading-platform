"""
Alpha Vantage API data collector for stock market data.
"""

from typing import List, Dict, Any, Optional
from decimal import Decimal
from datetime import datetime
import asyncio
from loguru import logger

from .base_collector import BaseDataCollector
from shared.models.market_data import (
    TickData, OHLCVData, MarketData, DataSource, AssetType
)
from config.settings import config


class AlphaVantageCollector(BaseDataCollector):
    """Data collector for Alpha Vantage API"""
    
    def __init__(self):
        super().__init__(config.api.alpha_vantage_key)
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 12  # 5 API calls per minute
        
    async def collect_tick_data(self, symbols: List[str]) -> List[TickData]:
        """Collect real-time tick data from Alpha Vantage"""
        tick_data = []
        symbols = self.validate_symbols(symbols)
        
        for symbol in symbols:
            try:
                await asyncio.sleep(self.rate_limit_delay)
                
                params = {
                    'function': 'GLOBAL_QUOTE',
                    'symbol': symbol,
                    'apikey': self.api_key
                }
                
                response = await self.make_request(self.base_url, params)
                
                if 'Global Quote' in response:
                    quote = response['Global Quote']
                    
                    tick = TickData(
                        symbol=symbol,
                        timestamp=datetime.utcnow(),
                        price=Decimal(quote.get('05. price', '0')),
                        volume=Decimal(quote.get('06. volume', '0')),
                        source=DataSource.ALPHA_VANTAGE,
                        asset_type=AssetType.STOCK
                    )
                    tick_data.append(tick)
                    
                    logger.info(f"Collected tick data for {symbol}")
                else:
                    logger.warning(f"No data received for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error collecting tick data for {symbol}: {e}")
                
        return tick_data
    
    async def collect_ohlcv_data(
        self,
        symbols: List[str],
        timeframe: str = "1min"
    ) -> List[OHLCVData]:
        """Collect OHLCV data from Alpha Vantage"""
        ohlcv_data = []
        symbols = self.validate_symbols(symbols)
        
        # Map timeframes to Alpha Vantage functions
        function_map = {
            "1min": "TIME_SERIES_INTRADAY",
            "5min": "TIME_SERIES_INTRADAY",
            "15min": "TIME_SERIES_INTRADAY",
            "30min": "TIME_SERIES_INTRADAY",
            "60min": "TIME_SERIES_INTRADAY",
            "1day": "TIME_SERIES_DAILY"
        }
        
        function = function_map.get(timeframe, "TIME_SERIES_INTRADAY")
        
        for symbol in symbols:
            try:
                await asyncio.sleep(self.rate_limit_delay)
                
                params = {
                    'function': function,
                    'symbol': symbol,
                    'apikey': self.api_key
                }
                
                if function == "TIME_SERIES_INTRADAY":
                    params['interval'] = timeframe
                    params['outputsize'] = 'compact'
                
                response = await self.make_request(self.base_url, params)
                
                # Extract time series data
                time_series_key = self._get_time_series_key(response)
                
                if time_series_key and time_series_key in response:
                    time_series = response[time_series_key]
                    
                    for timestamp_str, data in time_series.items():
                        try:
                            ohlcv = OHLCVData(
                                symbol=symbol,
                                timestamp=datetime.fromisoformat(
                                    timestamp_str.replace(' ', 'T')
                                ),
                                timeframe=timeframe,
                                open=Decimal(data['1. open']),
                                high=Decimal(data['2. high']),
                                low=Decimal(data['3. low']),
                                close=Decimal(data['4. close']),
                                volume=Decimal(data['5. volume']),
                                source=DataSource.ALPHA_VANTAGE,
                                asset_type=AssetType.STOCK
                            )
                            ohlcv_data.append(ohlcv)
                        except (KeyError, ValueError) as e:
                            logger.error(f"Error parsing OHLCV data: {e}")
                            continue
                    
                    logger.info(
                        f"Collected {len(time_series)} OHLCV records for {symbol}"
                    )
                else:
                    logger.warning(f"No OHLCV data received for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error collecting OHLCV data for {symbol}: {e}")
                
        return ohlcv_data
    
    async def collect_market_data(self, symbols: List[str]) -> List[MarketData]:
        """Collect comprehensive market data from Alpha Vantage"""
        market_data = []
        symbols = self.validate_symbols(symbols)
        
        for symbol in symbols:
            try:
                await asyncio.sleep(self.rate_limit_delay)
                
                # Get overview data
                params = {
                    'function': 'OVERVIEW',
                    'symbol': symbol,
                    'apikey': self.api_key
                }
                
                response = await self.make_request(self.base_url, params)
                
                if response and 'Symbol' in response:
                    data = MarketData(
                        symbol=symbol,
                        timestamp=datetime.utcnow(),
                        price=Decimal(response.get('Price', '0') or '0'),
                        volume=Decimal(
                            response.get('Volume', '0') or '0'
                        ),
                        market_cap=Decimal(
                            response.get('MarketCapitalization', '0') or '0'
                        ),
                        pe_ratio=Decimal(
                            response.get('PERatio', '0') or '0'
                        ),
                        fifty_two_week_high=Decimal(
                            response.get('52WeekHigh', '0') or '0'
                        ),
                        fifty_two_week_low=Decimal(
                            response.get('52WeekLow', '0') or '0'
                        ),
                        dividend_yield=Decimal(
                            response.get('DividendYield', '0') or '0'
                        ),
                        beta=Decimal(response.get('Beta', '0') or '0'),
                        source=DataSource.ALPHA_VANTAGE,
                        asset_type=AssetType.STOCK
                    )
                    market_data.append(data)
                    
                    logger.info(f"Collected market data for {symbol}")
                else:
                    logger.warning(f"No market data received for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error collecting market data for {symbol}: {e}")
                
        return market_data
    
    def _get_time_series_key(self, response: Dict[str, Any]) -> Optional[str]:
        """Get the time series key from Alpha Vantage response"""
        possible_keys = [
            'Time Series (1min)',
            'Time Series (5min)',
            'Time Series (15min)',
            'Time Series (30min)',
            'Time Series (60min)',
            'Time Series (Daily)',
            'Time Series (Weekly)',
            'Time Series (Monthly)'
        ]
        
        for key in possible_keys:
            if key in response:
                return key
        
        return None
