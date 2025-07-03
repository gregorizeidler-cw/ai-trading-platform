"""
Yahoo Finance data collector for stock market data.
"""

from typing import List
from decimal import Decimal
from datetime import datetime
import yfinance as yf
from loguru import logger

from .base_collector import BaseDataCollector
from shared.models.market_data import (
    TickData, OHLCVData, MarketData, DataSource, AssetType
)


class YahooFinanceCollector(BaseDataCollector):
    """Data collector for Yahoo Finance API"""
    
    def __init__(self):
        super().__init__()
        
    async def collect_tick_data(self, symbols: List[str]) -> List[TickData]:
        """Collect real-time tick data from Yahoo Finance"""
        tick_data = []
        symbols = self.validate_symbols(symbols)
        
        try:
            tickers = yf.Tickers(' '.join(symbols))
            
            for symbol in symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    info = ticker.info
                    
                    if 'regularMarketPrice' in info:
                        tick = TickData(
                            symbol=symbol,
                            timestamp=datetime.utcnow(),
                            price=Decimal(str(info['regularMarketPrice'])),
                            volume=Decimal(
                                str(info.get('regularMarketVolume', 0))
                            ),
                            bid=Decimal(str(info.get('bid', 0))),
                            ask=Decimal(str(info.get('ask', 0))),
                            bid_size=Decimal(str(info.get('bidSize', 0))),
                            ask_size=Decimal(str(info.get('askSize', 0))),
                            source=DataSource.YAHOO_FINANCE,
                            asset_type=AssetType.STOCK
                        )
                        tick_data.append(tick)
                        
                        logger.info(f"Collected tick data for {symbol}")
                    else:
                        logger.warning(f"No price data available for {symbol}")
                        
                except Exception as e:
                    logger.error(
                        f"Error collecting tick data for {symbol}: {e}"
                    )
                    
        except Exception as e:
            logger.error(f"Error initializing Yahoo Finance tickers: {e}")
            
        return tick_data
    
    async def collect_ohlcv_data(
        self,
        symbols: List[str],
        timeframe: str = "1m"
    ) -> List[OHLCVData]:
        """Collect OHLCV data from Yahoo Finance"""
        ohlcv_data = []
        symbols = self.validate_symbols(symbols)
        
        # Map timeframes to Yahoo Finance periods
        period_map = {
            "1m": "1d",
            "2m": "1d",
            "5m": "1d",
            "15m": "1d",
            "30m": "1d",
            "60m": "1d",
            "90m": "1d",
            "1h": "1d",
            "1d": "1mo",
            "5d": "3mo",
            "1wk": "6mo",
            "1mo": "1y"
        }
        
        period = period_map.get(timeframe, "1d")
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period, interval=timeframe)
                
                if not hist.empty:
                    for timestamp, row in hist.iterrows():
                        try:
                            ohlcv = OHLCVData(
                                symbol=symbol,
                                timestamp=timestamp.to_pydatetime(),
                                timeframe=timeframe,
                                open=Decimal(str(row['Open'])),
                                high=Decimal(str(row['High'])),
                                low=Decimal(str(row['Low'])),
                                close=Decimal(str(row['Close'])),
                                volume=Decimal(str(row['Volume'])),
                                source=DataSource.YAHOO_FINANCE,
                                asset_type=AssetType.STOCK
                            )
                            ohlcv_data.append(ohlcv)
                        except (KeyError, ValueError) as e:
                            logger.error(f"Error parsing OHLCV data: {e}")
                            continue
                    
                    logger.info(
                        f"Collected {len(hist)} OHLCV records for {symbol}"
                    )
                else:
                    logger.warning(f"No OHLCV data received for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error collecting OHLCV data for {symbol}: {e}")
                
        return ohlcv_data
    
    async def collect_market_data(
        self, symbols: List[str]
    ) -> List[MarketData]:
        """Collect comprehensive market data from Yahoo Finance"""
        market_data = []
        symbols = self.validate_symbols(symbols)
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                if info:
                    data = MarketData(
                        symbol=symbol,
                        timestamp=datetime.utcnow(),
                        price=Decimal(
                            str(info.get('regularMarketPrice', 0))
                        ),
                        volume=Decimal(
                            str(info.get('regularMarketVolume', 0))
                        ),
                        market_cap=Decimal(
                            str(info.get('marketCap', 0))
                        ),
                        pe_ratio=Decimal(
                            str(info.get('trailingPE', 0))
                        ),
                        fifty_two_week_high=Decimal(
                            str(info.get('fiftyTwoWeekHigh', 0))
                        ),
                        fifty_two_week_low=Decimal(
                            str(info.get('fiftyTwoWeekLow', 0))
                        ),
                        dividend_yield=Decimal(
                            str(info.get('dividendYield', 0) or 0)
                        ),
                        beta=Decimal(str(info.get('beta', 0) or 0)),
                        moving_avg_50=Decimal(
                            str(info.get('fiftyDayAverage', 0))
                        ),
                        moving_avg_200=Decimal(
                            str(info.get('twoHundredDayAverage', 0))
                        ),
                        source=DataSource.YAHOO_FINANCE,
                        asset_type=AssetType.STOCK
                    )
                    market_data.append(data)
                    
                    logger.info(f"Collected market data for {symbol}")
                else:
                    logger.warning(f"No market data received for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error collecting market data for {symbol}: {e}")
                
        return market_data
