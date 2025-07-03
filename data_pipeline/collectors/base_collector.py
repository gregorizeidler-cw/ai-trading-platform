"""
Base data collector class and interfaces for market data collection.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import asyncio
import aiohttp
from loguru import logger
from shared.models.market_data import (
    TickData, OHLCVData, MarketData, DataSource
)


class BaseDataCollector(ABC):
    """Abstract base class for all data collectors"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_running = False
        
    @abstractmethod
    async def collect_tick_data(self, symbols: List[str]) -> List[TickData]:
        """Collect real-time tick data for given symbols"""
        pass
    
    @abstractmethod
    async def collect_ohlcv_data(
        self,
        symbols: List[str],
        timeframe: str = "1m"
    ) -> List[OHLCVData]:
        """Collect OHLCV data for given symbols and timeframe"""
        pass
    
    @abstractmethod
    async def collect_market_data(
        self, symbols: List[str]
    ) -> List[MarketData]:
        """Collect comprehensive market data for given symbols"""
        pass
    
    async def start_session(self):
        """Start HTTP session for API calls"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
    async def close_session(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
            
    async def make_request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with error handling"""
        if not self.session:
            await self.start_session()
            
        try:
            async with self.session.get(
                url,
                params=params,
                headers=headers
            ) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"HTTP request error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in request: {e}")
            raise
    
    def validate_symbols(self, symbols: List[str]) -> List[str]:
        """Validate and clean symbol list"""
        valid_symbols = []
        for symbol in symbols:
            symbol = symbol.strip().upper()
            if symbol and len(symbol) <= 20:
                valid_symbols.append(symbol)
            else:
                logger.warning(f"Invalid symbol: {symbol}")
        return valid_symbols
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_session()


class DataCollectorRegistry:
    """Registry for managing multiple data collectors"""
    
    def __init__(self):
        self.collectors: Dict[DataSource, BaseDataCollector] = {}
        
    def register_collector(
        self,
        source: DataSource,
        collector: BaseDataCollector
    ):
        """Register a data collector for a specific source"""
        self.collectors[source] = collector
        logger.info(f"Registered collector for {source.value}")
        
    def get_collector(self, source: DataSource) -> Optional[BaseDataCollector]:
        """Get collector for a specific data source"""
        return self.collectors.get(source)
    
    def list_sources(self) -> List[DataSource]:
        """List all available data sources"""
        return list(self.collectors.keys())
    
    async def collect_from_all_sources(
        self,
        symbols: List[str],
        data_type: str = "tick"
    ) -> Dict[DataSource, List[Any]]:
        """Collect data from all registered sources"""
        results = {}
        
        async def collect_from_source(source: DataSource, collector: BaseDataCollector):
            try:
                async with collector:
                    if data_type == "tick":
                        data = await collector.collect_tick_data(symbols)
                    elif data_type == "ohlcv":
                        data = await collector.collect_ohlcv_data(symbols)
                    elif data_type == "market":
                        data = await collector.collect_market_data(symbols)
                    else:
                        logger.error(f"Unknown data type: {data_type}")
                        return
                    
                    results[source] = data
                    logger.info(
                        f"Collected {len(data)} records from {source.value}"
                    )
            except Exception as e:
                logger.error(f"Error collecting from {source.value}: {e}")
                results[source] = []
        
        # Collect from all sources concurrently
        tasks = [
            collect_from_source(source, collector)
            for source, collector in self.collectors.items()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        return results


# Global collector registry
collector_registry = DataCollectorRegistry()
