"""
Real-time financial market data collection, cleaning, and storage orchestrator.

This script automates the collection, processing, and storage of financial
market data from multiple sources in real-time.
"""

import asyncio
import signal
import sys
from datetime import datetime
from typing import List, Dict, Any
import uuid
from decimal import Decimal
from loguru import logger

from config.settings import config
from shared.models.market_data import DataSource, ProcessingStatus
from data_pipeline.collectors.base_collector import collector_registry
from data_pipeline.collectors.alpha_vantage_collector import (
    AlphaVantageCollector
)
from data_pipeline.collectors.yahoo_finance_collector import (
    YahooFinanceCollector
)
from data_pipeline.processors.data_processor import data_processor
from data_pipeline.storage.data_storage import data_storage


class MarketDataOrchestrator:
    """Main orchestrator for market data collection and processing"""
    
    def __init__(self):
        self.is_running = False
        self.collection_interval = 60  # seconds
        self.cleanup_interval = 3600 * 24  # 24 hours
        self.last_cleanup = datetime.utcnow()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self._initialize_collectors()
        self._initialize_storage()
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logger.remove()  # Remove default handler
        
        # Add file handler
        logger.add(
            config.logging.file_path,
            rotation=config.logging.rotation,
            retention=config.logging.retention,
            level=config.logging.level,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | {level} | "
                "{name}:{function}:{line} | {message}"
            )
        )
        
        # Add console handler
        logger.add(
            sys.stderr,
            level=config.logging.level,
            format=(
                "<green>{time:HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:"
                "<cyan>{line}</cyan> | <level>{message}</level>"
            )
        )
        
        logger.info("Logging configured successfully")
    
    def _initialize_collectors(self):
        """Initialize and register data collectors"""
        try:
            # Register Alpha Vantage collector
            if config.api.alpha_vantage_key:
                alpha_vantage = AlphaVantageCollector()
                collector_registry.register_collector(
                    DataSource.ALPHA_VANTAGE,
                    alpha_vantage
                )
                logger.info("Alpha Vantage collector registered")
            else:
                logger.warning("Alpha Vantage API key not configured")
            
            # Register Yahoo Finance collector
            yahoo_finance = YahooFinanceCollector()
            collector_registry.register_collector(
                DataSource.YAHOO_FINANCE,
                yahoo_finance
            )
            logger.info("Yahoo Finance collector registered")
            
            logger.info(
                f"Initialized {len(collector_registry.list_sources())} "
                f"data collectors"
            )
            
        except Exception as e:
            logger.error(f"Error initializing collectors: {e}")
            raise
    
    def _initialize_storage(self):
        """Initialize data storage"""
        try:
            data_storage.initialize_database()
            logger.info("Data storage initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing storage: {e}")
            raise
    
    async def collect_and_process_data(self) -> Dict[str, Any]:
        """Collect and process data from all sources"""
        batch_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        results = {}
        
        logger.info(f"Starting data collection batch: {batch_id}")
        
        try:
            # Get symbols to collect
            symbols = self._get_symbols_to_collect()
            
            if not symbols:
                logger.warning("No symbols configured for collection")
                return results
            
            # Collect tick data from all sources
            tick_results = await collector_registry.collect_from_all_sources(
                symbols, "tick"
            )
            
            # Collect OHLCV data from all sources
            ohlcv_results = await collector_registry.collect_from_all_sources(
                symbols, "ohlcv"
            )
            
            # Process and store data for each source
            for source, tick_data in tick_results.items():
                await self._process_and_store_tick_data(
                    tick_data, source, batch_id
                )
                results[f"{source.value}_tick_records"] = len(tick_data)
            
            for source, ohlcv_data in ohlcv_results.items():
                await self._process_and_store_ohlcv_data(
                    ohlcv_data, source, batch_id
                )
                results[f"{source.value}_ohlcv_records"] = len(ohlcv_data)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            results['processing_time_seconds'] = processing_time
            results['batch_id'] = batch_id
            
            logger.info(
                f"Completed data collection batch {batch_id} "
                f"in {processing_time:.2f} seconds"
            )
            
        except Exception as e:
            logger.error(f"Error in data collection batch {batch_id}: {e}")
            results['error'] = str(e)
        
        return results
    
    async def _process_and_store_tick_data(
        self, tick_data: List, source: DataSource, batch_id: str
    ):
        """Process and store tick data"""
        if not tick_data:
            return
        
        try:
            # Create processing status
            status = ProcessingStatus(
                batch_id=batch_id,
                symbol="ALL",
                source=source,
                status="processing",
                records_processed=0
            )
            
            # Process data
            processed_data = data_processor.process_tick_data(tick_data)
            
            # Store processed data
            if config.storage.save_processed_data:
                success = await data_storage.store_tick_data(processed_data)
                
                if success:
                    status.status = "completed"
                    status.records_processed = len(processed_data)
                    status.completed_at = datetime.utcnow()
                    status.processing_time = Decimal(str(
                        (status.completed_at - status.started_at).total_seconds()
                    ))
                else:
                    status.status = "failed"
                    status.error_message = "Failed to store processed data"
            
            # Store raw data if configured
            if config.storage.save_raw_data:
                await data_storage.store_tick_data(tick_data)
            
            # Generate and store quality metrics
            for symbol in set(data.symbol for data in tick_data):
                symbol_data = [d for d in tick_data if d.symbol == symbol]
                quality_metrics = data_processor.assess_data_quality(
                    symbol_data, symbol, source
                )
                await data_storage.store_data_quality_metrics([quality_metrics])
            
            # Store processing status
            await data_storage.store_processing_status(status)
            
            logger.info(
                f"Processed and stored {len(processed_data)} tick records "
                f"from {source.value}"
            )
            
        except Exception as e:
            logger.error(f"Error processing tick data from {source.value}: {e}")
    
    async def _process_and_store_ohlcv_data(
        self, ohlcv_data: List, source: DataSource, batch_id: str
    ):
        """Process and store OHLCV data"""
        if not ohlcv_data:
            return
        
        try:
            # Create processing status
            status = ProcessingStatus(
                batch_id=batch_id,
                symbol="ALL",
                source=source,
                status="processing",
                records_processed=0
            )
            
            # Process data
            processed_data = data_processor.process_ohlcv_data(ohlcv_data)
            
            # Store processed data
            if config.storage.save_processed_data:
                success = await data_storage.store_ohlcv_data(processed_data)
                
                if success:
                    status.status = "completed"
                    status.records_processed = len(processed_data)
                    status.completed_at = datetime.utcnow()
                    status.processing_time = Decimal(str(
                        (status.completed_at - status.started_at).total_seconds()
                    ))
                else:
                    status.status = "failed"
                    status.error_message = "Failed to store processed data"
            
            # Store raw data if configured
            if config.storage.save_raw_data:
                await data_storage.store_ohlcv_data(ohlcv_data)
            
            # Calculate technical indicators
            indicators = data_processor.calculate_technical_indicators(processed_data)
            logger.info(f"Calculated indicators for {len(indicators)} symbols")
            
            # Store processing status
            await data_storage.store_processing_status(status)
            
            logger.info(
                f"Processed and stored {len(processed_data)} OHLCV records "
                f"from {source.value}"
            )
            
        except Exception as e:
            logger.error(f"Error processing OHLCV data from {source.value}: {e}")
    
    def _get_symbols_to_collect(self) -> List[str]:
        """Get list of symbols to collect data for"""
        symbols = []
        
        if config.data_collection.collect_stocks:
            symbols.extend(config.data_collection.default_stock_symbols)
        
        if config.data_collection.collect_crypto:
            symbols.extend(config.data_collection.default_crypto_symbols)
        
        # Remove duplicates and clean
        symbols = list(set(symbol.strip().upper() for symbol in symbols))
        
        logger.info(f"Collecting data for {len(symbols)} symbols: {symbols}")
        return symbols
    
    async def cleanup_old_data(self):
        """Cleanup old data based on retention policy"""
        try:
            logger.info("Starting data cleanup process")
            success = await data_storage.cleanup_old_data()
            
            if success:
                self.last_cleanup = datetime.utcnow()
                logger.info("Data cleanup completed successfully")
            else:
                logger.error("Data cleanup failed")
                
        except Exception as e:
            logger.error(f"Error during data cleanup: {e}")
    
    async def generate_statistics(self) -> Dict[str, Any]:
        """Generate system statistics"""
        try:
            stats = await data_storage.get_data_statistics()
            stats['orchestrator_status'] = 'running' if self.is_running else 'stopped'
            stats['last_cleanup'] = self.last_cleanup
            stats['collection_interval'] = self.collection_interval
            
            return stats
            
        except Exception as e:
            logger.error(f"Error generating statistics: {e}")
            return {}
    
    async def run_collection_cycle(self):
        """Run a single data collection cycle"""
        try:
            # Collect and process data
            results = await self.collect_and_process_data()
            
            # Check if cleanup is needed
            time_since_cleanup = datetime.utcnow() - self.last_cleanup
            if time_since_cleanup.total_seconds() >= self.cleanup_interval:
                await self.cleanup_old_data()
            
            return results
            
        except Exception as e:
            logger.error(f"Error in collection cycle: {e}")
            return {'error': str(e)}
    
    async def run_continuous(self):
        """Run continuous data collection"""
        self.is_running = True
        logger.info("Starting continuous data collection")
        
        try:
            while self.is_running:
                cycle_start = datetime.utcnow()
                
                # Run collection cycle
                results = await self.run_collection_cycle()
                
                # Calculate sleep time
                cycle_duration = (datetime.utcnow() - cycle_start).total_seconds()
                sleep_time = max(0, self.collection_interval - cycle_duration)
                
                logger.info(
                    f"Collection cycle completed in {cycle_duration:.2f}s, "
                    f"sleeping for {sleep_time:.2f}s"
                )
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
        except asyncio.CancelledError:
            logger.info("Data collection cancelled")
        except Exception as e:
            logger.error(f"Error in continuous collection: {e}")
        finally:
            self.is_running = False
            logger.info("Stopped continuous data collection")
    
    def stop(self):
        """Stop the orchestrator"""
        logger.info("Stopping market data orchestrator...")
        self.is_running = False


# Global orchestrator instance
orchestrator = MarketDataOrchestrator()


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    orchestrator.stop()
    sys.exit(0)


async def main():
    """Main entry point for the data collection system"""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting Market Data Collection System")
    logger.info(f"Configuration: {config.data_collection.__dict__}")
    
    try:
        # Run a single collection cycle for testing
        logger.info("Running single collection cycle...")
        results = await orchestrator.run_collection_cycle()
        logger.info(f"Collection results: {results}")
        
        # Generate statistics
        stats = await orchestrator.generate_statistics()
        logger.info(f"System statistics: {stats}")
        
        # Uncomment the following line to run continuous collection
        # await orchestrator.run_continuous()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        logger.info("Market Data Collection System shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
