"""
Data storage management for financial market data.
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta
from loguru import logger

from shared.models.market_data import (
    TickData, OHLCVData, NewsData,
    DataQualityMetrics, ProcessingStatus
)
from shared.database.models import (
    db_manager, MarketDataTable, OHLCVTable, NewsTable,
    NewsSymbolTable, DataQualityTable, ProcessingStatusTable
)
from config.settings import config


class DataStorage:
    """Data storage manager for financial market data"""
    
    def __init__(self):
        self.db_manager = db_manager
        self.batch_size = config.data_collection.batch_insert_size
        
    def initialize_database(self):
        """Initialize database tables"""
        try:
            self.db_manager.create_tables()
            logger.info("Database tables initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    async def store_tick_data(self, tick_data: List[TickData]) -> bool:
        """Store tick data in the database"""
        if not tick_data:
            return True
            
        try:
            session = self.db_manager.get_session()
            
            # Process in batches
            for i in range(0, len(tick_data), self.batch_size):
                batch = tick_data[i:i + self.batch_size]
                
                for tick in batch:
                    db_record = MarketDataTable(
                        symbol=tick.symbol,
                        timestamp=tick.timestamp,
                        price=tick.price,
                        volume=tick.volume,
                        bid=tick.bid,
                        ask=tick.ask,
                        bid_size=tick.bid_size,
                        ask_size=tick.ask_size,
                        source=tick.source.value,
                        asset_type=tick.asset_type.value
                    )
                    session.add(db_record)
                
                session.commit()
                logger.info(f"Stored batch of {len(batch)} tick records")
            
            session.close()
            logger.info(f"Successfully stored {len(tick_data)} tick records")
            return True
            
        except Exception as e:
            logger.error(f"Error storing tick data: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    async def store_ohlcv_data(self, ohlcv_data: List[OHLCVData]) -> bool:
        """Store OHLCV data in the database"""
        if not ohlcv_data:
            return True
            
        try:
            session = self.db_manager.get_session()
            
            # Process in batches
            for i in range(0, len(ohlcv_data), self.batch_size):
                batch = ohlcv_data[i:i + self.batch_size]
                
                for ohlcv in batch:
                    db_record = OHLCVTable(
                        symbol=ohlcv.symbol,
                        timestamp=ohlcv.timestamp,
                        timeframe=ohlcv.timeframe,
                        open=ohlcv.open,
                        high=ohlcv.high,
                        low=ohlcv.low,
                        close=ohlcv.close,
                        volume=ohlcv.volume,
                        source=ohlcv.source.value,
                        asset_type=ohlcv.asset_type.value
                    )
                    session.add(db_record)
                
                session.commit()
                logger.info(f"Stored batch of {len(batch)} OHLCV records")
            
            session.close()
            logger.info(f"Successfully stored {len(ohlcv_data)} OHLCV records")
            return True
            
        except Exception as e:
            logger.error(f"Error storing OHLCV data: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    async def store_news_data(self, news_data: List[NewsData]) -> bool:
        """Store news data in the database"""
        if not news_data:
            return True
            
        try:
            session = self.db_manager.get_session()
            
            for news in news_data:
                # Store news record
                db_record = NewsTable(
                    headline=news.headline,
                    summary=news.summary,
                    url=news.url,
                    source=news.source,
                    timestamp=news.timestamp,
                    sentiment=news.sentiment,
                    relevance_score=news.relevance_score
                )
                session.add(db_record)
                session.flush()  # Get the ID
                
                # Store related symbols
                for symbol in news.symbols:
                    symbol_record = NewsSymbolTable(
                        news_id=db_record.id,
                        symbol=symbol
                    )
                    session.add(symbol_record)
            
            session.commit()
            session.close()
            logger.info(f"Successfully stored {len(news_data)} news records")
            return True
            
        except Exception as e:
            logger.error(f"Error storing news data: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    async def store_data_quality_metrics(
        self, metrics: List[DataQualityMetrics]
    ) -> bool:
        """Store data quality metrics"""
        if not metrics:
            return True
            
        try:
            session = self.db_manager.get_session()
            
            for metric in metrics:
                db_record = DataQualityTable(
                    symbol=metric.symbol,
                    timestamp=metric.timestamp,
                    completeness=metric.completeness,
                    timeliness=metric.timeliness,
                    accuracy=metric.accuracy,
                    consistency=metric.consistency,
                    anomaly_count=metric.anomaly_count,
                    missing_data_points=metric.missing_data_points,
                    duplicate_records=metric.duplicate_records,
                    source=metric.source.value
                )
                session.add(db_record)
            
            session.commit()
            session.close()
            logger.info(f"Successfully stored {len(metrics)} quality metrics")
            return True
            
        except Exception as e:
            logger.error(f"Error storing quality metrics: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    async def store_processing_status(
        self, status: ProcessingStatus
    ) -> bool:
        """Store processing status"""
        try:
            session = self.db_manager.get_session()
            
            db_record = ProcessingStatusTable(
                batch_id=status.batch_id,
                symbol=status.symbol,
                source=status.source.value,
                status=status.status,
                records_processed=status.records_processed,
                errors_count=status.errors_count,
                processing_time=status.processing_time,
                started_at=status.started_at,
                completed_at=status.completed_at,
                error_message=status.error_message
            )
            session.add(db_record)
            session.commit()
            session.close()
            
            logger.info(f"Stored processing status for batch {status.batch_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing processing status: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    async def get_latest_data(
        self, 
        symbol: str, 
        data_type: str = "tick",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get latest data for a symbol"""
        try:
            session = self.db_manager.get_session()
            
            if data_type == "tick":
                query = session.query(MarketDataTable).filter(
                    MarketDataTable.symbol == symbol
                ).order_by(
                    MarketDataTable.timestamp.desc()
                ).limit(limit)
            elif data_type == "ohlcv":
                query = session.query(OHLCVTable).filter(
                    OHLCVTable.symbol == symbol
                ).order_by(
                    OHLCVTable.timestamp.desc()
                ).limit(limit)
            else:
                logger.error(f"Unknown data type: {data_type}")
                return []
            
            results = query.all()
            session.close()
            
            # Convert to dictionaries
            data = []
            for result in results:
                record = {
                    column.name: getattr(result, column.name)
                    for column in result.__table__.columns
                }
                data.append(record)
            
            logger.info(f"Retrieved {len(data)} {data_type} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error retrieving data: {e}")
            if 'session' in locals():
                session.close()
            return []
    
    async def cleanup_old_data(self, retention_days: int = None) -> bool:
        """Clean up old data based on retention policy"""
        if retention_days is None:
            retention_days = config.storage.data_retention_days
            
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        try:
            session = self.db_manager.get_session()
            
            # Clean up old market data
            deleted_market = session.query(MarketDataTable).filter(
                MarketDataTable.timestamp < cutoff_date
            ).delete()
            
            # Clean up old OHLCV data
            deleted_ohlcv = session.query(OHLCVTable).filter(
                OHLCVTable.timestamp < cutoff_date
            ).delete()
            
            # Clean up old news data
            deleted_news = session.query(NewsTable).filter(
                NewsTable.timestamp < cutoff_date
            ).delete()
            
            session.commit()
            session.close()
            
            total_deleted = deleted_market + deleted_ohlcv + deleted_news
            logger.info(
                f"Cleaned up {total_deleted} old records "
                f"(older than {retention_days} days)"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    async def get_data_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            session = self.db_manager.get_session()
            
            # Count records in each table
            market_data_count = session.query(MarketDataTable).count()
            ohlcv_count = session.query(OHLCVTable).count()
            news_count = session.query(NewsTable).count()
            quality_count = session.query(DataQualityTable).count()
            
            # Get date ranges
            latest_market_data = session.query(
                MarketDataTable.timestamp
            ).order_by(
                MarketDataTable.timestamp.desc()
            ).first()
            
            oldest_market_data = session.query(
                MarketDataTable.timestamp
            ).order_by(
                MarketDataTable.timestamp.asc()
            ).first()
            
            session.close()
            
            stats = {
                'total_market_data_records': market_data_count,
                'total_ohlcv_records': ohlcv_count,
                'total_news_records': news_count,
                'total_quality_metrics': quality_count,
                'latest_data_timestamp': (
                    latest_market_data[0] if latest_market_data else None
                ),
                'oldest_data_timestamp': (
                    oldest_market_data[0] if oldest_market_data else None
                ),
                'generated_at': datetime.utcnow()
            }
            
            logger.info("Generated database statistics")
            return stats
            
        except Exception as e:
            logger.error(f"Error generating statistics: {e}")
            if 'session' in locals():
                session.close()
            return {}


# Global storage instance
data_storage = DataStorage()
