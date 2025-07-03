"""
Database models and connection utilities for the trading agent.
"""

from sqlalchemy import (
    create_engine, Column, String, DateTime, Numeric, Integer,
    Text, Index, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
from config.settings import config

Base = declarative_base()


class MarketDataTable(Base):
    """Market data storage table"""
    __tablename__ = 'market_data'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    price = Column(Numeric(18, 8), nullable=False)
    volume = Column(Numeric(18, 8))
    bid = Column(Numeric(18, 8))
    ask = Column(Numeric(18, 8))
    bid_size = Column(Numeric(18, 8))
    ask_size = Column(Numeric(18, 8))
    source = Column(String(50), nullable=False)
    asset_type = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_timestamp_source', 'timestamp', 'source'),
    )


class OHLCVTable(Base):
    """OHLCV data storage table"""
    __tablename__ = 'ohlcv_data'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    open = Column(Numeric(18, 8), nullable=False)
    high = Column(Numeric(18, 8), nullable=False)
    low = Column(Numeric(18, 8), nullable=False)
    close = Column(Numeric(18, 8), nullable=False)
    volume = Column(Numeric(18, 8), nullable=False)
    source = Column(String(50), nullable=False)
    asset_type = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index(
            'idx_symbol_timeframe_timestamp',
            'symbol',
            'timeframe',
            'timestamp'
        ),
    )


class NewsTable(Base):
    """News data storage table"""
    __tablename__ = 'news_data'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    headline = Column(Text, nullable=False)
    summary = Column(Text)
    url = Column(Text)
    source = Column(String(100), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    sentiment = Column(String(20))
    relevance_score = Column(Numeric(5, 4))
    created_at = Column(DateTime, default=datetime.utcnow)


class NewsSymbolTable(Base):
    """News-symbol relationship table"""
    __tablename__ = 'news_symbols'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    news_id = Column(UUID(as_uuid=True), ForeignKey('news_data.id'))
    symbol = Column(String(20), nullable=False, index=True)
    
    news = relationship("NewsTable")


class DataQualityTable(Base):
    """Data quality metrics storage table"""
    __tablename__ = 'data_quality'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    completeness = Column(Numeric(5, 2), nullable=False)
    timeliness = Column(Numeric(5, 2), nullable=False)
    accuracy = Column(Numeric(5, 2), nullable=False)
    consistency = Column(Numeric(5, 2), nullable=False)
    anomaly_count = Column(Integer, default=0)
    missing_data_points = Column(Integer, default=0)
    duplicate_records = Column(Integer, default=0)
    source = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ProcessingStatusTable(Base):
    """Data processing status tracking table"""
    __tablename__ = 'processing_status'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    batch_id = Column(String(100), nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    source = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False)
    records_processed = Column(Integer, default=0)
    errors_count = Column(Integer, default=0)
    processing_time = Column(Numeric(10, 4))
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    error_message = Column(Text)


class DatabaseManager:
    """Database connection and session management"""
    
    def __init__(self):
        self.engine = create_engine(
            config.database.url,
            pool_pre_ping=True,
            pool_recycle=300,
            echo=False
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
        
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
        
    def close_session(self, session):
        """Close database session"""
        session.close()


# Global database manager instance
db_manager = DatabaseManager()
