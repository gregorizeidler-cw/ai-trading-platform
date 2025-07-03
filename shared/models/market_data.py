"""
Data models for financial market data used across the trading agent system.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class AssetType(str, Enum):
    """Enumeration of supported asset types"""
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
    INDEX = "index"


class DataSource(str, Enum):
    """Enumeration of data sources"""
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    FINNHUB = "finnhub"
    TWELVE_DATA = "twelve_data"
    BINANCE = "binance"
    COINBASE = "coinbase"
    YAHOO_FINANCE = "yahoo_finance"
    WEBSOCKET = "websocket"


class MarketStatus(str, Enum):
    """Market status enumeration"""
    OPEN = "open"
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    AFTER_HOURS = "after_hours"
    HOLIDAY = "holiday"


class TickData(BaseModel):
    """Real-time tick data model"""
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Data timestamp")
    price: Decimal = Field(..., description="Current price")
    volume: Optional[Decimal] = Field(None, description="Volume traded")
    bid: Optional[Decimal] = Field(None, description="Bid price")
    ask: Optional[Decimal] = Field(None, description="Ask price")
    bid_size: Optional[Decimal] = Field(None, description="Bid size")
    ask_size: Optional[Decimal] = Field(None, description="Ask size")
    source: DataSource = Field(..., description="Data source")
    asset_type: AssetType = Field(..., description="Asset type")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            Decimal: lambda v: float(v),
            datetime: lambda v: v.isoformat()
        }


class OHLCVData(BaseModel):
    """OHLCV (Open, High, Low, Close, Volume) data model"""
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Data timestamp")
    timeframe: str = Field(..., description="Time interval (1m, 5m, 1h, 1d, etc.)")
    open: Decimal = Field(..., description="Opening price")
    high: Decimal = Field(..., description="Highest price")
    low: Decimal = Field(..., description="Lowest price")
    close: Decimal = Field(..., description="Closing price")
    volume: Decimal = Field(..., description="Trading volume")
    source: DataSource = Field(..., description="Data source")
    asset_type: AssetType = Field(..., description="Asset type")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            Decimal: lambda v: float(v),
            datetime: lambda v: v.isoformat()
        }


class MarketData(BaseModel):
    """Comprehensive market data model"""
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Data timestamp")
    price: Decimal = Field(..., description="Current price")
    volume: Optional[Decimal] = Field(None, description="Volume")
    market_cap: Optional[Decimal] = Field(None, description="Market capitalization")
    pe_ratio: Optional[Decimal] = Field(None, description="Price-to-earnings ratio")
    fifty_two_week_high: Optional[Decimal] = Field(None, description="52-week high")
    fifty_two_week_low: Optional[Decimal] = Field(None, description="52-week low")
    dividend_yield: Optional[Decimal] = Field(None, description="Dividend yield")
    beta: Optional[Decimal] = Field(None, description="Beta coefficient")
    moving_avg_50: Optional[Decimal] = Field(None, description="50-day moving average")
    moving_avg_200: Optional[Decimal] = Field(None, description="200-day moving average")
    rsi: Optional[Decimal] = Field(None, description="Relative Strength Index")
    macd: Optional[Decimal] = Field(None, description="MACD indicator")
    source: DataSource = Field(..., description="Data source")
    asset_type: AssetType = Field(..., description="Asset type")
    market_status: Optional[MarketStatus] = Field(None, description="Market status")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            Decimal: lambda v: float(v),
            datetime: lambda v: v.isoformat()
        }


class NewsData(BaseModel):
    """Financial news data model"""
    headline: str = Field(..., description="News headline")
    summary: Optional[str] = Field(None, description="News summary")
    url: Optional[str] = Field(None, description="News URL")
    source: str = Field(..., description="News source")
    timestamp: datetime = Field(..., description="Publication timestamp")
    symbols: List[str] = Field(default_factory=list, description="Related symbols")
    sentiment: Optional[str] = Field(None, description="Sentiment analysis")
    relevance_score: Optional[Decimal] = Field(None, description="Relevance score")
    
    class Config:
        json_encoders = {
            Decimal: lambda v: float(v),
            datetime: lambda v: v.isoformat()
        }


class EconomicIndicator(BaseModel):
    """Economic indicator data model"""
    indicator_name: str = Field(..., description="Indicator name")
    value: Decimal = Field(..., description="Indicator value")
    timestamp: datetime = Field(..., description="Data timestamp")
    country: Optional[str] = Field(None, description="Country code")
    frequency: Optional[str] = Field(None, description="Data frequency")
    unit: Optional[str] = Field(None, description="Value unit")
    source: DataSource = Field(..., description="Data source")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            Decimal: lambda v: float(v),
            datetime: lambda v: v.isoformat()
        }


class DataCollectionTask(BaseModel):
    """Data collection task configuration"""
    task_id: str = Field(..., description="Unique task identifier")
    symbols: List[str] = Field(..., description="Symbols to collect")
    asset_type: AssetType = Field(..., description="Asset type")
    data_sources: List[DataSource] = Field(..., description="Data sources")
    interval: int = Field(..., description="Collection interval in seconds")
    active: bool = Field(default=True, description="Task active status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_run: Optional[datetime] = Field(None, description="Last execution time")
    next_run: Optional[datetime] = Field(None, description="Next execution time")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DataQualityMetrics(BaseModel):
    """Data quality metrics for monitoring"""
    symbol: str = Field(..., description="Symbol")
    timestamp: datetime = Field(..., description="Metrics timestamp")
    completeness: Decimal = Field(..., description="Data completeness percentage")
    timeliness: Decimal = Field(..., description="Data timeliness score")
    accuracy: Decimal = Field(..., description="Data accuracy score")
    consistency: Decimal = Field(..., description="Data consistency score")
    anomaly_count: int = Field(default=0, description="Number of anomalies detected")
    missing_data_points: int = Field(default=0, description="Missing data points")
    duplicate_records: int = Field(default=0, description="Duplicate records")
    source: DataSource = Field(..., description="Data source")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            Decimal: lambda v: float(v),
            datetime: lambda v: v.isoformat()
        }


class ProcessingStatus(BaseModel):
    """Data processing status tracking"""
    batch_id: str = Field(..., description="Batch identifier")
    symbol: str = Field(..., description="Symbol")
    source: DataSource = Field(..., description="Data source")
    status: str = Field(..., description="Processing status")
    records_processed: int = Field(default=0, description="Records processed")
    errors_count: int = Field(default=0, description="Error count")
    processing_time: Optional[Decimal] = Field(None, description="Processing time in seconds")
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    error_message: Optional[str] = Field(None, description="Error details")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            Decimal: lambda v: float(v),
            datetime: lambda v: v.isoformat()
        }
