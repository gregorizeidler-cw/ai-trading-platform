import os
from pathlib import Path
from typing import List
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent / '.env')


class DatabaseConfig(BaseSettings):
    """Database configuration settings"""
    url: str = Field(
        default=os.getenv('DATABASE_URL', 'sqlite:///trading.db')
    )
    redis_url: str = Field(
        default=os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    )


class APIConfig(BaseSettings):
    """API keys and configuration for financial data sources"""
    alpha_vantage_key: str = Field(
        default=os.getenv('ALPHA_VANTAGE_API_KEY', '')
    )
    polygon_key: str = Field(default=os.getenv('POLYGON_API_KEY', ''))
    finnhub_key: str = Field(default=os.getenv('FINNHUB_API_KEY', ''))
    twelve_data_key: str = Field(default=os.getenv('TWELVE_DATA_API_KEY', ''))
    
    # AI/LLM API keys
    openai_api_key: str = Field(default=os.getenv('OPENAI_API_KEY', ''))
    anthropic_api_key: str = Field(default=os.getenv('ANTHROPIC_API_KEY', ''))
    
    # Crypto exchange keys
    binance_api_key: str = Field(default=os.getenv('BINANCE_API_KEY', ''))
    binance_secret: str = Field(default=os.getenv('BINANCE_SECRET_KEY', ''))
    coinbase_api_key: str = Field(default=os.getenv('COINBASE_API_KEY', ''))
    coinbase_secret: str = Field(default=os.getenv('COINBASE_SECRET', ''))


class DataCollectionConfig(BaseSettings):
    """Configuration for data collection settings"""
    websocket_reconnect_interval: int = Field(
        default=int(os.getenv('WEBSOCKET_RECONNECT_INTERVAL', 5))
    )
    max_reconnect_attempts: int = Field(
        default=int(os.getenv('MAX_RECONNECT_ATTEMPTS', 10))
    )
    data_buffer_size: int = Field(
        default=int(os.getenv('DATA_BUFFER_SIZE', 1000))
    )
    batch_insert_size: int = Field(
        default=int(os.getenv('BATCH_INSERT_SIZE', 100))
    )
    
    # Data types to collect
    collect_stocks: bool = Field(
        default=os.getenv('COLLECT_STOCKS', 'true').lower() == 'true'
    )
    collect_crypto: bool = Field(
        default=os.getenv('COLLECT_CRYPTO', 'true').lower() == 'true'
    )
    collect_forex: bool = Field(
        default=os.getenv('COLLECT_FOREX', 'false').lower() == 'true'
    )
    collect_commodities: bool = Field(
        default=os.getenv('COLLECT_COMMODITIES', 'false').lower() == 'true'
    )
    
    # Default symbols
    default_stock_symbols: List[str] = Field(
        default=os.getenv(
            'DEFAULT_STOCK_SYMBOLS',
            'AAPL,GOOGL,MSFT,TSLA,AMZN'
        ).split(',')
    )
    default_crypto_symbols: List[str] = Field(
        default=os.getenv(
            'DEFAULT_CRYPTO_SYMBOLS',
            'BTC/USD,ETH/USD,BNB/USD'
        ).split(',')
    )


class StorageConfig(BaseSettings):
    """Configuration for data storage"""
    save_raw_data: bool = Field(
        default=os.getenv('SAVE_RAW_DATA', 'true').lower() == 'true'
    )
    save_processed_data: bool = Field(
        default=os.getenv('SAVE_PROCESSED_DATA', 'true').lower() == 'true'
    )
    data_retention_days: int = Field(
        default=int(os.getenv('DATA_RETENTION_DAYS', 365))
    )
    cleanup_interval_hours: int = Field(
        default=int(os.getenv('CLEANUP_INTERVAL_HOURS', 24))
    )


class LoggingConfig(BaseSettings):
    """Logging configuration"""
    level: str = Field(default=os.getenv('LOG_LEVEL', 'INFO'))
    file_path: str = Field(
        default=os.getenv('LOG_FILE_PATH', 'logs/trading_agent.log')
    )
    rotation: str = Field(default=os.getenv('LOG_ROTATION', '1 day'))
    retention: str = Field(default=os.getenv('LOG_RETENTION', '30 days'))


class Config:
    """Main configuration class that combines all settings"""
    def __init__(self):
        self.database = DatabaseConfig()
        self.api = APIConfig()
        self.data_collection = DataCollectionConfig()
        self.storage = StorageConfig()
        self.logging = LoggingConfig()
        
        # Project paths
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / 'data'
        self.raw_data_dir = self.data_dir / 'raw'
        self.processed_data_dir = self.data_dir / 'processed'
        self.models_dir = self.data_dir / 'models'
        
        # Ensure directories exist
        directories = [
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = Config()
