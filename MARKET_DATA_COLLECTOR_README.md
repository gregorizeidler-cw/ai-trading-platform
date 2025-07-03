# Real-Time Financial Market Data Collection System

This system automates the collection, cleaning, and storage of financial market data from multiple sources in real-time. It supports stocks, cryptocurrencies, forex, and commodities data.

## Features

- **Multi-Source Data Collection**: Supports Alpha Vantage, Yahoo Finance, and more
- **Real-Time Processing**: Continuous data collection with configurable intervals
- **Data Quality Assurance**: Automated data cleaning, validation, and quality metrics
- **Scalable Storage**: PostgreSQL/SQLite database with optimized schemas
- **Technical Indicators**: Automatic calculation of RSI, MACD, Moving Averages, etc.
- **Error Handling**: Robust error handling with detailed logging
- **Configuration Management**: Environment-based configuration system

## Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Or using uv (recommended)
uv sync
```

### 2. Configure Environment

Copy the example configuration and update with your API keys:

```bash
cp config/.env.example config/.env
```

Edit `config/.env` and add your API keys:

```env
# API Keys
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
POLYGON_API_KEY=your_polygon_api_key
FINNHUB_API_KEY=your_finnhub_api_key

# Database (optional, defaults to SQLite)
DATABASE_URL=postgresql://username:password@localhost:5432/trading_db

# Symbols to track
DEFAULT_STOCK_SYMBOLS=AAPL,GOOGL,MSFT,TSLA,AMZN,NVDA,META
DEFAULT_CRYPTO_SYMBOLS=BTC/USD,ETH/USD,BNB/USD
```

### 3. Run the System

#### Single Collection Cycle
```bash
python run_data_collector.py --single
```

#### Continuous Collection
```bash
python run_data_collector.py --continuous
```

#### View Statistics
```bash
python run_data_collector.py --stats
```

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Market Data Orchestrator                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │   Data Sources  │  │   Data Processors│  │   Storage   │ │
│  │                 │  │                  │  │             │ │
│  │ • Alpha Vantage │  │ • Data Cleaning  │  │ • PostgreSQL│ │
│  │ • Yahoo Finance │→ │ • Quality Checks │→ │ • Redis     │ │
│  │ • Polygon       │  │ • Indicators     │  │ • File      │ │
│  │ • Binance       │  │ • Validation     │  │   System    │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                      Configuration                          │
│  • Environment Variables  • Logging  • Symbol Management   │
└─────────────────────────────────────────────────────────────┘
```

## Data Sources

### Supported Sources

1. **Alpha Vantage** (requires API key)
   - Real-time and historical stock data
   - Technical indicators
   - Fundamental data
   - Rate limit: 5 calls/minute (free tier)

2. **Yahoo Finance** (free)
   - Real-time stock quotes
   - Historical OHLCV data
   - Company fundamentals
   - No rate limits

3. **Polygon.io** (requires API key)
   - Real-time market data
   - Options and forex data
   - News and fundamentals

4. **Binance** (for crypto)
   - Real-time cryptocurrency data
   - Spot and futures markets
   - WebSocket support

### Adding New Data Sources

To add a new data source:

1. Create a new collector class inheriting from `BaseDataCollector`
2. Implement the required methods: `collect_tick_data`, `collect_ohlcv_data`, `collect_market_data`
3. Register the collector in the orchestrator
4. Add configuration for API keys/settings

Example:

```python
from data_pipeline.collectors.base_collector import BaseDataCollector

class MyDataCollector(BaseDataCollector):
    async def collect_tick_data(self, symbols: List[str]) -> List[TickData]:
        # Implementation here
        pass
    
    # ... other required methods

# Register in orchestrator
collector_registry.register_collector(DataSource.MY_SOURCE, MyDataCollector())
```

## Data Processing Pipeline

### 1. Data Collection
- Multi-threaded collection from various APIs
- Rate limiting and error handling
- Raw data validation

### 2. Data Cleaning
- Duplicate removal
- Missing value handling
- Outlier detection using statistical methods
- Data range validation

### 3. Quality Assessment
- Completeness scoring
- Timeliness evaluation
- Accuracy metrics
- Consistency checks

### 4. Technical Indicators
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands

### 5. Storage
- Batch insertion for performance
- Indexed database schema
- Data retention policies
- Automatic cleanup

## Database Schema

### Market Data Table
```sql
CREATE TABLE market_data (
    id UUID PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    price DECIMAL(18,8) NOT NULL,
    volume DECIMAL(18,8),
    bid DECIMAL(18,8),
    ask DECIMAL(18,8),
    source VARCHAR(50) NOT NULL,
    asset_type VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### OHLCV Data Table
```sql
CREATE TABLE ohlcv_data (
    id UUID PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    open DECIMAL(18,8) NOT NULL,
    high DECIMAL(18,8) NOT NULL,
    low DECIMAL(18,8) NOT NULL,
    close DECIMAL(18,8) NOT NULL,
    volume DECIMAL(18,8) NOT NULL,
    source VARCHAR(50) NOT NULL,
    asset_type VARCHAR(20) NOT NULL
);
```

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection string | `sqlite:///trading.db` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage API key | - |
| `COLLECT_STOCKS` | Enable stock data collection | `true` |
| `COLLECT_CRYPTO` | Enable crypto data collection | `true` |
| `DEFAULT_STOCK_SYMBOLS` | Default stocks to track | `AAPL,GOOGL,MSFT` |
| `DATA_RETENTION_DAYS` | Data retention period | `365` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Collection Settings

```python
# Collection interval (seconds)
COLLECTION_INTERVAL = 60

# Batch size for database operations
BATCH_INSERT_SIZE = 100

# Data quality thresholds
ANOMALY_THRESHOLD = 3.0  # Standard deviations
```

## Monitoring and Logging

### Log Files
- Default location: `logs/trading_agent.log`
- Rotation: Daily
- Retention: 30 days
- Format: Structured JSON logging

### Metrics Available
- Data collection rates
- Processing times
- Error rates
- Data quality scores
- Database statistics

### Example Log Output
```
2024-06-22 10:30:15 | INFO | collector:collect_tick_data:45 | Collected tick data for AAPL
2024-06-22 10:30:16 | INFO | processor:process_tick_data:123 | Processed 150 tick records (5 removed)
2024-06-22 10:30:17 | INFO | storage:store_tick_data:67 | Stored batch of 145 tick records
```

## Performance Optimization

### Database Optimization
- Use connection pooling
- Batch insert operations
- Proper indexing on timestamp and symbol columns
- Regular cleanup of old data

### Memory Management
- Process data in batches
- Use generators for large datasets
- Clear processed data from memory
- Monitor memory usage

### Rate Limiting
- Respect API rate limits
- Implement exponential backoff
- Use multiple API keys for higher limits
- Cache frequent requests

## Error Handling

### Retry Logic
- Automatic retry with exponential backoff
- Maximum retry attempts configurable
- Different strategies for different error types

### Graceful Degradation
- Continue with available data sources if one fails
- Skip problematic data points
- Maintain system operation during partial failures

### Error Notification
- Detailed error logging
- Optional email/Slack notifications
- Health check endpoints

## API Integration Examples

### Alpha Vantage
```python
# Get real-time quote
params = {
    'function': 'GLOBAL_QUOTE',
    'symbol': 'AAPL',
    'apikey': api_key
}
```

### Yahoo Finance
```python
import yfinance as yf

# Get ticker data
ticker = yf.Ticker('AAPL')
data = ticker.history(period='1d', interval='1m')
```

## Deployment

### Development
```bash
python run_data_collector.py --continuous
```

### Production (with systemd)
```bash
# Create service file
sudo cp deployment/trading-data-collector.service /etc/systemd/system/
sudo systemctl enable trading-data-collector
sudo systemctl start trading-data-collector
```

### Docker
```bash
docker build -t trading-data-collector .
docker run -d --name collector trading-data-collector
```

## Troubleshooting

### Common Issues

1. **API Rate Limits**
   - Check rate limits in logs
   - Increase collection intervals
   - Use multiple API keys

2. **Database Connection Issues**
   - Verify DATABASE_URL
   - Check database server status
   - Review connection pool settings

3. **Missing Data**
   - Check API key validity
   - Verify symbol formats
   - Review error logs

4. **High Memory Usage**
   - Reduce batch sizes
   - Increase cleanup frequency
   - Monitor data retention settings

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python run_data_collector.py --single
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code passes linting
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Document functions and classes
- Add unit tests for new features

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Check the troubleshooting section
- Review the logs for error details
- Create an issue on GitHub
- Contact the development team

---

**Note**: This system is designed for educational and research purposes. Ensure compliance with data provider terms of service and applicable financial regulations when using in production environments.
