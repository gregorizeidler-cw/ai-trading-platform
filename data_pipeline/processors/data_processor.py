"""
Data processing and cleaning utilities for financial market data.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from decimal import Decimal
from loguru import logger

from shared.models.market_data import (
    TickData, OHLCVData, MarketData, DataQualityMetrics, DataSource
)


class DataProcessor:
    """Data processing and cleaning for financial market data"""
    
    def __init__(self):
        self.anomaly_threshold = 3.0  # Standard deviations for anomaly detection
        
    def process_tick_data(self, tick_data: List[TickData]) -> List[TickData]:
        """Process and clean tick data"""
        if not tick_data:
            return []
            
        logger.info(f"Processing {len(tick_data)} tick data records")
        
        # Convert to DataFrame for processing
        df = self._tick_data_to_dataframe(tick_data)
        
        # Remove duplicates
        df = self._remove_duplicates(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Remove outliers
        df = self._remove_outliers(df, ['price', 'volume'])
        
        # Validate data ranges
        df = self._validate_data_ranges(df)
        
        # Convert back to TickData objects
        processed_data = self._dataframe_to_tick_data(df)
        
        logger.info(
            f"Processed data: {len(processed_data)} records remaining "
            f"({len(tick_data) - len(processed_data)} removed)"
        )
        
        return processed_data
    
    def process_ohlcv_data(
        self, ohlcv_data: List[OHLCVData]
    ) -> List[OHLCVData]:
        """Process and clean OHLCV data"""
        if not ohlcv_data:
            return []
            
        logger.info(f"Processing {len(ohlcv_data)} OHLCV data records")
        
        # Convert to DataFrame for processing
        df = self._ohlcv_data_to_dataframe(ohlcv_data)
        
        # Remove duplicates
        df = self._remove_duplicates(df, ['symbol', 'timestamp', 'timeframe'])
        
        # Validate OHLCV relationships
        df = self._validate_ohlcv_relationships(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Remove outliers
        df = self._remove_outliers(df, ['open', 'high', 'low', 'close', 'volume'])
        
        # Convert back to OHLCVData objects
        processed_data = self._dataframe_to_ohlcv_data(df)
        
        logger.info(
            f"Processed data: {len(processed_data)} records remaining "
            f"({len(ohlcv_data) - len(processed_data)} removed)"
        )
        
        return processed_data
    
    def calculate_technical_indicators(
        self, ohlcv_data: List[OHLCVData]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Calculate technical indicators from OHLCV data"""
        if not ohlcv_data:
            return {}
            
        # Group by symbol
        symbol_groups = {}
        for data in ohlcv_data:
            if data.symbol not in symbol_groups:
                symbol_groups[data.symbol] = []
            symbol_groups[data.symbol].append(data)
        
        indicators = {}
        
        for symbol, data_list in symbol_groups.items():
            try:
                df = self._ohlcv_data_to_dataframe(data_list)
                df = df.sort_values('timestamp')
                
                # Calculate moving averages
                df['sma_20'] = df['close'].rolling(window=20).mean()
                df['sma_50'] = df['close'].rolling(window=50).mean()
                df['ema_12'] = df['close'].ewm(span=12).mean()
                df['ema_26'] = df['close'].ewm(span=26).mean()
                
                # Calculate RSI
                df['rsi'] = self._calculate_rsi(df['close'])
                
                # Calculate MACD
                df['macd'] = df['ema_12'] - df['ema_26']
                df['macd_signal'] = df['macd'].ewm(span=9).mean()
                df['macd_histogram'] = df['macd'] - df['macd_signal']
                
                # Calculate Bollinger Bands
                bb_period = 20
                bb_std = 2
                df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
                bb_std_val = df['close'].rolling(window=bb_period).std()
                df['bb_upper'] = df['bb_middle'] + (bb_std_val * bb_std)
                df['bb_lower'] = df['bb_middle'] - (bb_std_val * bb_std)
                
                # Convert to list of dictionaries
                indicators[symbol] = df.to_dict('records')
                
                logger.info(f"Calculated indicators for {symbol}")
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {e}")
                
        return indicators
    
    def assess_data_quality(
        self, 
        data: List[Any], 
        symbol: str, 
        source: DataSource
    ) -> DataQualityMetrics:
        """Assess data quality and generate metrics"""
        if not data:
            return DataQualityMetrics(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                completeness=Decimal('0'),
                timeliness=Decimal('0'),
                accuracy=Decimal('0'),
                consistency=Decimal('0'),
                source=source
            )
        
        # Calculate completeness
        total_fields = len(data[0].__dict__)
        complete_records = 0
        
        for record in data:
            complete_fields = sum(
                1 for value in record.__dict__.values() 
                if value is not None
            )
            if complete_fields / total_fields >= 0.8:
                complete_records += 1
        
        completeness = Decimal(str(complete_records / len(data) * 100))
        
        # Calculate timeliness (data freshness)
        latest_timestamp = max(record.timestamp for record in data)
        time_diff = datetime.utcnow() - latest_timestamp
        timeliness_hours = max(0, 24 - time_diff.total_seconds() / 3600)
        timeliness = Decimal(str(min(100, timeliness_hours / 24 * 100)))
        
        # Calculate accuracy (price consistency)
        accuracy = self._calculate_accuracy(data)
        
        # Calculate consistency (temporal consistency)
        consistency = self._calculate_consistency(data)
        
        return DataQualityMetrics(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            completeness=completeness,
            timeliness=timeliness,
            accuracy=accuracy,
            consistency=consistency,
            source=source
        )
    
    def _tick_data_to_dataframe(self, tick_data: List[TickData]) -> pd.DataFrame:
        """Convert tick data to pandas DataFrame"""
        records = []
        for tick in tick_data:
            record = {
                'symbol': tick.symbol,
                'timestamp': tick.timestamp,
                'price': float(tick.price),
                'volume': float(tick.volume) if tick.volume else 0,
                'bid': float(tick.bid) if tick.bid else 0,
                'ask': float(tick.ask) if tick.ask else 0,
                'source': tick.source.value,
                'asset_type': tick.asset_type.value
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def _ohlcv_data_to_dataframe(
        self, ohlcv_data: List[OHLCVData]
    ) -> pd.DataFrame:
        """Convert OHLCV data to pandas DataFrame"""
        records = []
        for ohlcv in ohlcv_data:
            record = {
                'symbol': ohlcv.symbol,
                'timestamp': ohlcv.timestamp,
                'timeframe': ohlcv.timeframe,
                'open': float(ohlcv.open),
                'high': float(ohlcv.high),
                'low': float(ohlcv.low),
                'close': float(ohlcv.close),
                'volume': float(ohlcv.volume),
                'source': ohlcv.source.value,
                'asset_type': ohlcv.asset_type.value
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def _remove_duplicates(
        self, 
        df: pd.DataFrame, 
        subset: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Remove duplicate records"""
        if subset is None:
            subset = ['symbol', 'timestamp']
        
        initial_count = len(df)
        df = df.drop_duplicates(subset=subset, keep='last')
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate records")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data"""
        # Remove records with missing critical fields
        critical_fields = ['symbol', 'timestamp', 'price']
        df = df.dropna(subset=critical_fields)
        
        # Fill missing volume with 0
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0)
        
        return df
    
    def _remove_outliers(
        self, df: pd.DataFrame, columns: List[str]
    ) -> pd.DataFrame:
        """Remove outliers using z-score method"""
        initial_count = len(df)
        
        for column in columns:
            if column in df.columns:
                z_scores = np.abs(
                    (df[column] - df[column].mean()) / df[column].std()
                )
                df = df[z_scores < self.anomaly_threshold]
        
        removed_count = initial_count - len(df)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} outlier records")
        
        return df
    
    def _validate_data_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data ranges (e.g., prices > 0)"""
        initial_count = len(df)
        
        # Remove records with non-positive prices
        if 'price' in df.columns:
            df = df[df['price'] > 0]
        
        # Remove records with negative volume
        if 'volume' in df.columns:
            df = df[df['volume'] >= 0]
        
        removed_count = initial_count - len(df)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} records with invalid ranges")
        
        return df
    
    def _validate_ohlcv_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLCV relationships (high >= low, etc.)"""
        initial_count = len(df)
        
        # Validate high >= low
        df = df[df['high'] >= df['low']]
        
        # Validate high >= open and high >= close
        df = df[(df['high'] >= df['open']) & (df['high'] >= df['close'])]
        
        # Validate low <= open and low <= close
        df = df[(df['low'] <= df['open']) & (df['low'] <= df['close'])]
        
        removed_count = initial_count - len(df)
        if removed_count > 0:
            logger.info(
                f"Removed {removed_count} records with invalid OHLCV relationships"
            )
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_accuracy(self, data: List[Any]) -> Decimal:
        """Calculate data accuracy score"""
        # Simple accuracy calculation based on price reasonableness
        if not data:
            return Decimal('0')
        
        valid_prices = 0
        for record in data:
            if hasattr(record, 'price') and record.price > 0:
                valid_prices += 1
        
        accuracy = valid_prices / len(data) * 100
        return Decimal(str(accuracy))
    
    def _calculate_consistency(self, data: List[Any]) -> Decimal:
        """Calculate data consistency score"""
        if len(data) < 2:
            return Decimal('100')
        
        # Check temporal ordering
        sorted_data = sorted(data, key=lambda x: x.timestamp)
        consistent_ordering = (sorted_data == data)
        
        consistency = 100 if consistent_ordering else 80
        return Decimal(str(consistency))
    
    def _dataframe_to_tick_data(self, df: pd.DataFrame) -> List[TickData]:
        """Convert DataFrame back to TickData objects"""
        tick_data = []
        
        for _, row in df.iterrows():
            try:
                tick = TickData(
                    symbol=row['symbol'],
                    timestamp=row['timestamp'],
                    price=Decimal(str(row['price'])),
                    volume=Decimal(str(row['volume'])) if row['volume'] else None,
                    bid=Decimal(str(row['bid'])) if row['bid'] else None,
                    ask=Decimal(str(row['ask'])) if row['ask'] else None,
                    source=DataSource(row['source']),
                    asset_type=row['asset_type']
                )
                tick_data.append(tick)
            except Exception as e:
                logger.error(f"Error converting row to TickData: {e}")
                
        return tick_data
    
    def _dataframe_to_ohlcv_data(self, df: pd.DataFrame) -> List[OHLCVData]:
        """Convert DataFrame back to OHLCVData objects"""
        ohlcv_data = []
        
        for _, row in df.iterrows():
            try:
                ohlcv = OHLCVData(
                    symbol=row['symbol'],
                    timestamp=row['timestamp'],
                    timeframe=row['timeframe'],
                    open=Decimal(str(row['open'])),
                    high=Decimal(str(row['high'])),
                    low=Decimal(str(row['low'])),
                    close=Decimal(str(row['close'])),
                    volume=Decimal(str(row['volume'])),
                    source=DataSource(row['source']),
                    asset_type=row['asset_type']
                )
                ohlcv_data.append(ohlcv)
            except Exception as e:
                logger.error(f"Error converting row to OHLCVData: {e}")
                
        return ohlcv_data


# Global data processor instance
data_processor = DataProcessor()
