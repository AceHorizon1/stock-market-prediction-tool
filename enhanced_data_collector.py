"""
Enhanced Data Collector for Stock Market AI using Alpha Vantage API
Features: Caching, parallel processing, async operations, comprehensive error handling
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
from datetime import datetime, timedelta
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import time
import warnings
import requests
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
warnings.filterwarnings('ignore')

from config import config

logger = logging.getLogger(__name__)

class EnhancedDataCollector:
    """
    Enhanced data collector with Alpha Vantage API, caching, parallel processing, and async operations
    """
    
    def __init__(self, api_key: str = None):
        self.cache_dir = config.data.cache_dir
        self.max_workers = config.data.max_workers
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Alpha Vantage API setup
        self.api_key = api_key or config.api.alpha_vantage_key
        if not self.api_key:
            logger.warning("Alpha Vantage API key not found. Please set it in config or pass to constructor.")
        
        self.ts = TimeSeries(key=self.api_key, output_format='pandas')
        self.ti = TechIndicators(key=self.api_key, output_format='pandas')
        
        self._setup_cache()
    
    def _setup_cache(self) -> None:
        """Setup cache directory and structure"""
        self.cache_dir.mkdir(exist_ok=True)
        (self.cache_dir / 'raw').mkdir(exist_ok=True)
        (self.cache_dir / 'processed').mkdir(exist_ok=True)
    
    def _get_cache_key(self, symbols: List[str], period: str, **kwargs) -> str:
        """Generate cache key for data"""
        current_date = datetime.now()
        if hasattr(current_date, 'strftime'):
            date_str = current_date.strftime('%Y%m%d')
        else:
            date_str = str(current_date.date())
        key_data = f"{','.join(sorted(symbols))}_{period}_{date_str}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str, data_type: str = 'raw') -> Path:
        """Get cache file path"""
        return self.cache_dir / data_type / f"{cache_key}.pkl"
    
    def _load_from_cache(self, cache_key: str, data_type: str = 'raw') -> Optional[pd.DataFrame]:
        """Load data from cache"""
        cache_path = self._get_cache_path(cache_key, data_type)
        if cache_path.exists():
            try:
                # Check if cache is not older than 1 day
                if time.time() - cache_path.stat().st_mtime < 86400:  # 24 hours
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                    logger.info(f"Loaded data from cache: {cache_path}")
                    return data
                else:
                    logger.info(f"Cache expired: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None
    
    def _save_to_cache(self, data: pd.DataFrame, cache_key: str, data_type: str = 'raw') -> None:
        """Save data to cache"""
        try:
            cache_path = self._get_cache_path(cache_key, data_type)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved data to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _fetch_stock_data_alpha_vantage(self, symbol: str, period: str = 'daily') -> Optional[pd.DataFrame]:
        """
        Fetch stock data using Alpha Vantage API
        
        Args:
            symbol: Stock symbol
            period: Time period (daily, weekly, monthly, intraday)
            
        Returns:
            DataFrame with stock data
        """
        try:
            if not self.api_key:
                raise ValueError("Alpha Vantage API key not configured")
            
            # Map period to Alpha Vantage function
            period_mapping = {
                '1d': 'TIME_SERIES_INTRADAY',
                '5d': 'TIME_SERIES_INTRADAY',
                '1mo': 'TIME_SERIES_DAILY',
                '3mo': 'TIME_SERIES_DAILY',
                '6mo': 'TIME_SERIES_DAILY',
                '1y': 'TIME_SERIES_DAILY',
                '2y': 'TIME_SERIES_DAILY',
                '5y': 'TIME_SERIES_DAILY',
                '10y': 'TIME_SERIES_DAILY',
                'ytd': 'TIME_SERIES_DAILY',
                'max': 'TIME_SERIES_DAILY'
            }
            
            function = period_mapping.get(period, 'TIME_SERIES_DAILY')
            
            if function == 'TIME_SERIES_INTRADAY':
                # For intraday data
                data, meta_data = self.ts.get_intraday(symbol=symbol, interval='5min', outputsize='full')
            elif function == 'TIME_SERIES_DAILY':
                # For daily data
                data, meta_data = self.ts.get_daily(symbol=symbol, outputsize='full')
            else:
                # Default to daily
                data, meta_data = self.ts.get_daily(symbol=symbol, outputsize='full')
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return None
            
            # Standardize column names
            data.columns = [col.split('. ')[-1] if '. ' in col else col for col in data.columns]
            
            # Ensure we have the required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            available_cols = [col.lower() for col in data.columns]
            
            # Map Alpha Vantage columns to standard format
            column_mapping = {
                '1. open': 'Open',
                '2. high': 'High', 
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            
            # Rename columns
            data = data.rename(columns=column_mapping)
            
            # Add Symbol column
            data['Symbol'] = symbol
            
            # Convert index to datetime if it's not already
            if not isinstance(data.index, pd.DatetimeIndex):
                try:
                    data.index = pd.to_datetime(data.index)
                except Exception as e:
                    logger.warning(f"Could not convert index to datetime for {symbol}: {e}")
            
            logger.info(f"Successfully fetched data for {symbol}: {len(data)} rows")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    async def _fetch_stock_data_async(self, symbol: str, period: str) -> Tuple[str, Optional[pd.DataFrame]]:
        """
        Fetch stock data asynchronously using Alpha Vantage
        
        Args:
            symbol: Stock symbol
            period: Time period
            
        Returns:
            Tuple of (symbol, data)
        """
        try:
            # Use ThreadPoolExecutor to run the synchronous Alpha Vantage call
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None, self._fetch_stock_data_alpha_vantage, symbol, period
            )
            
            return symbol, data
            
        except Exception as e:
            logger.error(f"Error in async fetch for {symbol}: {e}")
            return symbol, None
    
    async def fetch_multiple_stocks_async(self, symbols: List[str], period: str) -> pd.DataFrame:
        """
        Fetch multiple stocks asynchronously
        
        Args:
            symbols: List of stock symbols
            period: Time period
            
        Returns:
            Combined DataFrame
        """
        # Check cache first
        cache_key = self._get_cache_key(symbols, period)
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        logger.info(f"Fetching data for {len(symbols)} symbols using async processing")
        start_time = time.time()
        
        # Create aiohttp session if not exists
        if self.session is None:
            self.session = aiohttp.ClientSession()
        
        # Fetch data for all symbols concurrently
        tasks = [self._fetch_stock_data_async(symbol, period) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        dataframes = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Exception in async fetch: {result}")
                continue
            
            symbol, data = result
            if data is not None:
                dataframes.append(data)
        
        if not dataframes:
            raise ValueError("No data could be fetched for any symbol")
        
        # Combine all dataframes
        combined_data = pd.concat(dataframes, ignore_index=True)
        
        # Save to cache
        self._save_to_cache(combined_data, cache_key)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Fetched data for {len(symbols)} symbols in {elapsed_time:.2f} seconds")
        
        return combined_data
    
    def fetch_stocks_parallel(self, symbols: List[str], period: str) -> pd.DataFrame:
        """
        Fetch stocks using parallel processing (synchronous interface)
        
        Args:
            symbols: List of stock symbols
            period: Time period
            
        Returns:
            Combined DataFrame
        """
        # Check cache first
        cache_key = self._get_cache_key(symbols, period)
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        logger.info(f"Fetching data for {len(symbols)} symbols using parallel processing")
        start_time = time.time()
        
        def fetch_single_stock(symbol: str) -> Tuple[str, Optional[pd.DataFrame]]:
            """Fetch single stock data"""
            try:
                data = self._fetch_stock_data_alpha_vantage(symbol, period)
                return symbol, data
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                return symbol, None
        
        # Use ThreadPoolExecutor for parallel processing
        dataframes = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(fetch_single_stock, symbol): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    symbol, data = future.result()
                    if data is not None:
                        dataframes.append(data)
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
        
        if not dataframes:
            raise ValueError("No data could be fetched for any symbol")
        
        # Combine all dataframes
        combined_data = pd.concat(dataframes, ignore_index=True)
        
        # Save to cache
        self._save_to_cache(combined_data, cache_key)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Fetched data for {len(symbols)} symbols in {elapsed_time:.2f} seconds")
        
        return combined_data
    
    def get_technical_indicators(self, symbol: str, indicator: str, interval: str = 'daily') -> Optional[pd.DataFrame]:
        """
        Get technical indicators from Alpha Vantage
        
        Args:
            symbol: Stock symbol
            indicator: Technical indicator (SMA, EMA, RSI, MACD, etc.)
            interval: Time interval
            
        Returns:
            DataFrame with technical indicators
        """
        try:
            if not self.api_key:
                raise ValueError("Alpha Vantage API key not configured")
            
            # Map indicator names to Alpha Vantage functions
            indicator_mapping = {
                'SMA': 'sma',
                'EMA': 'ema', 
                'RSI': 'rsi',
                'MACD': 'macd',
                'BBANDS': 'bbands',
                'STOCH': 'stoch',
                'ADX': 'adx',
                'CCI': 'cci',
                'ATR': 'atr',
                'OBV': 'obv'
            }
            
            function = indicator_mapping.get(indicator.upper(), indicator.lower())
            
            if hasattr(self.ti, function):
                method = getattr(self.ti, function)
                data, meta_data = method(symbol=symbol, interval=interval, series_type='close')
                return data
            else:
                logger.warning(f"Technical indicator {indicator} not supported")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching technical indicator {indicator} for {symbol}: {e}")
            return None
    
    def load_csv_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from CSV file with error handling
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with loaded data
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            data = None
            
            for encoding in encodings:
                try:
                    data = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if data is None:
                raise ValueError("Could not read file with any encoding")
            
            # Validate required columns
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            available_columns = [col.lower() for col in data.columns]
            
            missing_columns = []
            for col in required_columns:
                if col.lower() not in available_columns:
                    missing_columns.append(col)
            
            if missing_columns:
                logger.warning(f"Missing columns: {missing_columns}")
            
            # Convert Date column to datetime - try multiple possible column names
            date_column = None
            for col in data.columns:
                if col.lower() in ['date', 'time', 'datetime']:
                    date_column = col
                    break
            
            if date_column:
                try:
                    data[date_column] = pd.to_datetime(data[date_column])
                    data.set_index(date_column, inplace=True)
                    logger.info(f"Set {date_column} as datetime index")
                except Exception as e:
                    logger.warning(f"Could not convert {date_column} to datetime: {e}")
                    # If conversion fails, keep the original index
            else:
                logger.warning("No date column found, using original index")
            
            logger.info(f"Successfully loaded CSV file: {len(data)} rows")
            return data
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        if data is None or data.empty:
            return {
                'is_valid': False,
                'data_quality_score': 0.0,
                'missing_values': 100,
                'duplicates': 0,
                'price_consistency': 0,
                'errors': ['Data is None or empty']
            }
        
        validation_results = {}
        
        # Check for missing values
        missing_values = data.isnull().sum().sum()
        total_values = data.size
        missing_percentage = (missing_values / total_values) * 100 if total_values > 0 else 100
        
        validation_results['missing_values'] = missing_percentage
        
        # Check for duplicates
        duplicates = data.duplicated().sum()
        validation_results['duplicates'] = duplicates
        
        # Check price consistency
        price_consistency = self._check_price_consistency(data)
        validation_results['price_consistency'] = price_consistency['score']
        
        # Calculate overall quality score
        quality_score = max(0, 100 - missing_percentage - (duplicates * 0.1) - (100 - price_consistency['score']))
        validation_results['data_quality_score'] = quality_score / 100
        
        validation_results['is_valid'] = quality_score > 80
        
        return validation_results
    
    def _check_price_consistency(self, data: pd.DataFrame) -> Dict[str, int]:
        """Check price consistency (High >= Low, etc.)"""
        if data.empty:
            return {'score': 0, 'errors': 0}
        
        errors = 0
        total_checks = 0
        
        # Check if required columns exist
        price_cols = ['Open', 'High', 'Low', 'Close']
        available_cols = [col for col in price_cols if col in data.columns]
        
        if len(available_cols) >= 3:
            for col in available_cols:
                if col in data.columns:
                    # Check for negative prices
                    negative_prices = (data[col] < 0).sum()
                    errors += negative_prices
                    total_checks += len(data)
            
            # Check High >= Low
            if 'High' in data.columns and 'Low' in data.columns:
                high_low_errors = (data['High'] < data['Low']).sum()
                errors += high_low_errors
                total_checks += len(data)
        
        score = max(0, 100 - (errors / total_checks * 100)) if total_checks > 0 else 0
        return {'score': score, 'errors': errors}
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive data summary
        
        Args:
            data: DataFrame to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        if data is None or data.empty:
            return {'error': 'No data available'}
        
        summary = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
            'missing_values': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.to_dict()
        }
        
        # Date range
        if isinstance(data.index, pd.DatetimeIndex):
            summary['date_range'] = {
                'start': data.index.min().strftime('%Y-%m-%d'),
                'end': data.index.max().strftime('%Y-%m-%d'),
                'days': (data.index.max() - data.index.min()).days
            }
        
        # Symbol information
        if 'Symbol' in data.columns:
            summary['symbols'] = data['Symbol'].unique().tolist()
            summary['symbol_count'] = len(summary['symbols'])
        
        # Price statistics
        price_cols = ['Open', 'High', 'Low', 'Close']
        available_price_cols = [col for col in price_cols if col in data.columns]
        
        if available_price_cols:
            summary['price_statistics'] = {}
            for col in available_price_cols:
                summary['price_statistics'][col] = {
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std()),
                    'min': float(data[col].min()),
                    'max': float(data[col].max()),
                    'median': float(data[col].median())
                }
        
        # Volume statistics
        if 'Volume' in data.columns:
            summary['volume_statistics'] = {
                'mean': float(data['Volume'].mean()),
                'std': float(data['Volume'].std()),
                'min': float(data['Volume'].min()),
                'max': float(data['Volume'].max()),
                'median': float(data['Volume'].median())
            }
        
        return summary
    
    def clear_cache(self, data_type: Optional[str] = None) -> None:
        """Clear cache files"""
        try:
            if data_type:
                cache_dir = self.cache_dir / data_type
                if cache_dir.exists():
                    for file in cache_dir.glob('*.pkl'):
                        file.unlink()
                    logger.info(f"Cleared {data_type} cache")
            else:
                for cache_dir in [self.cache_dir / 'raw', self.cache_dir / 'processed']:
                    if cache_dir.exists():
                        for file in cache_dir.glob('*.pkl'):
                            file.unlink()
                logger.info("Cleared all cache")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        try:
            cache_info = {
                'raw_cache_size': 0,
                'processed_cache_size': 0,
                'raw_cache_files': 0,
                'processed_cache_files': 0
            }
            
            for cache_type in ['raw', 'processed']:
                cache_dir = self.cache_dir / cache_type
                if cache_dir.exists():
                    files = list(cache_dir.glob('*.pkl'))
                    cache_info[f'{cache_type}_cache_files'] = len(files)
                    cache_info[f'{cache_type}_cache_size'] = sum(f.stat().st_size for f in files) / 1024 / 1024
            
            return cache_info
        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
            return {}
    
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close() 