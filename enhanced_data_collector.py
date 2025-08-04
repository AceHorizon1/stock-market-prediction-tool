"""
Enhanced Data Collector for Stock Market AI
Features: Caching, parallel processing, async operations, comprehensive error handling
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import yfinance as yf
import logging
from datetime import datetime, timedelta
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import time
import warnings
warnings.filterwarnings('ignore')

from config import config

logger = logging.getLogger(__name__)

class EnhancedDataCollector:
    """
    Enhanced data collector with caching, parallel processing, and async operations
    """
    
    def __init__(self):
        self.cache_dir = config.data.cache_dir
        self.max_workers = config.data.max_workers
        self.session: Optional[aiohttp.ClientSession] = None
        self._setup_cache()
    
    def _setup_cache(self) -> None:
        """Setup cache directory and structure"""
        self.cache_dir.mkdir(exist_ok=True)
        (self.cache_dir / 'raw').mkdir(exist_ok=True)
        (self.cache_dir / 'processed').mkdir(exist_ok=True)
    
    def _get_cache_key(self, symbols: List[str], period: str, **kwargs) -> str:
        """Generate cache key for data"""
        key_data = f"{','.join(sorted(symbols))}_{period}_{datetime.now().strftime('%Y%m%d')}"
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
    
    async def _fetch_stock_data_async(self, symbol: str, period: str) -> Tuple[str, Optional[pd.DataFrame]]:
        """
        Fetch stock data asynchronously
        
        Args:
            symbol: Stock symbol
            period: Time period
            
        Returns:
            Tuple of (symbol, data)
        """
        try:
            ticker = yf.Ticker(symbol)
            data = await asyncio.get_event_loop().run_in_executor(
                None, ticker.history, period
            )
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return symbol, None
            
            # Add symbol column
            data['Symbol'] = symbol
            logger.info(f"Successfully fetched data for {symbol}: {len(data)} rows")
            return symbol, data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
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
        
        logger.info(f"Fetching data for {len(symbols)} symbols: {symbols}")
        start_time = time.time()
        
        # Create tasks for all symbols
        tasks = [self._fetch_stock_data_async(symbol, period) for symbol in symbols]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        dataframes = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed: {result}")
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
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                
                if data.empty:
                    logger.warning(f"No data found for {symbol}")
                    return symbol, None
                
                data['Symbol'] = symbol
                logger.info(f"Successfully fetched data for {symbol}: {len(data)} rows")
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
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert date column
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            
            # Add Symbol column if not present
            if 'Symbol' not in data.columns:
                data['Symbol'] = 'UNKNOWN'
            
            logger.info(f"Successfully loaded CSV file: {file_path} with {len(data)} rows")
            return data
            
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            raise
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and return statistics
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'total_rows': len(data),
            'total_symbols': data['Symbol'].nunique() if 'Symbol' in data.columns else 1,
            'date_range': {
                'start': data.index.min().strftime('%Y-%m-%d'),
                'end': data.index.max().strftime('%Y-%m-%d')
            },
            'missing_values': data.isnull().sum().to_dict(),
            'duplicate_dates': data.index.duplicated().sum(),
            'negative_prices': ((data[['Open', 'High', 'Low', 'Close']] < 0).any(axis=1)).sum(),
            'volume_issues': (data['Volume'] < 0).sum() if 'Volume' in data.columns else 0,
            'price_consistency': self._check_price_consistency(data),
            'data_quality_score': 0.0
        }
        
        # Calculate data quality score
        total_issues = (
            validation_results['missing_values'].get('Close', 0) +
            validation_results['duplicate_dates'] +
            validation_results['negative_prices'] +
            validation_results['volume_issues']
        )
        
        validation_results['data_quality_score'] = max(0, 1 - (total_issues / len(data)))
        
        return validation_results
    
    def _check_price_consistency(self, data: pd.DataFrame) -> Dict[str, int]:
        """Check price consistency (High >= Low, etc.)"""
        issues = {
            'high_below_low': ((data['High'] < data['Low']).sum()),
            'open_outside_range': ((data['Open'] < data['Low']) | (data['Open'] > data['High'])).sum(),
            'close_outside_range': ((data['Close'] < data['Low']) | (data['Close'] > data['High'])).sum()
        }
        return issues
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive data summary
        
        Args:
            data: DataFrame to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum(),
            'symbols': data['Symbol'].unique().tolist() if 'Symbol' in data.columns else ['UNKNOWN'],
            'date_range': {
                'start': data.index.min().strftime('%Y-%m-%d'),
                'end': data.index.max().strftime('%Y-%m-%d'),
                'days': (data.index.max() - data.index.min()).days
            },
            'price_statistics': {
                'mean': data[['Open', 'High', 'Low', 'Close']].mean().to_dict(),
                'std': data[['Open', 'High', 'Low', 'Close']].std().to_dict(),
                'min': data[['Open', 'High', 'Low', 'Close']].min().to_dict(),
                'max': data[['Open', 'High', 'Low', 'Close']].max().to_dict()
            },
            'volume_statistics': {
                'mean': data['Volume'].mean() if 'Volume' in data.columns else 0,
                'std': data['Volume'].std() if 'Volume' in data.columns else 0,
                'total': data['Volume'].sum() if 'Volume' in data.columns else 0
            }
        }
        
        return summary
    
    def clear_cache(self, data_type: Optional[str] = None) -> None:
        """
        Clear cache files
        
        Args:
            data_type: Type of cache to clear ('raw', 'processed', or None for all)
        """
        if data_type is None:
            cache_types = ['raw', 'processed']
        else:
            cache_types = [data_type]
        
        for cache_type in cache_types:
            cache_dir = self.cache_dir / cache_type
            if cache_dir.exists():
                for cache_file in cache_dir.glob('*.pkl'):
                    cache_file.unlink()
                logger.info(f"Cleared {cache_type} cache")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        cache_info = {}
        
        for cache_type in ['raw', 'processed']:
            cache_dir = self.cache_dir / cache_type
            if cache_dir.exists():
                files = list(cache_dir.glob('*.pkl'))
                cache_info[cache_type] = {
                    'file_count': len(files),
                    'total_size': sum(f.stat().st_size for f in files),
                    'oldest_file': min((f.stat().st_mtime for f in files), default=0),
                    'newest_file': max((f.stat().st_mtime for f in files), default=0)
                }
        
        return cache_info 