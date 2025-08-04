"""
Enhanced Feature Engineering for Stock Market AI
Features: Caching, parallel processing, advanced feature selection, memory optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import ta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import pickle
import hashlib
import time
import warnings
warnings.filterwarnings('ignore')

from config import config

logger = logging.getLogger(__name__)

class EnhancedFeatureEngineer:
    """
    Enhanced feature engineering with caching, parallel processing, and advanced techniques
    """
    
    def __init__(self):
        self.cache_dir = config.data.cache_dir / 'processed'
        self.cache_dir.mkdir(exist_ok=True)
        self.scalers = {}
        self.pca = None
        self.feature_importance = {}
        self.selected_features = []
        
    def _get_cache_key(self, data_hash: str, feature_config: Dict[str, Any]) -> str:
        """Generate cache key for feature engineering"""
        config_str = str(sorted(feature_config.items()))
        key_data = f"{data_hash}_{config_str}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_data_hash(self, data: pd.DataFrame) -> str:
        """Generate hash for data"""
        # Use shape, columns, and first/last few rows for hash
        hash_data = f"{data.shape}_{list(data.columns)}_{data.head(3).to_string()}_{data.tail(3).to_string()}"
        return hashlib.md5(hash_data.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load engineered features from cache"""
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            try:
                # Check if cache is not older than 1 day
                if time.time() - cache_path.stat().st_mtime < 86400:  # 24 hours
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                    logger.info(f"Loaded engineered features from cache: {cache_path}")
                    return data
                else:
                    logger.info(f"Cache expired: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None
    
    def _save_to_cache(self, data: pd.DataFrame, cache_key: str) -> None:
        """Save engineered features to cache"""
        try:
            cache_path = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved engineered features to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def add_technical_indicators_parallel(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators using parallel processing
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        df = data.copy()
        
        # Define indicator functions
        indicator_functions = [
            ('SMA_5', lambda x: ta.trend.sma_indicator(x['Close'], window=5)),
            ('SMA_20', lambda x: ta.trend.sma_indicator(x['Close'], window=20)),
            ('SMA_50', lambda x: ta.trend.sma_indicator(x['Close'], window=50)),
            ('EMA_12', lambda x: ta.trend.ema_indicator(x['Close'], window=12)),
            ('EMA_26', lambda x: ta.trend.ema_indicator(x['Close'], window=26)),
            ('MACD', lambda x: ta.trend.macd(x['Close'])),
            ('MACD_Signal', lambda x: ta.trend.macd_signal(x['Close'])),
            ('MACD_Histogram', lambda x: ta.trend.macd_diff(x['Close'])),
            ('BB_Upper', lambda x: ta.volatility.bollinger_hband(x['Close'])),
            ('BB_Lower', lambda x: ta.volatility.bollinger_lband(x['Close'])),
            ('BB_Middle', lambda x: ta.volatility.bollinger_mavg(x['Close'])),
            ('RSI', lambda x: ta.momentum.rsi(x['Close'])),
            ('Stoch_K', lambda x: ta.momentum.stoch(x['High'], x['Low'], x['Close'])),
            ('Stoch_D', lambda x: ta.momentum.stoch_signal(x['High'], x['Low'], x['Close'])),
            ('Williams_R', lambda x: ta.momentum.williams_r(x['High'], x['Low'], x['Close'])),
            ('CCI', lambda x: ta.trend.cci(x['High'], x['Low'], x['Close'])),
            ('ATR', lambda x: ta.volatility.average_true_range(x['High'], x['Low'], x['Close'])),
            ('PSAR', lambda x: ta.trend.psar_down(x['High'], x['Low'], x['Close']))
        ]
        
        def calculate_indicator(args: Tuple[str, callable]) -> Tuple[str, pd.Series]:
            """Calculate single indicator"""
            name, func = args
            try:
                result = func(df)
                return name, result
            except Exception as e:
                logger.warning(f"Failed to calculate {name}: {e}")
                return name, pd.Series(index=df.index, dtype=float)
        
        # Calculate indicators in parallel
        with ThreadPoolExecutor(max_workers=config.data.max_workers) as executor:
            futures = [executor.submit(calculate_indicator, (name, func)) for name, func in indicator_functions]
            
            for future in as_completed(futures):
                try:
                    name, result = future.result()
                    df[name] = result
                except Exception as e:
                    logger.error(f"Error in parallel indicator calculation: {e}")
        
        # Add derived indicators
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
        df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)
        
        return df
    
    def add_statistical_features_optimized(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add statistical features with memory optimization
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with statistical features
        """
        df = data.copy()
        
        # Calculate returns once
        df['Returns'] = df['Close'].pct_change()
        
        # Define windows for rolling calculations
        windows = [5, 10, 20, 50]
        
        # Calculate rolling statistics in batches to optimize memory
        for window in windows:
            # Returns statistics
            df[f'Returns_Mean_{window}d'] = df['Returns'].rolling(window=window, min_periods=1).mean()
            df[f'Returns_Std_{window}d'] = df['Returns'].rolling(window=window, min_periods=1).std()
            df[f'Returns_Skew_{window}d'] = df['Returns'].rolling(window=window, min_periods=1).skew()
            df[f'Returns_Kurt_{window}d'] = df['Returns'].rolling(window=window, min_periods=1).kurt()
            
            # Volume statistics
            if 'Volume' in df.columns:
                df[f'Volume_Mean_{window}d'] = df['Volume'].rolling(window=window, min_periods=1).mean()
                df[f'Volume_Std_{window}d'] = df['Volume'].rolling(window=window, min_periods=1).std()
                df[f'Volume_Ratio_{window}d'] = df['Volume'] / df[f'Volume_Mean_{window}d']
        
        # Price momentum and acceleration
        for period in [1, 3, 5, 10, 20]:
            df[f'Price_Momentum_{period}d'] = df['Close'].pct_change(period)
            df[f'Price_Acceleration_{period}d'] = df[f'Price_Momentum_{period}d'].diff()
        
        # Volatility measures
        for window in [5, 10, 20]:
            df[f'Volatility_{window}d'] = df['Returns'].rolling(window=window).std() * np.sqrt(252)
        
        return df
    
    def add_market_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add market microstructure features
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with microstructure features
        """
        df = data.copy()
        
        # Money Flow Index
        df['MFI'] = self._calculate_money_flow_index(df)
        
        # Price efficiency
        df['Price_Efficiency'] = self._calculate_price_efficiency(df)
        
        # Volume-price trend
        df['VPT'] = (df['Volume'] * df['Returns']).cumsum()
        
        # On-balance volume
        df['OBV'] = (np.sign(df['Returns']) * df['Volume']).cumsum()
        
        # Accumulation/Distribution Line
        df['ADL'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
        df['ADL'] = df['ADL'].cumsum()
        
        # Chaikin Money Flow
        df['CMF'] = df['ADL'] / df['Volume'].rolling(window=20).sum()
        
        # Volume-weighted average price
        df['VWAP'] = (df['Close'] * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
        
        return df
    
    def _calculate_money_flow_index(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        return mfi
    
    def _calculate_price_efficiency(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate price efficiency ratio"""
        price_changes = data['Close'].diff()
        cumulative_change = price_changes.rolling(window=window).sum()
        path_length = price_changes.abs().rolling(window=window).sum()
        
        efficiency = cumulative_change / path_length
        return efficiency
    
    def add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features
        
        Args:
            data: DataFrame with datetime index
            
        Returns:
            DataFrame with time features
        """
        df = data.copy()
        
        # Basic time features
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['Year'] = df.index.year
        df['Day_of_Year'] = df.index.dayofyear
        
        # Cyclical encoding for time features
        df['Day_of_Week_Sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
        df['Day_of_Week_Cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # Market session features
        df['Is_Monday'] = (df['Day_of_Week'] == 0).astype(int)
        df['Is_Friday'] = (df['Day_of_Week'] == 4).astype(int)
        df['Is_Month_End'] = (df.index.day >= 25).astype(int)
        df['Is_Quarter_End'] = ((df.index.month % 3 == 0) & (df.index.day >= 25)).astype(int)
        
        # Holiday effects (simplified)
        df['Is_January'] = (df['Month'] == 1).astype(int)
        df['Is_December'] = (df['Month'] == 12).astype(int)
        
        return df
    
    def create_target_variables(self, data: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
        """
        Create target variables for different prediction horizons
        
        Args:
            data: DataFrame with price data
            horizons: List of prediction horizons in days
            
        Returns:
            DataFrame with target variables
        """
        df = data.copy()
        
        for horizon in horizons:
            # Future returns
            df[f'Target_Return_{horizon}d'] = df['Close'].shift(-horizon) / df['Close'] - 1
            
            # Future price direction (classification)
            df[f'Target_Direction_{horizon}d'] = (df[f'Target_Return_{horizon}d'] > 0).astype(int)
            
            # Volatility targets
            if horizon > 1:
                future_returns = df['Close'].pct_change().shift(-horizon+1).rolling(window=horizon).std()
                df[f'Target_Volatility_{horizon}d'] = future_returns
        
        return df
    
    def engineer_all_features(self, data: pd.DataFrame, target_horizons: List[int] = None,
                            use_cache: bool = True) -> pd.DataFrame:
        """
        Engineer all features with caching and parallel processing
        
        Args:
            data: Input DataFrame
            target_horizons: List of target horizons
            use_cache: Whether to use caching
            
        Returns:
            DataFrame with all engineered features
        """
        if target_horizons is None:
            target_horizons = config.features.target_horizons
        
        # Generate cache key
        data_hash = self._get_data_hash(data)
        feature_config = {
            'target_horizons': target_horizons,
            'technical_indicators': config.features.technical_indicators,
            'statistical_features': config.features.statistical_features,
            'market_microstructure': config.features.market_microstructure,
            'time_features': config.features.time_features
        }
        cache_key = self._get_cache_key(data_hash, feature_config)
        
        # Try to load from cache
        if use_cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        logger.info("Starting feature engineering...")
        start_time = time.time()
        
        df = data.copy()
        
        # Engineer features in parallel where possible
        if config.features.technical_indicators:
            logger.info("Adding technical indicators...")
            df = self.add_technical_indicators_parallel(df)
        
        if config.features.statistical_features:
            logger.info("Adding statistical features...")
            df = self.add_statistical_features_optimized(df)
        
        if config.features.market_microstructure:
            logger.info("Adding market microstructure features...")
            df = self.add_market_microstructure_features(df)
        
        if config.features.time_features:
            logger.info("Adding time features...")
            df = self.add_time_features(df)
        
        # Create target variables
        logger.info("Creating target variables...")
        df = self.create_target_variables(df, target_horizons)
        
        # Handle missing values
        logger.info("Handling missing values...")
        df = self._handle_missing_values(df)
        
        # Save to cache
        if use_cache:
            self._save_to_cache(df, cache_key)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Feature engineering completed in {elapsed_time:.2f} seconds")
        logger.info(f"Final shape: {df.shape}")
        
        return df
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        df = data.copy()
        
        # Forward fill for technical indicators
        technical_cols = [col for col in df.columns if any(indicator in col for indicator in 
                        ['SMA', 'EMA', 'MACD', 'BB', 'RSI', 'Stoch', 'Williams', 'CCI', 'ATR', 'PSAR'])]
        df[technical_cols] = df[technical_cols].fillna(method='ffill')
        
        # Backward fill for remaining technical indicators
        df[technical_cols] = df[technical_cols].fillna(method='bfill')
        
        # Fill remaining missing values with 0
        df = df.fillna(0)
        
        return df
    
    def select_features_advanced(self, data: pd.DataFrame, target_column: str,
                               method: str = 'ensemble', n_features: int = 50) -> List[str]:
        """
        Advanced feature selection using multiple methods
        
        Args:
            data: DataFrame with features
            target_column: Target variable column
            method: Selection method ('correlation', 'mutual_info', 'random_forest', 'ensemble')
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        # Remove target columns and non-numeric columns
        target_cols = [col for col in data.columns if 'Target_' in col]
        non_numeric_cols = data.select_dtypes(include=['object', 'category']).columns
        
        feature_data = data.drop(columns=target_cols + list(non_numeric_cols))
        target_data = data[target_column].dropna()
        
        # Align data
        common_index = feature_data.index.intersection(target_data.index)
        X = feature_data.loc[common_index]
        y = target_data.loc[common_index]
        
        if method == 'correlation':
            correlations = X.corrwith(y).abs()
            selected_features = correlations.nlargest(n_features).index.tolist()
        
        elif method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(X, y, random_state=config.model.random_state)
            selected_features = X.columns[np.argsort(mi_scores)[-n_features:]].tolist()
        
        elif method == 'random_forest':
            if y.dtype == 'object' or len(y.unique()) < 10:
                # Classification
                rf = RandomForestClassifier(n_estimators=100, random_state=config.model.random_state)
            else:
                # Regression
                rf = RandomForestRegressor(n_estimators=100, random_state=config.model.random_state)
            
            rf.fit(X, y)
            feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
            selected_features = feature_importance.nlargest(n_features).index.tolist()
        
        elif method == 'ensemble':
            # Combine multiple methods
            methods = ['correlation', 'mutual_info', 'random_forest']
            feature_scores = {}
            
            for method_name in methods:
                try:
                    selected = self.select_features_advanced(data, target_column, method_name, n_features)
                    for feature in selected:
                        feature_scores[feature] = feature_scores.get(feature, 0) + 1
                except Exception as e:
                    logger.warning(f"Method {method_name} failed: {e}")
            
            # Select features that appear in multiple methods
            selected_features = [f for f, score in feature_scores.items() if score >= 2]
            if len(selected_features) < n_features // 2:
                # Fallback to correlation
                selected_features = self.select_features_advanced(data, target_column, 'correlation', n_features)
        
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        self.selected_features = selected_features
        logger.info(f"Selected {len(selected_features)} features using {method} method")
        
        return selected_features
    
    def scale_features(self, data: pd.DataFrame, feature_columns: List[str] = None,
                      method: str = 'standard', fit: bool = True) -> pd.DataFrame:
        """
        Scale features using various methods
        
        Args:
            data: DataFrame with features
            feature_columns: Columns to scale (if None, use all numeric columns)
            method: Scaling method ('standard', 'minmax', 'robust')
            fit: Whether to fit the scaler or use existing one
            
        Returns:
            DataFrame with scaled features
        """
        if feature_columns is None:
            feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        df = data.copy()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        if fit:
            df[feature_columns] = scaler.fit_transform(df[feature_columns])
            self.scalers[method] = scaler
        else:
            if method in self.scalers:
                df[feature_columns] = self.scalers[method].transform(df[feature_columns])
            else:
                logger.warning(f"No fitted scaler found for method: {method}")
        
        return df
    
    def get_feature_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive feature summary
        
        Args:
            data: DataFrame with engineered features
            
        Returns:
            Dictionary with feature summary
        """
        summary = {
            'total_features': len(data.columns),
            'feature_categories': {
                'technical_indicators': len([col for col in data.columns if any(indicator in col for indicator in 
                    ['SMA', 'EMA', 'MACD', 'BB', 'RSI', 'Stoch', 'Williams', 'CCI', 'ATR', 'PSAR'])]),
                'statistical_features': len([col for col in data.columns if any(stat in col for stat in 
                    ['Returns_', 'Volume_', 'Price_', 'Volatility_'])]),
                'microstructure_features': len([col for col in data.columns if any(micro in col for micro in 
                    ['MFI', 'Price_Efficiency', 'VPT', 'OBV', 'ADL', 'CMF', 'VWAP'])]),
                'time_features': len([col for col in data.columns if any(time_feat in col for time_feat in 
                    ['Day_of_', 'Month', 'Quarter', 'Year', 'Is_'])]),
                'target_variables': len([col for col in data.columns if 'Target_' in col])
            },
            'missing_values': data.isnull().sum().sum(),
            'memory_usage': data.memory_usage(deep=True).sum(),
            'data_types': data.dtypes.value_counts().to_dict()
        }
        
        return summary 