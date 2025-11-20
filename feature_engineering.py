import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
import ta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Comprehensive feature engineering for stock market prediction
    Creates technical indicators, statistical features, and market microstructure features
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None
        
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive technical indicators using the ta library
        
        Args:
            data: DataFrame with OHLCV data
        
        Returns:
            DataFrame with technical indicators
        """
        df = data.copy()
        
        # Trend indicators
        df['SMA_5'] = ta.trend.sma_indicator(df['Close'], window=5)
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        
        # MACD
        df['MACD'] = ta.trend.macd(df['Close'])
        df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
        df['MACD_Histogram'] = ta.trend.macd_diff(df['Close'])
        
        # Bollinger Bands
        df['BB_Upper'] = ta.volatility.bollinger_hband(df['Close'])
        df['BB_Lower'] = ta.volatility.bollinger_lband(df['Close'])
        df['BB_Middle'] = ta.volatility.bollinger_mavg(df['Close'])
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'])
        df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
        df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)
        
        # Stochastic
        df['Stoch_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
        df['Stoch_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
        
        # Williams %R
        df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
        
        # Commodity Channel Index
        df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
        
        # Average True Range
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        # Parabolic SAR
        df['PSAR'] = ta.trend.psar_down(df['High'], df['Low'], df['Close'])
        
        return df
    
    def add_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add statistical features based on price and volume
        
        Args:
            data: DataFrame with OHLCV data
        
        Returns:
            DataFrame with statistical features
        """
        df = data.copy()
        
        # Rolling statistics for returns
        for window in [5, 10, 20, 50]:
            df[f'Returns_Mean_{window}d'] = df['Returns'].rolling(window=window).mean()
            df[f'Returns_Std_{window}d'] = df['Returns'].rolling(window=window).std()
            df[f'Returns_Skew_{window}d'] = df['Returns'].rolling(window=window).skew()
            df[f'Returns_Kurt_{window}d'] = df['Returns'].rolling(window=window).kurt()
        
        # Rolling statistics for volume
        for window in [5, 10, 20]:
            df[f'Volume_Mean_{window}d'] = df['Volume'].rolling(window=window).mean()
            df[f'Volume_Std_{window}d'] = df['Volume'].rolling(window=window).std()
            df[f'Volume_Ratio_{window}d'] = df['Volume'] / df[f'Volume_Mean_{window}d']
        
        # Price momentum and acceleration
        for period in [1, 3, 5, 10, 20]:
            df[f'Price_Momentum_{period}d'] = df['Close'] / df['Close'].shift(period) - 1
            df[f'Price_Acceleration_{period}d'] = df[f'Price_Momentum_{period}d'].diff()
        
        # Volatility features
        for window in [5, 10, 20, 50]:
            df[f'Volatility_{window}d'] = df['Returns'].rolling(window=window).std() * np.sqrt(252)
            df[f'Realized_Vol_{window}d'] = np.sqrt((df['Returns']**2).rolling(window=window).sum())
        
        # Sharpe ratio (assuming risk-free rate of 0.02)
        for window in [20, 50]:
            returns_mean = df['Returns'].rolling(window=window).mean()
            returns_std = df['Returns'].rolling(window=window).std()
            df[f'Sharpe_Ratio_{window}d'] = (returns_mean - 0.02/252) / returns_std
        
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
        
        # Bid-ask spread proxy (using high-low spread)
        df['Spread_Proxy'] = (df['High'] - df['Low']) / df['Close']
        df['Spread_Proxy_MA'] = df['Spread_Proxy'].rolling(window=20).mean()
        
        # Volume-price relationship
        df['Volume_Price_Trend'] = (df['Volume'] * df['Returns']).rolling(window=20).sum()
        df['Money_Flow_Index'] = self._calculate_money_flow_index(df)
        
        # Price efficiency
        df['Price_Efficiency'] = self._calculate_price_efficiency(df)
        
        # Order flow imbalance (proxy)
        df['Order_Flow_Imbalance'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'])
        
        # Gap analysis
        df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['Gap_Absolute'] = abs(df['Gap'])
        
        # Intraday range
        df['Intraday_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Intraday_Range_MA'] = df['Intraday_Range'].rolling(window=20).mean()
        
        return df
    
    def add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features
        
        Args:
            data: DataFrame with datetime index
        
        Returns:
            DataFrame with time features
        """
        df = data.copy()
        
        # Extract time components
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        df['Day'] = df.index.day
        df['DayOfWeek'] = df.index.dayofweek
        df['Quarter'] = df.index.quarter
        
        # Cyclical encoding for time features
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        
        # Market session features
        df['Is_Monday'] = (df['DayOfWeek'] == 0).astype(int)
        df['Is_Friday'] = (df['DayOfWeek'] == 4).astype(int)
        df['Is_Month_End'] = (df['Day'] >= 25).astype(int)
        df['Is_Quarter_End'] = ((df['Month'] % 3 == 0) & (df['Day'] >= 25)).astype(int)
        
        # Trend features
        df['Days_From_Start'] = (df.index - df.index[0]).days
        df['Days_From_Start_Sin'] = np.sin(2 * np.pi * df['Days_From_Start'] / 365)
        df['Days_From_Start_Cos'] = np.cos(2 * np.pi * df['Days_From_Start'] / 365)
        
        return df
    
    def add_lag_features(self, data: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """
        Add lagged features
        
        Args:
            data: DataFrame
            columns: List of column names to lag
            lags: List of lag periods
        
        Returns:
            DataFrame with lagged features
        """
        df = data.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_Lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def add_rolling_features(self, data: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
        """
        Add rolling window features
        
        Args:
            data: DataFrame
            columns: List of column names
            windows: List of window sizes
        
        Returns:
            DataFrame with rolling features
        """
        df = data.copy()
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    df[f'{col}_Rolling_Mean_{window}'] = df[col].rolling(window=window).mean()
                    df[f'{col}_Rolling_Std_{window}'] = df[col].rolling(window=window).std()
                    df[f'{col}_Rolling_Min_{window}'] = df[col].rolling(window=window).min()
                    df[f'{col}_Rolling_Max_{window}'] = df[col].rolling(window=window).max()
                    df[f'{col}_Rolling_Median_{window}'] = df[col].rolling(window=window).median()
        
        return df
    
    def create_target_variables(self, data: pd.DataFrame, horizons: List[int] = [1, 3, 5, 10, 20]) -> pd.DataFrame:
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
            
            # Binary classification (up/down)
            df[f'Target_Binary_{horizon}d'] = (df[f'Target_Return_{horizon}d'] > 0).astype(int)
            
            # Volatility prediction
            df[f'Target_Volatility_{horizon}d'] = df['Returns'].rolling(window=horizon).std().shift(-horizon)
        
        return df
    
    def _calculate_money_flow_index(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']
        
        positive_flow = pd.Series(0, index=data.index)
        negative_flow = pd.Series(0, index=data.index)
        
        # Calculate positive and negative money flow
        for i in range(1, len(data)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.iloc[i] = money_flow.iloc[i]
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                negative_flow.iloc[i] = money_flow.iloc[i]
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    def _calculate_price_efficiency(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate price efficiency ratio"""
        price_changes = data['Close'].diff()
        cumulative_change = price_changes.rolling(window=window).sum()
        path_length = price_changes.abs().rolling(window=window).sum()
        
        efficiency = cumulative_change / path_length
        return efficiency
    
    def engineer_all_features(self, data: pd.DataFrame, target_horizons: List[int] = [1, 3, 5, 10]) -> pd.DataFrame:
        """
        Apply all feature engineering steps
        
        Args:
            data: Raw OHLCV DataFrame
            target_horizons: List of prediction horizons
        
        Returns:
            DataFrame with all engineered features
        """
        print("Starting feature engineering...")
        
        # Check if we have multi-stock data
        if 'Stock_Symbol' in data.columns or 'Symbol' in data.columns:
            symbol_col = 'Stock_Symbol' if 'Stock_Symbol' in data.columns else 'Symbol'
            symbols = data[symbol_col].unique()
            
            print(f"Processing {len(symbols)} stocks: {symbols}")
            
            # Process each stock separately
            engineered_dfs = []
            
            for symbol in symbols:
                print(f"  Processing {symbol}...")
                stock_data = data[data[symbol_col] == symbol].copy()
                
                if stock_data.empty:
                    print(f"    Warning: No data for {symbol}")
                    continue
                
                print(f"    Stock data shape: {stock_data.shape}")
                
                # Apply feature engineering to this stock
                stock_engineered = self._engineer_single_stock(stock_data, target_horizons)
                
                print(f"    Engineered shape for {symbol}: {stock_engineered.shape}")
                
                if not stock_engineered.empty:
                    engineered_dfs.append(stock_engineered)
                    print(f"    Added {symbol} to results")
                else:
                    print(f"    Warning: {symbol} produced empty result")
            
            # Combine all stocks
            if engineered_dfs:
                print(f"Combining {len(engineered_dfs)} stocks...")
                df = pd.concat(engineered_dfs, ignore_index=False)
                df = df.sort_index()
                print(f"Combined shape: {df.shape}")
            else:
                print("No valid engineered data to combine")
                df = pd.DataFrame()
        else:
            # Single stock data - process normally
            print("Processing single stock data...")
            df = self._engineer_single_stock(data, target_horizons)
        
        print(f"Feature engineering complete. Final shape: {df.shape}")
        return df
    
    def _engineer_single_stock(self, data: pd.DataFrame, target_horizons: List[int]) -> pd.DataFrame:
        """
        Apply feature engineering to a single stock's data
        
        Args:
            data: Single stock OHLCV DataFrame
            target_horizons: List of prediction horizons
        
        Returns:
            DataFrame with engineered features
        """
        print(f"      [DEBUG] Initial rows: {data.shape[0]}")
        # Add technical indicators
        print("Adding technical indicators...")
        df = self.add_technical_indicators(data)
        print(f"      [DEBUG] After technical indicators: {df.shape[0]}")
        # Add statistical features
        print("Adding statistical features...")
        df = self.add_statistical_features(df)
        print(f"      [DEBUG] After statistical features: {df.shape[0]}")
        # Add market microstructure features
        print("Adding market microstructure features...")
        df = self.add_market_microstructure_features(df)
        print(f"      [DEBUG] After market microstructure: {df.shape[0]}")
        # Add time features
        print("Adding time features...")
        df = self.add_time_features(df)
        print(f"      [DEBUG] After time features: {df.shape[0]}")
        # Add lag features for key columns
        key_columns = ['Close', 'Volume', 'Returns']
        df = self.add_lag_features(df, key_columns, [1, 2, 3, 5, 10])
        print(f"      [DEBUG] After lag features: {df.shape[0]}")
        # Add rolling features
        rolling_columns = ['Close', 'Volume', 'Returns']
        df = self.add_rolling_features(df, rolling_columns, [5, 10, 20])
        print(f"      [DEBUG] After rolling features: {df.shape[0]}")
        # Create target variables
        print("Creating target variables...")
        df = self.create_target_variables(df, target_horizons)
        print(f"      [DEBUG] After target variables: {df.shape[0]}")
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        # Only drop rows where main target columns are NaN (not all Target_ columns)
        target_columns = [col for col in df.columns if col.startswith('Target_Return_')]
        if target_columns:
            print(f"      [DEBUG] NaNs in main target columns before dropna:")
            for col in target_columns:
                print(f"        {col}: {df[col].isna().sum()} NaNs")
            df = df.dropna(subset=target_columns)
            print(f"      [DEBUG] After dropna on main targets: {df.shape[0]}")
        # Fill remaining NaN values with forward fill, then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        print(f"      [DEBUG] After fillna: {df.shape[0]}")
        return df
    
    def select_features(self, data: pd.DataFrame, target_column: str, 
                       method: str = 'correlation', threshold: float = 0.01) -> List[str]:
        """
        Select the most important features
        
        Args:
            data: DataFrame with features
            target_column: Name of target column
            method: Feature selection method ('correlation', 'variance', 'mutual_info')
            threshold: Threshold for feature selection
        
        Returns:
            List of selected feature names
        """
        if target_column not in data.columns:
            print(f"Target column {target_column} not found in data")
            return []
        
        # Remove target columns and non-numeric columns
        target_cols = [col for col in data.columns if 'Target_' in col]
        exclude_cols = target_cols + ['Symbol', 'Stock_Symbol']
        
        feature_cols = [col for col in data.columns if col not in exclude_cols and data[col].dtype in ['float64', 'int64']]
        
        if method == 'correlation':
            correlations = data[feature_cols].corrwith(data[target_column]).abs()
            selected_features = correlations[correlations > threshold].index.tolist()
        
        elif method == 'variance':
            variances = data[feature_cols].var()
            selected_features = variances[variances > threshold].index.tolist()
        
        elif method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(data[feature_cols], data[target_column])
            selected_features = [feature_cols[i] for i in range(len(feature_cols)) if mi_scores[i] > threshold]
        
        else:
            selected_features = feature_cols
        
        print(f"Selected {len(selected_features)} features out of {len(feature_cols)}")
        return selected_features
    
    def scale_features(self, data: pd.DataFrame, feature_columns: List[str], 
                      method: str = 'standard') -> pd.DataFrame:
        """
        Scale features for machine learning
        
        Args:
            data: DataFrame
            feature_columns: List of feature columns to scale
            method: Scaling method ('standard', 'minmax')
        
        Returns:
            DataFrame with scaled features
        """
        df = data.copy()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        df[feature_columns] = scaler.fit_transform(df[feature_columns])
        return df

# Example usage
if __name__ == "__main__":
    # Example with sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    sample_data = pd.DataFrame({
        'Open': np.random.randn(len(dates)).cumsum() + 100,
        'High': np.random.randn(len(dates)).cumsum() + 102,
        'Low': np.random.randn(len(dates)).cumsum() + 98,
        'Close': np.random.randn(len(dates)).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    # Add basic features
    sample_data['Returns'] = sample_data['Close'].pct_change()
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Engineer features
    engineered_data = fe.engineer_all_features(sample_data)
    
    print(f"Original shape: {sample_data.shape}")
    print(f"Engineered shape: {engineered_data.shape}")
    print(f"Number of features: {len(engineered_data.columns)}") 