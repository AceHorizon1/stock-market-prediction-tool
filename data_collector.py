import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
from typing import List, Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class DataCollector:
    """
    Comprehensive data collector for stock market prediction
    Collects data from multiple sources including:
    - Yahoo Finance (stock prices, volume, etc.)
    - Alpha Vantage (technical indicators, news sentiment)
    - FRED (economic indicators)
    """
    
    def __init__(self, alpha_vantage_key: Optional[str] = None, fred_key: Optional[str] = None):
        self.alpha_vantage_key = alpha_vantage_key
        self.fred_key = fred_key
        self.session = requests.Session()
        
    def get_stock_data(self, symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Add symbol column
            data['Symbol'] = symbol
            
            # Calculate additional features
            data = self._add_basic_features(data)
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_multiple_stocks(self, symbols: List[str], period: str = "5y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks
        
        Args:
            symbols: List of stock symbols
            period: Time period
            interval: Data interval
        
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        data_dict = {}
        
        for symbol in symbols:
            print(f"Fetching data for {symbol}...")
            data = self.get_stock_data(symbol, period, interval)
            if not data.empty:
                data_dict[symbol] = data
            time.sleep(0.1)  # Rate limiting
            
        return data_dict
    
    def get_market_data(self, symbols: List[str] = None) -> pd.DataFrame:
        """
        Get market-wide data including major indices
        
        Args:
            symbols: List of market symbols (default: major indices)
        
        Returns:
            DataFrame with market data
        """
        if symbols is None:
            symbols = ['^GSPC', '^DJI', '^IXIC', '^VIX', '^TNX', '^TYX']  # S&P 500, Dow, NASDAQ, VIX, 10Y, 30Y
        
        market_data = {}
        
        for symbol in symbols:
            data = self.get_stock_data(symbol, period="5y")
            if not data.empty:
                market_data[symbol] = data['Close']
        
        return pd.DataFrame(market_data)
    
    def get_economic_indicators(self) -> pd.DataFrame:
        """
        Fetch economic indicators from FRED (requires API key)
        
        Returns:
            DataFrame with economic indicators
        """
        if not self.fred_key:
            print("FRED API key required for economic indicators")
            return pd.DataFrame()
        
        try:
            from fredapi import Fred
            fred = Fred(api_key=self.fred_key)
            
            # Key economic indicators
            indicators = {
                'GDP': 'GDP',  # Gross Domestic Product
                'UNRATE': 'UNRATE',  # Unemployment Rate
                'CPIAUCSL': 'CPIAUCSL',  # Consumer Price Index
                'FEDFUNDS': 'FEDFUNDS',  # Federal Funds Rate
                'DGS10': 'DGS10',  # 10-Year Treasury Rate
                'DGS2': 'DGS2',  # 2-Year Treasury Rate
                'DEXUSEU': 'DEXUSEU',  # US/Euro Exchange Rate
                'DCOILWTICO': 'DCOILWTICO',  # WTI Crude Oil Price
                'GOLD': 'GOLD',  # Gold Price
            }
            
            econ_data = {}
            for name, series_id in indicators.items():
                try:
                    data = fred.get_series(series_id, observation_start='2019-01-01')
                    econ_data[name] = data
                except Exception as e:
                    print(f"Error fetching {name}: {str(e)}")
            
            return pd.DataFrame(econ_data)
            
        except Exception as e:
            print(f"Error fetching economic indicators: {str(e)}")
            return pd.DataFrame()
    
    def get_technical_indicators(self, symbol: str) -> pd.DataFrame:
        """
        Get technical indicators from Alpha Vantage (requires API key)
        
        Args:
            symbol: Stock symbol
        
        Returns:
            DataFrame with technical indicators
        """
        if not self.alpha_vantage_key:
            print("Alpha Vantage API key required for technical indicators")
            return pd.DataFrame()
        
        try:
            # SMA
            url = f'https://www.alphavantage.co/query?function=SMA&symbol={symbol}&interval=daily&time_period=20&series_type=close&apikey={self.alpha_vantage_key}'
            response = self.session.get(url)
            sma_data = response.json()
            
            # RSI
            url = f'https://www.alphavantage.co/query?function=RSI&symbol={symbol}&interval=daily&time_period=14&series_type=close&apikey={self.alpha_vantage_key}'
            response = self.session.get(url)
            rsi_data = response.json()
            
            # MACD
            url = f'https://www.alphavantage.co/query?function=MACD&symbol={symbol}&interval=daily&series_type=close&apikey={self.alpha_vantage_key}'
            response = self.session.get(url)
            macd_data = response.json()
            
            # Combine all indicators
            indicators = {}
            
            if 'Technical Analysis: SMA' in sma_data:
                indicators['SMA'] = pd.Series(sma_data['Technical Analysis: SMA'])
            
            if 'Technical Analysis: RSI' in rsi_data:
                indicators['RSI'] = pd.Series(rsi_data['Technical Analysis: RSI'])
            
            if 'Technical Analysis: MACD' in macd_data:
                indicators['MACD'] = pd.Series(macd_data['Technical Analysis: MACD'])
            
            return pd.DataFrame(indicators)
            
        except Exception as e:
            print(f"Error fetching technical indicators for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_news_sentiment(self, symbol: str) -> pd.DataFrame:
        """
        Get news sentiment data from Alpha Vantage (requires API key)
        
        Args:
            symbol: Stock symbol
        
        Returns:
            DataFrame with news sentiment
        """
        if not self.alpha_vantage_key:
            print("Alpha Vantage API key required for news sentiment")
            return pd.DataFrame()
        
        try:
            url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={self.alpha_vantage_key}'
            response = self.session.get(url)
            data = response.json()
            
            if 'feed' in data:
                sentiment_data = []
                for item in data['feed']:
                    sentiment_data.append({
                        'date': item.get('time_published', ''),
                        'title': item.get('title', ''),
                        'sentiment_score': item.get('overall_sentiment_score', 0),
                        'sentiment_label': item.get('overall_sentiment_label', ''),
                        'relevance_score': item.get('relevance_score', 0)
                    })
                
                return pd.DataFrame(sentiment_data)
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching news sentiment for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _add_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic technical features to the data
        
        Args:
            data: OHLCV DataFrame
        
        Returns:
            DataFrame with additional features
        """
        # Price changes
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Volatility (rolling standard deviation of returns)
        data['Volatility_5d'] = data['Returns'].rolling(window=5).std()
        data['Volatility_20d'] = data['Returns'].rolling(window=20).std()
        
        # Moving averages
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        
        # Price relative to moving averages
        data['Price_vs_SMA_20'] = data['Close'] / data['SMA_20'] - 1
        data['Price_vs_SMA_50'] = data['Close'] / data['SMA_50'] - 1
        data['Price_vs_SMA_200'] = data['Close'] / data['SMA_200'] - 1
        
        # Volume features
        data['Volume_SMA_20'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_20']
        
        # High-Low spread
        data['HL_Spread'] = (data['High'] - data['Low']) / data['Close']
        
        # Price momentum
        data['Momentum_5d'] = data['Close'] / data['Close'].shift(5) - 1
        data['Momentum_20d'] = data['Close'] / data['Close'].shift(20) - 1
        
        return data
    
    def create_comprehensive_dataset(self, symbols: List[str], include_market_data: bool = True, 
                                   include_economic_data: bool = True) -> pd.DataFrame:
        """
        Create a comprehensive dataset combining stock data, market data, and economic indicators
        
        Args:
            symbols: List of stock symbols
            include_market_data: Whether to include market indices
            include_economic_data: Whether to include economic indicators
        
        Returns:
            Comprehensive DataFrame for analysis
        """
        print("Creating comprehensive dataset...")
        
        # Get stock data
        stock_data = self.get_multiple_stocks(symbols)
        
        # Combine all stock data
        combined_data = pd.DataFrame()
        
        for symbol, data in stock_data.items():
            data_copy = data.copy()
            data_copy['Stock_Symbol'] = symbol
            combined_data = pd.concat([combined_data, data_copy])
        
        # Add market data if requested
        if include_market_data:
            print("Fetching market data...")
            market_data = self.get_market_data()
            if not market_data.empty:
                # Merge market data with stock data
                combined_data = combined_data.merge(market_data, left_index=True, right_index=True, how='left')
        
        # Add economic data if requested
        if include_economic_data:
            print("Fetching economic indicators...")
            econ_data = self.get_economic_indicators()
            if not econ_data.empty:
                # Forward fill economic data (monthly data)
                econ_data = econ_data.reindex(combined_data.index, method='ffill')
                combined_data = pd.concat([combined_data, econ_data], axis=1)
        
        # Clean up the data
        combined_data = combined_data.dropna()
        
        print(f"Dataset created with shape: {combined_data.shape}")
        return combined_data

    def load_from_file(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a CSV or Excel file.
        Args:
            file_path: Path to the file (.csv or .xlsx)
        Returns:
            DataFrame with loaded data
        """
        if file_path.lower().endswith('.csv'):
            return pd.read_csv(file_path, index_col=0, parse_dates=True)
        elif file_path.lower().endswith(('.xls', '.xlsx')):
            return pd.read_excel(file_path, index_col=0, parse_dates=True)
        else:
            raise ValueError('Unsupported file type. Please provide a .csv or .xlsx file.')

# Example usage
if __name__ == "__main__":
    # Initialize collector (add your API keys if you have them)
    collector = DataCollector()
    
    # Example: Get data for a few stocks
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    data = collector.create_comprehensive_dataset(symbols)
    
    print(f"Collected data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(data.head()) 