"""
Advanced ML Features for Stock Market AI
Features: Sentiment analysis, news impact, portfolio optimization, risk management
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import requests
import json
from datetime import datetime, timedelta
import yfinance as yf
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

from config import config

# Download required NLTK data
try:
    nltk.data.find("vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")


class SentimentAnalyzer:
    """
    Advanced sentiment analysis for stock market data
    """

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.news_cache = {}

    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using VADER

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment scores
        """
        if not text or pd.isna(text):
            return {"compound": 0, "pos": 0, "neg": 0, "neu": 0}

        # VADER sentiment analysis
        vader_scores = self.sia.polarity_scores(text)

        # TextBlob sentiment analysis
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity

        return {
            "compound": vader_scores["compound"],
            "pos": vader_scores["pos"],
            "neg": vader_scores["neg"],
            "neu": vader_scores["neu"],
            "textblob_polarity": textblob_polarity,
            "textblob_subjectivity": textblob_subjectivity,
        }

    def get_news_sentiment(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Get news sentiment for a stock symbol

        Args:
            symbol: Stock symbol
            days: Number of days to look back

        Returns:
            DataFrame with news sentiment data
        """
        try:
            # Get news from Yahoo Finance
            ticker = yf.Ticker(symbol)
            news = ticker.news

            if not news:
                return pd.DataFrame()

            # Process news articles
            sentiment_data = []
            for article in news:
                title = article.get("title", "")
                summary = article.get("summary", "")
                text = f"{title} {summary}"

                sentiment: Dict[str, Any] = self.analyze_text_sentiment(text)
                sentiment["date"] = datetime.fromtimestamp(
                    article.get("providerPublishTime", 0)
                )
                sentiment["title"] = title
                sentiment["summary"] = summary
                sentiment["url"] = article.get("link", "")

                sentiment_data.append(sentiment)

            df = pd.DataFrame(sentiment_data)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

            return df

        except Exception as e:
            print(f"Error getting news sentiment for {symbol}: {e}")
            return pd.DataFrame()

    def calculate_daily_sentiment(self, sentiment_df: pd.DataFrame) -> pd.Series:
        """
        Calculate daily sentiment scores

        Args:
            sentiment_df: DataFrame with sentiment data

        Returns:
            Series with daily sentiment scores
        """
        if sentiment_df.empty:
            return pd.Series()

        # Group by date and calculate average sentiment
        daily_sentiment = sentiment_df.groupby(sentiment_df.index.date).agg(
            {
                "compound": "mean",
                "pos": "mean",
                "neg": "mean",
                "neu": "mean",
                "textblob_polarity": "mean",
                "textblob_subjectivity": "mean",
            }
        )

        daily_sentiment.index = pd.to_datetime(daily_sentiment.index)

        return daily_sentiment


class NewsImpactAnalyzer:
    """
    Analyze the impact of news on stock prices
    """

    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()

    def analyze_news_impact(
        self, symbol: str, price_data: pd.DataFrame, lookback_days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze the impact of news sentiment on stock prices

        Args:
            symbol: Stock symbol
            price_data: DataFrame with price data
            lookback_days: Number of days to analyze

        Returns:
            Dictionary with analysis results
        """
        # Get news sentiment
        sentiment_df = self.sentiment_analyzer.get_news_sentiment(symbol, lookback_days)

        if sentiment_df.empty:
            return {"error": "No news data available"}

        # Calculate daily sentiment
        daily_sentiment = self.sentiment_analyzer.calculate_daily_sentiment(
            sentiment_df
        )

        # Align with price data
        aligned_data = price_data.join(daily_sentiment, how="inner")

        if aligned_data.empty:
            return {"error": "No overlapping data"}

        # Calculate correlations
        correlations = {}
        for sentiment_col in ["compound", "pos", "neg", "neu", "textblob_polarity"]:
            if sentiment_col in aligned_data.columns:
                corr = aligned_data["Returns"].corr(aligned_data[sentiment_col])
                correlations[sentiment_col] = corr

        # Calculate lagged correlations
        lagged_correlations = {}
        for lag in [1, 2, 3, 5]:
            lagged_returns = aligned_data["Returns"].shift(lag)
            for sentiment_col in ["compound", "pos", "neg", "neu"]:
                if sentiment_col in aligned_data.columns:
                    corr = lagged_returns.corr(aligned_data[sentiment_col])
                    lagged_correlations[f"{sentiment_col}_lag_{lag}"] = corr

        return {
            "correlations": correlations,
            "lagged_correlations": lagged_correlations,
            "aligned_data": aligned_data,
            "sentiment_summary": daily_sentiment.describe(),
        }


class PortfolioOptimizer:
    """
    Advanced portfolio optimization with risk management
    """

    def __init__(self):
        self.risk_free_rate = 0.02  # 2% risk-free rate

    def calculate_portfolio_metrics(
        self, returns: pd.DataFrame, weights: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics

        Args:
            returns: DataFrame with asset returns
            weights: Array of portfolio weights

        Returns:
            Dictionary with portfolio metrics
        """
        # Portfolio return
        portfolio_return = np.sum(returns.mean() * weights) * 252

        # Portfolio volatility
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

        # Sharpe ratio
        sharpe_ratio = (
            (portfolio_return - self.risk_free_rate) / portfolio_vol
            if portfolio_vol > 0
            else 0
        )

        # Maximum drawdown
        cumulative_returns = (1 + returns.dot(weights)).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Value at Risk (95% confidence)
        portfolio_returns = returns.dot(weights)
        var_95 = np.percentile(portfolio_returns, 5)

        return {
            "return": portfolio_return,
            "volatility": portfolio_vol,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "var_95": var_95,
        }

    def optimize_portfolio(
        self,
        returns: pd.DataFrame,
        method: str = "sharpe",
        constraints: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Optimize portfolio weights

        Args:
            returns: DataFrame with asset returns
            method: Optimization method ('sharpe', 'min_variance', 'max_return')
            constraints: Dictionary with optimization constraints

        Returns:
            Dictionary with optimization results
        """
        n_assets = len(returns.columns)

        # Default constraints
        if constraints is None:
            constraints = {"min_weight": 0.0, "max_weight": 1.0, "sum_weights": 1.0}

        # Objective function
        if method == "sharpe":

            def objective(weights):
                metrics = self.calculate_portfolio_metrics(returns, weights)
                return -metrics["sharpe_ratio"]  # Minimize negative Sharpe ratio

        elif method == "min_variance":

            def objective(weights):
                metrics = self.calculate_portfolio_metrics(returns, weights)
                return metrics["volatility"]

        elif method == "max_return":

            def objective(weights):
                metrics = self.calculate_portfolio_metrics(returns, weights)
                return -metrics["return"]  # Minimize negative return

        else:
            raise ValueError(f"Unknown optimization method: {method}")

        # Constraints
        constraints_list = [
            {"type": "eq", "fun": lambda x: np.sum(x) - constraints["sum_weights"]}
        ]

        # Bounds
        bounds = [
            (constraints["min_weight"], constraints["max_weight"])
            for _ in range(n_assets)
        ]

        # Initial weights (equal weight)
        initial_weights = np.array([1 / n_assets] * n_assets)

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_list,
        )

        if result.success:
            optimal_weights = result.x
            metrics = self.calculate_portfolio_metrics(returns, optimal_weights)

            return {
                "weights": optimal_weights,
                "metrics": metrics,
                "success": True,
                "message": result.message,
            }
        else:
            return {
                "weights": None,
                "metrics": None,
                "success": False,
                "message": result.message,
            }

    def efficient_frontier(
        self, returns: pd.DataFrame, n_points: int = 50
    ) -> pd.DataFrame:
        """
        Generate efficient frontier

        Args:
            returns: DataFrame with asset returns
            n_points: Number of points on the efficient frontier

        Returns:
            DataFrame with efficient frontier data
        """
        # Generate target returns
        min_return = returns.mean().min() * 252
        max_return = returns.mean().max() * 252
        target_returns = np.linspace(min_return, max_return, n_points)

        frontier_data = []

        for target_return in target_returns:
            # Optimize for minimum variance given target return
            def objective(weights):
                metrics = self.calculate_portfolio_metrics(returns, weights)
                return metrics["volatility"]

            def constraint_return(weights):
                metrics = self.calculate_portfolio_metrics(returns, weights)
                return metrics["return"] - target_return

            constraints = [
                {"type": "eq", "fun": lambda x: np.sum(x) - 1},
                {"type": "eq", "fun": constraint_return},
            ]

            bounds = [(0, 1) for _ in range(len(returns.columns))]
            initial_weights = np.array(
                [1 / len(returns.columns)] * len(returns.columns)
            )

            result = minimize(
                objective,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            if result.success:
                metrics = self.calculate_portfolio_metrics(returns, result.x)
                frontier_data.append(
                    {
                        "return": metrics["return"],
                        "volatility": metrics["volatility"],
                        "sharpe_ratio": metrics["sharpe_ratio"],
                        "weights": result.x,
                    }
                )

        return pd.DataFrame(frontier_data)


class RiskManager:
    """
    Advanced risk management tools
    """

    def __init__(self):
        self.var_confidence = 0.95
        self.cvar_confidence = 0.95

    def calculate_var(self, returns: pd.Series, confidence: float = None) -> float:
        """
        Calculate Value at Risk

        Args:
            returns: Series of returns
            confidence: Confidence level (default: 0.95)

        Returns:
            VaR value
        """
        if confidence is None:
            confidence = self.var_confidence

        return np.percentile(returns, (1 - confidence) * 100)

    def calculate_cvar(self, returns: pd.Series, confidence: float = None) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)

        Args:
            returns: Series of returns
            confidence: Confidence level (default: 0.95)

        Returns:
            CVaR value
        """
        if confidence is None:
            confidence = self.cvar_confidence

        var = self.calculate_var(returns, confidence)
        return returns[returns <= var].mean()

    def calculate_drawdown(self, prices: pd.Series) -> pd.Series:
        """
        Calculate drawdown series

        Args:
            prices: Series of prices

        Returns:
            Series of drawdown values
        """
        cumulative = prices.cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown

    def calculate_max_drawdown(self, prices: pd.Series) -> float:
        """
        Calculate maximum drawdown

        Args:
            prices: Series of prices

        Returns:
            Maximum drawdown value
        """
        drawdown = self.calculate_drawdown(prices)
        return drawdown.min()

    def calculate_volatility(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """
        Calculate rolling volatility

        Args:
            returns: Series of returns
            window: Rolling window size

        Returns:
            Series of volatility values
        """
        return returns.rolling(window=window).std() * np.sqrt(252)

    def calculate_beta(
        self, asset_returns: pd.Series, market_returns: pd.Series
    ) -> float:
        """
        Calculate beta relative to market

        Args:
            asset_returns: Asset returns
            market_returns: Market returns

        Returns:
            Beta value
        """
        # Align data
        aligned_data = pd.concat([asset_returns, market_returns], axis=1).dropna()

        if len(aligned_data) < 2:
            return np.nan

        asset_ret = aligned_data.iloc[:, 0]
        market_ret = aligned_data.iloc[:, 1]

        # Calculate beta
        covariance = np.cov(asset_ret, market_ret)[0, 1]
        market_variance = np.var(market_ret)

        return covariance / market_variance if market_variance > 0 else np.nan

    def generate_risk_report(
        self, returns: pd.DataFrame, prices: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive risk report

        Args:
            returns: DataFrame with asset returns
            prices: DataFrame with asset prices (optional)

        Returns:
            Dictionary with risk metrics
        """
        risk_report = {}

        for asset in returns.columns:
            asset_returns = returns[asset].dropna()

            risk_metrics = {
                "mean_return": asset_returns.mean() * 252,
                "volatility": asset_returns.std() * np.sqrt(252),
                "skewness": asset_returns.skew(),
                "kurtosis": asset_returns.kurtosis(),
                "var_95": self.calculate_var(asset_returns, 0.95),
                "cvar_95": self.calculate_cvar(asset_returns, 0.95),
                "sharpe_ratio": (asset_returns.mean() * 252)
                / (asset_returns.std() * np.sqrt(252)),
            }

            if prices is not None and asset in prices.columns:
                asset_prices = prices[asset].dropna()
                risk_metrics["max_drawdown"] = self.calculate_max_drawdown(asset_prices)

            risk_report[asset] = risk_metrics

        return risk_report


class AdvancedMLFeatures:
    """
    Main class combining all advanced ML features
    """

    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.news_analyzer = NewsImpactAnalyzer()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.risk_manager = RiskManager()

    def analyze_stock_with_sentiment(
        self, symbol: str, price_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis combining price data and sentiment

        Args:
            symbol: Stock symbol
            price_data: DataFrame with price data

        Returns:
            Dictionary with analysis results
        """
        # Get sentiment data
        sentiment_df = self.sentiment_analyzer.get_news_sentiment(symbol)
        daily_sentiment = self.sentiment_analyzer.calculate_daily_sentiment(
            sentiment_df
        )

        # Analyze news impact
        news_impact = self.news_analyzer.analyze_news_impact(symbol, price_data)

        # Calculate returns
        price_data["Returns"] = price_data["Close"].pct_change()

        # Align sentiment with price data
        aligned_data = price_data.join(daily_sentiment, how="inner")

        # Risk analysis
        risk_report = self.risk_manager.generate_risk_report(
            pd.DataFrame({"Returns": aligned_data["Returns"]})
        )

        return {
            "symbol": symbol,
            "sentiment_data": sentiment_df,
            "daily_sentiment": daily_sentiment,
            "news_impact": news_impact,
            "aligned_data": aligned_data,
            "risk_report": risk_report,
        }

    def optimize_portfolio_with_sentiment(
        self,
        symbols: List[str],
        price_data: Dict[str, pd.DataFrame],
        sentiment_weight: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Optimize portfolio considering sentiment

        Args:
            symbols: List of stock symbols
            price_data: Dictionary with price data for each symbol
            sentiment_weight: Weight for sentiment in optimization

        Returns:
            Dictionary with optimization results
        """
        # Calculate returns for all assets
        returns_data = {}
        sentiment_scores = {}

        for symbol in symbols:
            if symbol in price_data:
                data = price_data[symbol]
                returns_data[symbol] = data["Close"].pct_change().dropna()

                # Get sentiment score
                sentiment_df = self.sentiment_analyzer.get_news_sentiment(symbol)
                if not sentiment_df.empty:
                    avg_sentiment = sentiment_df["compound"].mean()
                    sentiment_scores[symbol] = avg_sentiment
                else:
                    sentiment_scores[symbol] = 0

        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()

        # Adjust returns based on sentiment
        sentiment_adjusted_returns = returns_df.copy()
        for symbol in symbols:
            if symbol in sentiment_scores:
                sentiment_adjusted_returns[symbol] += (
                    sentiment_scores[symbol] * sentiment_weight
                )

        # Optimize portfolio
        optimization_result = self.portfolio_optimizer.optimize_portfolio(
            sentiment_adjusted_returns, method="sharpe"
        )

        # Generate efficient frontier
        frontier = self.portfolio_optimizer.efficient_frontier(
            sentiment_adjusted_returns
        )

        return {
            "optimization_result": optimization_result,
            "efficient_frontier": frontier,
            "sentiment_scores": sentiment_scores,
            "returns_data": returns_df,
        }

    def create_sentiment_dashboard(
        self, symbol: str, analysis_results: Dict[str, Any]
    ) -> go.Figure:
        """
        Create interactive sentiment dashboard

        Args:
            symbol: Stock symbol
            analysis_results: Results from analyze_stock_with_sentiment

        Returns:
            Plotly figure with dashboard
        """
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Price and Sentiment",
                "Sentiment Distribution",
                "News Sentiment Over Time",
                "Sentiment vs Returns",
                "Risk Metrics",
                "Drawdown Analysis",
            ),
            specs=[
                [{"secondary_y": True}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatter"}],
            ],
        )

        aligned_data = analysis_results["aligned_data"]

        # Price and sentiment
        fig.add_trace(
            go.Scatter(x=aligned_data.index, y=aligned_data["Close"], name="Price"),
            row=1,
            col=1,
        )

        if "compound" in aligned_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=aligned_data.index, y=aligned_data["compound"], name="Sentiment"
                ),
                row=1,
                col=1,
                secondary_y=True,
            )

        # Sentiment distribution
        if "compound" in aligned_data.columns:
            fig.add_trace(
                go.Histogram(x=aligned_data["compound"], name="Sentiment Distribution"),
                row=1,
                col=2,
            )

        # News sentiment over time
        if "compound" in aligned_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=aligned_data.index, y=aligned_data["compound"], name="Sentiment"
                ),
                row=2,
                col=1,
            )

        # Sentiment vs returns
        if "compound" in aligned_data.columns and "Returns" in aligned_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=aligned_data["compound"],
                    y=aligned_data["Returns"],
                    mode="markers",
                    name="Sentiment vs Returns",
                ),
                row=2,
                col=2,
            )

        # Risk metrics
        risk_report = analysis_results["risk_report"]
        if "Returns" in risk_report:
            metrics = risk_report["Returns"]
            fig.add_trace(
                go.Bar(
                    x=list(metrics.keys()),
                    y=list(metrics.values()),
                    name="Risk Metrics",
                ),
                row=3,
                col=1,
            )

        # Drawdown
        if "Close" in aligned_data.columns:
            drawdown = self.risk_manager.calculate_drawdown(aligned_data["Close"])
            fig.add_trace(
                go.Scatter(x=aligned_data.index, y=drawdown, name="Drawdown"),
                row=3,
                col=2,
            )

        fig.update_layout(
            title=f"Sentiment Analysis Dashboard - {symbol}",
            height=800,
            showlegend=True,
        )

        return fig
