"""
Comprehensive test suite for enhanced Stock Market AI features
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

# Import modules to test
from enhanced_data_collector import EnhancedDataCollector
from enhanced_feature_engineering import EnhancedFeatureEngineer
from advanced_ml_features import (
    SentimentAnalyzer,
    NewsImpactAnalyzer,
    PortfolioOptimizer,
    RiskManager,
    AdvancedMLFeatures,
)
from config import config


class TestEnhancedDataCollector:
    """Test enhanced data collector functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.collector = EnhancedDataCollector()
        self.test_data = self._create_test_data()

    def _create_test_data(self) -> pd.DataFrame:
        """Create test data for testing"""
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        data = {
            "Open": np.random.uniform(100, 200, len(dates)),
            "High": np.random.uniform(150, 250, len(dates)),
            "Low": np.random.uniform(50, 150, len(dates)),
            "Close": np.random.uniform(100, 200, len(dates)),
            "Volume": np.random.uniform(1000000, 5000000, len(dates)),
            "Symbol": "TEST",
        }
        df = pd.DataFrame(data, index=dates)
        return df

    def test_data_validation(self):
        """Test data validation functionality"""
        validation = self.collector.validate_data(self.test_data)

        assert "total_rows" in validation
        assert "data_quality_score" in validation
        assert validation["total_rows"] == len(self.test_data)
        assert 0 <= validation["data_quality_score"] <= 1

    def test_data_summary(self):
        """Test data summary functionality"""
        summary = self.collector.get_data_summary(self.test_data)

        assert "shape" in summary
        assert "columns" in summary
        assert "date_range" in summary
        assert summary["shape"] == self.test_data.shape

    def test_cache_functionality(self):
        """Test caching functionality"""
        # Test cache info
        cache_info = self.collector.get_cache_info()
        assert isinstance(cache_info, dict)

        # Test cache clearing
        self.collector.clear_cache()
        cache_info_after = self.collector.get_cache_info()
        assert cache_info_after == {}

    def test_csv_loading(self):
        """Test CSV file loading"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            self.test_data.to_csv(f.name)
            temp_file = f.name

        try:
            loaded_data = self.collector.load_csv_file(temp_file)
            assert len(loaded_data) == len(self.test_data)
            assert "Symbol" in loaded_data.columns
        finally:
            # Cleanup
            Path(temp_file).unlink()


class TestEnhancedFeatureEngineer:
    """Test enhanced feature engineering functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.engineer = EnhancedFeatureEngineer()
        self.test_data = self._create_test_data()

    def _create_test_data(self) -> pd.DataFrame:
        """Create test data for feature engineering"""
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        data = {
            "Open": np.random.uniform(100, 200, len(dates)),
            "High": np.random.uniform(150, 250, len(dates)),
            "Low": np.random.uniform(50, 150, len(dates)),
            "Close": np.random.uniform(100, 200, len(dates)),
            "Volume": np.random.uniform(1000000, 5000000, len(dates)),
        }
        df = pd.DataFrame(data, index=dates)
        return df

    def test_technical_indicators(self):
        """Test technical indicators calculation"""
        result = self.engineer.add_technical_indicators_parallel(self.test_data)

        # Check that technical indicators were added
        expected_indicators = ["SMA_5", "SMA_20", "RSI", "MACD", "BB_Upper"]
        for indicator in expected_indicators:
            assert indicator in result.columns

    def test_statistical_features(self):
        """Test statistical features calculation"""
        result = self.engineer.add_statistical_features_optimized(self.test_data)

        # Check that statistical features were added
        expected_features = ["Returns_Mean_5d", "Returns_Std_5d", "Volume_Mean_5d"]
        for feature in expected_features:
            assert feature in result.columns

    def test_time_features(self):
        """Test time features calculation"""
        result = self.engineer.add_time_features(self.test_data)

        # Check that time features were added
        expected_features = ["Day_of_Week", "Month", "Year", "Day_of_Week_Sin"]
        for feature in expected_features:
            assert feature in result.columns

    def test_target_variables(self):
        """Test target variable creation"""
        horizons = [1, 3, 5]
        result = self.engineer.create_target_variables(self.test_data, horizons)

        # Check that target variables were created
        for horizon in horizons:
            assert f"Target_Return_{horizon}d" in result.columns
            assert f"Target_Direction_{horizon}d" in result.columns

    def test_feature_engineering_pipeline(self):
        """Test complete feature engineering pipeline"""
        result = self.engineer.engineer_all_features(self.test_data, use_cache=False)

        # Check that features were created
        assert len(result.columns) > len(self.test_data.columns)
        assert not result.isnull().all().any()  # No completely null columns

    def test_feature_selection(self):
        """Test feature selection functionality"""
        # Create data with target variable
        engineered_data = self.engineer.engineer_all_features(
            self.test_data, use_cache=False
        )
        target_column = "Target_Return_1d"

        selected_features = self.engineer.select_features_advanced(
            engineered_data, target_column, method="correlation", n_features=10
        )

        assert len(selected_features) <= 10
        assert all(feature in engineered_data.columns for feature in selected_features)

    def test_feature_scaling(self):
        """Test feature scaling functionality"""
        engineered_data = self.engineer.engineer_all_features(
            self.test_data, use_cache=False
        )
        numeric_columns = engineered_data.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        scaled_data = self.engineer.scale_features(
            engineered_data, numeric_columns, method="standard", fit=True
        )

        # Check that scaling was applied
        assert scaled_data.shape == engineered_data.shape
        # Check that scaled data has mean close to 0 and std close to 1
        scaled_numeric = scaled_data[numeric_columns]
        assert abs(scaled_numeric.mean().mean()) < 0.1
        assert abs(scaled_numeric.std().mean() - 1) < 0.1


class TestSentimentAnalyzer:
    """Test sentiment analysis functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.analyzer = SentimentAnalyzer()

    def test_text_sentiment_analysis(self):
        """Test text sentiment analysis"""
        positive_text = "This is a great stock with excellent performance!"
        negative_text = "This stock is terrible and performing poorly."
        neutral_text = "The stock price moved today."

        # Test positive sentiment
        positive_sentiment = self.analyzer.analyze_text_sentiment(positive_text)
        assert positive_sentiment["compound"] > 0
        assert positive_sentiment["pos"] > 0

        # Test negative sentiment
        negative_sentiment = self.analyzer.analyze_text_sentiment(negative_text)
        assert negative_sentiment["compound"] < 0
        assert negative_sentiment["neg"] > 0

        # Test neutral sentiment
        neutral_sentiment = self.analyzer.analyze_text_sentiment(neutral_text)
        assert abs(neutral_sentiment["compound"]) < 0.1

    def test_empty_text_handling(self):
        """Test handling of empty or null text"""
        empty_sentiment = self.analyzer.analyze_text_sentiment("")
        null_sentiment = self.analyzer.analyze_text_sentiment(None)

        assert empty_sentiment["compound"] == 0
        assert null_sentiment["compound"] == 0

    def test_daily_sentiment_calculation(self):
        """Test daily sentiment calculation"""
        # Create mock sentiment data
        dates = pd.date_range("2023-01-01", "2023-01-05")
        sentiment_data = []

        for date in dates:
            sentiment_data.append(
                {
                    "compound": np.random.uniform(-1, 1),
                    "pos": np.random.uniform(0, 1),
                    "neg": np.random.uniform(0, 1),
                    "neu": np.random.uniform(0, 1),
                    "textblob_polarity": np.random.uniform(-1, 1),
                    "textblob_subjectivity": np.random.uniform(0, 1),
                }
            )

        sentiment_df = pd.DataFrame(sentiment_data, index=dates)
        daily_sentiment = self.analyzer.calculate_daily_sentiment(sentiment_df)

        assert len(daily_sentiment) == len(dates)
        assert "compound" in daily_sentiment.columns


class TestPortfolioOptimizer:
    """Test portfolio optimization functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.optimizer = PortfolioOptimizer()
        self.test_returns = self._create_test_returns()

    def _create_test_returns(self) -> pd.DataFrame:
        """Create test returns data"""
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        returns_data = {
            "Asset_A": np.random.normal(0.001, 0.02, len(dates)),
            "Asset_B": np.random.normal(0.0008, 0.015, len(dates)),
            "Asset_C": np.random.normal(0.0012, 0.025, len(dates)),
        }
        return pd.DataFrame(returns_data, index=dates)

    def test_portfolio_metrics_calculation(self):
        """Test portfolio metrics calculation"""
        weights = np.array([0.4, 0.3, 0.3])
        metrics = self.optimizer.calculate_portfolio_metrics(self.test_returns, weights)

        assert "return" in metrics
        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "var_95" in metrics

        assert metrics["volatility"] >= 0
        assert metrics["max_drawdown"] <= 0

    def test_portfolio_optimization(self):
        """Test portfolio optimization"""
        result = self.optimizer.optimize_portfolio(self.test_returns, method="sharpe")

        assert result["success"] == True
        assert result["weights"] is not None
        assert len(result["weights"]) == len(self.test_returns.columns)
        assert abs(np.sum(result["weights"]) - 1.0) < 1e-6

    def test_efficient_frontier(self):
        """Test efficient frontier generation"""
        frontier = self.optimizer.efficient_frontier(self.test_returns, n_points=10)

        assert len(frontier) > 0
        assert "return" in frontier.columns
        assert "volatility" in frontier.columns
        assert "sharpe_ratio" in frontier.columns

        # Check that return increases with volatility (efficient frontier property)
        sorted_frontier = frontier.sort_values("volatility")
        assert sorted_frontier["return"].is_monotonic_increasing


class TestRiskManager:
    """Test risk management functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.risk_manager = RiskManager()
        self.test_returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
        self.test_prices = pd.Series(np.cumprod(1 + self.test_returns))

    def test_var_calculation(self):
        """Test Value at Risk calculation"""
        var_95 = self.risk_manager.calculate_var(self.test_returns, 0.95)
        var_99 = self.risk_manager.calculate_var(self.test_returns, 0.99)

        assert var_95 > var_99  # Higher confidence = more negative VaR
        assert var_95 < 0  # VaR should be negative for losses

    def test_cvar_calculation(self):
        """Test Conditional Value at Risk calculation"""
        cvar_95 = self.risk_manager.calculate_cvar(self.test_returns, 0.95)
        var_95 = self.risk_manager.calculate_var(self.test_returns, 0.95)

        assert cvar_95 <= var_95  # CVaR should be more negative than VaR

    def test_drawdown_calculation(self):
        """Test drawdown calculation"""
        drawdown = self.risk_manager.calculate_drawdown(self.test_prices)
        max_drawdown = self.risk_manager.calculate_max_drawdown(self.test_prices)

        assert len(drawdown) == len(self.test_prices)
        assert max_drawdown <= 0
        assert max_drawdown == drawdown.min()

    def test_volatility_calculation(self):
        """Test rolling volatility calculation"""
        volatility = self.risk_manager.calculate_volatility(
            self.test_returns, window=30
        )

        assert len(volatility) == len(self.test_returns)
        assert volatility.iloc[29:].notna().all()  # First 29 values should be NaN

    def test_beta_calculation(self):
        """Test beta calculation"""
        asset_returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        market_returns = pd.Series(np.random.normal(0.0008, 0.015, 100))

        beta = self.risk_manager.calculate_beta(asset_returns, market_returns)

        assert not pd.isna(beta)
        assert isinstance(beta, float)

    def test_risk_report_generation(self):
        """Test comprehensive risk report generation"""
        returns_df = pd.DataFrame(
            {
                "Asset_A": self.test_returns,
                "Asset_B": pd.Series(np.random.normal(0.0008, 0.015, 1000)),
            }
        )

        risk_report = self.risk_manager.generate_risk_report(returns_df)

        assert "Asset_A" in risk_report
        assert "Asset_B" in risk_report

        for asset in risk_report:
            metrics = risk_report[asset]
            assert "mean_return" in metrics
            assert "volatility" in metrics
            assert "sharpe_ratio" in metrics
            assert "var_95" in metrics
            assert "cvar_95" in metrics


class TestAdvancedMLFeatures:
    """Test advanced ML features integration"""

    def setup_method(self):
        """Setup test environment"""
        self.advanced_features = AdvancedMLFeatures()
        self.test_price_data = self._create_test_price_data()

    def _create_test_price_data(self) -> pd.DataFrame:
        """Create test price data"""
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        data = {
            "Open": np.random.uniform(100, 200, len(dates)),
            "High": np.random.uniform(150, 250, len(dates)),
            "Low": np.random.uniform(50, 150, len(dates)),
            "Close": np.random.uniform(100, 200, len(dates)),
            "Volume": np.random.uniform(1000000, 5000000, len(dates)),
        }
        return pd.DataFrame(data, index=dates)

    def test_sentiment_analyzer_integration(self):
        """Test sentiment analyzer integration"""
        analyzer = self.advanced_features.sentiment_analyzer

        # Test text sentiment analysis
        sentiment = analyzer.analyze_text_sentiment("This is a great stock!")
        assert "compound" in sentiment
        assert "pos" in sentiment
        assert "neg" in sentiment
        assert "neu" in sentiment

    def test_portfolio_optimizer_integration(self):
        """Test portfolio optimizer integration"""
        optimizer = self.advanced_features.portfolio_optimizer

        # Create test returns
        returns_data = {
            "Asset_A": np.random.normal(0.001, 0.02, 100),
            "Asset_B": np.random.normal(0.0008, 0.015, 100),
            "Asset_C": np.random.normal(0.0012, 0.025, 100),
        }
        returns_df = pd.DataFrame(returns_data)

        # Test optimization
        result = optimizer.optimize_portfolio(returns_df, method="sharpe")
        assert result["success"] == True

    def test_risk_manager_integration(self):
        """Test risk manager integration"""
        risk_manager = self.advanced_features.risk_manager

        # Test risk calculations
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        var = risk_manager.calculate_var(returns, 0.95)
        cvar = risk_manager.calculate_cvar(returns, 0.95)

        assert var < 0
        assert cvar <= var


# Integration tests
class TestIntegration:
    """Integration tests for the complete system"""

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # Create test data
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        test_data = pd.DataFrame(
            {
                "Open": np.random.uniform(100, 200, len(dates)),
                "High": np.random.uniform(150, 250, len(dates)),
                "Low": np.random.uniform(50, 150, len(dates)),
                "Close": np.random.uniform(100, 200, len(dates)),
                "Volume": np.random.uniform(1000000, 5000000, len(dates)),
                "Symbol": "TEST",
            },
            index=dates,
        )

        # Test data collection
        collector = EnhancedDataCollector()
        validation = collector.validate_data(test_data)
        assert validation["data_quality_score"] > 0

        # Test feature engineering
        engineer = EnhancedFeatureEngineer()
        engineered_data = engineer.engineer_all_features(test_data, use_cache=False)
        assert len(engineered_data.columns) > len(test_data.columns)

        # Test portfolio optimization
        optimizer = PortfolioOptimizer()
        returns_data = {
            "Asset_A": test_data["Close"].pct_change().dropna(),
            "Asset_B": test_data["Close"].pct_change().dropna() * 0.8,
        }
        returns_df = pd.DataFrame(returns_data)

        result = optimizer.optimize_portfolio(returns_df, method="sharpe")
        assert result["success"] == True

    def test_configuration_loading(self):
        """Test configuration loading"""
        # Test that config can be imported and used
        assert hasattr(config, "data")
        assert hasattr(config, "features")
        assert hasattr(config, "model")
        assert hasattr(config, "ui")

        # Test configuration values
        assert isinstance(config.data.default_symbols, list)
        assert isinstance(config.features.target_horizons, list)
        assert isinstance(config.model.random_state, int)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
