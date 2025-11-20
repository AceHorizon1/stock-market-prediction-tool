#!/usr/bin/env python3
"""
Test Installation Script for Stock Market Prediction Tool
This script tests all components to ensure they work correctly.
"""

import sys
import traceback
import warnings

warnings.filterwarnings("ignore")


def test_imports():
    """Test if all modules can be imported"""
    print("🔍 Testing imports...")

    try:
        import pandas as pd
        import numpy as np
        import yfinance as yf
        import streamlit as st
        import plotly.graph_objects as go
        import matplotlib.pyplot as plt
        import seaborn as sns
        import ta
        import sklearn
        import xgboost as xgb
        import lightgbm as lgb

        print("✅ All external libraries imported successfully")
        return True

    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        return False


def test_custom_modules():
    """Test if our custom modules can be imported"""
    print("\n🔍 Testing custom modules...")

    try:
        from data_collector import DataCollector
        from feature_engineering import FeatureEngineer
        from models import AdvancedStockPredictor
        from evaluation import ModelEvaluator

        print("✅ All custom modules imported successfully")
        return True

    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()
        return False


def test_data_collector():
    """Test data collector functionality"""
    print("\n🔍 Testing data collector...")

    try:
        from data_collector import DataCollector

        collector = DataCollector()

        # Test basic stock data collection
        data = collector.get_stock_data("AAPL", period="1mo")

        if data.empty:
            print("❌ No data collected for AAPL")
            return False

        print(f"✅ Data collector working - collected {len(data)} records for AAPL")
        print(f"   Columns: {list(data.columns)}")
        return True

    except Exception as e:
        print(f"❌ Data collector error: {str(e)}")
        traceback.print_exc()
        return False


def test_feature_engineering():
    """Test feature engineering functionality"""
    print("\n🔍 Testing feature engineering...")

    try:
        from feature_engineering import FeatureEngineer
        import pandas as pd
        import numpy as np

        # Create sample data
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        sample_data = pd.DataFrame(
            {
                "Open": np.random.randn(len(dates)).cumsum() + 100,
                "High": np.random.randn(len(dates)).cumsum() + 102,
                "Low": np.random.randn(len(dates)).cumsum() + 98,
                "Close": np.random.randn(len(dates)).cumsum() + 100,
                "Volume": np.random.randint(1000000, 10000000, len(dates)),
            },
            index=dates,
        )

        sample_data["Returns"] = sample_data["Close"].pct_change()

        # Test feature engineering
        fe = FeatureEngineer()
        engineered_data = fe.engineer_all_features(sample_data, target_horizons=[1])

        if engineered_data.empty:
            print("❌ Feature engineering produced empty data")
            return False

        print(
            f"✅ Feature engineering working - created {len(engineered_data.columns)} features"
        )
        print(f"   Shape: {engineered_data.shape}")
        return True

    except Exception as e:
        print(f"❌ Feature engineering error: {str(e)}")
        traceback.print_exc()
        return False


def test_models():
    """Test model functionality"""
    print("\n🔍 Testing models...")

    try:
        from models import AdvancedStockPredictor
        import pandas as pd
        import numpy as np

        # Create sample data
        n_samples = 100
        n_features = 10

        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        y = pd.Series(np.random.randn(n_samples))

        # Test model initialization
        predictor = AdvancedStockPredictor(model_type="tree", task="regression")

        # Test model training
        split_idx = int(0.8 * n_samples)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        results = predictor.train_models(X_train, y_train, X_val, y_val)

        if not results:
            print("❌ Model training produced no results")
            return False

        print(f"✅ Models working - trained {len(results)} models")
        for model_name, metrics in results.items():
            print(f"   {model_name}: RMSE={metrics.get('rmse', 0):.4f}")

        return True

    except Exception as e:
        print(f"❌ Models error: {str(e)}")
        traceback.print_exc()
        return False


def test_evaluation():
    """Test evaluation functionality"""
    print("\n🔍 Testing evaluation...")

    try:
        from evaluation import ModelEvaluator
        import pandas as pd
        import numpy as np

        # Create sample data
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        sample_data = pd.DataFrame(
            {
                "Close": np.random.randn(len(dates)).cumsum() + 100,
                "Returns": np.random.normal(0.001, 0.02, len(dates)),
            },
            index=dates,
        )

        # Create sample predictions
        predictions = sample_data["Returns"] + np.random.normal(0, 0.01, len(dates))

        # Test evaluation
        evaluator = ModelEvaluator()

        # Test regression metrics
        y_true = sample_data["Returns"].values[1:]
        y_pred = predictions.values[1:]

        metrics = evaluator.calculate_regression_metrics(y_true, y_pred)

        if not metrics:
            print("❌ Evaluation produced no metrics")
            return False

        print(f"✅ Evaluation working - calculated {len(metrics)} metrics")
        print(f"   RMSE: {metrics.get('rmse', 0):.4f}")
        print(f"   MAE: {metrics.get('mae', 0):.4f}")

        return True

    except Exception as e:
        print(f"❌ Evaluation error: {str(e)}")
        traceback.print_exc()
        return False


def test_streamlit():
    """Test if Streamlit can be imported and configured"""
    print("\n🔍 Testing Streamlit...")

    try:
        import streamlit as st

        # Test basic Streamlit functionality
        if hasattr(st, "title"):
            print("✅ Streamlit working correctly")
            return True
        else:
            print("❌ Streamlit not properly configured")
            return False

    except Exception as e:
        print(f"❌ Streamlit error: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("🧪 Stock Market Prediction Tool - Installation Test")
    print("=" * 60)

    tests = [
        ("External Libraries", test_imports),
        ("Custom Modules", test_custom_modules),
        ("Data Collector", test_data_collector),
        ("Feature Engineering", test_feature_engineering),
        ("Models", test_models),
        ("Evaluation", test_evaluation),
        ("Streamlit", test_streamlit),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {str(e)}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1

    print("-" * 60)
    print(f"Total: {total}, Passed: {passed}, Failed: {total - passed}")

    if passed == total:
        print("\n🎉 All tests passed! The tool is ready to use.")
        print("\nTo run the application:")
        print("   python run_app.py")
        print("   or")
        print("   streamlit run main.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Check your internet connection for data collection")
        print("3. Ensure all files are in the same directory")


if __name__ == "__main__":
    main()
