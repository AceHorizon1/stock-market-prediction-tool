#!/usr/bin/env python3
"""
Test script for the Stock Market Prediction Tool
"""

import sys
import traceback
from data_collector import DataCollector
from feature_engineering import FeatureEngineer


def test_data_collection():
    """Test data collection"""
    print("🧪 Testing Data Collection...")

    try:
        collector = DataCollector()
        symbols = ["AAPL", "MSFT"]

        # Test individual stock data
        print("  Testing individual stock data...")
        data = collector.get_stock_data("AAPL", period="1y")
        print(f"    AAPL data shape: {data.shape}")

        # Test comprehensive dataset
        print("  Testing comprehensive dataset...")
        combined_data = collector.create_comprehensive_dataset(
            symbols, include_market_data=False, include_economic_data=False
        )
        print(f"    Combined data shape: {combined_data.shape}")

        if not combined_data.empty:
            print("✅ Data collection test PASSED")
            return combined_data
        else:
            print("❌ Data collection test FAILED - empty dataset")
            return None

    except Exception as e:
        print(f"❌ Data collection test FAILED: {str(e)}")
        traceback.print_exc()
        return None


def test_feature_engineering(data):
    """Test feature engineering"""
    print("\n🧪 Testing Feature Engineering...")

    try:
        engineer = FeatureEngineer()

        # Debug: Check data structure
        print(f"    Original data shape: {data.shape}")
        print(f"    Columns: {list(data.columns)}")
        if "Stock_Symbol" in data.columns:
            print(f"    Stocks: {data['Stock_Symbol'].unique()}")
            print(f"    Stock counts: {data['Stock_Symbol'].value_counts()}")

        # Test feature engineering
        engineered_data = engineer.engineer_all_features(
            data, target_horizons=[1, 3, 5]
        )

        print(f"    Engineered data shape: {engineered_data.shape}")
        print(f"    Features created: {len(engineered_data.columns)}")

        if not engineered_data.empty:
            print("✅ Feature engineering test PASSED")
            return engineered_data
        else:
            print("❌ Feature engineering test FAILED - empty dataset")
            return None

    except Exception as e:
        print(f"❌ Feature engineering test FAILED: {str(e)}")
        traceback.print_exc()
        return None


def main():
    """Main test function"""
    print("🚀 Stock Market Prediction Tool - Test Suite")
    print("=" * 50)

    # Test data collection
    data = test_data_collection()
    if data is None:
        print("\n❌ Tests failed. Cannot proceed.")
        return

    # Test feature engineering
    engineered_data = test_feature_engineering(data)
    if engineered_data is None:
        print("\n❌ Tests failed. Cannot proceed.")
        return

    # Show sample data
    print("\n📊 Sample Engineered Data:")
    print(engineered_data.head())

    print("\n✅ All tests PASSED! The app should work properly.")
    print("\n🌐 You can now access the web app at: http://localhost:8501")


if __name__ == "__main__":
    main()
