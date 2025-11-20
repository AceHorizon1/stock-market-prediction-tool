#!/usr/bin/env python3
"""
Stock Market Prediction Tool - Demo
A simple demo that showcases the tool's capabilities.
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def run_demo():
    """Run a simple demo of the stock prediction tool"""
    print("🎬 Stock Market Prediction Tool - Demo")
    print("=" * 50)

    try:
        # Import our modules
        print("📦 Loading modules...")
        from data_collector import DataCollector
        from feature_engineering import FeatureEngineer
        from models import AdvancedStockPredictor
        from evaluation import ModelEvaluator

        print("✅ Modules loaded successfully")

        # Step 1: Load data
        print("\n📊 Step 1: Loading stock data...")
        collector = DataCollector()

        # Load data for a popular stock
        symbols = ["AAPL"]  # Apple Inc.
        data = collector.create_comprehensive_dataset(symbols, include_market_data=True)

        if data.empty:
            print("❌ No data loaded. This might be due to:")
            print("   - Internet connection issues")
            print("   - Yahoo Finance API limitations")
            print("   - Stock symbol not found")
            return

        print(f"✅ Data loaded successfully!")
        print(f"   Records: {len(data)}")
        print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"   Columns: {len(data.columns)}")

        # Show sample data
        print("\n📋 Sample data:")
        print(data[["Close", "Volume", "Returns"]].head())

        # Step 2: Engineer features
        print("\n🔧 Step 2: Engineering features...")
        fe = FeatureEngineer()
        engineered_data = fe.engineer_all_features(data, target_horizons=[1, 3, 5])

        print(f"✅ Features engineered!")
        print(f"   Original features: {len(data.columns)}")
        print(f"   Engineered features: {len(engineered_data.columns)}")

        # Show some new features
        new_features = [
            col for col in engineered_data.columns if col not in data.columns
        ]
        print(f"   New features created: {len(new_features)}")
        print(f"   Sample new features: {new_features[:10]}")

        # Step 3: Prepare data for training
        print("\n🎯 Step 3: Preparing data for training...")
        target_column = "Target_Return_1d"

        # Clean data
        data_clean = engineered_data.dropna(subset=[target_column])

        # Select features
        feature_columns = fe.select_features(
            data_clean, target_column, method="correlation", threshold=0.001
        )

        if not feature_columns:
            # Fallback to all numeric features
            feature_columns = [
                col
                for col in data_clean.columns
                if col not in ["Symbol", "Stock_Symbol"]
                and data_clean[col].dtype in ["float64", "int64"]
                and "Target_" not in col
            ]

        print(f"✅ Data prepared!")
        print(f"   Features selected: {len(feature_columns)}")
        print(f"   Clean records: {len(data_clean)}")

        # Prepare X and y
        X = data_clean[feature_columns]
        y = data_clean[target_column]

        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")

        # Step 4: Train model
        print("\n🤖 Step 4: Training machine learning model...")
        predictor = AdvancedStockPredictor(model_type="tree", task="regression")

        results = predictor.train_models(X_train, y_train, X_val, y_val)

        print("✅ Model training completed!")
        for model_name, metrics in results.items():
            print(f"   {model_name}:")
            print(f"     RMSE: {metrics.get('rmse', 0):.4f}")
            print(f"     MAE: {metrics.get('mae', 0):.4f}")
            print(f"     R²: {metrics.get('r_squared', 0):.4f}")

        # Step 5: Make predictions
        print("\n🔮 Step 5: Making predictions...")
        predictions = predictor.ensemble_predict(X_val)

        print(f"✅ Predictions made!")
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Sample predictions: {predictions[:5]}")

        # Step 6: Evaluate results
        print("\n📈 Step 6: Evaluating results...")
        evaluator = ModelEvaluator()

        # Calculate metrics
        metrics = evaluator.calculate_regression_metrics(y_val.values, predictions)

        print("✅ Evaluation completed!")
        print("Performance metrics:")
        print(f"   RMSE: {metrics.get('rmse', 0):.4f}")
        print(f"   MAE: {metrics.get('mae', 0):.4f}")
        print(f"   R²: {metrics.get('r_squared', 0):.4f}")
        print(f"   Directional Accuracy: {metrics.get('directional_accuracy', 0):.2%}")

        # Step 7: Backtest results
        print("\n💰 Step 7: Backtesting strategy...")
        backtest_data = data_clean.iloc[split_idx:].copy()
        backtest_data["Predictions"] = predictions

        backtest_results = evaluator.backtest_strategy(backtest_data, predictions)

        print("✅ Backtest completed!")
        print("Trading performance:")
        print(f"   Total Return: {backtest_results['total_return']:.2%}")
        print(f"   Annual Return: {backtest_results['annual_return']:.2%}")
        print(f"   Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {backtest_results['max_drawdown']:.2%}")
        print(f"   Win Rate: {backtest_results['win_rate']:.2%}")

        # Step 8: Show feature importance
        print("\n🎯 Step 8: Feature importance...")
        importance = predictor.get_feature_importance()

        if importance:
            # Get top 10 features
            sorted_importance = sorted(
                importance.items(), key=lambda x: x[1], reverse=True
            )
            top_features = sorted_importance[:10]

            print("✅ Feature importance calculated!")
            print("Top 10 most important features:")
            for feature, importance_score in top_features:
                print(f"   {feature}: {importance_score:.4f}")

        # Summary
        print("\n" + "=" * 50)
        print("🎉 Demo Completed Successfully!")
        print("=" * 50)

        print("\n📊 Summary:")
        print(f"   Data loaded: {len(data)} records")
        print(f"   Features created: {len(engineered_data.columns)}")
        print(f"   Model trained: {len(results)} models")
        print(f"   Predictions made: {len(predictions)}")
        print(f"   Performance: RMSE={metrics.get('rmse', 0):.4f}")
        print(f"   Trading return: {backtest_results['total_return']:.2%}")

        print("\n🚀 Ready to use the full tool!")
        print("Run 'python run_app.py' to start the web interface")

        return True

    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Make sure all dependencies are installed")
        print("3. Run 'python test_installation.py' to diagnose issues")
        return False


def main():
    """Main function"""
    success = run_demo()

    if success:
        print("\n✅ Demo completed successfully!")
    else:
        print("\n❌ Demo failed. Please check the errors above.")


if __name__ == "__main__":
    main()
