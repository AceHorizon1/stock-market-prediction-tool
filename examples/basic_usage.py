#!/usr/bin/env python3
"""
Basic Usage Example for Stock Market Prediction Tool
This example demonstrates the basic workflow of the tool.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_collector import DataCollector
from feature_engineering import FeatureEngineer
from models import AdvancedStockPredictor
from evaluation import ModelEvaluator

def basic_prediction_example():
    """
    Basic example: Predict stock returns for a single stock
    """
    print("🚀 Starting Basic Stock Prediction Example")
    print("=" * 50)
    
    # Step 1: Initialize components
    print("1. Initializing components...")
    collector = DataCollector()
    fe = FeatureEngineer()
    predictor = AdvancedStockPredictor(model_type='ensemble', task='regression')
    evaluator = ModelEvaluator()
    
    # Step 2: Load data
    print("2. Loading stock data...")
    symbols = ['AAPL']  # Apple stock
    data = collector.create_comprehensive_dataset(symbols, include_market_data=True)
    
    if data.empty:
        print("❌ No data loaded. Please check your internet connection.")
        return
    
    print(f"✅ Data loaded successfully! Shape: {data.shape}")
    
    # Step 3: Engineer features
    print("3. Engineering features...")
    engineered_data = fe.engineer_all_features(data, target_horizons=[1, 3, 5])
    
    print(f"✅ Features engineered! Shape: {engineered_data.shape}")
    print(f"   Features created: {len(engineered_data.columns)}")
    
    # Step 4: Prepare data for training
    print("4. Preparing data for training...")
    target_column = 'Target_Return_1d'
    
    # Remove rows with NaN targets
    data_clean = engineered_data.dropna(subset=[target_column])
    
    # Select features
    feature_columns = fe.select_features(
        data_clean, 
        target_column, 
        method='correlation', 
        threshold=0.01
    )
    
    if not feature_columns:
        print("❌ No features selected. Trying with lower threshold...")
        feature_columns = fe.select_features(
            data_clean, 
            target_column, 
            method='correlation', 
            threshold=0.001
        )
    
    if not feature_columns:
        print("❌ Still no features selected. Using all numeric features...")
        feature_columns = [col for col in data_clean.columns 
                          if col not in ['Symbol', 'Stock_Symbol'] and 
                          data_clean[col].dtype in ['float64', 'int64'] and
                          'Target_' not in col]
    
    print(f"✅ Selected {len(feature_columns)} features")
    
    # Prepare X and y
    X = data_clean[feature_columns]
    y = data_clean[target_column]
    
    # Split data (time series split)
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"✅ Data split - Train: {len(X_train)}, Validation: {len(X_val)}")
    
    # Step 5: Train model
    print("5. Training model...")
    results = predictor.train_models(X_train, y_train, X_val, y_val)
    
    print("✅ Model training completed!")
    print("Training results:")
    for model_name, metrics in results.items():
        print(f"   {model_name}: RMSE={metrics.get('rmse', 0):.4f}, MAE={metrics.get('mae', 0):.4f}")
    
    # Step 6: Make predictions
    print("6. Making predictions...")
    predictions = predictor.ensemble_predict(X_val)
    
    print(f"✅ Predictions made! Shape: {predictions.shape}")
    
    # Step 7: Evaluate results
    print("7. Evaluating results...")
    evaluation_report = evaluator.generate_evaluation_report(
        data_clean.iloc[split_idx:], predictions
    )
    
    print("✅ Evaluation completed!")
    print("Performance metrics:")
    for metric, value in evaluation_report['metrics'].items():
        if isinstance(value, float):
            print(f"   {metric}: {value:.4f}")
    
    # Step 8: Show backtest results
    print("8. Backtest results:")
    backtest = evaluation_report['backtest']
    print(f"   Total Return: {backtest['total_return']:.2%}")
    print(f"   Sharpe Ratio: {backtest['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {backtest['max_drawdown']:.2%}")
    print(f"   Win Rate: {backtest['win_rate']:.2%}")
    
    return evaluation_report

def multiple_stocks_example():
    """
    Example: Predict returns for multiple stocks
    """
    print("\n📊 Multiple Stocks Prediction Example")
    print("=" * 50)
    
    # Initialize components
    collector = DataCollector()
    fe = FeatureEngineer()
    predictor = AdvancedStockPredictor(model_type='tree', task='regression')
    
    # Load data for multiple stocks
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    print(f"Loading data for: {', '.join(symbols)}")
    
    data = collector.create_comprehensive_dataset(symbols, include_market_data=True)
    
    if data.empty:
        print("❌ No data loaded.")
        return
    
    print(f"✅ Data loaded! Shape: {data.shape}")
    
    # Engineer features
    engineered_data = fe.engineer_all_features(data, target_horizons=[1])
    
    # Train model for each stock
    results = {}
    
    for symbol in symbols:
        print(f"\nTraining model for {symbol}...")
        
        # Filter data for this symbol
        symbol_data = engineered_data[engineered_data['Stock_Symbol'] == symbol]
        
        if symbol_data.empty:
            print(f"❌ No data for {symbol}")
            continue
        
        # Prepare data
        target_column = 'Target_Return_1d'
        symbol_data_clean = symbol_data.dropna(subset=[target_column])
        
        if len(symbol_data_clean) < 100:
            print(f"❌ Insufficient data for {symbol}")
            continue
        
        # Select features
        feature_columns = fe.select_features(
            symbol_data_clean, 
            target_column, 
            method='correlation', 
            threshold=0.001
        )
        
        if not feature_columns:
            feature_columns = [col for col in symbol_data_clean.columns 
                              if col not in ['Symbol', 'Stock_Symbol'] and 
                              symbol_data_clean[col].dtype in ['float64', 'int64'] and
                              'Target_' not in col]
        
        X = symbol_data_clean[feature_columns]
        y = symbol_data_clean[target_column]
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train model
        try:
            model_results = predictor.train_models(X_train, y_train, X_val, y_val)
            results[symbol] = model_results
            print(f"✅ {symbol}: RMSE={list(model_results.values())[0].get('rmse', 0):.4f}")
        except Exception as e:
            print(f"❌ Error training {symbol}: {str(e)}")
    
    return results

def classification_example():
    """
    Example: Binary classification (up/down prediction)
    """
    print("\n🎯 Classification Example (Up/Down Prediction)")
    print("=" * 50)
    
    # Initialize components for classification
    collector = DataCollector()
    fe = FeatureEngineer()
    predictor = AdvancedStockPredictor(model_type='ensemble', task='classification')
    evaluator = ModelEvaluator()
    
    # Load data
    symbols = ['AAPL']
    data = collector.create_comprehensive_dataset(symbols)
    
    if data.empty:
        print("❌ No data loaded.")
        return
    
    # Engineer features with binary targets
    engineered_data = fe.engineer_all_features(data, target_horizons=[1])
    
    # Use binary target
    target_column = 'Target_Binary_1d'
    
    if target_column not in engineered_data.columns:
        print("❌ Binary target not found. Creating it...")
        # Create binary target from returns
        returns_col = 'Target_Return_1d'
        if returns_col in engineered_data.columns:
            engineered_data[target_column] = (engineered_data[returns_col] > 0).astype(int)
        else:
            print("❌ No return target found.")
            return
    
    # Prepare data
    data_clean = engineered_data.dropna(subset=[target_column])
    
    # Select features
    feature_columns = fe.select_features(
        data_clean, 
        target_column, 
        method='correlation', 
        threshold=0.001
    )
    
    if not feature_columns:
        feature_columns = [col for col in data_clean.columns 
                          if col not in ['Symbol', 'Stock_Symbol'] and 
                          data_clean[col].dtype in ['float64', 'int64'] and
                          'Target_' not in col]
    
    X = data_clean[feature_columns]
    y = data_clean[target_column]
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"✅ Data prepared - Train: {len(X_train)}, Validation: {len(X_val)}")
    print(f"   Class distribution: {y.value_counts().to_dict()}")
    
    # Train model
    results = predictor.train_models(X_train, y_train, X_val, y_val)
    
    print("✅ Classification model trained!")
    for model_name, metrics in results.items():
        print(f"   {model_name}: Accuracy={metrics.get('accuracy', 0):.4f}")
    
    # Make predictions
    predictions = predictor.ensemble_predict(X_val)
    
    # Evaluate
    classification_metrics = evaluator.calculate_classification_metrics(y_val, predictions)
    
    print("Classification metrics:")
    for metric, value in classification_metrics.items():
        if isinstance(value, float):
            print(f"   {metric}: {value:.4f}")
    
    return classification_metrics

def main():
    """
    Run all examples
    """
    print("📈 Stock Market Prediction Tool - Basic Examples")
    print("=" * 60)
    
    try:
        # Run basic prediction example
        basic_results = basic_prediction_example()
        
        # Run multiple stocks example
        multiple_results = multiple_stocks_example()
        
        # Run classification example
        classification_results = classification_example()
        
        print("\n🎉 All examples completed successfully!")
        print("\nSummary:")
        print("- Basic prediction: ✅")
        print("- Multiple stocks: ✅")
        print("- Classification: ✅")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    main() 