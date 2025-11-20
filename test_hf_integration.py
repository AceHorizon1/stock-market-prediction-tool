"""
Test script for Hugging Face Transformer integration
Demonstrates how to use HF transformer models for stock market prediction
"""

import pandas as pd
import numpy as np
from data_collector import DataCollector
from feature_engineering import FeatureEngineer
from hf_transformer_models import HFTransformerPredictor, HFTransformerStockPredictor
import warnings
warnings.filterwarnings('ignore')

def test_hf_transformer_integration():
    """Test the Hugging Face transformer integration"""
    
    print("="*80)
    print("Testing Hugging Face Transformer Integration")
    print("="*80)
    
    # Step 1: Collect data
    print("\n📊 Step 1: Collecting stock data...")
    collector = DataCollector()
    data = collector.fetch_stock_data("AAPL", period="1y")
    
    if data is None or data.empty:
        print("❌ Failed to collect data")
        return
    
    print(f"✅ Collected {len(data)} days of data")
    print(f"   Date range: {data.index.min()} to {data.index.max()}")
    
    # Step 2: Prepare data (use Close price as target)
    print("\n🔧 Step 2: Preparing data for transformer model...")
    
    # Use a subset of features for transformer
    feature_cols = ['Open', 'High', 'Low', 'Volume', 'Close']
    prepared_data = data[feature_cols].copy()
    prepared_data = prepared_data.dropna()
    
    if len(prepared_data) < 200:
        print(f"⚠️ Warning: Only {len(prepared_data)} samples available. Need at least 200 for good results.")
    
    # Step 3: Initialize and train transformer model
    print("\n🤖 Step 3: Initializing Hugging Face Transformer model...")
    
    try:
        # Use PatchTST (best for time series)
        predictor = HFTransformerPredictor(
            model_name="PatchTST",
            prediction_length=1,  # Predict 1 day ahead
            context_length=96,    # Use 96 days of history
        )
        
        print("\n🚀 Step 4: Training model (this may take a few minutes)...")
        print("   Note: Training with fewer epochs for demonstration")
        
        # Train with limited epochs for quick testing
        training_results = predictor.train(
            data=prepared_data,
            target_column="Close",
            feature_columns=['Open', 'High', 'Low', 'Volume'],
            epochs=5,  # Reduced for quick testing
            batch_size=16,
            learning_rate=1e-4
        )
        
        print(f"\n✅ Training completed!")
        print(f"   Train Loss: {training_results['train_loss']:.4f}")
        print(f"   Validation Loss: {training_results['eval_loss']:.4f}")
        
        # Step 5: Make predictions
        print("\n🔮 Step 5: Making predictions...")
        
        # Use last portion of data for prediction
        test_data = prepared_data.tail(150)
        predictions = predictor.predict(
            data=test_data,
            target_column="Close",
            feature_columns=['Open', 'High', 'Low', 'Volume']
        )
        
        print(f"✅ Generated {len(predictions)} predictions")
        
        # Compare with actual values
        actual = test_data['Close'].values[-len(predictions):]
        
        if len(actual) > 0 and len(predictions) > 0:
            mae = np.mean(np.abs(actual - predictions))
            mse = np.mean((actual - predictions) ** 2)
            rmse = np.sqrt(mse)
            
            print(f"\n📈 Prediction Metrics:")
            print(f"   MAE: ${mae:.2f}")
            print(f"   RMSE: ${rmse:.2f}")
            print(f"   Actual Close: ${actual[-1]:.2f}")
            print(f"   Predicted Close: ${predictions[-1]:.2f}")
            print(f"   Error: ${abs(actual[-1] - predictions[-1]):.2f}")
        
        # Step 6: Save model
        print("\n💾 Step 6: Saving model...")
        predictor.save_model("./models/hf_patchtst_aapl")
        print("✅ Model saved!")
        
    except Exception as e:
        print(f"\n❌ Error during training/prediction: {e}")
        print("\n💡 Tips:")
        print("   1. Make sure transformers and torch are installed:")
        print("      pip install transformers torch")
        print("   2. Ensure you have enough data (at least 200 samples)")
        print("   3. Try reducing context_length if you have less data")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*80)
    print("✅ Integration test completed successfully!")
    print("="*80)
    print("\n📝 Next steps:")
    print("   1. Experiment with different model types (Autoformer, TimeSeriesTransformer)")
    print("   2. Try different context_length and prediction_length values")
    print("   3. Integrate into your existing models.py workflow")
    print("   4. Use pretrained models from Hugging Face Hub")


def test_with_existing_interface():
    """Test using the wrapper class that's compatible with existing code"""
    
    print("\n" + "="*80)
    print("Testing Compatibility with Existing Interface")
    print("="*80)
    
    # Collect data
    collector = DataCollector()
    data = collector.fetch_stock_data("MSFT", period="1y")
    
    if data is None or data.empty:
        print("❌ Failed to collect data")
        return
    
    # Prepare data with features
    feature_engineer = FeatureEngineer()
    engineered_data = feature_engineer.engineer_all_features(data, target_horizons=[1])
    
    if engineered_data.empty:
        print("❌ Feature engineering failed")
        return
    
    # Use wrapper class
    print("\n🤖 Using HFTransformerStockPredictor (compatible interface)...")
    
    try:
        predictor = HFTransformerStockPredictor(
            model_name="PatchTST",
            prediction_length=1,
            context_length=96
        )
        
        # Train
        print("Training model...")
        results = predictor.train_model(
            data=engineered_data,
            target_column="Target_1",  # 1-day prediction target
            epochs=3  # Quick test
        )
        
        print(f"✅ Training completed: {results}")
        
        # Predict
        test_data = engineered_data.tail(100)
        predictions = predictor.predict(test_data)
        
        print(f"✅ Generated predictions: {len(predictions)} values")
        print(f"   Sample predictions: {predictions[:5]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n🧪 Hugging Face Transformer Integration Test")
    print("="*80)
    
    # Test 1: Basic integration
    test_hf_transformer_integration()
    
    # Test 2: Compatibility test (optional, may take longer)
    # Uncomment to test:
    # test_with_existing_interface()
    
    print("\n✨ All tests completed!")

