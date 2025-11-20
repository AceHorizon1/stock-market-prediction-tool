"""
Full Integration Test for Hugging Face Transformer in Stock Market AI
Tests the complete workflow: data collection -> feature engineering -> HF transformer training -> prediction
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported"""
    print("="*80)
    print("Testing Imports")
    print("="*80)
    
    try:
        import pandas as pd
        print("✅ pandas imported")
    except ImportError as e:
        print(f"❌ pandas not available: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported")
    except ImportError as e:
        print(f"❌ numpy not available: {e}")
        return False
    
    try:
        from data_collector import DataCollector
        print("✅ DataCollector imported")
    except ImportError as e:
        print(f"❌ DataCollector not available: {e}")
        return False
    
    try:
        from feature_engineering import FeatureEngineer
        print("✅ FeatureEngineer imported")
    except ImportError as e:
        print(f"❌ FeatureEngineer not available: {e}")
        return False
    
    try:
        from models import AdvancedStockPredictor
        print("✅ AdvancedStockPredictor imported")
    except ImportError as e:
        print(f"❌ AdvancedStockPredictor not available: {e}")
        return False
    
    # Test HF transformer import
    try:
        import torch
        print(f"✅ torch imported (version: {torch.__version__})")
    except ImportError:
        print("⚠️  torch not available - HF transformer will not work")
        print("   Install with: pip install torch")
        return False
    
    try:
        from transformers import PatchTSTForPrediction
        print("✅ transformers imported")
    except ImportError:
        print("⚠️  transformers not available - HF transformer will not work")
        print("   Install with: pip install transformers")
        return False
    
    try:
        from hf_transformer_models import HFTransformerPredictor, HFTransformerStockPredictor
        print("✅ HF transformer models imported")
    except ImportError as e:
        print(f"⚠️  HF transformer models not available: {e}")
        return False
    
    return True


def test_data_collection():
    """Test data collection"""
    print("\n" + "="*80)
    print("Testing Data Collection")
    print("="*80)
    
    try:
        from data_collector import DataCollector
        
        collector = DataCollector()
        print("🔄 Collecting stock data for AAPL...")
        data = collector.fetch_stock_data("AAPL", period="1y")
        
        if data is None or data.empty:
            print("❌ Failed to collect data")
            return None
        
        print(f"✅ Collected {len(data)} days of data")
        print(f"   Columns: {list(data.columns)}")
        print(f"   Date range: {data.index.min()} to {data.index.max()}")
        return data
        
    except Exception as e:
        print(f"❌ Error during data collection: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_feature_engineering(data):
    """Test feature engineering"""
    print("\n" + "="*80)
    print("Testing Feature Engineering")
    print("="*80)
    
    if data is None or data.empty:
        print("❌ No data available for feature engineering")
        return None
    
    try:
        from feature_engineering import FeatureEngineer
        
        engineer = FeatureEngineer()
        print("🔄 Engineering features...")
        engineered_data = engineer.engineer_all_features(data, target_horizons=[1, 3, 5])
        
        if engineered_data is None or engineered_data.empty:
            print("❌ Feature engineering produced no data")
            return None
        
        print(f"✅ Feature engineering complete")
        print(f"   Original shape: {data.shape}")
        print(f"   Engineered shape: {engineered_data.shape}")
        print(f"   Features created: {len(engineered_data.columns)}")
        
        return engineered_data
        
    except Exception as e:
        print(f"❌ Error during feature engineering: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_hf_transformer_basic(data):
    """Test HF transformer with basic data"""
    print("\n" + "="*80)
    print("Testing Hugging Face Transformer (Basic)")
    print("="*80)
    
    if data is None or data.empty:
        print("❌ No data available")
        return False
    
    try:
        from hf_transformer_models import HFTransformerPredictor
        
        # Use a subset of data for quick testing
        test_data = data[['Open', 'High', 'Low', 'Volume', 'Close']].copy()
        test_data = test_data.dropna()
        
        if len(test_data) < 200:
            print(f"⚠️  Only {len(test_data)} samples available. Need at least 200 for good results.")
            print("   Continuing with available data...")
        
        print("🔄 Initializing HF Transformer (PatchTST)...")
        predictor = HFTransformerPredictor(
            model_name="PatchTST",
            prediction_length=1,
            context_length=min(96, len(test_data) // 3)  # Adjust context length
        )
        
        print("✅ Model initialized")
        print(f"   Model type: {predictor.model_name}")
        print(f"   Device: {predictor.device}")
        print(f"   Context length: {predictor.context_length}")
        print(f"   Prediction length: {predictor.prediction_length}")
        
        # Test data preparation
        print("\n🔄 Testing data preparation...")
        try:
            past_values, future_values, feature_cols = predictor.prepare_data(
                test_data,
                target_column="Close",
                feature_columns=['Open', 'High', 'Low', 'Volume']
            )
            print(f"✅ Data prepared successfully")
            print(f"   Past values shape: {past_values.shape}")
            print(f"   Future values shape: {future_values.shape}")
            print(f"   Features used: {feature_cols}")
            
            # Quick training test (just 2 epochs for testing)
            print("\n🔄 Testing training (2 epochs for quick test)...")
            results = predictor.train(
                data=test_data,
                target_column="Close",
                feature_columns=['Open', 'High', 'Low', 'Volume'],
                epochs=2,
                batch_size=16
            )
            
            print(f"✅ Training completed")
            print(f"   Train loss: {results.get('train_loss', 'N/A'):.4f}")
            print(f"   Validation loss: {results.get('eval_loss', 'N/A'):.4f}")
            
            # Test prediction
            print("\n🔄 Testing prediction...")
            predictions = predictor.predict(
                data=test_data.tail(100),
                target_column="Close",
                feature_columns=['Open', 'High', 'Low', 'Volume']
            )
            
            print(f"✅ Predictions generated")
            print(f"   Predictions shape: {predictions.shape}")
            print(f"   Sample predictions: {predictions[:5]}")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Error during training/prediction: {e}")
            print("   This might be due to insufficient data or model configuration")
            import traceback
            traceback.print_exc()
            return False
        
    except ImportError as e:
        print(f"❌ HF transformer not available: {e}")
        print("   Install with: pip install transformers torch")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_with_existing_code(engineered_data):
    """Test integration with existing AdvancedStockPredictor"""
    print("\n" + "="*80)
    print("Testing Integration with Existing Code")
    print("="*80)
    
    if engineered_data is None or engineered_data.empty:
        print("❌ No engineered data available")
        return False
    
    try:
        from models import AdvancedStockPredictor
        
        # Check if HF transformer is available
        try:
            from models import HF_AVAILABLE
            if not HF_AVAILABLE:
                print("⚠️  HF transformer not available in models.py")
                print("   This is expected if transformers is not installed")
                return False
        except:
            print("⚠️  Could not check HF availability")
            return False
        
        # Prepare data
        target_column = "Target_1"
        if target_column not in engineered_data.columns:
            # Try alternative target
            target_cols = [col for col in engineered_data.columns if 'Target' in col]
            if target_cols:
                target_column = target_cols[0]
                print(f"⚠️  Using alternative target: {target_column}")
            else:
                print("❌ No target columns found")
                return False
        
        # Clean data
        data_clean = engineered_data.dropna(subset=[target_column])
        
        if len(data_clean) < 200:
            print(f"⚠️  Only {len(data_clean)} samples. Need 200+ for HF transformer")
            return False
        
        print(f"🔄 Testing with AdvancedStockPredictor...")
        print(f"   Target: {target_column}")
        print(f"   Data shape: {data_clean.shape}")
        
        predictor = AdvancedStockPredictor(
            model_type='hf_transformer',
            task='regression'
        )
        
        print("🔄 Training model (2 epochs for quick test)...")
        # Note: This will use the existing train_model method
        # The HF transformer training might take longer
        results = predictor.train_model(
            data=data_clean,
            target_column=target_column,
            model_type='hf_transformer',
            task='regression'
        )
        
        print("✅ Integration test completed")
        if 'metrics' in results:
            print("   Metrics:")
            for metric, value in results['metrics'].items():
                print(f"     {metric}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during integration test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("FULL INTEGRATION TEST - Hugging Face Transformer")
    print("="*80)
    print("\nThis test verifies the complete integration of HF transformers")
    print("into the Stock Market AI tool.\n")
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import test failed. Please install missing dependencies.")
        print("\nInstallation commands:")
        print("  pip install pandas numpy")
        print("  pip install transformers torch")
        return
    
    # Test 2: Data collection
    data = test_data_collection()
    if data is None:
        print("\n⚠️  Data collection failed. Some tests will be skipped.")
    
    # Test 3: Feature engineering
    engineered_data = None
    if data is not None:
        engineered_data = test_feature_engineering(data)
    
    # Test 4: HF transformer basic
    hf_success = False
    if data is not None:
        hf_success = test_hf_transformer_basic(data)
    
    # Test 5: Integration with existing code
    integration_success = False
    if engineered_data is not None:
        integration_success = test_integration_with_existing_code(engineered_data)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"✅ Imports: Passed")
    print(f"{'✅' if data is not None else '❌'} Data Collection: {'Passed' if data is not None else 'Failed'}")
    print(f"{'✅' if engineered_data is not None else '❌'} Feature Engineering: {'Passed' if engineered_data is not None else 'Failed'}")
    print(f"{'✅' if hf_success else '⚠️ '} HF Transformer Basic: {'Passed' if hf_success else 'Skipped/Failed'}")
    print(f"{'✅' if integration_success else '⚠️ '} Integration Test: {'Passed' if integration_success else 'Skipped/Failed'}")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    if hf_success:
        print("✅ HF Transformer integration is working!")
        print("\nYou can now:")
        print("  1. Use 'hf_transformer' as model_type in the GUI or Streamlit app")
        print("  2. Train models using AdvancedStockPredictor with model_type='hf_transformer'")
        print("  3. Compare HF transformer performance with other models")
    else:
        print("⚠️  Some tests failed or were skipped.")
        print("\nTo enable full functionality:")
        print("  1. Install dependencies: pip install transformers torch")
        print("  2. Ensure you have sufficient data (200+ samples)")
        print("  3. Run the test again")
    
    print("\n✨ Test completed!")


if __name__ == "__main__":
    main()

