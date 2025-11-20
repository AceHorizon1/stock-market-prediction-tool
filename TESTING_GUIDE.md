# Testing Guide - Hugging Face Transformer Integration

## Quick Test

Run the comprehensive integration test:

```bash
cd "Stock Market AI"
python test_full_integration.py
```

This will test:
1. ✅ All imports (pandas, numpy, transformers, torch)
2. ✅ Data collection from Yahoo Finance
3. ✅ Feature engineering
4. ✅ HF Transformer initialization and training
5. ✅ Integration with existing code

## Testing in GUI Application

1. **Start the GUI**:
   ```bash
   python app_gui.py
   ```

2. **Workflow**:
   - Load Data tab: Fetch stock data (e.g., AAPL, MSFT)
   - Feature Engineering tab: Click "Engineer Features"
   - Model Training tab: 
     - Select Model Type: **hf_transformer**
     - Select Prediction Task: regression
     - Select Target Horizon: 1 day
     - Click "Train Model"

3. **Expected Behavior**:
   - Training will take longer than other models (especially first time)
   - Progress will be shown in the training info box
   - Results will be available in the Results tab

## Testing in Streamlit App

1. **Start the Streamlit app**:
   ```bash
   streamlit run main.py
   ```

2. **Workflow**:
   - Sidebar: Enter stock symbols (e.g., AAPL)
   - Click "Load Data"
   - Sidebar: Select Model Type: **hf_transformer**
   - Sidebar: Select Prediction Task: regression
   - Sidebar: Select Target Horizon: 1 day
   - Click "Train Model"

3. **Expected Behavior**:
   - Info message about HF transformer
   - Training progress shown
   - Results displayed after training

## Testing via Python Script

### Basic Test
```python
from data_collector import DataCollector
from feature_engineering import FeatureEngineer
from models import AdvancedStockPredictor

# Collect data
collector = DataCollector()
data = collector.fetch_stock_data("AAPL", period="1y")

# Engineer features
engineer = FeatureEngineer()
engineered_data = engineer.engineer_all_features(data, target_horizons=[1])

# Train HF transformer
predictor = AdvancedStockPredictor(
    model_type='hf_transformer',
    task='regression'
)

results = predictor.train_model(
    data=engineered_data,
    target_column="Target_1"
)

print(f"Metrics: {results['metrics']}")
```

### Direct HF Transformer Test
```python
from hf_transformer_models import HFTransformerPredictor
from data_collector import DataCollector

# Collect data
collector = DataCollector()
data = collector.fetch_stock_data("AAPL", period="1y")

# Initialize and train
predictor = HFTransformerPredictor(
    model_name="PatchTST",
    prediction_length=1,
    context_length=96
)

results = predictor.train(
    data=data,
    target_column="Close",
    feature_columns=['Open', 'High', 'Low', 'Volume'],
    epochs=10
)

# Make predictions
predictions = predictor.predict(
    data=data.tail(100),
    target_column="Close"
)
```

## Requirements for Testing

### Minimum Requirements
- Python 3.8+
- pandas, numpy installed
- transformers, torch installed
- At least 200 samples of stock data

### Recommended
- 500+ samples for better performance
- GPU (CUDA) for faster training
- 8GB+ RAM

## Troubleshooting

### Import Errors
```bash
# Install missing dependencies
pip install transformers torch accelerate
pip install pandas numpy scikit-learn
```

### "Not enough data" Error
- Collect more historical data (use longer period like "2y" or "5y")
- Reduce `context_length` parameter
- Use daily data instead of intraday

### Out of Memory
- Reduce `batch_size` (try 8 or 16)
- Reduce `context_length` (try 48 or 64)
- Use fewer features

### Slow Training
- Use GPU if available (CUDA)
- Reduce number of epochs for testing
- Reduce context_length

### Model Not Found in GUI/Streamlit
- Make sure you've updated the files with 'hf_transformer' option
- Restart the application
- Check that transformers is installed

## Expected Results

### Successful Integration
- ✅ Model trains without errors
- ✅ Predictions are generated
- ✅ Metrics are calculated (MAE, RMSE, etc.)
- ✅ Results displayed in GUI/Streamlit

### Performance Expectations
- **Training Time**: 2-10 minutes (depending on data size and hardware)
- **Prediction Time**: < 1 second per prediction
- **Accuracy**: Varies, but should be competitive with LSTM models

## Comparison Testing

Test HF transformer against other models:

```python
from models import AdvancedStockPredictor

models_to_test = ['ensemble', 'deep', 'hf_transformer']

for model_type in models_to_test:
    predictor = AdvancedStockPredictor(model_type=model_type)
    results = predictor.train_model(data, target_column="Target_1")
    print(f"{model_type}: {results['metrics']}")
```

## Next Steps After Testing

1. **Compare Performance**: Benchmark HF transformer vs other models
2. **Tune Hyperparameters**: Adjust context_length, prediction_length, epochs
3. **Try Different Models**: Test Autoformer and TimeSeriesTransformer
4. **Production Use**: Integrate into your trading strategy

---

**Last Updated**: November 19, 2025

