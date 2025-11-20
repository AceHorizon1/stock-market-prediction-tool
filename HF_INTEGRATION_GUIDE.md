# Hugging Face Transformer Integration Guide

## Overview

This guide explains how to use Hugging Face transformer models for stock market prediction in the Stock Market AI project. The integration adds state-of-the-art transformer architectures specifically designed for time series forecasting.

## Available Models

The integration supports three transformer models from Hugging Face:

1. **PatchTST** (Recommended) - Patch-based Time Series Transformer
   - Best for long-term forecasting
   - Efficient patch-based architecture
   - State-of-the-art performance on time series

2. **Autoformer** - Decomposition Transformers with Auto-Correlation
   - Good for capturing seasonal patterns
   - Auto-correlation mechanism for dependencies

3. **TimeSeriesTransformer** - Standard transformer for time series
   - General-purpose transformer architecture
   - Good baseline model

## Installation

### Step 1: Install Dependencies

```bash
pip install transformers torch accelerate
```

Or update your requirements:

```bash
pip install -r requirements.txt
```

The requirements.txt has been updated to include:
- `transformers>=4.35.0`
- `torch>=2.0.0`
- `accelerate>=0.24.0`

### Step 2: Verify Installation

```python
from transformers import PatchTSTForPrediction
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Usage

### Basic Usage

```python
from hf_transformer_models import HFTransformerPredictor
from data_collector import DataCollector

# Collect data
collector = DataCollector()
data = collector.fetch_stock_data("AAPL", period="1y")

# Initialize predictor
predictor = HFTransformerPredictor(
    model_name="PatchTST",
    prediction_length=1,  # Predict 1 day ahead
    context_length=96,    # Use 96 days of history
)

# Train model
results = predictor.train(
    data=data,
    target_column="Close",
    feature_columns=['Open', 'High', 'Low', 'Volume'],
    epochs=10,
    batch_size=32
)

# Make predictions
predictions = predictor.predict(
    data=data.tail(100),
    target_column="Close"
)
```

### Integration with Existing Code

The HF transformer can be used with the existing `StockPredictor` interface:

```python
from models import AdvancedStockPredictor

# Use 'hf_transformer' as model_type
predictor = AdvancedStockPredictor(
    model_type='hf_transformer',
    task='regression'
)

# Train using existing interface
results = predictor.train_model(
    data=engineered_data,
    target_column="Target_1",  # 1-day prediction
    model_type='hf_transformer',
    task='regression'
)

# Make predictions
predictions = predictor.predict(X_test, model_name='hf_transformer')
```

### Using Pretrained Models

You can use pretrained models from Hugging Face Hub:

```python
predictor = HFTransformerPredictor(
    model_name="PatchTST",
    model_id="ibm/PatchesTST-finetuned-exchange_rate",  # Example pretrained model
    prediction_length=1,
    context_length=96
)
```

Note: Most pretrained models are for different domains. You'll typically need to fine-tune on stock data.

## Parameters

### HFTransformerPredictor Parameters

- **model_name**: `'PatchTST'`, `'Autoformer'`, or `'TimeSeriesTransformer'`
- **model_id**: Optional Hugging Face model ID for pretrained models
- **prediction_length**: Number of time steps to predict (default: 1)
- **context_length**: Number of historical time steps to use (default: 96)
- **num_parallel_samples**: Number of parallel samples for probabilistic forecasting (default: 100)
- **device**: `'cuda'` or `'cpu'` (auto-detected if None)

### Training Parameters

- **epochs**: Number of training epochs (default: 10)
- **batch_size**: Batch size for training (default: 32)
- **learning_rate**: Learning rate (default: 1e-4)
- **train_split**: Train/validation split ratio (default: 0.8)

## Data Requirements

### Minimum Data Requirements

- **Minimum samples**: `context_length + prediction_length + 50` (recommended: 200+)
- **Features**: At least 1 numeric feature (recommended: 4-10 features)
- **Data quality**: No missing values in selected features

### Recommended Data

- **Samples**: 500+ for good performance
- **Features**: 4-10 relevant features (Open, High, Low, Volume, technical indicators)
- **Time period**: At least 1 year of daily data

## Example: Complete Workflow

```python
import pandas as pd
from data_collector import DataCollector
from feature_engineering import FeatureEngineer
from hf_transformer_models import HFTransformerPredictor

# 1. Collect data
collector = DataCollector()
raw_data = collector.fetch_stock_data("AAPL", period="2y")

# 2. Engineer features (optional, can use raw OHLCV)
feature_engineer = FeatureEngineer()
engineered_data = feature_engineer.engineer_all_features(raw_data)

# 3. Select features for transformer
feature_cols = ['Open', 'High', 'Low', 'Volume', 'RSI', 'MACD']
prepared_data = engineered_data[feature_cols + ['Close']].dropna()

# 4. Initialize and train
predictor = HFTransformerPredictor(
    model_name="PatchTST",
    prediction_length=1,
    context_length=96
)

training_results = predictor.train(
    data=prepared_data,
    target_column="Close",
    feature_columns=feature_cols,
    epochs=20,
    batch_size=32
)

# 5. Make predictions
test_data = prepared_data.tail(150)
predictions = predictor.predict(
    data=test_data,
    target_column="Close",
    feature_columns=feature_cols
)

# 6. Evaluate
actual = test_data['Close'].values[-len(predictions):]
mae = np.mean(np.abs(actual - predictions))
print(f"MAE: ${mae:.2f}")

# 7. Save model
predictor.save_model("./models/hf_patchtst_aapl")
```

## Performance Tips

### 1. Data Preparation
- Use relevant features (price, volume, technical indicators)
- Ensure sufficient data (200+ samples minimum)
- Handle missing values before training

### 2. Model Configuration
- **context_length**: Longer context (96-192) for better long-term patterns
- **prediction_length**: Start with 1, increase for multi-step forecasting
- **batch_size**: Adjust based on available memory (16-64)

### 3. Training
- Start with fewer epochs (5-10) for testing
- Use learning rate 1e-4 to 1e-3
- Monitor validation loss to avoid overfitting

### 4. Hardware
- GPU recommended for faster training (CUDA)
- CPU works but will be slower
- 8GB+ RAM recommended

## Comparison with Existing Models

| Model Type | Pros | Cons | Best For |
|-----------|------|------|----------|
| **HF Transformer** | State-of-the-art, long-term forecasting, captures complex patterns | Requires more data, slower training, more memory | Long-term predictions, complex patterns |
| **LSTM** | Good for sequences, moderate complexity | May struggle with very long sequences | Medium-term predictions |
| **Ensemble (XGBoost, etc.)** | Fast training, interpretable, good for short-term | Limited long-term forecasting | Short-term predictions, feature importance |
| **Linear Models** | Very fast, interpretable | Limited pattern capture | Baseline, quick predictions |

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce `batch_size` (try 8 or 16)
   - Reduce `context_length` (try 48 or 64)
   - Use fewer features

2. **Not Enough Data Error**
   - Collect more historical data
   - Reduce `context_length`
   - Use daily data instead of intraday

3. **Slow Training**
   - Use GPU if available
   - Reduce `epochs` for testing
   - Reduce `context_length`

4. **Poor Predictions**
   - Ensure sufficient training data (500+ samples)
   - Try different features
   - Increase training epochs
   - Adjust learning rate

### Getting Help

- Check the test script: `python test_hf_integration.py`
- Review Hugging Face documentation: https://huggingface.co/docs/transformers
- Check model-specific documentation in `hf_transformer_models.py`

## Advanced Usage

### Custom Model Configuration

```python
from transformers import PatchTSTConfig, PatchTSTForPrediction

config = PatchTSTConfig(
    prediction_length=5,
    context_length=192,
    num_input_channels=10,
    patch_len=16,
    stride=8,
    d_model=256,  # Larger model
    n_heads=16,
    num_layers=6,
    dropout=0.1
)

model = PatchTSTForPrediction(config)
```

### Multi-Step Forecasting

```python
predictor = HFTransformerPredictor(
    model_name="PatchTST",
    prediction_length=5,  # Predict 5 days ahead
    context_length=96
)
```

### Ensemble with Other Models

```python
from models import AdvancedStockPredictor

# Train multiple models
predictor = AdvancedStockPredictor(model_type='ensemble')
results = predictor.train_models(X_train, y_train, X_val, y_val)

# Add HF transformer
hf_predictor = HFTransformerStockPredictor()
hf_results = hf_predictor.train_model(data, target_column="Close")

# Combine predictions
ensemble_pred = (predictor.ensemble_predict(X_test) + 
                 hf_predictor.predict(X_test)) / 2
```

## Files Created

1. **hf_transformer_models.py** - Main integration module
2. **test_hf_integration.py** - Test and example script
3. **HF_INTEGRATION_GUIDE.md** - This guide
4. **requirements.txt** - Updated with transformers dependencies

## Next Steps

1. **Run the test script**: `python test_hf_integration.py`
2. **Experiment with different models**: Try Autoformer and TimeSeriesTransformer
3. **Fine-tune hyperparameters**: Adjust context_length, prediction_length, etc.
4. **Compare with existing models**: Benchmark against LSTM and ensemble methods
5. **Use pretrained models**: Explore Hugging Face Hub for relevant pretrained models

## References

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [PatchTST Paper](https://huggingface.co/papers/2211.14730)
- [Autoformer Paper](https://huggingface.co/papers/2106.13008)
- [Time Series Forecasting with Transformers](https://huggingface.co/blog/time-series-transformers)

---

**Created**: November 19, 2025  
**Last Updated**: November 19, 2025

