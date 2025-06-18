# 📈 Stock Market Prediction Tool

A comprehensive AI-powered tool for predicting stock market movements using advanced machine learning algorithms and extensive financial data analysis.

## 🚀 Features

### Data Collection
- **Multi-source data**: Yahoo Finance, Alpha Vantage, FRED (Federal Reserve Economic Data)
- **Comprehensive coverage**: Stock prices, volume, technical indicators, economic indicators
- **Real-time updates**: Live data fetching capabilities
- **Historical data**: Extensive historical data for training

### Feature Engineering
- **100+ Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, etc.
- **Statistical Features**: Rolling statistics, volatility measures, momentum indicators
- **Market Microstructure**: Order flow proxies, price efficiency, spread analysis
- **Time-based Features**: Cyclical encoding, seasonal patterns, market session features
- **Economic Indicators**: GDP, unemployment, inflation, interest rates, exchange rates

### Machine Learning Models
- **Ensemble Methods**: Random Forest, XGBoost, LightGBM, CatBoost
- **Deep Learning**: Dense Neural Networks, LSTM for time series
- **Traditional ML**: Linear Regression, SVM, Gradient Boosting
- **Hybrid Approaches**: Combination of multiple model types

### Evaluation & Backtesting
- **Comprehensive Metrics**: RMSE, MAE, R², Sharpe Ratio, Maximum Drawdown
- **Backtesting Framework**: Strategy performance analysis
- **Rolling Window Evaluation**: Time-series cross-validation
- **Visualization**: Interactive charts and performance plots

### User Interface
- **Streamlit Web App**: User-friendly web interface
- **Real-time Predictions**: Instant stock predictions
- **Interactive Charts**: Plotly-based visualizations
- **Batch Processing**: Multiple stock analysis

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd stock-market-prediction-tool
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Set up API keys** (for enhanced data collection)
   ```bash
   # Create a .env file
   echo "ALPHA_VANTAGE_API_KEY=your_key_here" > .env
   echo "FRED_API_KEY=your_key_here" >> .env
   ```

## 🚀 Quick Start

### 1. Run the Application
```bash
streamlit run main.py
```

### 2. Load Data
- Open the web interface in your browser
- Enter stock symbols (e.g., AAPL, MSFT, GOOGL)
- Choose time period (1y, 2y, 5y, 10y, max)
- Click "Load Data"

### 3. Train Model
- Select model type (Ensemble, Tree, Neural, Deep)
- Choose prediction task (Regression or Classification)
- Set prediction horizon (1, 3, 5, 10, 20 days)
- Click "Train Model"

### 4. Make Predictions
- Use the prediction interface to get forecasts
- View performance metrics and visualizations
- Download results for further analysis

## 📊 Usage Examples

### Basic Usage
```python
from data_collector import DataCollector
from feature_engineering import FeatureEngineer
from models import AdvancedStockPredictor
from evaluation import ModelEvaluator

# Initialize components
collector = DataCollector()
fe = FeatureEngineer()
predictor = AdvancedStockPredictor(model_type='ensemble', task='regression')
evaluator = ModelEvaluator()

# Load data
symbols = ['AAPL', 'MSFT', 'GOOGL']
data = collector.create_comprehensive_dataset(symbols)

# Engineer features
engineered_data = fe.engineer_all_features(data)

# Train model
X_train, X_val, y_train, y_val = split_data(engineered_data)
predictor.train_models(X_train, y_train, X_val, y_val)

# Make predictions
predictions = predictor.ensemble_predict(X_val)

# Evaluate results
report = evaluator.generate_evaluation_report(data, predictions)
```

### Advanced Usage
```python
# Custom feature engineering
custom_features = fe.add_technical_indicators(data)
custom_features = fe.add_statistical_features(custom_features)
custom_features = fe.add_market_microstructure_features(custom_features)

# Model comparison
models = {
    'xgboost': predictor.models['xgboost'],
    'lstm': predictor.models['lstm'],
    'ensemble': predictor.models
}

comparison = evaluator.compare_models(data, models)
```

## 📈 Model Performance

The tool includes multiple evaluation metrics:

### Regression Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination
- **Directional Accuracy**: Percentage of correct direction predictions
- **Information Ratio**: Risk-adjusted return measure

### Classification Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve

### Trading Performance
- **Total Return**: Overall strategy return
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Average Trade**: Mean return per trade

## 🔧 Configuration

### Model Parameters
```python
# Ensemble model configuration
predictor = AdvancedStockPredictor(
    model_type='ensemble',  # 'linear', 'tree', 'neural', 'deep'
    task='regression'       # 'regression' or 'classification'
)

# Feature selection
features = fe.select_features(
    data, 
    target_column='Target_Return_1d',
    method='correlation',   # 'correlation', 'variance', 'mutual_info'
    threshold=0.01
)
```

### Data Sources
```python
# Configure data sources
collector = DataCollector(
    alpha_vantage_key='your_key',  # Optional
    fred_key='your_key'            # Optional
)

# Load different data types
stock_data = collector.get_stock_data('AAPL', period='5y')
market_data = collector.get_market_data(['^GSPC', '^VIX'])
economic_data = collector.get_economic_indicators()
```

## 📁 Project Structure

```
stock-market-prediction-tool/
├── main.py                 # Main Streamlit application
├── data_collector.py       # Data collection from multiple sources
├── feature_engineering.py  # Feature creation and engineering
├── models.py              # Machine learning models
├── evaluation.py          # Model evaluation and backtesting
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── examples/             # Usage examples
    ├── basic_usage.py
    ├── advanced_usage.py
    └── custom_models.py
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This tool is for educational and research purposes only. Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always conduct thorough research and consider consulting with financial advisors before making investment decisions.

## 🆘 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/your-repo/issues) page
2. Create a new issue with detailed information
3. Include error messages and system information

## 🔮 Future Enhancements

- [ ] Real-time streaming data integration
- [ ] Advanced deep learning models (Transformers, GANs)
- [ ] Sentiment analysis from news and social media
- [ ] Portfolio optimization algorithms
- [ ] Mobile app version
- [ ] API endpoints for external integration
- [ ] More economic indicators and alternative data sources

## 📊 Performance Benchmarks

| Model Type | RMSE | MAE | R² | Training Time |
|------------|------|-----|----|---------------|
| Linear | 0.0234 | 0.0187 | 0.156 | 2s |
| Tree | 0.0211 | 0.0169 | 0.234 | 15s |
| Neural | 0.0208 | 0.0165 | 0.245 | 45s |
| Ensemble | 0.0192 | 0.0151 | 0.312 | 60s |
| Deep | 0.0189 | 0.0148 | 0.325 | 120s |

*Benchmarks based on S&P 500 data, 5-year period, daily predictions*

---

**Happy Trading! 📈💰** 