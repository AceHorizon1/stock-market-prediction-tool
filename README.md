# 📈 Stock Market Prediction Tool

A comprehensive AI-powered tool for predicting stock market movements using advanced machine learning algorithms and extensive feature engineering.

## 🚀 Features

### **Data Collection**
- **Multi-source data**: Yahoo Finance, Alpha Vantage, FRED
- **File upload support**: Import CSV/Excel files with OHLCV data
- **Real-time data**: Fetch live stock data for any symbol
- **Multiple stocks**: Analyze multiple stocks simultaneously

### **Advanced Feature Engineering**
- **100+ Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Statistical Features**: Rolling statistics, volatility measures, momentum
- **Market Microstructure**: Volume analysis, price efficiency, money flow
- **Time Features**: Day, month, quarter, year patterns
- **Target Variables**: Multiple prediction horizons (1, 3, 5, 10, 20 days)

### **Machine Learning Models**
- **Ensemble Methods**: Random Forest, Gradient Boosting, XGBoost, LightGBM
- **Deep Learning**: Neural Networks, LSTM for time series
- **Traditional ML**: Linear Regression, Ridge, Lasso, SVM
- **Multiple Tasks**: Regression (price prediction) and Classification (up/down)

### **User Interfaces**
- **Desktop GUI**: PySimpleGUI application with tabs and interactive features
- **Web Interface**: Streamlit web app (alternative option)
- **Command Line**: Direct Python scripts for automation

### **Visualization & Analysis**
- **Interactive Plots**: Actual vs Predicted, Feature Importance, Time Series
- **Performance Metrics**: MSE, MAE, RMSE, Accuracy
- **Results Tables**: Predictions, feature rankings, model comparisons

## 🛠️ Installation

### Prerequisites
- Python 3.10 or higher
- macOS (tested on Apple Silicon M1/M2/M3)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd stock-market-prediction-tool

# Install Homebrew dependencies (macOS)
brew install python@3.10 tcl-tk
brew install python-tk@3.10

# Create virtual environment
python3.10 -m venv venv310
source venv310/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install --extra-index-url https://PySimpleGUI.net/install PySimpleGUI
```

## 🎯 Quick Start

### Desktop GUI (Recommended)
```bash
source venv310/bin/activate
python3 app_gui.py
```

### Web Interface
```bash
source venv310/bin/activate
streamlit run main.py
```

### Command Line
```bash
source venv310/bin/activate
python demo.py
```

## 📱 Using the Desktop GUI

### 1. **Data Loading Tab**
- **Upload CSV/Excel**: Browse and select your OHLCV data file
- **Fetch Online**: Enter stock symbols (e.g., AAPL, MSFT, GOOGL)
- **Data Information**: View shape, date range, and sample data

### 2. **Feature Engineering Tab**
- **Engineer Features**: Creates 100+ technical and statistical indicators
- **Feature Information**: Shows feature categories and counts
- **Sample Data**: Displays first 5 rows of engineered features

### 3. **Model Training Tab**
- **Model Type**: Choose from ensemble, tree, linear, neural, deep
- **Prediction Task**: Regression (price) or Classification (up/down)
- **Target Horizon**: 1, 3, 5, 10, or 20 days
- **Training Progress**: Real-time status and metrics

### 4. **Results Tab**
- **Show Results**: Display model performance metrics
- **Predictions Table**: Actual vs predicted values
- **Generate Plots**: Interactive visualizations

## 📊 Supported Data Formats

### CSV/Excel Files
Your data file should contain these columns:
- `Date` (index): Date in YYYY-MM-DD format
- `Open`: Opening price
- `High`: Highest price
- `Low`: Lowest price
- `Close`: Closing price
- `Volume`: Trading volume

### Stock Symbols
Popular symbols to try:
- **AAPL**: Apple Inc.
- **MSFT**: Microsoft Corporation
- **GOOGL**: Alphabet Inc.
- **AMZN**: Amazon.com Inc.
- **TSLA**: Tesla Inc.

## 🔧 Configuration

### Model Settings
- **Ensemble**: Best overall performance (recommended)
- **Tree**: Good for non-linear patterns
- **Linear**: Fast and interpretable
- **Neural**: Complex pattern recognition
- **Deep**: Advanced deep learning models

### Feature Engineering
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Statistical Features**: Rolling windows, volatility, momentum
- **Time Features**: Calendar effects, seasonal patterns
- **Target Variables**: Multiple prediction horizons

## 📈 Example Workflow

1. **Load Data**: Upload CSV file or fetch AAPL, MSFT, GOOGL
2. **Engineer Features**: Creates 188 features from raw data
3. **Train Model**: Ensemble model for 10-day return prediction
4. **View Results**: Check MSE, MAE, and prediction accuracy
5. **Generate Plots**: Visualize actual vs predicted values

## 🏗️ Project Structure

```
stock-market-prediction-tool/
├── app_gui.py              # PySimpleGUI desktop application
├── main.py                 # Streamlit web application
├── data_collector.py       # Data collection from multiple sources
├── feature_engineering.py  # 100+ feature engineering functions
├── models.py              # Machine learning models and training
├── evaluation.py          # Model evaluation and backtesting
├── demo.py                # Command-line demo script
├── test_app.py            # Testing and validation scripts
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── examples/              # Example usage scripts
```

## 🔍 Troubleshooting

### Common Issues

**Tkinter Error**: 
```bash
brew install python-tk@3.10
```

**PySimpleGUI Import Error**:
```bash
pip install --extra-index-url https://PySimpleGUI.net/install PySimpleGUI
```

**Feature Engineering Empty Data**:
- Ensure your data has sufficient rows (at least 200)
- Check that date index is properly formatted
- Verify OHLCV columns are present

**Model Training Errors**:
- Use numerical data only (excludes Symbol columns automatically)
- Ensure target column exists in engineered data
- Check for sufficient training samples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Yahoo Finance**: Stock data API
- **Alpha Vantage**: Technical indicators and news sentiment
- **FRED**: Economic indicators
- **Scikit-learn**: Machine learning algorithms
- **PySimpleGUI**: Desktop GUI framework
- **Streamlit**: Web application framework

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the example scripts
3. Open an issue on GitHub

---

**Happy Trading! 📈💰** 