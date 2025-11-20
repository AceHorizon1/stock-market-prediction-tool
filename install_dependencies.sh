#!/bin/bash
# Install all dependencies for Stock Market AI

echo "📦 Installing Stock Market AI Dependencies..."
echo ""

# Core dependencies
echo "Installing core dependencies..."
pip3 install pandas numpy scikit-learn scipy --quiet

# Data collection
echo "Installing data collection libraries..."
pip3 install yfinance requests aiohttp --quiet

# Visualization
echo "Installing visualization libraries..."
pip3 install matplotlib seaborn plotly altair --quiet

# Web Interface
echo "Installing web interface..."
pip3 install streamlit dash dash-bootstrap-components --quiet

# Technical Analysis
echo "Installing technical analysis..."
pip3 install ta statsmodels --quiet

# Machine Learning
echo "Installing machine learning libraries..."
pip3 install tensorflow keras xgboost lightgbm catboost --quiet

# Hugging Face Transformers
echo "Installing Hugging Face transformers..."
pip3 install transformers torch accelerate --quiet

# Utilities
echo "Installing utilities..."
pip3 install pyyaml joblib tqdm psutil --quiet

echo ""
echo "✅ All dependencies installed!"
echo ""
echo "To verify, run:"
echo "  python3 -c 'import streamlit, pandas, plotly, ta; print(\"✅ All packages available\")'"

