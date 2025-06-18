#!/usr/bin/env python3
"""
Stock Market Prediction Tool
A comprehensive tool for predicting stock market movements using machine learning
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_collector import DataCollector
from feature_engineering import FeatureEngineer
from models import AdvancedStockPredictor
from evaluation import ModelEvaluator

class StockPredictionApp:
    """
    Main application class for the stock prediction tool
    """
    
    def __init__(self):
        self.data_collector = DataCollector()
        self.feature_engineer = FeatureEngineer()
        self.predictor = None
        self.evaluator = ModelEvaluator()
        self.data = None
        self.engineered_data = None
        
    def run(self):
        """Run the Streamlit application"""
        st.set_page_config(
            page_title="Stock Market Prediction Tool",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("📈 Stock Market Prediction Tool")
        st.markdown("A comprehensive AI-powered tool for predicting stock market movements")
        
        # Sidebar
        self.sidebar()
        
        # Main content
        if st.session_state.get('data_loaded', False):
            self.main_content()
        else:
            self.welcome_page()
    
    def sidebar(self):
        """Create the sidebar with controls"""
        st.sidebar.header("Settings")
        
        # Data collection settings
        st.sidebar.subheader("Data Collection")
        
        # Stock symbols
        default_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        symbols_input = st.sidebar.text_input(
            "Stock Symbols (comma-separated)",
            value=", ".join(default_symbols)
        )
        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        
        # Time period
        period = st.sidebar.selectbox(
            "Time Period",
            options=['1y', '2y', '5y', '10y', 'max'],
            index=2
        )
        
        # Include additional data
        include_market_data = st.sidebar.checkbox("Include Market Data", value=True)
        include_economic_data = st.sidebar.checkbox("Include Economic Data", value=False)
        
        # Load data button
        if st.sidebar.button("Load Data", type="primary"):
            with st.spinner("Loading data..."):
                self.load_data(symbols, period, include_market_data, include_economic_data)
        
        # Model settings
        if st.session_state.get('data_loaded', False):
            st.sidebar.subheader("Model Settings")
            
            model_type = st.sidebar.selectbox(
                "Model Type",
                options=['ensemble', 'tree', 'linear', 'neural', 'deep'],
                index=0
            )
            
            task = st.sidebar.selectbox(
                "Prediction Task",
                options=['regression', 'classification'],
                index=0
            )
            
            target_horizon = st.sidebar.selectbox(
                "Prediction Horizon (days)",
                options=[1, 3, 5, 10, 20],
                index=0
            )
            
            # Train model button
            if st.sidebar.button("Train Model", type="primary"):
                with st.spinner("Training model..."):
                    self.train_model(model_type, task, target_horizon)
    
    def welcome_page(self):
        """Display welcome page"""
        st.markdown("""
        ## Welcome to the Stock Market Prediction Tool! 🚀
        
        This tool uses advanced machine learning algorithms to predict stock market movements.
        
        ### Features:
        - **Multi-source data collection**: Yahoo Finance, Alpha Vantage, FRED
        - **Advanced feature engineering**: 100+ technical and statistical indicators
        - **Multiple ML models**: Ensemble, Deep Learning, LSTM, XGBoost, and more
        - **Comprehensive evaluation**: Backtesting, performance metrics, visualizations
        - **Real-time predictions**: Get predictions for any stock
        
        ### How to use:
        1. **Load Data**: Enter stock symbols and click "Load Data"
        2. **Configure Model**: Choose model type and prediction task
        3. **Train Model**: Click "Train Model" to start training
        4. **View Results**: Explore predictions, performance metrics, and visualizations
        
        ### Getting Started:
        Use the sidebar to configure your settings and load data for your desired stocks.
        """)
        
        # Example usage
        st.subheader("Example Usage")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Popular Stocks:**
            - AAPL (Apple)
            - MSFT (Microsoft)
            - GOOGL (Google)
            - AMZN (Amazon)
            - TSLA (Tesla)
            """)
        
        with col2:
            st.markdown("""
            **Model Types:**
            - **Ensemble**: Best overall performance
            - **Tree**: Good for non-linear patterns
            - **Deep**: Neural networks for complex patterns
            - **Linear**: Fast and interpretable
            """)
    
    def load_data(self, symbols, period, include_market_data, include_economic_data):
        """Load and process data"""
        try:
            # Load data
            self.data = self.data_collector.create_comprehensive_dataset(
                symbols, include_market_data, include_economic_data
            )
            
            if self.data.empty:
                st.error("No data loaded. Please check your symbols and try again.")
                return
            
            # Engineer features
            self.engineered_data = self.feature_engineer.engineer_all_features(
                self.data, target_horizons=[1, 3, 5, 10, 20]
            )
            
            st.session_state['data_loaded'] = True
            st.session_state['symbols'] = symbols
            st.session_state['period'] = period
            
            st.success(f"✅ Data loaded successfully! Shape: {self.engineered_data.shape}")
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    
    def train_model(self, model_type, task, target_horizon):
        """Train the prediction model"""
        try:
            # Prepare data
            target_column = f'Target_Return_{target_horizon}d'
            
            if target_column not in self.engineered_data.columns:
                st.error(f"Target column {target_column} not found in data")
                return
            
            # Remove rows with NaN targets
            data_clean = self.engineered_data.dropna(subset=[target_column])
            
            if data_clean.empty:
                st.error("No valid data after cleaning")
                return
            
            # Select features
            feature_columns = self.feature_engineer.select_features(
                data_clean, target_column, method='correlation', threshold=0.01
            )
            
            if not feature_columns:
                st.error("No features selected. Try lowering the correlation threshold.")
                return
            
            # Prepare X and y
            X = data_clean[feature_columns]
            y = data_clean[target_column]
            
            # Split data (time series split)
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Initialize and train model
            self.predictor = AdvancedStockPredictor(model_type=model_type, task=task)
            
            # Train models
            results = self.predictor.train_models(X_train, y_train, X_val, y_val)
            
            # Store results
            st.session_state['model_trained'] = True
            st.session_state['training_results'] = results
            st.session_state['feature_columns'] = feature_columns
            st.session_state['model_type'] = model_type
            st.session_state['task'] = task
            st.session_state['target_horizon'] = target_horizon
            
            st.success("✅ Model trained successfully!")
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
    
    def main_content(self):
        """Display main content after data is loaded"""
        # Data overview
        self.show_data_overview()
        
        # Model results
        if st.session_state.get('model_trained', False):
            self.show_model_results()
        
        # Prediction interface
        self.show_prediction_interface()
    
    def show_data_overview(self):
        """Show data overview"""
        st.header("📊 Data Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(self.engineered_data))
        
        with col2:
            st.metric("Features", len(self.engineered_data.columns))
        
        with col3:
            st.metric("Date Range", f"{self.engineered_data.index[0].date()} to {self.engineered_data.index[-1].date()}")
        
        # Show data sample
        st.subheader("Data Sample")
        st.dataframe(self.engineered_data.head(10))
        
        # Show price trends
        if 'Close' in self.engineered_data.columns:
            st.subheader("Price Trends")
            
            # Get unique symbols
            if 'Stock_Symbol' in self.engineered_data.columns:
                symbols = self.engineered_data['Stock_Symbol'].unique()
                
                for symbol in symbols[:3]:  # Show first 3 symbols
                    symbol_data = self.engineered_data[self.engineered_data['Stock_Symbol'] == symbol]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=symbol_data.index,
                        y=symbol_data['Close'],
                        name=symbol,
                        line=dict(width=2)
                    ))
                    
                    fig.update_layout(
                        title=f"{symbol} Price Chart",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    def show_model_results(self):
        """Show model training results"""
        st.header("🤖 Model Results")
        
        results = st.session_state.get('training_results', {})
        
        if results:
            # Display metrics
            st.subheader("Model Performance")
            
            # Create metrics display
            cols = st.columns(len(results))
            
            for i, (model_name, metrics) in enumerate(results.items()):
                with cols[i]:
                    st.metric("Model", model_name)
                    
                    if st.session_state.get('task') == 'regression':
                        st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
                        st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
                        st.metric("R²", f"{metrics.get('r_squared', 0):.4f}")
                    else:
                        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                        st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                        st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
            
            # Feature importance
            if self.predictor and hasattr(self.predictor, 'get_feature_importance'):
                st.subheader("Feature Importance")
                
                importance = self.predictor.get_feature_importance()
                if importance:
                    importance_df = pd.DataFrame(
                        list(importance.items()),
                        columns=['Feature', 'Importance']
                    ).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        importance_df.head(20),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Top 20 Most Important Features"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    def show_prediction_interface(self):
        """Show prediction interface"""
        st.header("🔮 Make Predictions")
        
        if not st.session_state.get('model_trained', False):
            st.info("Please train a model first to make predictions.")
            return
        
        # Prediction form
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Single Stock Prediction")
            
            # Get latest data for prediction
            if self.engineered_data is not None:
                latest_data = self.engineered_data.iloc[-1:]
                feature_columns = st.session_state.get('feature_columns', [])
                
                if feature_columns and all(col in latest_data.columns for col in feature_columns):
                    X_pred = latest_data[feature_columns]
                    
                    if st.button("Predict Next Day"):
                        try:
                            prediction = self.predictor.ensemble_predict(X_pred)
                            st.success(f"Predicted Return: {prediction[0]:.4f} ({prediction[0]*100:.2f}%)")
                            
                            # Interpret prediction
                            if prediction[0] > 0:
                                st.info("📈 Bullish prediction - Stock expected to rise")
                            else:
                                st.info("📉 Bearish prediction - Stock expected to fall")
                                
                        except Exception as e:
                            st.error(f"Error making prediction: {str(e)}")
        
        with col2:
            st.subheader("Batch Prediction")
            
            # Allow user to input custom data
            st.info("Upload CSV file with features for batch prediction")
            
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="File should contain the same features used for training"
            )
            
            if uploaded_file is not None:
                try:
                    batch_data = pd.read_csv(uploaded_file)
                    st.dataframe(batch_data.head())
                    
                    if st.button("Predict Batch"):
                        feature_columns = st.session_state.get('feature_columns', [])
                        
                        if all(col in batch_data.columns for col in feature_columns):
                            X_batch = batch_data[feature_columns]
                            predictions = self.predictor.ensemble_predict(X_batch)
                            
                            # Display results
                            results_df = pd.DataFrame({
                                'Prediction': predictions,
                                'Prediction_Percent': predictions * 100
                            })
                            
                            st.dataframe(results_df)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Predictions",
                                data=csv,
                                file_name="predictions.csv",
                                mime="text/csv"
                            )
                        else:
                            st.error("Missing required features in uploaded file")
                            
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

def main():
    """Main function to run the application"""
    app = StockPredictionApp()
    app.run()

if __name__ == "__main__":
    main() 