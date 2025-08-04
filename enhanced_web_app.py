"""
Enhanced Web Application for Stock Market AI
Features: Modern UI/UX, real-time data, interactive dashboards, responsive design
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import altair as alt
from datetime import datetime, timedelta
import asyncio
import time
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced modules
from enhanced_data_collector import EnhancedDataCollector
from enhanced_feature_engineering import EnhancedFeatureEngineer
from models import AdvancedStockPredictor
from evaluation import ModelEvaluator
from config import config

# Page configuration
st.set_page_config(
    page_title="Stock Market AI - Enhanced",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/stock-market-ai',
        'Report a bug': 'https://github.com/your-repo/stock-market-ai/issues',
        'About': 'Enhanced Stock Market AI with modern UI/UX and advanced features'
    }
)

class EnhancedStockPredictionApp:
    """
    Enhanced web application with modern UI/UX and advanced features
    """
    
    def __init__(self):
        self.data_collector = EnhancedDataCollector()
        self.feature_engineer = EnhancedFeatureEngineer()
        self.predictor = None
        self.evaluator = ModelEvaluator()
        self.data = None
        self.engineered_data = None
        self.model_results = None
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'features_engineered' not in st.session_state:
            st.session_state.features_engineered = False
        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = False
        
    def run(self):
        """Run the enhanced Streamlit application"""
        # Custom CSS for modern styling
        self._load_custom_css()
        
        # Header
        self._create_header()
        
        # Sidebar
        self._create_sidebar()
        
        # Main content
        if st.session_state.data_loaded:
            self._create_main_content()
        else:
            self._create_welcome_page()
    
    def _load_custom_css(self):
        """Load custom CSS for modern styling"""
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-success { background-color: #28a745; }
        .status-warning { background-color: #ffc107; }
        .status-error { background-color: #dc3545; }
        .feature-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _create_header(self):
        """Create modern header"""
        st.markdown("""
        <div class="main-header">
            <h1>📈 Enhanced Stock Market AI</h1>
            <p>Advanced AI-powered stock prediction with real-time data, interactive dashboards, and modern UI/UX</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _create_sidebar(self):
        """Create enhanced sidebar with modern controls"""
        with st.sidebar:
            st.header("🎛️ Control Panel")
            
            # Data Collection Section
            st.subheader("📊 Data Collection")
            
            # Data source selection
            data_source = st.radio(
                "Data Source",
                ["Online Data", "Upload File"],
                help="Choose between fetching online data or uploading a CSV file"
            )
            
            if data_source == "Online Data":
                # Stock symbols input with autocomplete
                default_symbols = config.data.default_symbols
                symbols_input = st.text_area(
                    "Stock Symbols",
                    value=", ".join(default_symbols),
                    help="Enter stock symbols separated by commas (e.g., AAPL, MSFT, GOOGL)"
                )
                symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
                
                # Time period selection
                period = st.selectbox(
                    "Time Period",
                    options=['1y', '2y', '5y', '10y', 'max'],
                    index=1,
                    help="Select the time period for data collection"
                )
                
                # Load data button with progress
                if st.button("🚀 Load Data", type="primary", use_container_width=True):
                    with st.spinner("Loading data..."):
                        self._load_data_online(symbols, period)
            
            else:
                # File upload
                uploaded_file = st.file_uploader(
                    "Upload CSV/Excel File",
                    type=['csv', 'xlsx'],
                    help="Upload your OHLCV data file"
                )
                
                if uploaded_file is not None:
                    if st.button("📁 Load File", type="primary", use_container_width=True):
                        with st.spinner("Processing file..."):
                            self._load_data_file(uploaded_file)
            
            # Feature Engineering Section
            if st.session_state.data_loaded:
                st.subheader("🔧 Feature Engineering")
                
                # Feature engineering options
                col1, col2 = st.columns(2)
                with col1:
                    technical_indicators = st.checkbox("Technical Indicators", value=True)
                    statistical_features = st.checkbox("Statistical Features", value=True)
                with col2:
                    microstructure_features = st.checkbox("Market Microstructure", value=True)
                    time_features = st.checkbox("Time Features", value=True)
                
                if st.button("⚙️ Engineer Features", type="secondary", use_container_width=True):
                    with st.spinner("Engineering features..."):
                        self._engineer_features(technical_indicators, statistical_features, 
                                             microstructure_features, time_features)
            
            # Model Training Section
            if st.session_state.features_engineered:
                st.subheader("🤖 Model Training")
                
                # Model configuration
                model_type = st.selectbox(
                    "Model Type",
                    options=['ensemble', 'tree', 'linear', 'neural', 'deep'],
                    index=0,
                    help="Choose the type of machine learning model"
                )
                
                task = st.selectbox(
                    "Prediction Task",
                    options=['regression', 'classification'],
                    index=0,
                    help="Choose between price prediction (regression) or direction prediction (classification)"
                )
                
                target_horizon = st.selectbox(
                    "Target Horizon (days)",
                    options=[1, 3, 5, 10, 20],
                    index=0,
                    help="Select the prediction horizon"
                )
                
                if st.button("🎯 Train Model", type="primary", use_container_width=True):
                    with st.spinner("Training model..."):
                        self._train_model(model_type, task, target_horizon)
            
            # Cache Management
            st.subheader("🗄️ Cache Management")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear Cache", use_container_width=True):
                    self.data_collector.clear_cache()
                    st.success("Cache cleared!")
            with col2:
                if st.button("Cache Info", use_container_width=True):
                    cache_info = self.data_collector.get_cache_info()
                    st.json(cache_info)
    
    def _create_welcome_page(self):
        """Create modern welcome page"""
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h2>🚀 Welcome to Enhanced Stock Market AI</h2>
            <p style="font-size: 1.2rem; color: #666;">
                Advanced AI-powered stock prediction with real-time data, interactive dashboards, and modern UI/UX
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h3>📊 Real-time Data</h3>
                <p>Fetch live stock data from multiple sources with caching and parallel processing</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h3>🔧 Advanced Features</h3>
                <p>100+ technical indicators, statistical features, and market microstructure analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h3>🤖 ML Models</h3>
                <p>Ensemble methods, deep learning, and advanced model stacking</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h3>📈 Interactive Dashboards</h3>
                <p>Modern visualizations with real-time updates and interactive charts</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick start guide
        st.subheader("🚀 Quick Start Guide")
        
        with st.expander("Step-by-step instructions"):
            st.markdown("""
            1. **Load Data**: Choose between online data or file upload
            2. **Engineer Features**: Create advanced technical and statistical features
            3. **Train Model**: Select model type and train with your data
            4. **View Results**: Analyze predictions and performance metrics
            5. **Interactive Dashboard**: Explore data with modern visualizations
            """)
    
    def _create_main_content(self):
        """Create main content with tabs"""
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Overview", 
            "🔧 Feature Engineering", 
            "🤖 Model Training", 
            "📈 Results & Predictions",
            "📊 Interactive Dashboard"
        ])
        
        with tab1:
            self._show_data_overview()
        
        with tab2:
            self._show_feature_engineering()
        
        with tab3:
            self._show_model_training()
        
        with tab4:
            self._show_results()
        
        with tab5:
            self._show_interactive_dashboard()
    
    def _load_data_online(self, symbols: list, period: str):
        """Load data from online sources"""
        try:
            # Use parallel processing for faster data loading
            self.data = self.data_collector.fetch_stocks_parallel(symbols, period)
            
            # Validate data
            validation = self.data_collector.validate_data(self.data)
            
            if validation['data_quality_score'] > 0.8:
                st.session_state.data_loaded = True
                st.success(f"✅ Data loaded successfully! {len(self.data)} rows from {len(symbols)} symbols")
                
                # Show data summary
                summary = self.data_collector.get_data_summary(self.data)
                st.json(summary)
            else:
                st.warning(f"⚠️ Data quality score: {validation['data_quality_score']:.2f}")
                st.json(validation)
                
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
    
    def _load_data_file(self, uploaded_file):
        """Load data from uploaded file"""
        try:
            # Save uploaded file temporarily
            with open("temp_upload.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load and validate data
            self.data = self.data_collector.load_csv_file("temp_upload.csv")
            validation = self.data_collector.validate_data(self.data)
            
            if validation['data_quality_score'] > 0.8:
                st.session_state.data_loaded = True
                st.success(f"✅ File loaded successfully! {len(self.data)} rows")
                st.json(validation)
            else:
                st.warning(f"⚠️ Data quality score: {validation['data_quality_score']:.2f}")
                st.json(validation)
                
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    def _engineer_features(self, technical: bool, statistical: bool, microstructure: bool, time: bool):
        """Engineer features with selected options"""
        try:
            # Update config based on user selection
            config.features.technical_indicators = technical
            config.features.statistical_features = statistical
            config.features.market_microstructure = microstructure
            config.features.time_features = time
            
            # Engineer features
            self.engineered_data = self.feature_engineer.engineer_all_features(self.data)
            
            st.session_state.features_engineered = True
            st.success(f"✅ Features engineered successfully! {self.engineered_data.shape[1]} features created")
            
            # Show feature summary
            summary = self.feature_engineer.get_feature_summary(self.engineered_data)
            st.json(summary)
            
        except Exception as e:
            st.error(f"❌ Error engineering features: {str(e)}")
    
    def _train_model(self, model_type: str, task: str, target_horizon: int):
        """Train model with selected parameters"""
        try:
            # Create target column name
            target_column = f"Target_Return_{target_horizon}d" if task == 'regression' else f"Target_Direction_{target_horizon}d"
            
            # Initialize predictor
            self.predictor = AdvancedStockPredictor(model_type=model_type, task=task)
            
            # Train model
            self.model_results = self.predictor.train_model(
                self.engineered_data, 
                target_column, 
                model_type, 
                task
            )
            
            st.session_state.model_trained = True
            st.success(f"✅ Model trained successfully! {model_type} model for {task}")
            
            # Show training results
            st.json(self.model_results)
            
        except Exception as e:
            st.error(f"❌ Error training model: {str(e)}")
    
    def _show_data_overview(self):
        """Show enhanced data overview"""
        if self.data is not None:
            st.subheader("📊 Data Overview")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Rows", len(self.data))
            
            with col2:
                st.metric("Symbols", self.data['Symbol'].nunique() if 'Symbol' in self.data.columns else 1)
            
            with col3:
                st.metric("Date Range", f"{self.data.index.min().strftime('%Y-%m-%d')} to {self.data.index.max().strftime('%Y-%m-%d')}")
            
            with col4:
                st.metric("Memory Usage", f"{self.data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
            
            # Data preview
            st.subheader("📋 Data Preview")
            st.dataframe(self.data.head(10), use_container_width=True)
            
            # Price chart
            st.subheader("📈 Price Chart")
            if 'Symbol' in self.data.columns:
                symbols = self.data['Symbol'].unique()
                selected_symbol = st.selectbox("Select Symbol", symbols)
                symbol_data = self.data[self.data['Symbol'] == selected_symbol]
            else:
                symbol_data = self.data
            
            # Create interactive price chart
            fig = go.Figure()
            
            fig.add_trace(go.Candlestick(
                x=symbol_data.index,
                open=symbol_data['Open'],
                high=symbol_data['High'],
                low=symbol_data['Low'],
                close=symbol_data['Close'],
                name='OHLC'
            ))
            
            fig.update_layout(
                title=f"Price Chart - {selected_symbol if 'Symbol' in self.data.columns else 'Data'}",
                xaxis_title="Date",
                yaxis_title="Price",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_feature_engineering(self):
        """Show feature engineering results"""
        if self.engineered_data is not None:
            st.subheader("🔧 Feature Engineering Results")
            
            # Feature summary
            summary = self.feature_engineer.get_feature_summary(self.engineered_data)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Features", summary['total_features'])
            
            with col2:
                st.metric("Technical Indicators", summary['feature_categories']['technical_indicators'])
            
            with col3:
                st.metric("Statistical Features", summary['feature_categories']['statistical_features'])
            
            # Feature categories chart
            st.subheader("📊 Feature Categories")
            
            categories = summary['feature_categories']
            fig = px.pie(
                values=list(categories.values()),
                names=list(categories.keys()),
                title="Feature Distribution by Category"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature preview
            st.subheader("📋 Engineered Features Preview")
            st.dataframe(self.engineered_data.head(10), use_container_width=True)
    
    def _show_model_training(self):
        """Show model training interface"""
        st.subheader("🤖 Model Training")
        
        if not st.session_state.features_engineered:
            st.warning("⚠️ Please engineer features first")
            return
        
        # Model configuration form
        with st.form("model_training_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                model_type = st.selectbox("Model Type", ['ensemble', 'tree', 'linear', 'neural', 'deep'])
                task = st.selectbox("Task", ['regression', 'classification'])
            
            with col2:
                target_horizon = st.selectbox("Target Horizon", [1, 3, 5, 10, 20])
                feature_selection = st.checkbox("Use Feature Selection", value=True)
            
            submitted = st.form_submit_button("🎯 Train Model", type="primary")
            
            if submitted:
                with st.spinner("Training model..."):
                    self._train_model(model_type, task, target_horizon)
    
    def _show_results(self):
        """Show model results and predictions"""
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train a model first")
            return
        
        st.subheader("📈 Model Results")
        
        # Performance metrics
        if 'metrics' in self.model_results:
            metrics = self.model_results['metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("MSE", f"{metrics.get('mse', 0):.4f}")
            
            with col2:
                st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
            
            with col3:
                st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
            
            with col4:
                st.metric("R²", f"{metrics.get('r_squared', 0):.4f}")
        
        # Predictions vs Actual
        if 'predictions' in self.model_results:
            st.subheader("📊 Predictions vs Actual")
            
            predictions_df = self.model_results['predictions']
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=predictions_df.index,
                y=predictions_df['Actual'],
                mode='lines',
                name='Actual',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=predictions_df.index,
                y=predictions_df['Predicted'],
                mode='lines',
                name='Predicted',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title="Predictions vs Actual Values",
                xaxis_title="Date",
                yaxis_title="Value",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_interactive_dashboard(self):
        """Show interactive dashboard"""
        st.subheader("📊 Interactive Dashboard")
        
        if self.data is None:
            st.warning("⚠️ Please load data first")
            return
        
        # Dashboard controls
        col1, col2 = st.columns(2)
        
        with col1:
            chart_type = st.selectbox(
                "Chart Type",
                ["Price Chart", "Volume Chart", "Returns Distribution", "Correlation Matrix"]
            )
        
        with col2:
            if 'Symbol' in self.data.columns:
                selected_symbol = st.selectbox("Symbol", self.data['Symbol'].unique())
                symbol_data = self.data[self.data['Symbol'] == selected_symbol]
            else:
                symbol_data = self.data
        
        # Create interactive charts based on selection
        if chart_type == "Price Chart":
            self._create_price_chart(symbol_data)
        elif chart_type == "Volume Chart":
            self._create_volume_chart(symbol_data)
        elif chart_type == "Returns Distribution":
            self._create_returns_chart(symbol_data)
        elif chart_type == "Correlation Matrix":
            self._create_correlation_chart(symbol_data)
    
    def _create_price_chart(self, data):
        """Create interactive price chart"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price', 'Volume'),
            row_width=[0.7, 0.3]
        )
        
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="OHLC"
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name="Volume"
        ), row=2, col=1)
        
        fig.update_layout(
            title="Interactive Price Chart",
            xaxis_rangeslider_visible=False,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_volume_chart(self, data):
        """Create volume analysis chart"""
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name="Volume"
        ))
        
        fig.update_layout(
            title="Volume Analysis",
            xaxis_title="Date",
            yaxis_title="Volume",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_returns_chart(self, data):
        """Create returns distribution chart"""
        returns = data['Close'].pct_change().dropna()
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name="Returns Distribution"
        ))
        
        fig.update_layout(
            title="Returns Distribution",
            xaxis_title="Returns",
            yaxis_title="Frequency",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_correlation_chart(self, data):
        """Create correlation matrix chart"""
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title="Correlation Matrix",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main function to run the enhanced application"""
    app = EnhancedStockPredictionApp()
    app.run()

if __name__ == "__main__":
    main() 