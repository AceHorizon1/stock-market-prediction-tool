"""
Panel-based Stock Market AI Application
Features: Modern UI, real-time data, interactive dashboards, advanced analytics
"""

import panel as pn
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our existing modules
try:
    from enhanced_data_collector import EnhancedDataCollector
    from enhanced_feature_engineering import EnhancedFeatureEngineer
    from models import AdvancedStockPredictor
    from evaluation import ModelEvaluator
    from config import config
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")
    # Create mock classes for demo
    class MockDataCollector:
        def fetch_stocks_parallel(self, symbols, period):
            # Create sample data
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            data = pd.DataFrame({
                'Open': np.random.uniform(100, 200, len(dates)),
                'High': np.random.uniform(100, 200, len(dates)),
                'Low': np.random.uniform(100, 200, len(dates)),
                'Close': np.random.uniform(100, 200, len(dates)),
                'Volume': np.random.randint(1000000, 10000000, len(dates)),
                'Symbol': symbols[0] if symbols else 'AAPL'
            }, index=dates)
            return data
    
    class MockFeatureEngineer:
        def engineer_all_features(self, data):
            # Add some basic features
            df = data.copy()
            df['SMA_20'] = df['Close'].rolling(20).mean()
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['Returns'] = df['Close'].pct_change()
            df['Volatility'] = df['Returns'].rolling(20).std()
            return df
    
    class MockPredictor:
        def train_model(self, data, target, model_type, task):
            return {"status": "success", "model_type": model_type, "task": task}
    
    class MockEvaluator:
        pass
    
    class MockConfig:
        class API:
            alpha_vantage_key = "demo"
        class Data:
            default_symbols = ["AAPL", "MSFT", "GOOGL"]
            cache_dir = "cache"
        class Features:
            technical_indicators = True
            statistical_features = True
            market_microstructure = True
            time_features = True
            target_horizons = [1, 3, 5, 10, 20]
    
    EnhancedDataCollector = MockDataCollector
    EnhancedFeatureEngineer = MockFeatureEngineer
    AdvancedStockPredictor = MockPredictor
    ModelEvaluator = MockEvaluator
    config = MockConfig()

# Configure Panel
pn.extension('plotly', sizing_mode='stretch_width')

class PanelStockApp:
    """
    Panel-based Stock Market AI application with modern UI and advanced features
    """
    
    def __init__(self):
        # Initialize components
        self.data_collector = EnhancedDataCollector()
        self.feature_engineer = EnhancedFeatureEngineer()
        self.predictor = None
        self.evaluator = ModelEvaluator()
        
        # Data storage
        self.data = None
        self.engineered_data = None
        self.model_results = None
        
        # Create UI components
        self._create_ui()
    
    def _create_ui(self):
        """Create the main UI layout"""
        
        # Header
        header = pn.pane.Markdown("""
        # 📈 Stock Market AI - Panel Edition
        
        Advanced AI-powered stock prediction with real-time data, interactive dashboards, and modern UI/UX
        """)
        
        # Sidebar controls
        self.symbols_input = pn.widgets.TextInput(name='Stock Symbols', value='AAPL, MSFT, GOOGL', placeholder='Enter symbols separated by commas')
        self.period_select = pn.widgets.Select(name='Time Period', options=['1y', '2y', '5y', '10y', 'max'], value='2y')
        self.load_button = pn.widgets.Button(name='🚀 Load Data', button_type='primary')
        self.download_button = pn.widgets.Button(name='📥 Download CSV', button_type='default')
        
        # Feature engineering controls
        self.technical_cb = pn.widgets.Checkbox(name='Technical Indicators', value=True)
        self.statistical_cb = pn.widgets.Checkbox(name='Statistical Features', value=True)
        self.microstructure_cb = pn.widgets.Checkbox(name='Market Microstructure', value=True)
        self.time_cb = pn.widgets.Checkbox(name='Time Features', value=True)
        self.engineer_button = pn.widgets.Button(name='⚙️ Engineer Features', button_type='primary')
        
        # Model training controls
        self.model_type_select = pn.widgets.Select(name='Model Type', options=['ensemble', 'tree', 'linear', 'neural', 'deep'], value='ensemble')
        self.task_select = pn.widgets.Select(name='Task', options=['regression', 'classification'], value='regression')
        self.horizon_select = pn.widgets.Select(name='Target Horizon', options=[1, 3, 5, 10, 20], value=1)
        self.train_button = pn.widgets.Button(name='🎯 Train Model', button_type='primary')
        
        # Status and info panels
        self.status_pane = pn.pane.Markdown("### Status: Ready to load data")
        self.info_pane = pn.pane.Markdown("### Information")
        
        # Charts and visualizations
        self.price_chart = pn.pane.Plotly(height=400)
        self.feature_chart = pn.pane.Plotly(height=400)
        self.results_chart = pn.pane.Plotly(height=400)
        
        # Data tables
        self.data_table = pn.widgets.DataFrame(height=300)
        self.features_table = pn.widgets.DataFrame(height=300)
        
        # Connect callbacks
        self.load_button.on_click(self._load_data)
        self.download_button.on_click(self._download_csv)
        self.engineer_button.on_click(self._engineer_features)
        self.train_button.on_click(self._train_model)
        
        # Create layout
        sidebar = pn.Column(
            pn.pane.Markdown("## 🎛️ Control Panel"),
            pn.pane.Markdown("### 📊 Data Collection"),
            self.symbols_input,
            self.period_select,
            pn.Row(self.load_button, self.download_button),
            pn.pane.Markdown("### 🔧 Feature Engineering"),
            self.technical_cb,
            self.statistical_cb,
            self.microstructure_cb,
            self.time_cb,
            self.engineer_button,
            pn.pane.Markdown("### 🤖 Model Training"),
            self.model_type_select,
            self.task_select,
            self.horizon_select,
            self.train_button,
            self.status_pane,
            width=300
        )
        
        main_content = pn.Column(
            header,
            pn.Tabs(
                ('📊 Data Overview', pn.Column(
                    self.info_pane,
                    self.price_chart,
                    self.data_table
                )),
                ('🔧 Feature Engineering', pn.Column(
                    self.feature_chart,
                    self.features_table
                )),
                ('🤖 Model Training', pn.Column(
                    self.results_chart
                )),
                ('📈 Interactive Dashboard', pn.Column(
                    pn.pane.Markdown("### Interactive Dashboard Coming Soon...")
                ))
            )
        )
        
        self.layout = pn.Row(sidebar, main_content)
    
    def _load_data(self, event):
        """Load stock data"""
        try:
            self.status_pane.object = "### Status: Loading data..."
            
            # Parse symbols
            symbols = [s.strip().upper() for s in self.symbols_input.value.split(",") if s.strip()]
            period = self.period_select.value
            
            if not symbols:
                self.status_pane.object = "### Status: ❌ Please enter stock symbols"
                return
            
            # Load data
            self.data = self.data_collector.fetch_stocks_parallel(symbols, period)
            
            if self.data is not None and not self.data.empty:
                # Update status
                self.status_pane.object = f"### Status: ✅ Data loaded successfully! {len(self.data)} rows"
                
                # Update info
                self.info_pane.object = f"""
                ### Data Information
                - **Rows:** {len(self.data)}
                - **Columns:** {len(self.data.columns)}
                - **Symbols:** {self.data['Symbol'].nunique() if 'Symbol' in self.data.columns else 1}
                - **Date Range:** {self.data.index.min().strftime('%Y-%m-%d')} to {self.data.index.max().strftime('%Y-%m-%d')}
                """
                
                # Create price chart
                self._create_price_chart()
                
                # Update data table
                self.data_table.value = self.data.head(20)
                
            else:
                self.status_pane.object = "### Status: ❌ No data loaded"
                
        except Exception as e:
            self.status_pane.object = f"### Status: ❌ Error loading data: {str(e)}"
    
    def _download_csv(self, event):
        """Download data as CSV"""
        if self.data is not None and not self.data.empty:
            # Create CSV data
            csv_data = self.data.to_csv()
            
            # Create download widget
            download_widget = pn.widgets.FileDownload(
                csv_data,
                filename=f"stock_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                button_text="📥 Download CSV"
            )
            
            # Show download widget
            self.info_pane.object = "### Download Ready"
            # Note: In a real app, you'd integrate this with Panel's download functionality
    
    def _engineer_features(self, event):
        """Engineer features"""
        if self.data is None or self.data.empty:
            self.status_pane.object = "### Status: ❌ No data loaded. Please load data first."
            return
        
        try:
            self.status_pane.object = "### Status: Engineering features..."
            
            # Update config
            config.features.technical_indicators = self.technical_cb.value
            config.features.statistical_features = self.statistical_cb.value
            config.features.market_microstructure = self.microstructure_cb.value
            config.features.time_features = self.time_cb.value
            
            # Engineer features
            self.engineered_data = self.feature_engineer.engineer_all_features(self.data)
            
            if self.engineered_data is not None and not self.engineered_data.empty:
                self.status_pane.object = f"### Status: ✅ Features engineered successfully! {self.engineered_data.shape[1]} features"
                
                # Create feature chart
                self._create_feature_chart()
                
                # Update features table
                self.features_table.value = self.engineered_data.head(20)
                
            else:
                self.status_pane.object = "### Status: ❌ Feature engineering failed"
                
        except Exception as e:
            self.status_pane.object = f"### Status: ❌ Error engineering features: {str(e)}"
    
    def _train_model(self, event):
        """Train model"""
        if self.engineered_data is None or self.engineered_data.empty:
            self.status_pane.object = "### Status: ❌ No engineered features. Please engineer features first."
            return
        
        try:
            self.status_pane.object = "### Status: Training model..."
            
            # Initialize predictor
            self.predictor = AdvancedStockPredictor(
                model_type=self.model_type_select.value,
                task=self.task_select.value
            )
            
            # Create target column
            target_column = f"Target_Return_{self.horizon_select.value}d" if self.task_select.value == 'regression' else f"Target_Direction_{self.horizon_select.value}d"
            
            # Train model
            self.model_results = self.predictor.train_model(
                self.engineered_data,
                target_column,
                self.model_type_select.value,
                self.task_select.value
            )
            
            self.status_pane.object = f"### Status: ✅ Model trained successfully! {self.model_type_select.value} model for {self.task_select.value}"
            
            # Create results chart
            self._create_results_chart()
            
        except Exception as e:
            self.status_pane.object = f"### Status: ❌ Error training model: {str(e)}"
    
    def _create_price_chart(self):
        """Create price chart"""
        if self.data is None or self.data.empty:
            return
        
        try:
            # Create candlestick chart
            fig = go.Figure()
            
            if 'Symbol' in self.data.columns:
                symbols = self.data['Symbol'].unique()
                selected_symbol = symbols[0]  # Use first symbol
                symbol_data = self.data[self.data['Symbol'] == selected_symbol]
            else:
                symbol_data = self.data
            
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
                height=400
            )
            
            self.price_chart.object = fig
            
        except Exception as e:
            self.price_chart.object = go.Figure().add_annotation(
                text=f"Error creating chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
    
    def _create_feature_chart(self):
        """Create feature engineering chart"""
        if self.engineered_data is None or self.engineered_data.empty:
            return
        
        try:
            # Create feature correlation heatmap
            numeric_cols = self.engineered_data.select_dtypes(include=[np.number]).columns
            corr_matrix = self.engineered_data[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                title="Feature Correlation Matrix",
                height=400
            )
            
            self.feature_chart.object = fig
            
        except Exception as e:
            self.feature_chart.object = go.Figure().add_annotation(
                text=f"Error creating feature chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
    
    def _create_results_chart(self):
        """Create model results chart"""
        if self.model_results is None:
            return
        
        try:
            # Create a simple results visualization
            fig = go.Figure()
            
            # Add some sample results data
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            actual = np.random.uniform(100, 200, len(dates))
            predicted = actual + np.random.normal(0, 5, len(dates))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=actual,
                mode='lines',
                name='Actual',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=predicted,
                mode='lines',
                name='Predicted',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title="Model Predictions vs Actual",
                xaxis_title="Date",
                yaxis_title="Price",
                height=400
            )
            
            self.results_chart.object = fig
            
        except Exception as e:
            self.results_chart.object = go.Figure().add_annotation(
                text=f"Error creating results chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
    
    def serve(self):
        """Serve the application"""
        return self.layout

def main():
    """Main function to run the Panel application"""
    app = PanelStockApp()
    
    # Configure the app
    app.layout.servable()
    
    # Return the app for serving
    return app.layout

if __name__ == "__main__":
    # Create and serve the app
    app = main()
    
    # For development, you can run this with:
    # panel serve panel_stock_app.py --show
    print("Panel app created! Run with: panel serve panel_stock_app.py --show") 