import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Comprehensive model evaluation for stock prediction
    Includes backtesting, performance metrics, and visualization
    """
    
    def __init__(self):
        self.results = {}
        self.backtest_results = {}
        
    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # Percentage errors
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        metrics['mape_trimmed'] = np.mean(np.abs((y_true - y_pred) / y_true)[np.abs(y_true) > 0.01]) * 100
        
        # Directional accuracy
        direction_true = np.diff(y_true) > 0
        direction_pred = np.diff(y_pred) > 0
        metrics['directional_accuracy'] = np.mean(direction_true == direction_pred)
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r_squared'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Adjusted R-squared
        n = len(y_true)
        p = 1  # Number of features (simplified)
        metrics['adjusted_r_squared'] = 1 - (1 - metrics['r_squared']) * (n - 1) / (n - p - 1)
        
        # Information ratio (assuming returns)
        metrics['information_ratio'] = np.mean(y_pred - y_true) / np.std(y_pred - y_true) if np.std(y_pred - y_true) != 0 else 0
        
        # Sharpe ratio (assuming returns)
        metrics['sharpe_ratio'] = np.mean(y_pred) / np.std(y_pred) if np.std(y_pred) != 0 else 0
        
        return metrics
    
    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       y_prob: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Derived metrics
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # AUC-ROC if probabilities are available
        if y_prob is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            except:
                metrics['auc_roc'] = 0
        
        # Directional accuracy for trading
        metrics['directional_accuracy'] = np.mean(y_true == y_pred)
        
        return metrics
    
    def backtest_strategy(self, data: pd.DataFrame, predictions: np.ndarray, 
                         initial_capital: float = 10000, transaction_cost: float = 0.001,
                         strategy_type: str = 'long_short') -> Dict[str, float]:
        """
        Backtest a trading strategy
        
        Args:
            data: DataFrame with price data
            predictions: Model predictions
            initial_capital: Initial capital
            transaction_cost: Transaction cost as percentage
            strategy_type: Strategy type ('long_short', 'long_only', 'threshold')
        
        Returns:
            Dictionary with backtest results
        """
        # Create a copy of the data
        df = data.copy()
        df['Predictions'] = predictions
        
        # Calculate returns
        df['Returns'] = df['Close'].pct_change()
        df['Strategy_Returns'] = 0.0
        
        if strategy_type == 'long_short':
            # Long when prediction > 0, short when prediction < 0
            df.loc[df['Predictions'] > 0, 'Strategy_Returns'] = df.loc[df['Predictions'] > 0, 'Returns']
            df.loc[df['Predictions'] < 0, 'Strategy_Returns'] = -df.loc[df['Predictions'] < 0, 'Returns']
        
        elif strategy_type == 'long_only':
            # Long only when prediction > threshold
            threshold = np.percentile(predictions, 70)  # Top 30% predictions
            df.loc[df['Predictions'] > threshold, 'Strategy_Returns'] = df.loc[df['Predictions'] > threshold, 'Returns']
        
        elif strategy_type == 'threshold':
            # Long when prediction > threshold, short when prediction < -threshold
            threshold = np.std(predictions) * 0.5
            df.loc[df['Predictions'] > threshold, 'Strategy_Returns'] = df.loc[df['Predictions'] > threshold, 'Returns']
            df.loc[df['Predictions'] < -threshold, 'Strategy_Returns'] = -df.loc[df['Predictions'] < -threshold, 'Returns']
        
        # Calculate cumulative returns
        df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
        df['Strategy_Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
        
        # Calculate portfolio value
        df['Portfolio_Value'] = initial_capital * df['Strategy_Cumulative_Returns']
        
        # Calculate metrics
        total_return = (df['Portfolio_Value'].iloc[-1] / initial_capital) - 1
        annual_return = total_return * (252 / len(df))
        volatility = df['Strategy_Returns'].std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        
        # Maximum drawdown
        peak = df['Portfolio_Value'].expanding().max()
        drawdown = (df['Portfolio_Value'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = (df['Strategy_Returns'] > 0).sum()
        total_trades = (df['Strategy_Returns'] != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Average trade
        avg_trade = df['Strategy_Returns'].mean()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_trade': avg_trade,
            'total_trades': total_trades,
            'final_portfolio_value': df['Portfolio_Value'].iloc[-1]
        }
    
    def rolling_window_evaluation(self, data: pd.DataFrame, predictions: np.ndarray,
                                window_size: int = 252, step_size: int = 63) -> pd.DataFrame:
        """
        Evaluate model performance using rolling windows
        
        Args:
            data: DataFrame with price data
            predictions: Model predictions
            window_size: Size of rolling window
            step_size: Step size for rolling window
        
        Returns:
            DataFrame with rolling performance metrics
        """
        results = []
        
        for i in range(0, len(data) - window_size, step_size):
            window_data = data.iloc[i:i+window_size]
            window_predictions = predictions[i:i+window_size]
            
            # Calculate metrics for this window
            metrics = self.calculate_regression_metrics(
                window_data['Returns'].values[1:],  # Skip first NaN
                window_predictions[1:]
            )
            
            # Add window information
            metrics['start_date'] = window_data.index[0]
            metrics['end_date'] = window_data.index[-1]
            metrics['window_size'] = len(window_data)
            
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def plot_predictions(self, data: pd.DataFrame, predictions: np.ndarray, 
                        title: str = "Model Predictions vs Actual") -> go.Figure:
        """
        Create interactive plot of predictions vs actual values
        
        Args:
            data: DataFrame with actual values
            predictions: Model predictions
            title: Plot title
        
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Price Predictions', 'Returns Predictions', 'Cumulative Returns'),
            vertical_spacing=0.1
        )
        
        # Price predictions
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Close'], name='Actual Price', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=predictions, name='Predicted Price', line=dict(color='red')),
            row=1, col=1
        )
        
        # Returns predictions
        actual_returns = data['Returns']
        fig.add_trace(
            go.Scatter(x=data.index, y=actual_returns, name='Actual Returns', line=dict(color='blue')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=predictions, name='Predicted Returns', line=dict(color='red')),
            row=2, col=1
        )
        
        # Cumulative returns
        cum_actual = (1 + actual_returns).cumprod()
        cum_pred = (1 + pd.Series(predictions, index=data.index)).cumprod()
        
        fig.add_trace(
            go.Scatter(x=data.index, y=cum_actual, name='Actual Cumulative', line=dict(color='blue')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=cum_pred, name='Predicted Cumulative', line=dict(color='red')),
            row=3, col=1
        )
        
        fig.update_layout(height=800, title_text=title)
        return fig
    
    def plot_backtest_results(self, data: pd.DataFrame, backtest_results: Dict[str, float]) -> go.Figure:
        """
        Create interactive plot of backtest results
        
        Args:
            data: DataFrame with price and strategy data
            backtest_results: Results from backtest_strategy
        
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Portfolio Value', 'Cumulative Returns', 'Drawdown', 'Monthly Returns'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Portfolio_Value'], name='Portfolio Value'),
            row=1, col=1
        )
        
        # Cumulative returns comparison
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Cumulative_Returns'], name='Buy & Hold'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Strategy_Cumulative_Returns'], name='Strategy'),
            row=1, col=2
        )
        
        # Drawdown
        peak = data['Portfolio_Value'].expanding().max()
        drawdown = (data['Portfolio_Value'] - peak) / peak
        fig.add_trace(
            go.Scatter(x=data.index, y=drawdown, name='Drawdown', fill='tonexty'),
            row=2, col=1
        )
        
        # Monthly returns
        monthly_returns = data['Strategy_Returns'].resample('M').sum()
        fig.add_trace(
            go.Bar(x=monthly_returns.index, y=monthly_returns, name='Monthly Returns'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text="Backtest Results")
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
        """
        Create confusion matrix plot
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Plotly figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16}
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual"
        )
        
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> go.Figure:
        """
        Create ROC curve plot
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
        
        Returns:
            Plotly figure
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc:.3f})'
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash')
        ))
        
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def generate_evaluation_report(self, data: pd.DataFrame, predictions: np.ndarray,
                                 y_true: np.ndarray = None, task: str = 'regression') -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report
        
        Args:
            data: DataFrame with price data
            predictions: Model predictions
            y_true: True values (if different from data)
            task: Task type ('regression' or 'classification')
        
        Returns:
            Dictionary with comprehensive evaluation results
        """
        report = {}
        
        if y_true is None:
            y_true = data['Returns'].values[1:]  # Skip first NaN
            predictions = predictions[1:]  # Align with y_true
        
        # Calculate metrics
        if task == 'regression':
            report['metrics'] = self.calculate_regression_metrics(y_true, predictions)
        else:
            report['metrics'] = self.calculate_classification_metrics(y_true, predictions)
        
        # Backtest results
        report['backtest'] = self.backtest_strategy(data, predictions)
        
        # Rolling window evaluation
        report['rolling_evaluation'] = self.rolling_window_evaluation(data, predictions)
        
        # Create plots
        report['plots'] = {
            'predictions': self.plot_predictions(data, predictions),
            'backtest': self.plot_backtest_results(data, report['backtest'])
        }
        
        if task == 'classification':
            report['plots']['confusion_matrix'] = self.plot_confusion_matrix(y_true, predictions)
            # Note: ROC curve requires probabilities, which might not be available
        
        return report
    
    def compare_models(self, data: pd.DataFrame, model_predictions: Dict[str, np.ndarray],
                      task: str = 'regression') -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            data: DataFrame with price data
            model_predictions: Dictionary with model names and predictions
            task: Task type
        
        Returns:
            DataFrame with comparison results
        """
        comparison_results = []
        
        for model_name, predictions in model_predictions.items():
            # Calculate metrics
            y_true = data['Returns'].values[1:]
            pred = predictions[1:]
            
            if task == 'regression':
                metrics = self.calculate_regression_metrics(y_true, pred)
            else:
                metrics = self.calculate_classification_metrics(y_true, pred)
            
            # Backtest results
            backtest = self.backtest_strategy(data, predictions)
            
            # Combine results
            result = {**metrics, **backtest}
            result['model_name'] = model_name
            
            comparison_results.append(result)
        
        return pd.DataFrame(comparison_results)

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_samples = len(dates)
    
    # Generate sample price data
    returns = np.random.normal(0.0005, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'Close': prices,
        'Returns': returns
    }, index=dates)
    
    # Generate sample predictions
    predictions = returns + np.random.normal(0, 0.01, n_samples)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Generate evaluation report
    report = evaluator.generate_evaluation_report(data, predictions)
    
    print("Evaluation Report:")
    print(f"Metrics: {report['metrics']}")
    print(f"Backtest Results: {report['backtest']}")
    
    # Show plots
    report['plots']['predictions'].show()
    report['plots']['backtest'].show() 