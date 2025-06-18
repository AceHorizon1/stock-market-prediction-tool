import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    """
    Comprehensive stock prediction model with multiple algorithms
    Supports both regression (price prediction) and classification (up/down prediction)
    """
    
    def __init__(self, model_type: str = 'ensemble', task: str = 'regression'):
        """
        Initialize the predictor
        
        Args:
            model_type: Type of model ('linear', 'tree', 'neural', 'ensemble', 'deep')
            task: Prediction task ('regression' or 'classification')
        """
        self.model_type = model_type
        self.task = task
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.history = {}
        
    def create_linear_models(self) -> Dict[str, Any]:
        """Create linear models"""
        if self.task == 'regression':
            return {
                'linear': LinearRegression(),
                'ridge': Ridge(alpha=1.0),
                'lasso': Lasso(alpha=0.1)
            }
        else:
            return {
                'logistic': LogisticRegression(random_state=42, max_iter=1000),
                'ridge_classifier': Ridge(alpha=1.0)
            }
    
    def create_tree_models(self) -> Dict[str, Any]:
        """Create tree-based models"""
        if self.task == 'regression':
            return {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
                'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42),
                'catboost': CatBoostRegressor(iterations=100, random_state=42, verbose=False)
            }
        else:
            return {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42),
                'lightgbm': lgb.LGBMClassifier(n_estimators=100, random_state=42),
                'catboost': CatBoostClassifier(iterations=100, random_state=42, verbose=False)
            }
    
    def create_neural_models(self) -> Dict[str, Any]:
        """Create neural network models"""
        if self.task == 'regression':
            return {
                'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            }
        else:
            return {
                'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            }
    
    def create_svm_models(self) -> Dict[str, Any]:
        """Create SVM models"""
        if self.task == 'regression':
            return {
                'svr': SVR(kernel='rbf', C=1.0, gamma='scale')
            }
        else:
            return {
                'svc': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
            }
    
    def create_deep_learning_model(self, input_shape: int) -> keras.Model:
        """Create a deep learning model"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='linear' if self.task == 'regression' else 'sigmoid')
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        loss = 'mse' if self.task == 'regression' else 'binary_crossentropy'
        metrics = ['mae'] if self.task == 'regression' else ['accuracy']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model
    
    def create_lstm_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Create an LSTM model for time series prediction"""
        model = keras.Sequential([
            layers.LSTM(50, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(50, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(25),
            layers.Dense(1, activation='linear' if self.task == 'regression' else 'sigmoid')
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        loss = 'mse' if self.task == 'regression' else 'binary_crossentropy'
        metrics = ['mae'] if self.task == 'regression' else ['accuracy']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model
    
    def prepare_sequences(self, data: np.ndarray, target: np.ndarray, 
                         sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM model"""
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(target[i])
        return np.array(X), np.array(y)
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_val: pd.DataFrame = None, y_val: pd.Series = None,
                    sequence_length: int = 60) -> Dict[str, Any]:
        """
        Train multiple models
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            sequence_length: Sequence length for LSTM
        
        Returns:
            Dictionary with training results
        """
        results = {}
        
        # Create models based on type
        if self.model_type == 'linear':
            models = self.create_linear_models()
        elif self.model_type == 'tree':
            models = self.create_tree_models()
        elif self.model_type == 'neural':
            models = self.create_neural_models()
        elif self.model_type == 'svm':
            models = self.create_svm_models()
        elif self.model_type == 'ensemble':
            models = {**self.create_tree_models(), **self.create_linear_models()}
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train traditional ML models
        for name, model in models.items():
            print(f"Training {name}...")
            try:
                model.fit(X_train, y_train)
                self.models[name] = model
                
                # Get feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(X_train.columns, model.feature_importances_))
                
                # Make predictions
                if X_val is not None:
                    y_pred = model.predict(X_val)
                    results[name] = self.evaluate_predictions(y_val, y_pred)
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
        
        # Train deep learning models
        if self.model_type in ['deep', 'ensemble']:
            print("Training deep learning models...")
            
            # Dense neural network
            try:
                dense_model = self.create_deep_learning_model(X_train.shape[1])
                history = dense_model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val) if X_val is not None else None,
                    epochs=50,
                    batch_size=32,
                    verbose=0
                )
                self.models['dense_nn'] = dense_model
                self.history['dense_nn'] = history.history
                
                if X_val is not None:
                    y_pred = dense_model.predict(X_val).flatten()
                    results['dense_nn'] = self.evaluate_predictions(y_val, y_pred)
                    
            except Exception as e:
                print(f"Error training dense neural network: {str(e)}")
            
            # LSTM model
            try:
                # Prepare sequences
                X_seq, y_seq = self.prepare_sequences(X_train.values, y_train.values, sequence_length)
                X_val_seq, y_val_seq = None, None
                
                if X_val is not None:
                    X_val_seq, y_val_seq = self.prepare_sequences(X_val.values, y_val.values, sequence_length)
                
                lstm_model = self.create_lstm_model((sequence_length, X_train.shape[1]))
                history = lstm_model.fit(
                    X_seq, y_seq,
                    validation_data=(X_val_seq, y_val_seq) if X_val_seq is not None else None,
                    epochs=50,
                    batch_size=32,
                    verbose=0
                )
                self.models['lstm'] = lstm_model
                self.history['lstm'] = history.history
                
                if X_val_seq is not None:
                    y_pred = lstm_model.predict(X_val_seq).flatten()
                    results['lstm'] = self.evaluate_predictions(y_val_seq, y_pred)
                    
            except Exception as e:
                print(f"Error training LSTM: {str(e)}")
        
        return results
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate predictions"""
        if self.task == 'regression':
            return {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            }
        else:
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'classification_report': classification_report(y_true, y_pred, output_dict=True)
            }
    
    def predict(self, X: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """
        Make predictions using trained models
        
        Args:
            X: Features for prediction
            model_name: Specific model to use (if None, uses best model)
        
        Returns:
            Predictions
        """
        if model_name is None:
            # Use the first available model
            model_name = list(self.models.keys())[0]
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if model_name == 'lstm':
            # Prepare sequences for LSTM
            sequence_length = 60  # Default sequence length
            if len(X) < sequence_length:
                raise ValueError(f"Need at least {sequence_length} samples for LSTM prediction")
            
            # Use the last sequence_length samples
            X_seq = X.iloc[-sequence_length:].values.reshape(1, sequence_length, -1)
            return model.predict(X_seq).flatten()
        
        elif model_name == 'dense_nn':
            return model.predict(X).flatten()
        
        else:
            return model.predict(X)
    
    def ensemble_predict(self, X: pd.DataFrame, method: str = 'average') -> np.ndarray:
        """
        Make ensemble predictions
        
        Args:
            X: Features for prediction
            method: Ensemble method ('average', 'weighted', 'voting')
        
        Returns:
            Ensemble predictions
        """
        predictions = {}
        
        for name, model in self.models.items():
            try:
                if name == 'lstm':
                    sequence_length = 60
                    if len(X) >= sequence_length:
                        X_seq = X.iloc[-sequence_length:].values.reshape(1, sequence_length, -1)
                        predictions[name] = model.predict(X_seq).flatten()
                elif name == 'dense_nn':
                    predictions[name] = model.predict(X).flatten()
                else:
                    predictions[name] = model.predict(X)
            except Exception as e:
                print(f"Error making prediction with {name}: {str(e)}")
        
        if method == 'average':
            return np.mean(list(predictions.values()), axis=0)
        elif method == 'weighted':
            # Simple equal weighting
            return np.mean(list(predictions.values()), axis=0)
        elif method == 'voting':
            if self.task == 'classification':
                # Majority vote
                votes = np.array(list(predictions.values()))
                return np.round(np.mean(votes, axis=0))
            else:
                return np.mean(list(predictions.values()), axis=0)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    def get_feature_importance(self, model_name: str = None) -> Dict[str, float]:
        """Get feature importance for a model"""
        if model_name is None:
            # Return importance from the first available model
            for name in self.models.keys():
                if name in self.feature_importance:
                    return self.feature_importance[name]
            return {}
        
        return self.feature_importance.get(model_name, {})
    
    def save_models(self, filepath: str):
        """Save trained models"""
        model_data = {
            'models': self.models,
            'model_type': self.model_type,
            'task': self.task,
            'feature_importance': self.feature_importance,
            'history': self.history
        }
        joblib.dump(model_data, filepath)
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.model_type = model_data['model_type']
        self.task = model_data['task']
        self.feature_importance = model_data['feature_importance']
        self.history = model_data.get('history', {})
        print(f"Models loaded from {filepath}")

class AdvancedStockPredictor(StockPredictor):
    """
    Advanced stock predictor with additional features
    """
    
    def __init__(self, model_type: str = 'ensemble', task: str = 'regression'):
        super().__init__(model_type, task)
        self.ensemble_weights = {}
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                               model_name: str, param_grid: Dict[str, List]) -> Dict[str, Any]:
        """Optimize hyperparameters using grid search"""
        from sklearn.model_selection import GridSearchCV
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Create a new instance of the same model type
        if 'RandomForest' in str(type(model)):
            base_model = RandomForestRegressor() if self.task == 'regression' else RandomForestClassifier()
        elif 'XGB' in str(type(model)):
            base_model = xgb.XGBRegressor() if self.task == 'regression' else xgb.XGBClassifier()
        elif 'LGBM' in str(type(model)):
            base_model = lgb.LGBMRegressor() if self.task == 'regression' else lgb.LGBMClassifier()
        else:
            print(f"Hyperparameter optimization not implemented for {model_name}")
            return {}
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model, param_grid, cv=3, scoring='neg_mean_squared_error' if self.task == 'regression' else 'accuracy'
        )
        grid_search.fit(X_train, y_train)
        
        # Update the model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
    
    def create_ensemble_weights(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """Create ensemble weights based on validation performance"""
        weights = {}
        
        for name, model in self.models.items():
            try:
                y_pred = self.predict(X_val, name)
                if self.task == 'regression':
                    # Use inverse MSE as weight
                    mse = mean_squared_error(y_val, y_pred)
                    weights[name] = 1 / (mse + 1e-8)
                else:
                    # Use accuracy as weight
                    accuracy = accuracy_score(y_val, y_pred)
                    weights[name] = accuracy
            except Exception as e:
                print(f"Error calculating weight for {name}: {str(e)}")
                weights[name] = 0
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        self.ensemble_weights = weights
        return weights
    
    def weighted_ensemble_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make weighted ensemble predictions"""
        if not self.ensemble_weights:
            return self.ensemble_predict(X, method='average')
        
        predictions = {}
        for name, model in self.models.items():
            if name in self.ensemble_weights:
                try:
                    if name == 'lstm':
                        sequence_length = 60
                        if len(X) >= sequence_length:
                            X_seq = X.iloc[-sequence_length:].values.reshape(1, sequence_length, -1)
                            predictions[name] = model.predict(X_seq).flatten()
                    elif name == 'dense_nn':
                        predictions[name] = model.predict(X).flatten()
                    else:
                        predictions[name] = model.predict(X)
                except Exception as e:
                    print(f"Error making prediction with {name}: {str(e)}")
        
        # Calculate weighted average
        weighted_pred = np.zeros(len(X))
        total_weight = 0
        
        for name, pred in predictions.items():
            weight = self.ensemble_weights[name]
            weighted_pred += weight * pred
            total_weight += weight
        
        if total_weight > 0:
            weighted_pred /= total_weight
        
        return weighted_pred

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                    columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.randn(n_samples))
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Initialize predictor
    predictor = AdvancedStockPredictor(model_type='ensemble', task='regression')
    
    # Train models
    results = predictor.train_models(X_train, y_train, X_val, y_val)
    
    print("Training results:")
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics}")
    
    # Make predictions
    predictions = predictor.ensemble_predict(X_val)
    print(f"Ensemble prediction shape: {predictions.shape}")
    
    # Get feature importance
    importance = predictor.get_feature_importance()
    print(f"Feature importance: {len(importance)} features") 