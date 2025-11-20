"""
Hugging Face Transformer Models Integration for Stock Market Prediction
Integrates state-of-the-art transformer models from Hugging Face for time series forecasting
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
import torch
from transformers import (
    PatchTSTConfig,
    PatchTSTForPrediction,
    AutoformerConfig,
    AutoformerForPrediction,
    TimeSeriesTransformerConfig,
    TimeSeriesTransformerForPrediction,
    Trainer,
    TrainingArguments
)
from transformers.utils import logging
import warnings
warnings.filterwarnings('ignore')
logging.set_verbosity_error()

class HFTransformerPredictor:
    """
    Hugging Face Transformer-based stock market predictor
    Supports PatchTST, Autoformer, and TimeSeriesTransformer models
    """
    
    def __init__(
        self,
        model_name: str = "PatchTST",
        model_id: Optional[str] = None,
        prediction_length: int = 1,
        context_length: int = 96,
        num_parallel_samples: int = 100,
        device: Optional[str] = None
    ):
        """
        Initialize the Hugging Face transformer predictor
        
        Args:
            model_name: Name of the model ('PatchTST', 'Autoformer', 'TimeSeriesTransformer')
            model_id: Hugging Face model ID (e.g., 'ibm/PatchesTST-finetuned-exchange_rate')
            prediction_length: Number of time steps to predict
            context_length: Number of historical time steps to use as context
            num_parallel_samples: Number of parallel samples for probabilistic forecasting
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.model_id = model_id
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.num_parallel_samples = num_parallel_samples
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.config = None
        self.scaler = None
        self.feature_columns = None
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the transformer model"""
        try:
            if self.model_name == "PatchTST":
                self._initialize_patchtst()
            elif self.model_name == "Autoformer":
                self._initialize_autoformer()
            elif self.model_name == "TimeSeriesTransformer":
                self._initialize_timeseries_transformer()
            else:
                raise ValueError(f"Unknown model name: {self.model_name}")
            
            self.model = self.model.to(self.device)
            print(f"✅ Initialized {self.model_name} model on {self.device}")
            
        except Exception as e:
            print(f"⚠️ Error initializing model: {e}")
            print("Falling back to default PatchTST configuration...")
            self._initialize_patchtst_fallback()
    
    def _initialize_patchtst(self):
        """Initialize PatchTST model"""
        if self.model_id:
            try:
                self.model = PatchTSTForPrediction.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32
                )
                self.config = self.model.config
                return
            except Exception as e:
                print(f"⚠️ Could not load pretrained model {self.model_id}: {e}")
                print("Using default configuration...")
        
        # Default PatchTST configuration
        self.config = PatchTSTConfig(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            num_input_channels=1,  # Will be updated based on data
            patch_len=16,
            stride=8,
            d_model=128,
            n_heads=8,
            num_layers=3,
            dropout=0.1,
            num_parallel_samples=self.num_parallel_samples
        )
        self.model = PatchTSTForPrediction(self.config)
    
    def _initialize_patchtst_fallback(self):
        """Fallback PatchTST initialization"""
        self.config = PatchTSTConfig(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            num_input_channels=1,
            patch_len=16,
            stride=8,
            d_model=128,
            n_heads=8,
            num_layers=3,
            dropout=0.1
        )
        self.model = PatchTSTForPrediction(self.config)
    
    def _initialize_autoformer(self):
        """Initialize Autoformer model"""
        if self.model_id:
            try:
                self.model = AutoformerForPrediction.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32
                )
                self.config = self.model.config
                return
            except Exception as e:
                print(f"⚠️ Could not load pretrained model {self.model_id}: {e}")
        
        self.config = AutoformerConfig(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            num_input_channels=1,
            d_model=128,
            n_heads=8,
            num_layers=3,
            dropout=0.1
        )
        self.model = AutoformerForPrediction(self.config)
    
    def _initialize_timeseries_transformer(self):
        """Initialize TimeSeriesTransformer model"""
        if self.model_id:
            try:
                self.model = TimeSeriesTransformerForPrediction.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32
                )
                self.config = self.model.config
                return
            except Exception as e:
                print(f"⚠️ Could not load pretrained model {self.model_id}: {e}")
        
        self.config = TimeSeriesTransformerConfig(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            num_input_channels=1,
            d_model=128,
            n_heads=8,
            num_layers=3,
            dropout=0.1
        )
        self.model = TimeSeriesTransformerForPrediction(self.config)
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str = "Close",
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Prepare data for transformer model
        
        Args:
            data: DataFrame with time series data
            target_column: Name of the target column to predict
            feature_columns: List of feature columns to use (if None, uses all numeric columns)
        
        Returns:
            Tuple of (past_values, future_values, feature_names)
        """
        # Select feature columns
        if feature_columns is None:
            # Use numeric columns, excluding target
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numeric_cols:
                numeric_cols.remove(target_column)
            feature_columns = numeric_cols[:10]  # Limit to 10 features for simplicity
        
        self.feature_columns = feature_columns
        
        # Prepare data
        df = data[feature_columns + [target_column]].copy()
        df = df.dropna()
        
        if len(df) < self.context_length + self.prediction_length:
            raise ValueError(
                f"Need at least {self.context_length + self.prediction_length} samples, "
                f"got {len(df)}"
            )
        
        # Normalize data
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(df.values)
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(self.context_length, len(scaled_data) - self.prediction_length + 1):
            past = scaled_data[i - self.context_length:i]
            future = scaled_data[i:i + self.prediction_length, -1]  # Target is last column
            sequences.append(past)
            targets.append(future)
        
        past_values = torch.tensor(np.array(sequences), dtype=torch.float32)
        future_values = torch.tensor(np.array(targets), dtype=torch.float32)
        
        return past_values, future_values, feature_columns
    
    def train(
        self,
        data: pd.DataFrame,
        target_column: str = "Close",
        feature_columns: Optional[List[str]] = None,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        train_split: float = 0.8
    ) -> Dict[str, Any]:
        """
        Train the transformer model
        
        Args:
            data: Training data
            target_column: Target column name
            feature_columns: Feature columns to use
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            train_split: Train/validation split ratio
        
        Returns:
            Training history dictionary
        """
        print(f"🔄 Preparing data for {self.model_name}...")
        
        # Prepare data
        past_values, future_values, feature_cols = self.prepare_data(
            data, target_column, feature_columns
        )
        
        # Update model config if needed
        num_features = past_values.shape[-1]
        if hasattr(self.config, 'num_input_channels') and self.config.num_input_channels != num_features:
            print(f"⚠️ Updating model config: num_input_channels from {self.config.num_input_channels} to {num_features}")
            # Note: This requires recreating the model, which is complex
            # For now, we'll use the first feature only if mismatch occurs
            if num_features > 1:
                print("⚠️ Using first feature only due to model configuration")
                past_values = past_values[:, :, 0:1]
                num_features = 1
        
        # Split data
        split_idx = int(len(past_values) * train_split)
        train_past = past_values[:split_idx]
        train_future = future_values[:split_idx]
        val_past = past_values[split_idx:]
        val_future = future_values[split_idx:]
        
        print(f"📊 Training samples: {len(train_past)}, Validation samples: {len(val_past)}")
        
        # Simple training loop (more reliable than Trainer for time series)
        print(f"🚀 Training {self.model_name} model...")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        self.model.train()
        for epoch in range(epochs):
            # Training
            train_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(train_past), batch_size):
                batch_past = train_past[i:i+batch_size].to(self.device)
                batch_future = train_future[i:i+batch_size].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(past_values=batch_past)
                predictions = outputs.prediction_outputs
                
                # Calculate loss
                loss = criterion(predictions, batch_future.unsqueeze(-1))
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = train_loss / num_batches if num_batches > 0 else 0
            train_losses.append(avg_train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for i in range(0, len(val_past), batch_size):
                    batch_past = val_past[i:i+batch_size].to(self.device)
                    batch_future = val_future[i:i+batch_size].to(self.device)
                    
                    outputs = self.model(past_values=batch_past)
                    predictions = outputs.prediction_outputs
                    
                    loss = criterion(predictions, batch_future.unsqueeze(-1))
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
            val_losses.append(avg_val_loss)
            
            self.model.train()
            
            if (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"   Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        print(f"✅ Training completed!")
        
        return {
            "train_loss": train_losses[-1] if train_losses else 0,
            "eval_loss": val_losses[-1] if val_losses else 0,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "epochs": epochs
        }
    
    def predict(
        self,
        data: pd.DataFrame,
        target_column: str = "Close",
        feature_columns: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            data: Input data for prediction
            target_column: Target column name (for data preparation)
            feature_columns: Feature columns to use
        
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare data
        past_values, _, _ = self.prepare_data(data, target_column, feature_columns)
        
        # Use only the last sequence
        if len(past_values) > 0:
            last_sequence = past_values[-1:].to(self.device)
        else:
            raise ValueError("No valid sequences found in data")
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(past_values=last_sequence)
            predictions = outputs.prediction_outputs.cpu().numpy()
        
        # Inverse transform predictions
        if self.scaler is not None:
            # Create dummy array for inverse transform
            dummy = np.zeros((len(predictions[0]), len(self.feature_columns) + 1))
            dummy[:, -1] = predictions[0].flatten()
            predictions = self.scaler.inverse_transform(dummy)[:, -1]
        else:
            predictions = predictions[0].flatten()
        
        return predictions
    
    def save_model(self, path: str):
        """Save the trained model"""
        if self.model is not None:
            self.model.save_pretrained(path)
            print(f"✅ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        if self.model_name == "PatchTST":
            self.model = PatchTSTForPrediction.from_pretrained(path)
        elif self.model_name == "Autoformer":
            self.model = AutoformerForPrediction.from_pretrained(path)
        elif self.model_name == "TimeSeriesTransformer":
            self.model = TimeSeriesTransformerForPrediction.from_pretrained(path)
        
        self.model = self.model.to(self.device)
        print(f"✅ Model loaded from {path}")


class TimeSeriesDataset(torch.utils.data.Dataset):
    """Dataset class for time series data"""
    
    def __init__(self, past_values: torch.Tensor, future_values: torch.Tensor):
        self.past_values = past_values
        self.future_values = future_values
    
    def __len__(self):
        return len(self.past_values)
    
    def __getitem__(self, idx):
        return {
            "past_values": self.past_values[idx],
            "future_values": self.future_values[idx]
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    past_values = torch.stack([item["past_values"] for item in batch])
    future_values = torch.stack([item["future_values"] for item in batch])
    return {
        "past_values": past_values,
        "future_values": future_values
    }


# Integration with existing StockPredictor class
class HFTransformerStockPredictor:
    """
    Wrapper class that integrates HF transformers with existing StockPredictor interface
    """
    
    def __init__(
        self,
        model_name: str = "PatchTST",
        prediction_length: int = 1,
        context_length: int = 96
    ):
        self.predictor = HFTransformerPredictor(
            model_name=model_name,
            prediction_length=prediction_length,
            context_length=context_length
        )
        self.is_trained = False
    
    def train_model(
        self,
        data: pd.DataFrame,
        target_column: str,
        epochs: int = 10
    ) -> Dict[str, Any]:
        """Train the model (compatible with existing interface)"""
        results = self.predictor.train(
            data=data,
            target_column=target_column,
            epochs=epochs
        )
        self.is_trained = True
        return results
    
    def predict(self, X: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """Make predictions (compatible with existing interface)"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Use the last column as target (will be ignored in prediction)
        target_col = X.columns[-1] if len(X.columns) > 0 else "Close"
        
        predictions = self.predictor.predict(
            data=X,
            target_column=target_col
        )
        
        # Repeat prediction for all rows if needed
        if len(predictions) == 1 and len(X) > 1:
            predictions = np.repeat(predictions, len(X))
        
        return predictions[:len(X)]

