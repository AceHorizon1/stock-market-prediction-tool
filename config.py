"""
Configuration management for Stock Market AI
Centralized settings and configuration for the entire application
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import yaml
import logging


@dataclass
class DataConfig:
    """Configuration for data collection and processing"""

    default_symbols: List[str] = field(
        default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    )
    default_period: str = "2y"
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    max_workers: int = 4
    chunk_size: int = 1000


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""

    technical_indicators: bool = True
    statistical_features: bool = True
    market_microstructure: bool = True
    time_features: bool = True
    target_horizons: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    feature_selection_threshold: float = 0.01
    scaling_method: str = "standard"  # 'standard', 'minmax', 'robust'


@dataclass
class ModelConfig:
    """Configuration for machine learning models"""

    default_model_type: str = "ensemble"
    default_task: str = "regression"
    default_horizon: int = 1
    cross_validation_folds: int = 5
    random_state: int = 42
    model_cache_dir: Path = field(default_factory=lambda: Path("models"))


@dataclass
class UIConfig:
    """Configuration for user interfaces"""

    theme: str = "light"  # 'light', 'dark'
    default_port: int = 8501
    debug_mode: bool = False
    max_display_rows: int = 100


@dataclass
class APIConfig:
    """Configuration for external APIs"""

    yahoo_finance_enabled: bool = True
    alpha_vantage_enabled: bool = False
    fred_enabled: bool = False
    news_api_enabled: bool = False

    # API Keys (should be loaded from environment variables)
    alpha_vantage_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ALPHAVANTAGE_API_KEY", "R4Y9XE7WYXU6ZNJZ")
    )
    fred_api_key: Optional[str] = None
    news_api_key: Optional[str] = None


@dataclass
class LoggingConfig:
    """Configuration for logging"""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[Path] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


class AppConfig:
    """Main application configuration"""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config.yaml")
        self.data = DataConfig()
        self.features = FeatureConfig()
        self.model = ModelConfig()
        self.ui = UIConfig()
        self.api = APIConfig()
        self.logging = LoggingConfig()

        # Load configuration from file if it exists
        self._load_config()
        self._setup_logging()
        self._setup_directories()

    def _load_config(self) -> None:
        """Load configuration from YAML file"""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    config_data = yaml.safe_load(f)
                    self._update_from_dict(config_data)
            except Exception as e:
                logging.warning(f"Failed to load config file: {e}")

    def _update_from_dict(self, config_data: Dict[str, Any]) -> None:
        """Update configuration from dictionary"""
        for section_name, section_data in config_data.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.logging.level),
            format=self.logging.format,
            handlers=self._get_log_handlers(),
        )

    def _get_log_handlers(self) -> List[logging.Handler]:
        """Get log handlers based on configuration"""
        handlers = [logging.StreamHandler()]

        if self.logging.file_path:
            from logging.handlers import RotatingFileHandler

            handlers.append(
                RotatingFileHandler(
                    self.logging.file_path,
                    maxBytes=self.logging.max_file_size,
                    backupCount=self.logging.backup_count,
                )
            )

        return handlers

    def _setup_directories(self) -> None:
        """Create necessary directories"""
        directories = [
            self.data.cache_dir,
            self.model.model_cache_dir,
            Path("logs"),
            Path("data"),
            Path("reports"),
        ]

        for directory in directories:
            directory.mkdir(exist_ok=True)

    def save_config(self) -> None:
        """Save current configuration to file"""
        config_data = {
            "data": self.data.__dict__,
            "features": self.features.__dict__,
            "model": self.model.__dict__,
            "ui": self.ui.__dict__,
            "api": self.api.__dict__,
            "logging": self.logging.__dict__,
        }

        with open(self.config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

    def get_cache_path(self, filename: str) -> Path:
        """Get cache file path"""
        return self.data.cache_dir / filename

    def get_model_path(self, model_name: str) -> Path:
        """Get model file path"""
        return self.model.model_cache_dir / f"{model_name}.joblib"


# Global configuration instance
config = AppConfig()
