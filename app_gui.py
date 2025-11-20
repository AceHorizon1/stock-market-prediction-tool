import PySimpleGUI as sg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import io
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Import our custom modules
from data_collector import DataCollector
from feature_engineering import FeatureEngineer
from models import AdvancedStockPredictor
from evaluation import ModelEvaluator

# Set matplotlib to use TkAgg backend
matplotlib.use("TkAgg")

# Configure PySimpleGUI
sg.theme("LightBlue2")
sg.set_options(font=("Arial", 10))


class StockPredictionGUI:
    def __init__(self):
        self.data_collector = DataCollector()
        self.feature_engineer = FeatureEngineer()
        self.predictor = None
        self.evaluator = ModelEvaluator()
        self.data = None
        self.engineered_data = None
        self.model_results = None

    def create_layout(self):
        """Create the main layout with tabs"""

        # Data Loading Tab
        data_tab = [
            [sg.Text("Data Source Selection", font=("Arial", 14, "bold"))],
            [sg.HorizontalSeparator()],
            [
                sg.Radio(
                    "Upload CSV/Excel File",
                    "DATASRC",
                    key="-FILE-",
                    default=True,
                    font=("Arial", 11),
                ),
                sg.Radio(
                    "Fetch Online Data", "DATASRC", key="-ONLINE-", font=("Arial", 11)
                ),
            ],
            [
                sg.Text("File Path:"),
                sg.Input(key="-FILEPATH-", size=(50, 1)),
                sg.FileBrowse(
                    "Browse", file_types=(("CSV/Excel Files", "*.csv;*.xlsx"),)
                ),
            ],
            [
                sg.Text("Stock Symbols (comma separated):"),
                sg.Input(
                    key="-SYMBOLS-", size=(40, 1), default_text="AAPL, MSFT, GOOGL"
                ),
            ],
            [sg.Button("Load Data", size=(15, 1), button_color=("white", "blue"))],
            [sg.HorizontalSeparator()],
            [sg.Text("Data Information", font=("Arial", 12, "bold"))],
            [
                sg.Multiline(
                    size=(80, 10),
                    key="-DATA_INFO-",
                    disabled=True,
                    background_color="lightgray",
                )
            ],
        ]

        # Feature Engineering Tab
        features_tab = [
            [sg.Text("Feature Engineering", font=("Arial", 14, "bold"))],
            [sg.HorizontalSeparator()],
            [
                sg.Button(
                    "Engineer Features", size=(15, 1), button_color=("white", "green")
                )
            ],
            [sg.HorizontalSeparator()],
            [sg.Text("Feature Information", font=("Arial", 12, "bold"))],
            [
                sg.Multiline(
                    size=(80, 8),
                    key="-FEATURE_INFO-",
                    disabled=True,
                    background_color="lightgray",
                )
            ],
            [
                sg.Text(
                    "Sample Engineered Data (First 5 rows):", font=("Arial", 11, "bold")
                )
            ],
            [
                sg.Table(
                    values=[],
                    headings=[],
                    auto_size_columns=True,
                    display_row_numbers=True,
                    num_rows=5,
                    key="-FEATURE_TABLE-",
                    size=(80, 10),
                    background_color="white",
                    text_color="black",
                )
            ],
        ]

        # Model Training Tab
        model_tab = [
            [sg.Text("Model Training", font=("Arial", 14, "bold"))],
            [sg.HorizontalSeparator()],
            [sg.Text("Model Settings:", font=("Arial", 11, "bold"))],
            [
                sg.Text("Model Type:"),
                sg.Combo(
                    ["ensemble", "tree", "linear", "neural", "deep", "hf_transformer"],
                    default_value="ensemble",
                    key="-MODEL_TYPE-",
                    size=(15, 1),
                ),
            ],
            [
                sg.Text("Prediction Task:"),
                sg.Combo(
                    ["regression", "classification"],
                    default_value="regression",
                    key="-TASK-",
                    size=(15, 1),
                ),
            ],
            [
                sg.Text("Target Horizon (days):"),
                sg.Combo(
                    [1, 3, 5, 10, 20], default_value=1, key="-HORIZON-", size=(15, 1)
                ),
            ],
            [sg.Button("Train Model", size=(15, 1), button_color=("white", "orange"))],
            [sg.HorizontalSeparator()],
            [sg.Text("Training Progress:", font=("Arial", 11, "bold"))],
            [
                sg.Multiline(
                    size=(80, 10),
                    key="-TRAINING_INFO-",
                    disabled=True,
                    background_color="lightgray",
                )
            ],
        ]

        # Results Tab
        results_tab = [
            [sg.Text("Model Results & Predictions", font=("Arial", 14, "bold"))],
            [sg.HorizontalSeparator()],
            [sg.Button("Show Results", size=(15, 1), button_color=("white", "purple"))],
            [sg.Button("Generate Plots", size=(15, 1), button_color=("white", "red"))],
            [sg.HorizontalSeparator()],
            [sg.Text("Model Performance:", font=("Arial", 11, "bold"))],
            [
                sg.Multiline(
                    size=(80, 8),
                    key="-RESULTS_INFO-",
                    disabled=True,
                    background_color="lightgray",
                )
            ],
            [sg.Text("Predictions Table:", font=("Arial", 11, "bold"))],
            [
                sg.Table(
                    values=[],
                    headings=[],
                    auto_size_columns=True,
                    display_row_numbers=True,
                    num_rows=10,
                    key="-PREDICTIONS_TABLE-",
                    size=(80, 12),
                    background_color="white",
                    text_color="black",
                )
            ],
            [sg.Canvas(key="-CANVAS-", size=(800, 600))],
        ]

        # Main layout with tabs
        layout = [
            [
                sg.Text(
                    "Stock Market Prediction Tool",
                    font=("Arial", 18, "bold"),
                    justification="center",
                    expand_x=True,
                )
            ],
            [
                sg.TabGroup(
                    [
                        [
                            sg.Tab("Data Loading", data_tab, key="-TAB1-"),
                            sg.Tab("Feature Engineering", features_tab, key="-TAB2-"),
                            sg.Tab("Model Training", model_tab, key="-TAB3-"),
                            sg.Tab("Results", results_tab, key="-TAB4-"),
                        ]
                    ],
                    expand_x=True,
                    expand_y=True,
                )
            ],
            [sg.Button("Exit", size=(10, 1), button_color=("white", "red"))],
        ]

        return layout

    def load_data(self, values):
        """Load data from file or online"""
        try:
            if values["-FILE-"] and values["-FILEPATH-"]:
                # Load from file
                self.data = self.data_collector.load_from_file(values["-FILEPATH-"])
                info_text = f"✅ Data loaded from file: {values['-FILEPATH-']}\n"
                info_text += f"📊 Shape: {self.data.shape}\n"
                info_text += f"📅 Date range: {self.data.index.min()} to {self.data.index.max()}\n"
                info_text += f"📈 Columns: {', '.join(self.data.columns)}\n\n"
                info_text += "Sample data:\n"
                info_text += str(self.data.head())

            elif values["-ONLINE-"] and values["-SYMBOLS-"]:
                # Load from online
                symbols = [
                    s.strip().upper()
                    for s in values["-SYMBOLS-"].split(",")
                    if s.strip()
                ]
                self.data = self.data_collector.create_comprehensive_dataset(
                    symbols, include_market_data=False, include_economic_data=False
                )
                info_text = f"✅ Data fetched online for: {', '.join(symbols)}\n"
                info_text += f"📊 Shape: {self.data.shape}\n"
                info_text += f"📅 Date range: {self.data.index.min()} to {self.data.index.max()}\n"
                info_text += f"📈 Columns: {', '.join(self.data.columns)}\n\n"
                info_text += "Sample data:\n"
                info_text += str(self.data.head())

            else:
                info_text = (
                    "❌ Please select a data source and provide the required input."
                )
                return False, info_text

            return True, info_text

        except Exception as e:
            return False, f"❌ Error loading data: {str(e)}"

    def engineer_features(self):
        """Engineer features from the loaded data"""
        try:
            if self.data is None or self.data.empty:
                return False, "❌ No data loaded. Please load data first."

            # Engineer features
            self.engineered_data = self.feature_engineer.engineer_all_features(
                self.data, target_horizons=[1, 3, 5, 10, 20]
            )

            if self.engineered_data.empty:
                return False, "❌ Feature engineering produced no usable data."

            # Create info text
            info_text = f"✅ Feature engineering completed!\n"
            info_text += f"📊 Original shape: {self.data.shape}\n"
            info_text += f"📊 Engineered shape: {self.engineered_data.shape}\n"
            info_text += f"🔧 Features created: {len(self.engineered_data.columns)}\n"
            info_text += f"📅 Date range: {self.engineered_data.index.min()} to {self.engineered_data.index.max()}\n\n"
            info_text += "Feature categories:\n"

            # Categorize features
            feature_categories = {
                "Technical Indicators": [
                    col
                    for col in self.engineered_data.columns
                    if any(x in col for x in ["RSI", "MACD", "BB", "SMA", "EMA"])
                ],
                "Statistical Features": [
                    col
                    for col in self.engineered_data.columns
                    if any(x in col for x in ["Rolling", "Std", "Mean", "Skew", "Kurt"])
                ],
                "Target Variables": [
                    col for col in self.engineered_data.columns if "Target_" in col
                ],
                "Time Features": [
                    col
                    for col in self.engineered_data.columns
                    if any(x in col for x in ["Day", "Month", "Year", "Quarter"])
                ],
                "Other Features": [
                    col
                    for col in self.engineered_data.columns
                    if not any(
                        x in col
                        for x in [
                            "RSI",
                            "MACD",
                            "BB",
                            "SMA",
                            "EMA",
                            "Rolling",
                            "Std",
                            "Mean",
                            "Skew",
                            "Kurt",
                            "Target_",
                            "Day",
                            "Month",
                            "Year",
                            "Quarter",
                        ]
                    )
                ],
            }

            for category, features in feature_categories.items():
                if features:
                    info_text += f"  • {category}: {len(features)} features\n"

            return True, info_text

        except Exception as e:
            return False, f"❌ Error in feature engineering: {str(e)}"

    def train_model(self, values):
        """Train the prediction model"""
        try:
            if self.engineered_data is None or self.engineered_data.empty:
                return (
                    False,
                    "❌ No engineered data available. Please engineer features first.",
                )

            # Get model parameters
            model_type = values["-MODEL_TYPE-"]
            task = values["-TASK-"]
            target_horizon = values["-HORIZON-"]

            # Prepare data
            target_column = f"Target_Return_{target_horizon}d"

            if target_column not in self.engineered_data.columns:
                return False, f"❌ Target column {target_column} not found in data."

            # Remove rows with NaN targets
            data_clean = self.engineered_data.dropna(subset=[target_column])

            if data_clean.empty:
                return False, "❌ No valid data after removing NaN targets."

            # Initialize predictor
            if model_type == "hf_transformer":
                training_info = (
                    f"🚀 Training Hugging Face Transformer (PatchTST) model...\n"
                )
                training_info += f"📊 Target: {target_column}\n"
                training_info += f"📈 Data shape: {data_clean.shape}\n"
                training_info += (
                    f"⏱️  Training started at: {datetime.now().strftime('%H:%M:%S')}\n"
                )
                training_info += f"⚠️  Note: HF Transformer requires sufficient data (200+ samples)\n\n"
            else:
                training_info = f"🚀 Training {model_type} model for {task} task...\n"
                training_info += f"📊 Target: {target_column}\n"
                training_info += f"📈 Data shape: {data_clean.shape}\n"
                training_info += (
                    f"⏱️  Training started at: {datetime.now().strftime('%H:%M:%S')}\n\n"
                )

            self.predictor = AdvancedStockPredictor(model_type=model_type, task=task)

            # Train the model
            self.model_results = self.predictor.train_model(
                data_clean, target_column, model_type=model_type, task=task
            )

            training_info += (
                f"✅ Training completed at: {datetime.now().strftime('%H:%M:%S')}\n"
            )
            training_info += f"📊 Model performance:\n"

            if "metrics" in self.model_results:
                for metric, value in self.model_results["metrics"].items():
                    training_info += f"  • {metric}: {value:.4f}\n"

            return True, training_info

        except Exception as e:
            return False, f"❌ Error training model: {str(e)}"

    def show_results(self):
        """Display model results and predictions"""
        try:
            if self.model_results is None:
                return (
                    False,
                    "❌ No model results available. Please train a model first.",
                )

            # Create results info
            results_info = "📊 Model Results Summary:\n"
            results_info += "=" * 40 + "\n\n"

            if "metrics" in self.model_results:
                results_info += "Performance Metrics:\n"
                for metric, value in self.model_results["metrics"].items():
                    results_info += f"  • {metric}: {value:.4f}\n"
                results_info += "\n"

            if "predictions" in self.model_results:
                results_info += f"Predictions generated: {len(self.model_results['predictions'])} samples\n"
                results_info += f"Prediction range: {self.model_results['predictions'].min():.4f} to {self.model_results['predictions'].max():.4f}\n"

            return True, results_info

        except Exception as e:
            return False, f"❌ Error showing results: {str(e)}"

    def generate_plots(self, window):
        """Generate and display plots"""
        try:
            if self.model_results is None:
                return (
                    False,
                    "❌ No model results available. Please train a model first.",
                )

            # Clear any existing plots
            window["-CANVAS-"].TKCanvas.delete("all")

            # Create figure with better sizing and spacing
            fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=100)
            fig.suptitle(
                "Stock Market Prediction Results", fontsize=14, fontweight="bold"
            )

            # Plot 1: Actual vs Predicted
            if "y_test" in self.model_results and "predictions" in self.model_results:
                y_test = self.model_results["y_test"]
                predictions = self.model_results["predictions"]

                axes[0, 0].scatter(y_test, predictions, alpha=0.6, s=20)
                # Add perfect prediction line
                min_val = min(y_test.min(), predictions.min())
                max_val = max(y_test.max(), predictions.max())
                axes[0, 0].plot(
                    [min_val, max_val],
                    [min_val, max_val],
                    "r--",
                    lw=2,
                    label="Perfect Prediction",
                )
                axes[0, 0].set_xlabel("Actual Values", fontsize=10)
                axes[0, 0].set_ylabel("Predicted Values", fontsize=10)
                axes[0, 0].set_title(
                    "Actual vs Predicted", fontsize=12, fontweight="bold"
                )
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].legend()

            # Plot 2: Prediction Distribution
            if "predictions" in self.model_results:
                predictions = self.model_results["predictions"]
                axes[0, 1].hist(
                    predictions, bins=30, alpha=0.7, color="skyblue", edgecolor="black"
                )
                axes[0, 1].set_xlabel("Predicted Values", fontsize=10)
                axes[0, 1].set_ylabel("Frequency", fontsize=10)
                axes[0, 1].set_title(
                    "Prediction Distribution", fontsize=12, fontweight="bold"
                )
                axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Feature Importance (if available)
            if (
                "feature_importance" in self.model_results
                and self.model_results["feature_importance"] is not None
            ):
                feature_importance = self.model_results["feature_importance"]
                if len(feature_importance) > 0:
                    top_features = feature_importance.head(10)
                    y_pos = np.arange(len(top_features))
                    axes[1, 0].barh(
                        y_pos,
                        top_features.values,
                        color="lightgreen",
                        edgecolor="black",
                    )
                    axes[1, 0].set_yticks(y_pos)
                    axes[1, 0].set_yticklabels(
                        [
                            f[:20] + "..." if len(f) > 20 else f
                            for f in top_features.index
                        ],
                        fontsize=8,
                    )
                    axes[1, 0].set_xlabel("Importance", fontsize=10)
                    axes[1, 0].set_title(
                        "Top 10 Feature Importance", fontsize=12, fontweight="bold"
                    )
                    axes[1, 0].grid(True, alpha=0.3)
                else:
                    axes[1, 0].text(
                        0.5,
                        0.5,
                        "No feature importance\navailable",
                        ha="center",
                        va="center",
                        transform=axes[1, 0].transAxes,
                    )
                    axes[1, 0].set_title(
                        "Feature Importance", fontsize=12, fontweight="bold"
                    )
            else:
                axes[1, 0].text(
                    0.5,
                    0.5,
                    "No feature importance\navailable",
                    ha="center",
                    va="center",
                    transform=axes[1, 0].transAxes,
                )
                axes[1, 0].set_title(
                    "Feature Importance", fontsize=12, fontweight="bold"
                )

            # Plot 4: Time Series of Predictions
            if "predictions" in self.model_results and self.engineered_data is not None:
                predictions = self.model_results["predictions"]
                test_indices = self.model_results.get(
                    "test_indices", range(len(predictions))
                )

                if len(test_indices) == len(predictions):
                    axes[1, 1].plot(
                        test_indices,
                        predictions,
                        label="Predictions",
                        alpha=0.7,
                        linewidth=1,
                    )
                    if "y_test" in self.model_results:
                        y_test = self.model_results["y_test"]
                        axes[1, 1].plot(
                            test_indices, y_test, label="Actual", alpha=0.7, linewidth=1
                        )
                    axes[1, 1].set_xlabel("Time Index", fontsize=10)
                    axes[1, 1].set_ylabel("Values", fontsize=10)
                    axes[1, 1].set_title(
                        "Predictions Over Time", fontsize=12, fontweight="bold"
                    )
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
                else:
                    axes[1, 1].text(
                        0.5,
                        0.5,
                        "Time series data\nnot available",
                        ha="center",
                        va="center",
                        transform=axes[1, 1].transAxes,
                    )
                    axes[1, 1].set_title("Time Series", fontsize=12, fontweight="bold")
            else:
                axes[1, 1].text(
                    0.5,
                    0.5,
                    "Time series data\nnot available",
                    ha="center",
                    va="center",
                    transform=axes[1, 1].transAxes,
                )
                axes[1, 1].set_title("Time Series", fontsize=12, fontweight="bold")

            # Adjust layout with more space
            plt.tight_layout(pad=3.0)

            # Convert plot to canvas with proper sizing
            canvas = FigureCanvasTkAgg(fig, window["-CANVAS-"].TKCanvas)
            canvas.draw()

            # Get the canvas widget and pack it properly
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(side="top", fill="both", expand=True, padx=10, pady=10)

            return True, "✅ Plots generated successfully!"

        except Exception as e:
            return False, f"❌ Error generating plots: {str(e)}"

    def update_table(self, window, table_key, data, max_rows=10):
        """Update a table with data"""
        if data is None or data.empty:
            window[table_key].update(values=[])
            return

        # Convert DataFrame to list of lists for table
        if isinstance(data, pd.DataFrame):
            table_data = data.head(max_rows).values.tolist()
        else:
            table_data = []

        # Only update values, not headings
        window[table_key].update(values=table_data)

    def run(self):
        """Run the GUI application"""
        layout = self.create_layout()
        window = sg.Window(
            "Stock Market Prediction Tool",
            layout,
            size=(1000, 800),
            resizable=True,
            finalize=True,
        )

        while True:
            event, values = window.read()

            if event == sg.WIN_CLOSED or event == "Exit":
                break

            elif event == "Load Data":
                success, message = self.load_data(values)
                window["-DATA_INFO-"].update(message)
                if success:
                    sg.popup(
                        "Success!",
                        "Data loaded successfully.",
                        auto_close=True,
                        auto_close_duration=2,
                    )

            elif event == "Engineer Features":
                success, message = self.engineer_features()
                window["-FEATURE_INFO-"].update(message)
                if success:
                    # Update feature table
                    self.update_table(
                        window, "-FEATURE_TABLE-", self.engineered_data, max_rows=5
                    )
                    sg.popup(
                        "Success!",
                        "Features engineered successfully.",
                        auto_close=True,
                        auto_close_duration=2,
                    )

            elif event == "Train Model":
                success, message = self.train_model(values)
                window["-TRAINING_INFO-"].update(message)
                if success:
                    sg.popup(
                        "Success!",
                        "Model trained successfully.",
                        auto_close=True,
                        auto_close_duration=2,
                    )

            elif event == "Show Results":
                success, message = self.show_results()
                window["-RESULTS_INFO-"].update(message)
                if success:
                    # Update predictions table
                    if "predictions" in self.model_results:
                        pred_df = pd.DataFrame(
                            {
                                "Predictions": self.model_results["predictions"],
                                "Actual": self.model_results.get(
                                    "y_test",
                                    [None] * len(self.model_results["predictions"]),
                                ),
                            }
                        )
                        self.update_table(
                            window, "-PREDICTIONS_TABLE-", pred_df, max_rows=10
                        )
                    sg.popup(
                        "Success!",
                        "Results displayed successfully.",
                        auto_close=True,
                        auto_close_duration=2,
                    )

            elif event == "Generate Plots":
                success, message = self.generate_plots(window)
                if not success:
                    sg.popup("Error", message)

        window.close()


def main():
    """Main function to run the GUI app"""
    app = StockPredictionGUI()
    app.run()


if __name__ == "__main__":
    main()
