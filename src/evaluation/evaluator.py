"""Model evaluation and analysis utilities for BERT classifier.

This module provides comprehensive evaluation functionality including:
- Model checkpoint loading and validation
- Multi-metric performance evaluation
- Confusion matrix visualization
- Error analysis with word clouds
- Confidence threshold analysis
- Performance visualization
- Detailed results logging and export

The module generates detailed analysis in both textual and visual formats,
supporting both basic evaluation and in-depth error analysis.

Typical usage:
    ```python
    config = EvaluationConfig.from_args(args)
    evaluator = ModelEvaluator.from_config(config)
    metrics, results = evaluator.evaluate(save_predictions=True)
    ```

Attributes:
    DEFAULT_METRICS (List[str]): Default metrics for evaluation ['accuracy', 'f1', 'precision', 'recall']

Note:
    Requires matplotlib, seaborn, and wordcloud for visualization.
    Expects saved model checkpoints to include both model state and configuration.
"""

import argparse
import logging
import warnings
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.jit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)  # Add back classification_report
from sklearn.model_selection import KFold
from wordcloud import WordCloud

from src.config.configuration import CONFIG, EvaluationConfig, ModelConfig
from src.data_utils import create_dataloaders, load_and_preprocess_data
from src.models.model import BERTClassifier  # Add this import
from src.utils.logging_manager import (
    get_logger,
    setup_logging,
)  # Change from setup_logger
from src.utils.metrics import calculate_metrics
from src.utils.model_loading import (
    safe_load_checkpoint,
    load_model_checkpoint,
)  # Add this import
from ..utils.model_info import display_model_info  # Add import

# Configure matplotlib once at module level
plt.switch_backend("agg")  # Use non-interactive backend
mpl.use("agg")  # Force agg backend
mpl.rcParams["figure.max_open_warning"] = 0  # Disable max figure warning

logger = get_logger(__name__)  # Change to get_logger


@contextmanager
def suppress_evaluation_warnings():
    """Temporarily suppress specific warnings during evaluation."""
    with warnings.catch_warnings():
        # Suppress specific warning types
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        # Suppress specific pandas/numpy warnings
        warnings.filterwarnings("ignore", module="pandas")
        warnings.filterwarnings("ignore", module="numpy")
        # Suppress matplotlib warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
        warnings.filterwarnings(
            "ignore", message=".*weights_only.*"
        )  # Add torch load warning
        # Suppress backend and locator warnings
        mpl.logging.getLogger("matplotlib").setLevel(logging.WARNING)
        mpl.logging.getLogger("matplotlib.ticker").disabled = True
        mpl.logging.getLogger("matplotlib.font_manager").disabled = True
        mpl.logging.getLogger("matplotlib.backends").disabled = True
        yield
        # Re-enable logging after the context
        mpl.logging.getLogger("matplotlib").setLevel(logging.INFO)
        mpl.logging.getLogger("matplotlib.ticker").disabled = False
        mpl.logging.getLogger("matplotlib.font_manager").disabled = False
        mpl.logging.getLogger("matplotlib.backends").disabled = False


class ModelEvaluator:
    """Model evaluation class for BERT classifiers.

    This class handles model loading and evaluation, providing detailed metrics
    and result analysis capabilities.

    Attributes:
        model_path (Path): Path to the saved model checkpoint.
        config (Union[EvaluationConfig, ModelConfig]): Model configuration.
        device (torch.device): Device to run evaluation on.
        metrics (List[str]): Metrics to compute during evaluation.
        model (BERTClassifier): Loaded model instance.
    """

    DEFAULT_METRICS = ["accuracy", "f1", "precision", "recall"]

    def __init__(
        self,
        model_path: Path,
        config: Union[EvaluationConfig, ModelConfig],
        num_classes: int,
        metric_value: float,
        model: BERTClassifier,
    ):
        """Initialize the evaluator.

        Args:
            model_path (Path): Path to the saved model checkpoint.
            config (Union[EvaluationConfig, ModelConfig]): Model configuration.
        """
        self.model_path = Path(model_path)  # Store model_path directly in the instance
        self.config = config
        self.device = torch.device(config.device)
        self.metrics = getattr(config, "metrics", self.DEFAULT_METRICS)
        self.model = model
        self.num_classes = num_classes
        self.metric_value = metric_value

    def _calculate_layer_sizes(
        self, input_size: int, hidden_dims: List[int], num_classes: int
    ) -> List[int]:
        """Calculate the layer sizes from input through hidden layers to output.

        Args:
            input_size: Size of input layer (BERT hidden size)
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes

        Returns:
            List of layer sizes including input and output
        """
        # Simple concatenation of input size, hidden dims, and output size
        return [input_size] + hidden_dims + [num_classes]

    def _load_model(self) -> None:
        """Load trained model from checkpoint."""
        try:
            checkpoint = torch.load(
                self.config.best_model, map_location=self.config.device
            )

            # Create model with configuration from checkpoint
            self.model = BERTClassifier.from_checkpoint(
                checkpoint=checkpoint,
                num_classes=self.config.num_classes,
                device=self.config.device,
            )

            logger.info("Successfully loaded model")
            return self.model

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}") from e

    def _plot_confusion_matrix(
        self, actual_labels: List[str], predicted_labels: List[str], output_dir: Path
    ) -> None:
        """Generate and save confusion matrix visualization."""
        # Use default fonts and style settings
        plt.style.use("default")
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.size": 10,
                "figure.figsize": (12, 8),
                "figure.dpi": 100,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
            }
        )

        plt.figure()
        cm = pd.crosstab(
            pd.Series(actual_labels, name="Actual"),
            pd.Series(predicted_labels, name="Predicted"),
        )

        # Create heatmap with basic text annotations
        ax = sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            annot_kws={"size": 10},
            cbar=True,
            square=True,
        )

        ax.set_title("Confusion Matrix", pad=20)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png")
        plt.close()

    def _plot_error_analysis(self, error_df: pd.DataFrame, output_dir: Path) -> None:
        """Generate visualizations for error analysis."""
        num_classes = len(error_df["true_label"].unique())

        # Convert inches to pixels (assuming 100 DPI)
        inches_width = 7  # Changed from 5 to 7 inches
        inches_height = 3  # Increased height for better proportion
        dpi = 100
        width_pixels = inches_width * dpi
        height_pixels = inches_height * dpi

        plt.figure(figsize=(inches_width, inches_height * num_classes))
        plt.style.use("default")
        plt.rcParams.update(
            {
                "font.size": 12,
                "axes.titlesize": 16,  # Increased from 14
                "axes.titlepad": 20,
                "figure.constrained_layout.use": True,
            }
        )

        for idx, true_label in enumerate(error_df["true_label"].unique(), 1):
            mask = error_df["true_label"] == true_label
            texts = " ".join(error_df[mask]["text"])

            plt.subplot(num_classes, 1, idx)

            wordcloud = WordCloud(
                width=width_pixels,
                height=height_pixels,
                background_color="white",
                max_words=100,
                prefer_horizontal=0.7,
                min_font_size=8,
                max_font_size=100,  # Increased from 80
            ).generate(texts)

            plt.imshow(wordcloud, interpolation="bilinear", aspect="equal")
            plt.axis("off")
            # Make title more prominent
            plt.title(
                f"Misclassified Words for True Label: {true_label}",
                pad=25,  # Increased padding
                fontsize=16,  # Increased font size
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )  # Added background

        plt.tight_layout(h_pad=2.0)  # Increased padding between subplots
        plt.savefig(
            output_dir / "error_wordclouds.png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close()

        # 2. Confidence distribution for errors
        plt.figure(figsize=(10, 6))
        sns.histplot(data=error_df, x="confidence", hue="true_label", bins=20)
        plt.title("Confidence Distribution of Errors by True Label")
        plt.savefig(output_dir / "error_confidence_dist.png")
        plt.close()

        # 3. Error confusion patterns
        confusion = pd.crosstab(
            error_df["true_label"],
            error_df["predicted_label"],
            values=error_df["confidence"],
            aggfunc="mean",
        )
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion, annot=True, fmt=".2f", cmap="YlOrRd")
        plt.title("Average Confidence of Misclassifications")
        plt.savefig(output_dir / "error_confusion_patterns.png")
        plt.close()

        # 4. Text length analysis
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=error_df, x="true_label", y="text_length")
        plt.title("Text Length Distribution by True Label")
        plt.xticks(rotation=45)
        plt.savefig(output_dir / "error_length_dist.png")
        plt.close()

    def _plot_classification_report(
        self, report_df: pd.DataFrame, output_dir: Path
    ) -> None:
        """Generate heatmap visualization of classification report metrics."""
        plt.figure(figsize=(8, 6))

        # Drop support column and last rows (avg rows) for the heatmap
        metrics_df = report_df.drop("support", axis=1).drop(
            ["accuracy", "macro avg", "weighted avg"]
        )

        # Calculate appropriate left margin based on label length
        max_label_length = max(len(str(label)) for label in metrics_df.index)
        left_margin = max(0.2, max_label_length * 0.015)

        # Calculate minimum width needed for cells (4 chars: "0.00")
        num_columns = len(metrics_df.columns)
        min_cell_width = 0.5  # minimum width in inches per cell
        min_figure_width = num_columns * min_cell_width + 2  # add margin space

        # Use the larger of minimum width or default width
        fig_width = max(8, min_figure_width)
        plt.figure(figsize=(fig_width, 6))

        # Adjust subplot parameters
        plt.subplots_adjust(left=left_margin, right=0.95, top=0.90, bottom=0.15)

        # Create heatmap with adjusted layout
        ax = sns.heatmap(
            metrics_df,
            annot=True,
            fmt=".2f",
            cmap="RdYlBu",
            center=0.5,
            vmin=0,
            vmax=1,
            square=False,
            cbar_kws={"label": "Score", "shrink": 0.8},
            annot_kws={
                "size": 6,
                "weight": "light",
                "va": "center",
                "ha": "center",  # Center horizontally
            },
        )

        # Set fixed cell width based on figure size
        ax.set_aspect(
            fig_width / (6 * num_columns)
        )  # Scale aspect ratio to maintain minimum width

        plt.title("Classification Report Heatmap", pad=10)
        plt.xlabel("Metrics")
        plt.ylabel("Classes")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            output_dir / "classification_report_heatmap.png",
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.2,
            facecolor="white",
        )
        plt.close()

    def analyze_confidence_thresholds(
        self, probabilities: List[float], predictions: List[int], labels: List[int]
    ) -> Dict[float, Dict[str, float]]:
        """Analyze model performance across different confidence thresholds."""
        thresholds = np.arange(0.1, 1.0, 0.1)
        results = {}

        for threshold in thresholds:
            # Filter predictions above threshold
            high_conf_mask = np.array(probabilities) >= threshold
            if not any(high_conf_mask):
                continue

            filtered_preds = [
                pred for pred, mask in zip(predictions, high_conf_mask) if mask
            ]
            filtered_labels = [
                label for label, mask in zip(labels, high_conf_mask) if mask
            ]

            # Calculate metrics
            results[threshold] = {
                "accuracy": accuracy_score(filtered_labels, filtered_preds),
                "coverage": sum(high_conf_mask) / len(high_conf_mask),
                "f1": f1_score(filtered_labels, filtered_preds, average="macro"),
            }

        return results

    def analyze_errors(
        self,
        texts: List[str],
        true_labels: List[str],
        pred_labels: List[str],
        confidences: List[float],
    ) -> pd.DataFrame:
        """Analyze misclassified examples."""
        errors = defaultdict(list)

        for text, true, pred, conf in zip(texts, true_labels, pred_labels, confidences):
            if true != pred:
                errors["text"].append(text)
                errors["true_label"].append(true)
                errors["predicted_label"].append(pred)
                errors["confidence"].append(conf)
                # Add text length as potential feature
                errors["text_length"].append(len(text.split()))

        # Create error_df with default empty DataFrame if no errors found
        if not errors:
            logger.info("No errors found in evaluation")
            error_df = pd.DataFrame(
                columns=[
                    "text",
                    "true_label",
                    "predicted_label",
                    "confidence",
                    "text_length",
                ]
            )
        else:
            error_df = pd.DataFrame(errors)
            error_df = error_df.sort_values("confidence", ascending=False)
            # Generate error analysis plots only if there are errors
            self._plot_error_analysis(error_df, self.config.evaluation_dir)

        return error_df

    def evaluate(
        self, save_predictions: bool = True, output_dir: Optional[Path] = None
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        """Evaluate model using k-fold cross validation and generate overall metrics."""
        with suppress_evaluation_warnings():
            output_dir = output_dir or self.config.evaluation_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            # Load all data
            data = load_and_preprocess_data(self.config, validation_mode=True)

            # Initialize k-fold cross validation
            kf = KFold(n_splits=self.config.n_folds, shuffle=True, random_state=42)

            # Store metrics for each fold
            fold_metrics = []
            all_predictions = []

            # Run k-fold cross validation
            for fold, (_, val_idx) in enumerate(kf.split(data.test_texts)):
                logger.info("\nEvaluating fold %d/%d", fold + 1, self.config.n_folds)

                # Split data for this fold
                fold_texts = [data.test_texts[i] for i in val_idx]
                fold_labels = [data.test_labels[i] for i in val_idx]

                # Create dataloader for this fold
                fold_dataloader = create_dataloaders(
                    fold_texts, fold_labels, self.config, self.config.batch_size
                )

                # Run evaluation on this fold
                fold_preds = []
                fold_probs = []

                self.model.eval()
                with torch.no_grad():
                    for batch in fold_dataloader:
                        input_ids = batch["input_ids"]
                        attention_mask = batch["attention_mask"]

                        outputs = self.model(
                            input_ids=input_ids, attention_mask=attention_mask
                        )
                        probs = torch.softmax(outputs, dim=1)
                        preds = torch.argmax(probs, dim=1)

                        fold_preds.extend(preds.cpu().tolist())
                        fold_probs.extend(probs.cpu().tolist())

                # Calculate metrics for this fold
                fold_metrics.append(
                    calculate_metrics(fold_labels, fold_preds, self.metrics)
                )

                # Store predictions
                fold_df = pd.DataFrame(
                    {
                        "text": fold_texts,
                        "true_label": data.label_encoder.inverse_transform(fold_labels),
                        "predicted_label": data.label_encoder.inverse_transform(
                            fold_preds
                        ),
                        "confidence": [max(probs) for probs in fold_probs],
                        "fold": fold + 1,
                    }
                )
                all_predictions.append(fold_df)

                # Save fold-specific results
                if save_predictions:
                    fold_dir = output_dir / f"fold_{fold + 1}"
                    fold_dir.mkdir(exist_ok=True)
                    fold_df.to_csv(fold_dir / "predictions.csv", index=False)

                    # Generate fold-specific visualizations
                    self._plot_confusion_matrix(
                        fold_df["true_label"], fold_df["predicted_label"], fold_dir
                    )

                    self.analyze_errors(
                        fold_texts,
                        fold_df["true_label"],
                        fold_df["predicted_label"],
                        fold_df["confidence"],
                    ).to_csv(fold_dir / "error_analysis.csv")

            # Aggregate results across folds
            results_df = pd.concat(all_predictions, ignore_index=True)

            # Calculate mean and std of metrics across folds
            mean_metrics = {
                metric: np.mean([fold[metric] for fold in fold_metrics])
                for metric in self.metrics
            }
            std_metrics = {
                f"{metric}_std": np.std([fold[metric] for fold in fold_metrics])
                for metric in self.metrics
            }

            # Combine means and stds
            final_metrics = {**mean_metrics, **std_metrics}

            if self.config.verbosity > 0:
                logger.info("\nCross Validation Results:")
                for metric in self.metrics:
                    logger.info(
                        "%s: %.4f ± %.4f",
                        metric,
                        mean_metrics[metric],
                        std_metrics[f"{metric}_std"],
                    )

            if save_predictions:
                # Save aggregated results
                results_df.to_csv(output_dir / "all_predictions.csv", index=False)
                pd.DataFrame([final_metrics]).to_csv(
                    output_dir / "cv_metrics.csv", index=False
                )

                # Generate confusion matrix for entire dataset
                self._plot_confusion_matrix(
                    results_df["true_label"],
                    results_df["predicted_label"],
                    output_dir,  # Save in main output directory
                )

                # Generate classification report for entire dataset
                overall_report = classification_report(
                    results_df["true_label"],
                    results_df["predicted_label"],
                    output_dict=True,
                )
                report_df = pd.DataFrame(overall_report).T

                # Plot and save overall classification report heatmap
                self._plot_classification_report(report_df, output_dir)

                # Save classification report as CSV
                report_df.to_csv(output_dir / "classification_report.csv")

            return final_metrics, results_df

    @classmethod
    def from_config(cls, config: ModelConfig) -> "ModelEvaluator":
        """Create evaluator instance from configuration."""
        # Load model checkpoint
        checkpoint = load_model_checkpoint(
            config.best_model, default_root=config.output_root
        )

        # Create model instance from checkpoint
        model = BERTClassifier.from_checkpoint(
            checkpoint=checkpoint.__dict__,  # Pass the dictionary
            num_classes=checkpoint.num_classes,
            device=config.device,
        )

        logger.info("Successfully loaded model")

        # Log model architecture details
        logger.info("\nModel Architecture:")
        logger.info("=" * 50)
        logger.info("BERT hidden size: %d", model.bert_hidden_size)
        logger.info("Hidden layers: %s", model.classifier_config["hidden_dim"])
        logger.info("Activation: %s", model.classifier_config.get("activation", "gelu"))
        logger.info("Dropout: %.3f", model.classifier_config.get("dropout_rate", 0.1))
        logger.info("=" * 50)

        return cls(
            model_path=config.best_model,
            config=config,
            model=model,
            num_classes=checkpoint.num_classes,
            metric_value=checkpoint.metric_value or float("nan"),  # Access directly
        )

    @classmethod
    def add_model_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add model-specific arguments to argument parser.

        Args:
            parser (argparse.ArgumentParser): Argument parser to extend.
        """
        model_args = parser.add_argument_group("Model Configuration")
        model_args.add_argument(
            "--device",
            type=str,
            default="cpu",
            choices=["cpu", "cuda"],
            help="Device to use for inference",
        )
        model_args.add_argument(
            "--batch_size", type=int, default=32, help="Batch size for evaluation"
        )


def parse_eval_args() -> argparse.Namespace:
    """Parse command-line arguments for model evaluation.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate trained BERT classifier")

    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(CONFIG["output_root"]),
        help="Root directory for all operations",
    )
    parser.add_argument(
        "--best_model",
        type=str,
        required=True,
        help="Model filename (relative to best_trials_dir)",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=CONFIG.get("logging", {}).get("verbosity", 1),
        choices=[0, 1, 2],
        help="Verbosity level",
    )
    parser.add_argument(
        "--n_folds", type=int, default=7, help="Number of cross-validation folds"
    )

    return parser.parse_args()


def main():
    """Command-line interface entry point for model evaluation."""
    args = parse_eval_args()
    config = EvaluationConfig.from_args(args)
    setup_logging(config)

    # Resolve both paths:
    # 1. Model checkpoint path
    config.best_model = config.best_trials_dir / args.best_model
    # 2. BERT encoder path is now handled by ModelConfig's __post_init__

    logger.info("Using BERT encoder from: %s", config.bert_encoder_path)
    logger.info("Loading model checkpoint from: %s", config.best_model)

    # Create evaluator and run evaluation
    evaluator = ModelEvaluator.from_config(config)
    metrics, _ = evaluator.evaluate(save_predictions=True)

    if config.verbosity > 0:
        print("\nEvaluation Results:")
        print("-" * 30)
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")


if __name__ == "__main__":
    main()
