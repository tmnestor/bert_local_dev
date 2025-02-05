from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse
import torch
import pandas as pd
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score  # Add these imports
import logging
import matplotlib.font_manager as fm
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Suppress matplotlib font manager debug messages
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

from ..config.config import EvaluationConfig, ModelConfig
from ..models.model import BERTClassifier
from ..utils.metrics import calculate_metrics
from ..data_utils import load_and_preprocess_data, create_dataloaders  # Changed import
from ..utils.logging_manager import (
    get_logger,
    setup_logging,
)  # Change from setup_logger
from ..utils.model_loading import safe_load_checkpoint

logger = get_logger(__name__)  # Change to get_logger


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

    def __init__(self, model_path: Path, config: Union[EvaluationConfig, ModelConfig]):
        """Initialize the evaluator.

        Args:
            model_path (Path): Path to the saved model checkpoint.
            config (Union[EvaluationConfig, ModelConfig]): Model configuration.
        """
        self.model_path = Path(model_path)  # Store model_path directly in the instance
        self.config = config
        self.device = torch.device(config.device)
        self.metrics = getattr(config, "metrics", self.DEFAULT_METRICS)
        self.model = self._load_model()

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
            # Remove redundant logging since this is already logged in main()
            checkpoint = torch.load(
                self.config.best_model, map_location=self.config.device
            )

            # Get classifier config and ensure it exists
            if "config" in checkpoint:
                classifier_config = checkpoint["config"]
                if (
                    isinstance(classifier_config, dict)
                    and "classifier_config" in classifier_config
                ):
                    classifier_config = classifier_config["classifier_config"]
            else:
                raise ValueError("Checkpoint missing classifier configuration")

            # Debug log only configuration details, not loading message
            if self.config.verbosity > 1:  # Only show in debug mode
                logger.debug("\nLoaded checkpoint configuration:")
                for k, v in classifier_config.items():
                    logger.debug("  %s: %s", k, str(v))

            # Extract model configuration
            num_classes = checkpoint.get("num_classes", self.config.num_classes)

            # Validate critical parameters
            if num_classes is None:
                raise ValueError("num_classes not found in checkpoint")
            if "hidden_dim" not in classifier_config:
                raise ValueError("hidden_dim not found in classifier_config")

            # Create model with configuration
            self.model = BERTClassifier(
                self.config.bert_model_name, num_classes, classifier_config
            )

            # Load state dict
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                raise ValueError("No state dict found in checkpoint")

            # Verify state dict matches
            current_state = self.model.state_dict()
            current_shapes = {k: v.shape for k, v in current_state.items()}
            checkpoint_shapes = {k: v.shape for k, v in state_dict.items()}

            if current_shapes != checkpoint_shapes:
                logger.error("Model architecture mismatch:")
                for key in set(current_shapes) | set(checkpoint_shapes):
                    if key in current_shapes and key in checkpoint_shapes:
                        if current_shapes[key] != checkpoint_shapes[key]:
                            logger.error(
                                "  %s: checkpoint=%s, current=%s",
                                key,
                                checkpoint_shapes[key],
                                current_shapes[key],
                            )
                    elif key in current_shapes:
                        logger.error(
                            "  %s: missing in checkpoint (current=%s)",
                            key,
                            current_shapes[key],
                        )
                    else:
                        logger.error(
                            "  %s: unexpected in checkpoint (%s)",
                            key,
                            checkpoint_shapes[key],
                        )
                raise ValueError("Model architecture does not match checkpoint")

            # Load the state dict
            self.model.load_state_dict(state_dict)
            self.model.to(self.config.device)
            self.model.eval()

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

    def _plot_error_analysis(
        self, error_df: pd.DataFrame, output_dir: Path
    ) -> None:
        """Generate visualizations for error analysis."""
        # 1. Create word clouds by error type with matched widths
        num_classes = len(error_df['true_label'].unique())
        fig_width = 20  # Fixed figure width
        fig_height = 6 * num_classes  # Height per class
        
        plt.figure(figsize=(fig_width, fig_height))
        
        # Set consistent style for all subplots
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.titlepad': 20,
            'figure.constrained_layout.use': True  # Better spacing
        })
        
        for idx, true_label in enumerate(error_df['true_label'].unique(), 1):
            mask = error_df['true_label'] == true_label
            texts = ' '.join(error_df[mask]['text'])
            
            # Create subplot with specific size
            plt.subplot(num_classes, 1, idx)
            
            # Create and generate wordcloud
            wordcloud = WordCloud(
                width=2000,           # Increased for better resolution
                height=400,           # Fixed height
                background_color='white',
                max_words=100,
                prefer_horizontal=0.7,  # More horizontal words
                min_font_size=8,
                max_font_size=80
            ).generate(texts)
            
            plt.imshow(wordcloud, interpolation='bilinear', aspect='auto')
            plt.axis('off')
            plt.title(f'Misclassified Words for True Label: {true_label}', 
                     pad=20, fontsize=14, fontweight='bold')
        
        # Adjust layout to prevent overlap
        plt.tight_layout(h_pad=3.0)
        plt.savefig(output_dir / 'error_wordclouds.png', 
                    dpi=300,
                    bbox_inches='tight',
                    facecolor='white')
        plt.close()

        # 2. Confidence distribution for errors
        plt.figure(figsize=(10, 6))
        sns.histplot(data=error_df, x='confidence', hue='true_label', bins=20)
        plt.title('Confidence Distribution of Errors by True Label')
        plt.savefig(output_dir / 'error_confidence_dist.png')
        plt.close()

        # 3. Error confusion patterns
        confusion = pd.crosstab(
            error_df['true_label'],
            error_df['predicted_label'],
            values=error_df['confidence'],
            aggfunc='mean'
        )
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion, annot=True, fmt='.2f', cmap='YlOrRd')
        plt.title('Average Confidence of Misclassifications')
        plt.savefig(output_dir / 'error_confusion_patterns.png')
        plt.close()

        # 4. Text length analysis
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=error_df, x='true_label', y='text_length')
        plt.title('Text Length Distribution by True Label')
        plt.xticks(rotation=45)
        plt.savefig(output_dir / 'error_length_dist.png')
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

            filtered_preds = [p for p, m in zip(predictions, high_conf_mask) if m]
            filtered_labels = [l for l, m in zip(labels, high_conf_mask) if m]

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

        error_df = pd.DataFrame(errors)
        error_df = error_df.sort_values("confidence", ascending=False)
        
        # Generate error analysis plots
        self._plot_error_analysis(error_df, self.config.evaluation_dir)
        
        return error_df

    def evaluate(
        self, save_predictions: bool = True, output_dir: Optional[Path] = None
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        """Evaluate model on test dataset."""
        output_dir = output_dir or self.config.evaluation_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load and prepare data first
        test_texts, test_labels, label_encoder = load_and_preprocess_data(
            self.config, validation_mode=True
        )

        # Create dataloader and run evaluation - remove validation_mode parameter
        test_dataloader = create_dataloaders(
            test_texts, test_labels, self.config, self.config.batch_size
        )

        # Run evaluation
        all_preds = []
        all_labels = []
        all_probs = []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"]

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.tolist())
                all_probs.extend(probs.cpu().tolist())

        # Calculate metrics and create results DataFrame
        metrics = calculate_metrics(all_labels, all_preds, self.metrics)
        results_df = pd.DataFrame(
            {
                "text": test_texts,
                "true_label": label_encoder.inverse_transform(all_labels),
                "predicted_label": label_encoder.inverse_transform(all_preds),
                "confidence": [max(probs) for probs in all_probs],
            }
        )

        # Create confusion matrix
        confusion_df = pd.crosstab(
            results_df["true_label"], results_df["predicted_label"], margins=True
        )

        # Show results based on verbosity
        if self.config.verbosity > 0:
            tqdm.write("\nEvaluation Results:")
            tqdm.write("-" * 30)
            for metric, value in metrics.items():
                tqdm.write(f"{metric.capitalize()}: {value:.4f}")

            if self.config.verbosity > 1:  # Show confusion matrix only in debug mode
                tqdm.write("\nConfusion Matrix:")
                tqdm.write("-" * 30)
                tqdm.write("\n" + confusion_df.to_string())

            tqdm.write("\n" + "=" * 50)

        if save_predictions:
            # Save results
            results_df.to_csv(output_dir / "predictions.csv", index=False)
            confusion_df.to_csv(output_dir / "confusion_matrix.csv")
            tqdm.write(f"\nDetailed results saved to: {output_dir}")

            # Plot confusion matrix
            self._plot_confusion_matrix(
                results_df["true_label"], results_df["predicted_label"], output_dir
            )

            # Analyze confidence thresholds
            threshold_results = self.analyze_confidence_thresholds(
                [max(probs) for probs in all_probs],
                all_preds,  # Add predictions parameter
                all_labels,
            )
            pd.DataFrame(threshold_results).T.to_csv(
                output_dir / "confidence_analysis.csv"
            )

            # Analyze errors
            error_analysis = self.analyze_errors(
                test_texts,
                results_df["true_label"],
                results_df["predicted_label"],
                results_df["confidence"],
            )
            error_analysis.to_csv(output_dir / "error_analysis.csv")
            
            # Add error analysis metrics to results
            error_metrics = {
                'error_rate_by_class': (confusion_df.iloc[:-1, :-1].sum(axis=1) / 
                                      confusion_df.iloc[:-1, -1]).to_dict(),
                'avg_confidence_errors': error_analysis['confidence'].mean(),
                'avg_text_length_errors': error_analysis['text_length'].mean(),
            }
            
            # Save error metrics
            pd.DataFrame([error_metrics]).to_csv(
                output_dir / 'error_metrics.csv', index=False
            )

        return metrics, results_df

    @classmethod
    def from_config(cls, config: EvaluationConfig) -> "ModelEvaluator":
        """Create evaluator instance from configuration."""
        model_files = list(config.best_trials_dir.glob("best_model_*.pt"))
        if model_files:
            logger.info("\nFound model files:")
            for f in model_files:
                try:
                    # Use safe loading for scanning
                    checkpoint = safe_load_checkpoint(f, "cpu", strict=False)
                    logger.info(
                        "  %s (score: %.4f)",
                        f.name,
                        checkpoint.get("metric_value", float("nan")),
                    )
                except (RuntimeError, ValueError, FileNotFoundError, KeyError) as e:
                    logger.warning("  Failed to load %s: %s", f.name, e)

        logger.info("\nSelected model: %s", config.best_model)
        return cls(model_path=config.best_model, config=config)

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


def main():
    """Command-line interface entry point for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate trained BERT classifier")
    EvaluationConfig.add_argparse_args(parser)
    args = parser.parse_args()
    config = EvaluationConfig.from_args(args)
    setup_logging(config)  # Initialize logging configuration first

    try:
        best_model = Path(args.best_model)
        if not best_model.is_absolute():
            best_model = config.output_root / best_model

        logger.info(
            f"Loading model from: {best_model}"
        )  # Keep this single loading message
        evaluator = ModelEvaluator(
            model_path=best_model,  # Pass resolved path
            config=config,
        )
        _, _ = evaluator.evaluate(
            save_predictions=True, output_dir=config.evaluation_dir
        )
        logger.info("Evaluation completed successfully")
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
