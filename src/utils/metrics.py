from typing import Dict, List, Union, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(
    predictions: Union[List, np.ndarray],
    labels: Union[List, np.ndarray],
    average: str = 'macro'
) -> Dict[str, float]:
    """Compute classification metrics.
    
    Args:
        predictions: Model predictions
        labels: True labels
        average: Averaging strategy for multi-class metrics ('macro', 'micro', 'weighted')
        
    Returns:
        Dictionary containing computed metrics:
            - accuracy: Overall accuracy
            - precision: Precision score
            - recall: Recall score
            - f1: F1 score
    """
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    if isinstance(labels, list):
        labels = np.array(labels)
        
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average=average,
        zero_division=0
    )
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }
