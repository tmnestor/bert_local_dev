from typing import List, Dict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def calculate_metrics(true_labels: List[str], pred_labels: List[str], metrics: List[str]) -> Dict[str, float]:
    """Calculate specified evaluation metrics.
    
    Args:
        true_labels (List[str]): Ground truth labels.
        pred_labels (List[str]): Model predicted labels.
        metrics (List[str]): List of metric names to compute.
    
    Returns:
        Dict[str, float]: Dictionary containing metric name to score mappings.
            Supported metrics include:
            - 'accuracy': Classification accuracy
            - 'f1': Weighted F1 score
            - 'precision': Weighted precision
            - 'recall': Weighted recall
    """
    metric_funcs = {
        'accuracy': accuracy_score,
        'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
        'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
        'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted')
    }
    
    return {
        metric: metric_funcs[metric](true_labels, pred_labels)
        for metric in metrics
        if metric in metric_funcs
    }
