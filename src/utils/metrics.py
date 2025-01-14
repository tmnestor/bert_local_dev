from typing import List, Dict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def calculate_metrics(true_labels: List[str], pred_labels: List[str], metrics: List[str]) -> Dict[str, float]:
    """Calculate specified metrics"""
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
