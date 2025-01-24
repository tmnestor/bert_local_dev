import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
import logging
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
from transformers import PreTrainedTokenizer

class CrossValidator:
  def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
    """
    Initialize the cross-validator.
    
    Args:
      n_splits: Number of folds for cross-validation
      shuffle: Whether to shuffle data before splitting
      random_state: Random seed for reproducibility
    """
    self.n_splits = n_splits
    self.kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    self.logger = logging.getLogger(__name__)
    
  def validate(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict:
    """
    Perform k-fold cross-validation with progress tracking.
    
    Args:
      model: BERT model with tokenizer attribute
      X: Text data array
      y: Target labels array
      
    Returns:
      Dictionary containing validation results
    """
    results = {
      'fold_scores': [],
      'confusion_matrices': [],
      'class_metrics': [],
      'overall_accuracy': 0.0
    }
    
    self.logger.info(f"Starting {self.n_splits}-fold cross-validation")
    device = next(model.parameters()).device
    
    # Get tokenizer from model
    tokenizer = model.bert.tokenizer if hasattr(model.bert, 'tokenizer') else model.tokenizer
    
    # Initialize progress bar
    fold_iterator = tqdm(enumerate(self.kf.split(X), 1), 
               total=self.n_splits, 
               desc="Cross-validation")
    
    for fold, (train_idx, val_idx) in fold_iterator:
      # Get validation split
      X_val = X[val_idx]
      y_val = y[val_idx]
      
      # Tokenize text data
      encoded = tokenizer(
        list(X_val),
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=512
      ).to(device)
      
      # Convert labels to tensor
      y_val_tensor = torch.tensor(y_val).to(device)
      
      # Evaluate
      model.eval()
      with torch.no_grad():
        outputs = model(
          input_ids=encoded['input_ids'],
          attention_mask=encoded['attention_mask']
        )
        _, y_pred = torch.max(outputs.logits, dim=1)
        y_pred = y_pred.cpu().numpy()
        y_val = y_val_tensor.cpu().numpy()
      
      # Calculate metrics
      fold_cm = confusion_matrix(y_val, y_pred)
      fold_report = classification_report(y_val, y_pred, output_dict=True)
      fold_accuracy = fold_report['accuracy']
      
      # Store results
      results['fold_scores'].append(fold_accuracy)
      results['confusion_matrices'].append(fold_cm)
      results['class_metrics'].append(fold_report)
      
      # Update progress
      fold_iterator.set_postfix({'Accuracy': f'{fold_accuracy:.3f}'})
      
      self.logger.info(f"Fold {fold} completed - Accuracy: {fold_accuracy:.3f}")
    
    # Calculate overall metrics
    results['overall_accuracy'] = np.mean(results['fold_scores'])
    self.logger.info(f"Cross-validation completed - Average accuracy: {results['overall_accuracy']:.3f}")
    
    return results
  
  def get_summary(self, results: Dict) -> Dict:
    """
    Generate a summary of cross-validation results.
    
    Args:
      results: Dictionary containing validation results
      
    Returns:
      Dictionary containing summary statistics
    """
    summary = {
      'mean_accuracy': np.mean(results['fold_scores']),
      'std_accuracy': np.std(results['fold_scores']),
      'fold_scores': results['fold_scores'],
      'per_class_metrics': self._aggregate_class_metrics(results['class_metrics'])
    }
    
    return summary
  
  def _aggregate_class_metrics(self, class_metrics: List[Dict]) -> Dict:
    """
    Aggregate per-class metrics across folds.
    
    Args:
      class_metrics: List of classification reports from each fold
      
    Returns:
      Dictionary containing averaged per-class metrics
    """
    aggregated = {}
    metrics = ['precision', 'recall', 'f1-score']
    
    # Get all classes (excluding 'accuracy', 'macro avg', 'weighted avg')
    classes = [k for k in class_metrics[0].keys() 
          if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    for class_name in classes:
      aggregated[class_name] = {}
      for metric in metrics:
        values = [fold[class_name][metric] for fold in class_metrics]
        aggregated[class_name][metric] = {
          'mean': np.mean(values),
          'std': np.std(values)
        }
        
    return aggregated