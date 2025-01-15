from pathlib import Path
from typing import NamedTuple, List, Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import pickle

from ..utils.logging_manager import setup_logger

logger = setup_logger(__name__)

class DataSplit(NamedTuple):
    """Container for dataset splits with their labels"""
    train_texts: List[str]
    train_labels: List[int]
    val_texts: List[str]
    val_labels: List[int]
    test_texts: List[str]
    test_labels: List[int]
    label_encoder: LabelEncoder
    num_classes: int

class DataSplitter:
    """Handles dataset splitting and persistence"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.splits_dir = self.data_dir / "splits"
        self.splits_dir.mkdir(exist_ok=True, parents=True)
        self.label_encoder = LabelEncoder()
    
    def create_splits(self, 
                     data_file: Path,
                     train_size: float = 0.6,
                     val_size: float = 0.2,
                     random_state: int = 42,
                     force: bool = False) -> DataSplit:
        """
        Create train/val/test splits from data file
        
        Args:
            data_file: Path to CSV file with 'text' and 'category' columns
            train_size: Proportion of data for training (default: 0.6)
            val_size: Proportion of data for validation (default: 0.2)
            random_state: Random seed for reproducibility
            force: If True, recreate splits even if they exist
        """
        if not force and self._splits_exist():
            logger.info("Loading existing splits...")
            return self.load_splits()
            
        logger.info("Creating new data splits...")
        df = pd.read_csv(data_file)
        
        # Fit label encoder on all categories
        self.label_encoder.fit(sorted(df['category'].unique()))
        labels = self.label_encoder.transform(df['category'])
        texts = df['text'].tolist()
        
        # First split: separate test set
        test_size = 1.0 - train_size - val_size
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
        
        # Second split: separate train and validation
        val_ratio = val_size / (train_size + val_size)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts,
            train_val_labels,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val_labels
        )
        
        splits = DataSplit(
            train_texts=train_texts,
            train_labels=train_labels.tolist(),
            val_texts=val_texts,
            val_labels=val_labels.tolist(),
            test_texts=test_texts,
            test_labels=test_labels.tolist(),
            label_encoder=self.label_encoder,
            num_classes=len(self.label_encoder.classes_)
        )
        
        self._save_splits(splits)
        logger.info(f"Created splits: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test")
        return splits
    
    def _splits_exist(self) -> bool:
        """Check if splits already exist"""
        return all((
            (self.splits_dir / "train.csv").exists(),
            (self.splits_dir / "val.csv").exists(),
            (self.splits_dir / "test.csv").exists(),
            (self.splits_dir / "label_encoder.pkl").exists()
        ))
    
    def _save_splits(self, splits: DataSplit) -> None:
        """Save splits to disk"""
        # Save data splits
        for name, texts, labels in [
            ("train", splits.train_texts, splits.train_labels),
            ("val", splits.val_texts, splits.val_labels),
            ("test", splits.test_texts, splits.test_labels)
        ]:
            df = pd.DataFrame({
                'text': texts,
                'category': self.label_encoder.inverse_transform(labels)
            })
            df.to_csv(self.splits_dir / f"{name}.csv", index=False)
        
        # Save label encoder
        with open(self.splits_dir / "label_encoder.pkl", 'wb') as f:
            pickle.dump(splits.label_encoder, f)
            
        # Save metadata
        metadata = {
            'num_classes': splits.num_classes,
            'class_labels': splits.label_encoder.classes_.tolist(),
            'split_sizes': {
                'train': len(splits.train_texts),
                'val': len(splits.val_texts),
                'test': len(splits.test_texts)
            }
        }
        with open(self.splits_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _validate_split_files(self) -> bool:
        """Check if all required split files exist and are valid"""
        required_files = {
            'train.csv': False,
            'val.csv': False,
            'test.csv': False,
            'label_encoder.pkl': False,
            'metadata.json': False
        }
        
        for file in self.splits_dir.glob('*'):
            if file.name in required_files:
                required_files[file.name] = True
        
        return all(required_files.values())
    
    def load_splits(self) -> DataSplit:
        """Load splits from disk with validation"""
        if not self._validate_split_files():
            raise FileNotFoundError(
                "One or more split files missing. Required files: "
                "train.csv, val.csv, test.csv, label_encoder.pkl, metadata.json"
            )
        
        try:
            # Load label encoder
            with open(self.splits_dir / "label_encoder.pkl", 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # Load splits
            splits = {}
            for name in ["train", "val", "test"]:
                df = pd.read_csv(self.splits_dir / f"{name}.csv")
                splits[f"{name}_texts"] = df['text'].tolist()
                splits[f"{name}_labels"] = self.label_encoder.transform(df['category']).tolist()
            
            # Load and verify metadata
            with open(self.splits_dir / "metadata.json", 'r') as f:
                metadata = json.load(f)
                
            if len(self.label_encoder.classes_) != metadata['num_classes']:
                raise ValueError("Inconsistent number of classes in saved splits")
            
            return DataSplit(
                **splits,
                label_encoder=self.label_encoder,
                num_classes=metadata['num_classes']
            )
            
        except Exception as e:
            raise FileNotFoundError(f"Error loading splits: {str(e)}") from e
