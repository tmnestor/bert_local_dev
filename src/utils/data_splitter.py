import json
import joblib  # Replace pickle with joblib
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ..utils.logging_manager import setup_logger

logger = setup_logger(__name__)

@dataclass
class DataSplit:
    """Data split container for text classification tasks.
    
    Args:
        train_texts (List[str]): Training text samples
        train_labels (List[str]): Training labels
        val_texts (List[str]): Validation text samples
        val_labels (List[str]): Validation labels
        test_texts (Optional[List[str]], optional): Test text samples. Defaults to None.
        test_labels (Optional[List[str]], optional): Test labels. Defaults to None.
        label_encoder (LabelEncoder): Fitted label encoder for categories
        num_classes (int): Number of unique classes in dataset
    """
    train_texts: List[str]
    train_labels: List[str]
    val_texts: List[str]
    val_labels: List[str]
    test_texts: Optional[List[str]] = None
    test_labels: Optional[List[str]] = None
    label_encoder: Optional[LabelEncoder] = None 
    num_classes: Optional[int] = None

    def __post_init__(self):
        """Validate and set num_classes if not provided"""
        if self.label_encoder and not self.num_classes:
            self.num_classes = len(self.label_encoder.classes_)

class DataSplitter:
    """Handles dataset splitting and persistence.

    Creates and manages train/validation/test splits of text classification datasets,
    with support for saving and loading splits to disk.

    Args:
        data_dir: Base directory for data storage.
    """
    
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
        """Create train/val/test splits from data file.
        
        Args:
            data_file: Path to CSV file with 'text' and 'category' columns.
            train_size: Proportion of data for training (default: 0.6).
            val_size: Proportion of data for validation (default: 0.2).
            random_state: Random seed for reproducibility.
            force: If True, recreate splits even if they exist.
            
        Returns:
            DataSplit object containing the splits and metadata.

        Raises:
            FileNotFoundError: If data_file doesn't exist.
            ValueError: If split proportions are invalid.
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
            (self.splits_dir / "label_encoder.joblib").exists()  # Update filename
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
        
        # Save label encoder using joblib instead of pickle
        joblib.dump(splits.label_encoder, self.splits_dir / "label_encoder.joblib")
            
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
        with open(self.splits_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    def _validate_split_files(self) -> bool:
        """Check if all required split files exist and are valid"""
        required_files = {
            'train.csv': False,
            'val.csv': False,
            'test.csv': False,
            'label_encoder.joblib': False,  # Update filename
            'metadata.json': False
        }
        
        for file in self.splits_dir.glob('*'):
            if file.name in required_files:
                required_files[file.name] = True
        
        return all(required_files.values())
    
    def load_splits(self) -> DataSplit:
        """Load existing splits from disk.

        Returns:
            DataSplit object containing the loaded splits and metadata.

        Raises:
            FileNotFoundError: If split files are missing or corrupted.
            ValueError: If loaded splits are inconsistent.
        """
        if not self._validate_split_files():
            raise FileNotFoundError(
                "One or more split files missing. Required files: "
                "train.csv, val.csv, test.csv, label_encoder.joblib, metadata.json"  # Update filename
            )
        
        try:
            # Load label encoder using joblib
            self.label_encoder = joblib.load(self.splits_dir / "label_encoder.joblib")
            
            # Load splits
            splits = {}
            for name in ["train", "val", "test"]:
                df = pd.read_csv(self.splits_dir / f"{name}.csv")
                splits[f"{name}_texts"] = df['text'].tolist()
                splits[f"{name}_labels"] = self.label_encoder.transform(df['category']).tolist()
            
            # Load and verify metadata
            with open(self.splits_dir / "metadata.json", 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
            if len(self.label_encoder.classes_) != metadata['num_classes']:
                raise ValueError("Inconsistent number of classes in saved splits")
            
            return DataSplit(
                train_texts=splits['train_texts'],
                train_labels=splits['train_labels'],
                val_texts=splits['val_texts'],
                val_labels=splits['val_labels'],
                test_texts=splits['test_texts'],
                test_labels=splits['test_labels'],
                label_encoder=self.label_encoder,
                num_classes=len(self.label_encoder.classes_)
            )
            
        except Exception as e:
            raise FileNotFoundError(f"Error loading splits: {str(e)}") from e
