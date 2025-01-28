from pathlib import Path
import pandas as pd
import os
import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

@dataclass
class DataSplit:
    """Container for split data."""
    train_texts: list
    val_texts: list
    test_texts: list
    train_labels: list
    val_labels: list
    test_labels: list
    label_encoder: LabelEncoder
    num_classes: int

class DataSplitter:
    def __init__(self, data_file, val_size=0.2, test_size=0.2, random_state=42):
        """Initialize DataSplitter.
        
        Args:
            data_file: Path to data file or Path object
            val_size: Validation split size (default: 0.2)
            test_size: Test split size (default: 0.2)
            random_state: Random seed (default: 42)
        """
        self.data_file = Path(data_file)
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.split_dir = self.data_file.parent / "splits"
        self.logger = logging.getLogger(__name__)

    def load_splits(self):
        """Load existing splits or create new ones."""
        if not self.split_dir.exists():
            self.split_dir.mkdir(exist_ok=True)
            return self._create_and_save_splits()

        try:
            train_data = pd.read_csv(self.split_dir / "train.csv")
            val_data = pd.read_csv(self.split_dir / "val.csv")
            test_data = pd.read_csv(self.split_dir / "test.csv")
            
            # Validate split sizes
            total = len(train_data) + len(val_data) + len(test_data)
            train_ratio = len(train_data) / total
            val_ratio = len(val_data) / total
            
            if abs(train_ratio - 0.6) > 0.01 or abs(val_ratio - 0.2) > 0.01:
                self.logger.warning("Existing splits have unexpected ratios. Creating new splits.")
                return self._create_and_save_splits()

            # Create label encoder
            label_encoder = LabelEncoder()
            all_labels = pd.concat([train_data['category'], val_data['category'], test_data['category']])
            label_encoder.fit(all_labels)
            
            return DataSplit(
                train_texts=train_data['text'].tolist(),
                val_texts=val_data['text'].tolist(),
                test_texts=test_data['text'].tolist(),
                train_labels=label_encoder.transform(train_data['category']).tolist(),
                val_labels=label_encoder.transform(val_data['category']).tolist(),
                test_labels=label_encoder.transform(test_data['category']).tolist(),
                label_encoder=label_encoder,
                num_classes=len(label_encoder.classes_)
            )
            
        except (FileNotFoundError, pd.errors.EmptyDataError):
            self.logger.info("Creating new data splits")
            return self._create_and_save_splits()

    def _create_and_save_splits(self):
        """Create and save new data splits."""
        data = pd.read_csv(self.data_file)
        
        # Create label encoder
        label_encoder = LabelEncoder()
        data['encoded_category'] = label_encoder.fit_transform(data['category'])
        
        # First split: separate test set
        train_val_data, test_data = train_test_split(
            data, 
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=data['category']
        )
        
        # Second split: separate validation set
        val_size_adjusted = self.val_size / (1 - self.test_size)
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=train_val_data['category']
        )
        
        # Save splits
        train_data.to_csv(self.split_dir / "train.csv", index=False)
        val_data.to_csv(self.split_dir / "val.csv", index=False)
        test_data.to_csv(self.split_dir / "test.csv", index=False)
        
        self.logger.info(f"Created splits: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        return DataSplit(
            train_texts=train_data['text'].tolist(),
            val_texts=val_data['text'].tolist(),
            test_texts=test_data['text'].tolist(),
            train_labels=train_data['encoded_category'].tolist(),
            val_labels=val_data['encoded_category'].tolist(),
            test_labels=test_data['encoded_category'].tolist(),
            label_encoder=label_encoder,
            num_classes=len(label_encoder.classes_)
        )
