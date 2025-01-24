# main.py to test refactorisation
import argparse
from pathlib import Path
import torch
from src.config.config import ModelConfig
from src.models.model import BERTClassifier
from src.models.encoders import BERTEncoder
from src.models.heads import StandardClassifierHead, PlaneResNetHead
from src.utils.train_utils import load_and_preprocess_data, create_dataloaders
from src.utils.logging_manager import setup_logger

logger = setup_logger(__name__)

def test_classifier(config_path: str = None):
    """Test the refactored BERTClassifier with both head architectures"""
    # Setup basic configuration
    config = ModelConfig(
        bert_model_name='bert-base-uncased',
        num_classes=5,
        batch_size=8,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Load sample data
    logger.info("Loading data...")
    train_texts, val_texts, train_labels, val_labels, label_encoder = load_and_preprocess_data(config)
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        [train_texts, val_texts],
        [train_labels, val_labels],
        config,
        config.batch_size
    )
    
    # Test both architectures
    architectures = {
        'standard': {
            'num_layers': 2,
            'hidden_dim': 256,
            'activation': 'gelu',
            'regularization': 'dropout',
            'dropout_rate': 0.1
        },
        'plane_resnet': {
            'num_planes': 8,
            'plane_width': 256
        }
    }
    
    for arch_name, arch_config in architectures.items():
        logger.info(f"\nTesting {arch_name} architecture...")
        
        # Create encoder
        encoder = BERTEncoder(
            model_name=config.bert_model_name,
            cls_pooling=True
        )
        
        # Create appropriate head
        if arch_name == 'standard':
            head = StandardClassifierHead(
                input_dim=encoder.get_output_dim(),
                num_classes=config.num_classes,
                config=arch_config
            )
        else:
            head = PlaneResNetHead(
                input_dim=encoder.get_output_dim(),  # Changed from input_size
                num_classes=config.num_classes,
                num_planes=arch_config['num_planes'],
                plane_width=arch_config['plane_width']
            )
        
        # Create classifier
        model = BERTClassifier(encoder, head)
        model.to(config.device)
        
        # Test forward pass
        logger.info("Testing forward pass...")
        batch = next(iter(train_dataloader))
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        
        try:
            outputs = model(input_ids, attention_mask)
            logger.info(f"Output shape: {outputs.logits.shape}")
            logger.info(f"Embeddings shape: {outputs.embeddings.shape}")
            logger.info(f"Test successful for {arch_name} architecture")
        except Exception as e:
            logger.error(f"Error testing {arch_name} architecture: {str(e)}")

def parse_args():
    parser = argparse.ArgumentParser(description='Test BERTClassifier architectures')
    parser.add_argument('--config', type=str, help='Path to config file (optional)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    try:
        test_classifier(args.config)
        logger.info("\nAll tests completed successfully!")
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}", exc_info=True)