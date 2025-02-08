import os
import argparse
from pathlib import Path

def parse_args():
    """Parse command line args with environment variable defaults."""
    parser = argparse.ArgumentParser()
    
    # Use environment variable or fallback to default
    parser.add_argument(
        "--output_root",
        type=Path,
        default=os.environ.get("BERT_OUTPUT_ROOT", "/tmp/bert_output"),
        help="Root directory (env: BERT_OUTPUT_ROOT)"
    )
    
    parser.add_argument(
        "--model_path", 
        type=str,
        default=os.environ.get("BERT_MODEL_PATH", "./bert_encoder"),
        help="Path to BERT model (env: BERT_MODEL_PATH)"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"Output root: {args.output_root}")
    print(f"Model path: {args.model_path}")
