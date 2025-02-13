# Installation Guide

## Requirements
- Python 3.8+
- CUDA capable GPU (recommended)
- 16GB RAM minimum

## Quick Install
```bash
git clone https://github.com/your-org/bert-classifier
cd bert-classifier
pip install -r requirements.txt
```

## Docker Installation
```bash
docker pull your-org/bert-classifier
docker run -it --gpus all your-org/bert-classifier
```

## Verify Installation
```bash
python -m src.verify_install
```
