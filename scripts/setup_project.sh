#!/bin/bash

# Create root directory
mkdir -p bert_local_dev
cd bert_local_dev

# Create root level directories
mkdir -p src/{config,models,training,tuning,evaluation,utils} tests logs models data evaluation_results

# Create src/config files
touch src/config/__init__.py
touch src/config/base_config.py
touch src/config/config.py

# Create src/models files
touch src/models/__init__.py
touch src/models/model.py
touch src/models/bert_classifier.py

# Create src/training files
touch src/training/__init__.py
touch src/training/train.py
touch src/training/trainer.py
touch src/training/dataset.py
touch src/training/validate.py

# Create src/tuning files
touch src/tuning/__init__.py
touch src/tuning/optimize.py

# Create src/evaluation files
touch src/evaluation/__init__.py
touch src/evaluation/evaluator.py

# Create src/utils files
touch src/utils/__init__.py
touch src/utils/data_splitter.py
touch src/utils/logging_manager.py
touch src/utils/metrics.py
touch src/utils/train_utils.py

# Create root level files
touch README.md
touch LICENSE
touch nlp_env.yml

# Create .gitkeep files for empty directories
touch logs/.gitkeep
touch models/.gitkeep
touch data/.gitkeep
touch evaluation_results/.gitkeep

# Create .gitignore
cat > .gitignore << 'EOL'
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class
**/*.pth
**/*.pt
**/*.safetensors
**/*.png
**/*.json
**/*.log
**/*.ipynb
**/*.pyc

.DS_Store

# Jupyter Notebook checkpoints
**/.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# Keep directories but ignore contents
logs/*
!logs/.gitkeep

monitoring/*
!monitoring/.gitkeep

figures/*
!figures/.gitkeep

# Python artifacts
__pycache__/
*.py[cod]
*$py.class

# Coverage reports
.coverage
coverage.xml
htmlcov/

# pytest
.pytest_cache/

# mypy
.mypy_cache/

# Profiling
*.prof
*.pstats

# Experiment tracking
wandb/
mlruns/
tensorboard/

# Distribution / packaging
dist/
build/
*.egg-info/

# Virtual environments
venv/
env/
.env/

# IDE specific files
.vscode/
.idea/

# BERT MODELS
bert_encoder/
EOL

# Make script executable
chmod +x setup_project.sh

echo "Project structure created successfully!"
