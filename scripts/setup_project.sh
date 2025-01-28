#!/bin/bash

# Create base directory
mkdir -p _bert_local_dev
cd _bert_local_dev

# Create main project structure
mkdir -p src/{config,models,training,tuning,utils,evaluation}
mkdir -p logs
mkdir -p data
mkdir -p tests
mkdir -p notebooks

# Create Python package files
touch src/__init__.py
touch src/config/__init__.py
touch src/models/__init__.py
touch src/training/__init__.py
touch src/tuning/__init__.py
touch src/utils/__init__.py
touch src/evaluation/__init__.py

# Create main module files
touch src/config/config.py
touch src/models/model.py
touch src/training/trainer.py
touch src/tuning/optimize.py
touch src/utils/{train_utils.py,logging_manager.py}
touch src/evaluation/metrics.py

# Create test directories
mkdir -p tests/{config,models,training,tuning,utils}
touch tests/__init__.py
touch tests/config/__init__.py
touch tests/models/__init__.py
touch tests/training/__init__.py
touch tests/tuning/__init__.py
touch tests/utils/__init__.py

# Create project files
touch README.md
touch requirements.txt
touch setup.py
touch .gitignore

# Make script executable
chmod +x setup_project.sh

# Print directory structure
echo "Created project structure:"
tree .

echo "Project setup complete!"
