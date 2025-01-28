#!/bin/bash

# Create base src directory structure
mkdir -p src/{config,models,training,tuning,evaluation,utils,data_utils}

# Create directories for data and results
mkdir -p {data,logs,best_trials,evaluation_results}/

# Create __init__.py files in each directory
find src -type d -exec touch {}/__init__.py \;

# Config files
touch src/config/{base_config,config,defaults}.py

# Models
touch src/models/model.py

# Training
touch src/training/{train,trainer,validate}.py

# Tuning 
touch src/tuning/optimize.py

# Evaluation
touch src/evaluation/evaluator.py

# Utils
touch src/utils/{train_utils,logging_manager,metrics,model_loading}.py

# Data Utils
touch src/data_utils/{loaders,dataset,splitter,validation}.py

# Add .gitkeep files to empty dirs to track in git
touch {logs,best_trials,evaluation_results}/.gitkeep

# Create placeholder for data
touch data/.gitkeep

echo "Directory structure created successfully"

# Make script executable
chmod +x "$0"
