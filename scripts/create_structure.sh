#!/bin/bash

# Create base src directory
mkdir -p src/{config,models,training,tuning,evaluation,utils,data_utils}

# Create __init__.py files in each directory
touch src/__init__.py
touch src/config/__init__.py
touch src/models/__init__.py
touch src/training/__init__.py
touch src/tuning/__init__.py 
touch src/evaluation/__init__.py
touch src/utils/__init__.py
touch src/data_utils/__init__.py

# Create source files in each directory
# Config
touch src/config/{base_config,config}.py

# Models
touch src/models/model.py

# Training
touch src/training/{train,trainer,dataset,validate}.py

# Tuning
touch src/tuning/optimize.py

# Evaluation  
touch src/evaluation/evaluator.py

# Utils
touch src/utils/{data_splitter,logging_manager,metrics,train_utils}.py

# Data Utils
touch src/data_utils/{dataset,splitter,loaders,validation}.py

echo "Directory structure created successfully"
