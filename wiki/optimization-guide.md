# Hyperparameter Optimization Guide

## Population Based Training (PBT)

PBT dynamically optimizes hyperparameters during training by maintaining a population of models that train in parallel and learn from each other.

### How It Works

1. **Population Management**
   - Maintains 4 parallel training runs
   - Each model has different hyperparameters
   - Models can copy parameters from better performers

2. **Adaptation Strategies**
   - **Exploit**: Copy parameters from better models (50% chance)
   - **Explore**: Perturb parameters by random factors (0.8-1.2)
   - Triggers when model is in bottom 20% of population

### Optimized Parameters
