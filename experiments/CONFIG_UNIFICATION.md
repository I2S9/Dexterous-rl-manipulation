# Configuration Unification

## Overview

This document describes the unified experiment configuration system that centralizes all experiment parameters, eliminating hardcoded values and ensuring reproducibility.

## Architecture

The configuration system is organized hierarchically:

```
ExperimentConfig
├── TrainingConfig          # Training parameters
├── CurriculumSchedulerConfig  # Curriculum learning parameters
├── EvaluationConfig        # Evaluation parameters
├── RobustnessConfig       # Robustness test parameters
├── SeedVarianceConfig     # Seed variance analysis parameters
└── ComponentAblationConfig  # Component ablation parameters
```

## Usage

### Loading Configurations

```python
from experiments import load_config, load_named_config, ExperimentConfig

# Load default configuration
config = ExperimentConfig.default()

# Load from JSON file
config = load_config("path/to/config.json")

# Load named configuration
config = load_named_config("default")  # or "quick_test"
```

### Using Configuration in Scripts

```python
from experiments import load_named_config

# Load configuration
exp_config = load_named_config("default")

# Use training parameters
num_episodes = exp_config.training.num_episodes
seeds = exp_config.training.seeds
learning_rate = exp_config.training.learning_rate

# Use evaluation parameters
num_episodes_per_object = exp_config.evaluation.num_episodes_per_object
num_heldout_objects = exp_config.evaluation.num_heldout_objects
```

### Command Line Usage

```bash
# Use default configuration
python evaluation/run_component_ablation.py

# Use named configuration
python evaluation/run_component_ablation.py --config-name quick_test

# Use custom configuration file
python evaluation/run_component_ablation.py --config path/to/config.json
```

## Configuration Files

### Default Configuration (`config_default.json`)

Full configuration with standard parameters for production experiments:
- 200 episodes per training run
- 5 seeds for robust statistics
- Standard curriculum scheduler settings

### Quick Test Configuration (`config_quick_test.json`)

Reduced parameters for fast testing:
- 50 episodes per training run
- 2 seeds
- Reduced evaluation parameters

## Benefits

1. **Reproducibility**: All parameters are explicitly defined and version-controlled
2. **No Hardcoding**: Eliminates magic numbers scattered throughout codebase
3. **Easy Experimentation**: Change parameters by editing JSON files
4. **Consistency**: Same configuration structure across all experiments
5. **Documentation**: Configuration files serve as documentation

## Migration Guide

### Before (Hardcoded)

```python
def train(num_episodes=200, seed=42, learning_rate=0.01):
    # Hardcoded values
    pass
```

### After (Configuration-Based)

```python
from experiments import load_config

def train(exp_config):
    num_episodes = exp_config.training.num_episodes
    seed = exp_config.training.seed
    learning_rate = exp_config.training.learning_rate
    # All values from configuration
    pass
```

## Configuration Structure

### TrainingConfig

- `num_episodes`: Number of training episodes
- `max_episode_steps`: Maximum steps per episode
- `learning_rate`: Policy learning rate
- `seed`: Default random seed
- `seeds`: List of seeds for multi-seed experiments
- `reward_type`: "dense" or "sparse"
- `convergence_window_size`: Window for convergence detection
- `convergence_threshold`: Success rate threshold for convergence

### CurriculumSchedulerConfig

- `success_rate_threshold`: Threshold for curriculum progression
- `window_size`: Window size for success rate calculation
- `min_episodes_before_progression`: Minimum episodes before progression
- `progression_steps`: Number of difficulty levels
- `initial_difficulty`: Starting difficulty ("easy", "medium", "hard")
- `target_difficulty`: Target difficulty

### EvaluationConfig

- `num_episodes_per_object`: Episodes per held-out object
- `num_heldout_objects`: Number of held-out objects
- `max_episode_steps`: Maximum steps per episode
- `seeds`: List of seeds for evaluation
- `reward_type`: Reward type for evaluation

## Best Practices

1. **Always use configuration files** for experiments
2. **Version control configuration files** alongside code
3. **Document custom configurations** with descriptions
4. **Use named configurations** for common setups
5. **Validate configurations** before running experiments

## Future Extensions

The configuration system can be extended with:
- Environment-specific configurations
- Policy architecture configurations
- Hyperparameter search configurations
- Distributed training configurations
