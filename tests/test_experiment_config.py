"""
Test experiment configuration system.

This script validates that the unified configuration system works correctly
and ensures reproducibility.
"""

import sys
from pathlib import Path
import json
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments import (
    ExperimentConfig,
    TrainingConfig,
    CurriculumSchedulerConfig,
    EvaluationConfig,
    load_config,
    load_named_config,
)


def test_default_config():
    """Test default configuration creation."""
    print("Testing default configuration...")
    
    config = ExperimentConfig.default()
    
    assert config.experiment_name == "default"
    assert config.training.num_episodes == 200
    assert config.training.seed == 42
    assert len(config.training.seeds) > 0
    
    print("  PASS: Default configuration works")


def test_quick_test_config():
    """Test quick test configuration."""
    print("Testing quick test configuration...")
    
    config = ExperimentConfig.quick_test()
    
    assert config.experiment_name == "quick_test"
    assert config.training.num_episodes == 50  # Reduced for quick test
    assert len(config.training.seeds) == 2  # Fewer seeds for quick test
    
    print("  PASS: Quick test configuration works")


def test_config_serialization():
    """Test configuration JSON serialization."""
    print("Testing configuration serialization...")
    
    config = ExperimentConfig.default()
    
    # Convert to dict and back
    config_dict = config.to_dict()
    config_restored = ExperimentConfig.from_dict(config_dict)
    
    assert config_restored.experiment_name == config.experiment_name
    assert config_restored.training.num_episodes == config.training.num_episodes
    assert config_restored.training.seed == config.training.seed
    
    print("  PASS: Configuration serialization works")


def test_config_json_io():
    """Test configuration JSON file I/O."""
    print("Testing configuration JSON I/O...")
    
    config = ExperimentConfig.default()
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        # Save to JSON
        config.to_json(temp_path)
        
        # Load from JSON
        config_loaded = ExperimentConfig.from_json(temp_path)
        
        assert config_loaded.experiment_name == config.experiment_name
        assert config_loaded.training.num_episodes == config.training.num_episodes
        assert config_loaded.training.seed == config.training.seed
        
        print("  PASS: Configuration JSON I/O works")
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_load_named_config():
    """Test loading named configurations."""
    print("Testing named configuration loading...")
    
    try:
        config = load_named_config("default")
        assert config.experiment_name == "default"
        print("  PASS: Named configuration loading works")
    except FileNotFoundError:
        print("  SKIP: Named configuration file not found (expected in some setups)")


def test_config_modification():
    """Test modifying configuration values."""
    print("Testing configuration modification...")
    
    config = ExperimentConfig.default()
    
    # Modify values
    config.training.num_episodes = 100
    config.training.seed = 123
    config.evaluation.num_episodes_per_object = 10
    
    assert config.training.num_episodes == 100
    assert config.training.seed == 123
    assert config.evaluation.num_episodes_per_object == 10
    
    print("  PASS: Configuration modification works")


def test_nested_configs():
    """Test nested configuration structures."""
    print("Testing nested configurations...")
    
    config = ExperimentConfig.default()
    
    # Check nested configs exist
    assert isinstance(config.training, TrainingConfig)
    assert isinstance(config.curriculum_scheduler, CurriculumSchedulerConfig)
    assert isinstance(config.evaluation, EvaluationConfig)
    
    # Check nested config values
    assert config.curriculum_scheduler.success_rate_threshold > 0
    assert config.evaluation.num_heldout_objects > 0
    
    print("  PASS: Nested configurations work")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Experiment Configuration System Tests")
    print("=" * 60)
    
    try:
        test_default_config()
        test_quick_test_config()
        test_config_serialization()
        test_config_json_io()
        test_load_named_config()
        test_config_modification()
        test_nested_configs()
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
