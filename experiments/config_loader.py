"""
Configuration loader utilities.

This module provides helper functions to load experiment configurations
from JSON files or create default configurations.
"""

from pathlib import Path
from typing import Optional
from experiments.experiment_config import ExperimentConfig


def load_config(config_path: Optional[str] = None) -> ExperimentConfig:
    """
    Load experiment configuration from file or return default.
    
    Args:
        config_path: Path to JSON configuration file. If None, returns default config.
        
    Returns:
        ExperimentConfig instance
    """
    if config_path is None:
        return ExperimentConfig.default()
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return ExperimentConfig.from_json(str(config_file))


def get_config_path(config_name: str) -> str:
    """
    Get path to a predefined configuration file.
    
    Args:
        config_name: Name of configuration ("default", "quick_test", etc.)
        
    Returns:
        Path to configuration file
    """
    config_dir = Path(__file__).parent
    config_file = config_dir / f"config_{config_name}.json"
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration '{config_name}' not found. "
            f"Expected file: {config_file}"
        )
    
    return str(config_file)


def load_named_config(config_name: str) -> ExperimentConfig:
    """
    Load a named configuration from the experiments directory.
    
    Args:
        config_name: Name of configuration (without "config_" prefix and ".json" suffix)
        
    Returns:
        ExperimentConfig instance
    """
    config_path = get_config_path(config_name)
    return load_config(config_path)
