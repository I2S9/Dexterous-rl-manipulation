"""
Unified experiment configuration system.

This module provides a centralized configuration system for all experiments,
eliminating hardcoded values and ensuring reproducibility.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json
from pathlib import Path
import os


@dataclass
class TrainingConfig:
    """Configuration for training experiments."""
    
    # Training parameters
    num_episodes: int = 200
    max_episode_steps: int = 200
    learning_rate: float = 0.01
    
    # Random seed
    seed: int = 42
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1000])
    
    # Reward configuration
    reward_type: str = "dense"  # "dense" or "sparse"
    
    # Environment parameters
    num_fingers: int = 5
    joints_per_finger: int = 3
    
    # Convergence detection
    convergence_window_size: int = 20
    convergence_threshold: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "num_episodes": self.num_episodes,
            "max_episode_steps": self.max_episode_steps,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "seeds": self.seeds,
            "reward_type": self.reward_type,
            "num_fingers": self.num_fingers,
            "joints_per_finger": self.joints_per_finger,
            "convergence_window_size": self.convergence_window_size,
            "convergence_threshold": self.convergence_threshold,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary."""
        return cls(**config_dict)


@dataclass
class CurriculumSchedulerConfig:
    """Configuration for curriculum scheduler."""
    
    # Progression thresholds
    success_rate_threshold: float = 0.3
    window_size: int = 15
    min_episodes_before_progression: int = 20
    progression_steps: int = 5
    
    # Initial and target configurations
    initial_difficulty: str = "easy"  # "easy", "medium", "hard"
    target_difficulty: str = "hard"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success_rate_threshold": self.success_rate_threshold,
            "window_size": self.window_size,
            "min_episodes_before_progression": self.min_episodes_before_progression,
            "progression_steps": self.progression_steps,
            "initial_difficulty": self.initial_difficulty,
            "target_difficulty": self.target_difficulty,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CurriculumSchedulerConfig":
        """Create from dictionary."""
        return cls(**config_dict)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation experiments."""
    
    # Evaluation parameters
    num_episodes_per_object: int = 5
    num_heldout_objects: int = 10
    max_episode_steps: int = 200
    
    # Random seed
    seed: int = 42
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1000])
    
    # Reward type for evaluation
    reward_type: str = "dense"
    
    # Convergence detection
    convergence_window_size: int = 20
    convergence_threshold: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_episodes_per_object": self.num_episodes_per_object,
            "num_heldout_objects": self.num_heldout_objects,
            "max_episode_steps": self.max_episode_steps,
            "seed": self.seed,
            "seeds": self.seeds,
            "reward_type": self.reward_type,
            "convergence_window_size": self.convergence_window_size,
            "convergence_threshold": self.convergence_threshold,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EvaluationConfig":
        """Create from dictionary."""
        return cls(**config_dict)


@dataclass
class RobustnessConfig:
    """Configuration for robustness tests."""
    
    # Noise levels to test
    observation_noise_levels: List[float] = field(default_factory=lambda: [0.0, 0.01, 0.05, 0.1, 0.2])
    dynamics_noise_levels: List[float] = field(default_factory=lambda: [0.0, 0.01, 0.05, 0.1, 0.2])
    
    # Evaluation parameters
    num_episodes_per_noise: int = 10
    max_episode_steps: int = 200
    
    # Random seed
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "observation_noise_levels": self.observation_noise_levels,
            "dynamics_noise_levels": self.dynamics_noise_levels,
            "num_episodes_per_noise": self.num_episodes_per_noise,
            "max_episode_steps": self.max_episode_steps,
            "seed": self.seed,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RobustnessConfig":
        """Create from dictionary."""
        return cls(**config_dict)


@dataclass
class SeedVarianceConfig:
    """Configuration for seed variance analysis."""
    
    # Seeds to test (minimum 3)
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1000, 2024, 3000])
    
    # Evaluation parameters
    num_episodes_per_object: int = 5
    max_episode_steps: int = 200
    
    # Variance threshold
    max_cv_threshold: float = 0.2  # Maximum coefficient of variation
    
    # Reward type
    reward_type: str = "dense"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "seeds": self.seeds,
            "num_episodes_per_object": self.num_episodes_per_object,
            "max_episode_steps": self.max_episode_steps,
            "max_cv_threshold": self.max_cv_threshold,
            "reward_type": self.reward_type,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SeedVarianceConfig":
        """Create from dictionary."""
        return cls(**config_dict)
    
    def validate(self):
        """Validate configuration."""
        if len(self.seeds) < 3:
            raise ValueError(f"Seed variance analysis requires at least 3 seeds, got {len(self.seeds)}")


@dataclass
class ComponentAblationConfig:
    """Configuration for component ablation study."""
    
    # Training parameters
    num_episodes: int = 200
    max_episode_steps: int = 200
    
    # Seeds for multiple runs
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1000])
    
    # Learning rate
    learning_rate: float = 0.01
    
    # Curriculum scheduler config
    curriculum_scheduler: CurriculumSchedulerConfig = field(default_factory=CurriculumSchedulerConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_episodes": self.num_episodes,
            "max_episode_steps": self.max_episode_steps,
            "seeds": self.seeds,
            "learning_rate": self.learning_rate,
            "curriculum_scheduler": self.curriculum_scheduler.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ComponentAblationConfig":
        """Create from dictionary."""
        config_dict = config_dict.copy()
        if "curriculum_scheduler" in config_dict:
            config_dict["curriculum_scheduler"] = CurriculumSchedulerConfig.from_dict(
                config_dict["curriculum_scheduler"]
            )
        return cls(**config_dict)


@dataclass
class ExperimentConfig:
    """
    Unified experiment configuration.
    
    This class centralizes all experiment parameters to ensure
    reproducibility and eliminate hardcoded values.
    """
    
    # Experiment metadata
    experiment_name: str = "default"
    description: str = ""
    
    # Training configuration
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Curriculum scheduler configuration
    curriculum_scheduler: CurriculumSchedulerConfig = field(default_factory=CurriculumSchedulerConfig)
    
    # Evaluation configuration
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Robustness test configuration
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    
    # Seed variance analysis configuration
    seed_variance: SeedVarianceConfig = field(default_factory=SeedVarianceConfig)
    
    # Component ablation configuration
    component_ablation: ComponentAblationConfig = field(default_factory=ComponentAblationConfig)
    
    # Output directory
    output_dir: str = "logs"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment_name": self.experiment_name,
            "description": self.description,
            "training": self.training.to_dict(),
            "curriculum_scheduler": self.curriculum_scheduler.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "robustness": self.robustness.to_dict(),
            "seed_variance": self.seed_variance.to_dict(),
            "component_ablation": self.component_ablation.to_dict(),
            "output_dir": self.output_dir,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        config_dict = config_dict.copy()
        
        # Convert nested configs
        if "training" in config_dict:
            config_dict["training"] = TrainingConfig.from_dict(config_dict["training"])
        if "curriculum_scheduler" in config_dict:
            config_dict["curriculum_scheduler"] = CurriculumSchedulerConfig.from_dict(
                config_dict["curriculum_scheduler"]
            )
        if "evaluation" in config_dict:
            config_dict["evaluation"] = EvaluationConfig.from_dict(config_dict["evaluation"])
        if "robustness" in config_dict:
            config_dict["robustness"] = RobustnessConfig.from_dict(config_dict["robustness"])
        if "seed_variance" in config_dict:
            config_dict["seed_variance"] = SeedVarianceConfig.from_dict(config_dict["seed_variance"])
        if "component_ablation" in config_dict:
            config_dict["component_ablation"] = ComponentAblationConfig.from_dict(
                config_dict["component_ablation"]
            )
        
        return cls(**config_dict)
    
    def to_json(self, json_path: str):
        """
        Save configuration to JSON file.
        
        Args:
            json_path: Path to save JSON file
        """
        os.makedirs(Path(json_path).parent, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, json_path: str) -> "ExperimentConfig":
        """
        Load configuration from JSON file.
        
        Args:
            json_path: Path to JSON configuration file
            
        Returns:
            ExperimentConfig instance
        """
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def default(cls) -> "ExperimentConfig":
        """
        Create default experiment configuration.
        
        Returns:
            Default configuration
        """
        return cls(
            experiment_name="default",
            description="Default experiment configuration"
        )
    
    @classmethod
    def quick_test(cls) -> "ExperimentConfig":
        """
        Create quick test configuration with reduced parameters.
        
        Returns:
            Quick test configuration
        """
        return cls(
            experiment_name="quick_test",
            description="Quick test configuration with reduced parameters",
            training=TrainingConfig(
                num_episodes=50,
                max_episode_steps=100,
                seeds=[42, 123]
            ),
            evaluation=EvaluationConfig(
                num_episodes_per_object=3,
                num_heldout_objects=5,
                seeds=[42, 123]
            ),
            seed_variance=SeedVarianceConfig(
                seeds=[42, 123, 456]
            ),
            component_ablation=ComponentAblationConfig(
                num_episodes=50,
                seeds=[42, 123]
            )
        )
