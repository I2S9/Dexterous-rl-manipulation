"""
Simple learning policy for demonstration purposes.

This module provides a simplified learning mechanism to demonstrate
training effects without requiring a full RL implementation.
"""

import numpy as np
from typing import Optional
from gymnasium import spaces


class SimpleLearner:
    """
    Simple learning policy that improves over time.
    
    This is a simplified learning mechanism to demonstrate
    training effects without full RL implementation.
    
    The policy maintains a mean action vector that gets updated
    based on received rewards, with exploration noise added.
    """
    
    def __init__(
        self,
        action_space: spaces.Box,
        learning_rate: float = 0.01,
        exploration_noise: float = 0.3,
        action_clip_range: float = 0.5
    ):
        """
        Initialize simple learner.
        
        Args:
            action_space: Gymnasium action space
            learning_rate: Learning rate for policy updates
            exploration_noise: Standard deviation of exploration noise
            action_clip_range: Range for clipping mean action updates
        """
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.exploration_noise = exploration_noise
        self.action_clip_range = action_clip_range
        
        # Simple policy: mean action that gets updated
        self.mean_action = np.zeros(action_space.shape[0], dtype=np.float32)
        self.best_reward = -np.inf
    
    def select_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Select action with exploration.
        
        Args:
            observation: Current observation (not used, but kept for interface consistency)
            
        Returns:
            action: Action with exploration noise
        """
        # Add exploration noise
        noise = np.random.normal(
            0,
            self.exploration_noise,
            size=self.mean_action.shape
        ).astype(np.float32)
        
        action = self.mean_action + noise
        
        # Clip to action space bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        return action
    
    def update(self, reward: float):
        """
        Update policy based on reward.
        
        Simple update: if reward improved, slightly adjust mean action.
        
        Args:
            reward: Reward received
        """
        if reward > self.best_reward:
            # Small random adjustment towards better performance
            adjustment = np.random.normal(
                0,
                self.learning_rate,
                size=self.mean_action.shape
            )
            self.mean_action += adjustment
            self.mean_action = np.clip(
                self.mean_action,
                -self.action_clip_range,
                self.action_clip_range
            )
            self.best_reward = reward
    
    def reset(self):
        """Reset policy state between episodes."""
        self.best_reward = -np.inf
