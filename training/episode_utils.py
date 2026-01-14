"""
Episode execution utilities.

This module provides common functions for running episodes
across different training and evaluation contexts.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from envs import DexterousManipulationEnv


def run_episode(
    env: DexterousManipulationEnv,
    policy: Any,
    max_steps: Optional[int] = None,
    reset_policy: bool = True
) -> Tuple[bool, int, float]:
    """
    Run a single episode in the environment.
    
    Args:
        env: Environment to run episode in
        policy: Policy to use for action selection
        max_steps: Maximum steps per episode (uses env default if None)
        reset_policy: Whether to reset policy state at episode start
        
    Returns:
        success: Whether the episode was successful
        steps: Number of steps taken
        total_reward: Total reward accumulated
    """
    obs, info = env.reset()
    
    if reset_policy and hasattr(policy, 'reset'):
        policy.reset()
    
    max_steps = max_steps or env.max_episode_steps
    total_reward = 0.0
    success = False
    
    for step in range(max_steps):
        action = policy.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Update policy if it has an update method
        if hasattr(policy, 'update'):
            policy.update(reward)
        
        if terminated or truncated:
            success = info.get("success", False)
            break
    
    return success, step + 1, total_reward


def run_training_episode(
    env: DexterousManipulationEnv,
    policy: Any,
    max_steps: Optional[int] = None
) -> Dict[str, float]:
    """
    Run a single training episode and return detailed statistics.
    
    Args:
        env: Environment to run episode in
        policy: Policy to use for action selection
        max_steps: Maximum steps per episode (uses env default if None)
        
    Returns:
        Dictionary with episode statistics:
        - success: Whether episode was successful
        - steps: Number of steps taken
        - total_reward: Total reward accumulated
        - mean_reward: Mean reward per step
    """
    success, steps, total_reward = run_episode(env, policy, max_steps=max_steps)
    
    return {
        "success": success,
        "steps": steps,
        "total_reward": total_reward,
        "mean_reward": total_reward / steps if steps > 0 else 0.0,
    }
