"""
Robustness stress tests for dexterous manipulation.

This module implements stress tests that inject noise into observations
and dynamics to evaluate policy robustness.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from envs import DexterousManipulationEnv
from experiments.config import CurriculumConfig
from evaluation.metrics import EvaluationMetrics, format_metrics_report


class NoisyObservationWrapper:
    """
    Wrapper that adds noise to observations.
    
    Simulates sensor noise and measurement uncertainty.
    """
    
    def __init__(
        self,
        env: DexterousManipulationEnv,
        observation_noise_std: float = 0.0,
        seed: Optional[int] = None
    ):
        """
        Initialize noisy observation wrapper.
        
        Args:
            env: Base environment
            observation_noise_std: Standard deviation of observation noise
            seed: Random seed
        """
        self.env = env
        self.observation_noise_std = observation_noise_std
        self.rng = np.random.default_rng(seed)
        
        # Inherit environment properties
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = env.metadata
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment and add noise to initial observation."""
        obs, info = self.env.reset(seed=seed, options=options)
        noisy_obs = self._add_noise(obs)
        return noisy_obs, info
    
    def step(self, action):
        """Step environment and add noise to observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        noisy_obs = self._add_noise(obs)
        return noisy_obs, reward, terminated, truncated, info
    
    def _add_noise(self, observation: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to observation.
        
        Args:
            observation: Original observation
            
        Returns:
            Noisy observation
        """
        if self.observation_noise_std <= 0.0:
            return observation
        
        noise = self.rng.normal(0, self.observation_noise_std, size=observation.shape)
        noisy_obs = observation + noise.astype(observation.dtype)
        
        return noisy_obs
    
    def close(self):
        """Close environment."""
        self.env.close()


class NoisyDynamicsWrapper:
    """
    Wrapper that adds noise to dynamics.
    
    Simulates actuator noise and model uncertainty.
    """
    
    def __init__(
        self,
        env: DexterousManipulationEnv,
        dynamics_noise_std: float = 0.0,
        seed: Optional[int] = None
    ):
        """
        Initialize noisy dynamics wrapper.
        
        Args:
            env: Base environment
            dynamics_noise_std: Standard deviation of dynamics noise
            seed: Random seed
        """
        self.env = env
        self.dynamics_noise_std = dynamics_noise_std
        self.rng = np.random.default_rng(seed)
        
        # Inherit environment properties
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = env.metadata
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment."""
        return self.env.reset(seed=seed, options=options)
    
    def step(self, action):
        """
        Step environment with noisy dynamics.
        
        Adds noise to actions before applying them.
        """
        if self.dynamics_noise_std > 0.0:
            # Add noise to actions (simulating actuator noise)
            noise = self.rng.normal(0, self.dynamics_noise_std, size=action.shape)
            noisy_action = action + noise.astype(action.dtype)
            # Clip to action space
            noisy_action = np.clip(
                noisy_action,
                self.action_space.low,
                self.action_space.high
            )
        else:
            noisy_action = action
        
        return self.env.step(noisy_action)
    
    def close(self):
        """Close environment."""
        self.env.close()


class CombinedNoiseWrapper:
    """
    Wrapper that adds both observation and dynamics noise.
    """
    
    def __init__(
        self,
        env: DexterousManipulationEnv,
        observation_noise_std: float = 0.0,
        dynamics_noise_std: float = 0.0,
        seed: Optional[int] = None
    ):
        """
        Initialize combined noise wrapper.
        
        Args:
            env: Base environment
            observation_noise_std: Standard deviation of observation noise
            dynamics_noise_std: Standard deviation of dynamics noise
            seed: Random seed
        """
        self.env = env
        self.observation_noise_std = observation_noise_std
        self.dynamics_noise_std = dynamics_noise_std
        self.rng = np.random.default_rng(seed)
        
        # Inherit environment properties
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = env.metadata
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment and add noise to initial observation."""
        obs, info = self.env.reset(seed=seed, options=options)
        noisy_obs = self._add_observation_noise(obs)
        return noisy_obs, info
    
    def step(self, action):
        """Step environment with noisy dynamics and observations."""
        # Add noise to actions
        if self.dynamics_noise_std > 0.0:
            noise = self.rng.normal(0, self.dynamics_noise_std, size=action.shape)
            noisy_action = action + noise.astype(action.dtype)
            noisy_action = np.clip(
                noisy_action,
                self.action_space.low,
                self.action_space.high
            )
        else:
            noisy_action = action
        
        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(noisy_action)
        
        # Add noise to observations
        noisy_obs = self._add_observation_noise(obs)
        
        return noisy_obs, reward, terminated, truncated, info
    
    def _add_observation_noise(self, observation: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to observation."""
        if self.observation_noise_std <= 0.0:
            return observation
        
        noise = self.rng.normal(0, self.observation_noise_std, size=observation.shape)
        noisy_obs = observation + noise.astype(observation.dtype)
        
        return noisy_obs
    
    def close(self):
        """Close environment."""
        self.env.close()


class RobustnessTester:
    """
    Tester for evaluating policy robustness under noise.
    """
    
    def __init__(
        self,
        policy,
        eval_config: CurriculumConfig,
        reward_type: str = "dense",
        max_episode_steps: int = 200
    ):
        """
        Initialize robustness tester.
        
        Args:
            policy: Policy to test
            eval_config: Evaluation configuration
            reward_type: Type of reward
            max_episode_steps: Maximum steps per episode
        """
        self.policy = policy
        self.eval_config = eval_config
        self.reward_type = reward_type
        self.max_episode_steps = max_episode_steps
    
    def evaluate_with_noise(
        self,
        observation_noise_std: float = 0.0,
        dynamics_noise_std: float = 0.0,
        num_episodes: int = 20,
        seed: Optional[int] = None
    ) -> Dict:
        """
        Evaluate policy with specified noise levels.
        
        Args:
            observation_noise_std: Standard deviation of observation noise
            dynamics_noise_std: Standard deviation of dynamics noise
            num_episodes: Number of evaluation episodes
            seed: Random seed
            
        Returns:
            Dictionary with evaluation results and metrics
        """
        # Create environment with noise
        base_env = DexterousManipulationEnv(
            curriculum_config=self.eval_config,
            reward_type=self.reward_type,
            max_episode_steps=self.max_episode_steps
        )
        
        if observation_noise_std > 0.0 or dynamics_noise_std > 0.0:
            env = CombinedNoiseWrapper(
                base_env,
                observation_noise_std=observation_noise_std,
                dynamics_noise_std=dynamics_noise_std,
                seed=seed
            )
        else:
            env = base_env
        
        # Run episodes
        episodes = []
        rng = np.random.default_rng(seed)
        
        for episode in range(num_episodes):
            episode_seed = rng.integers(0, 2**31) if seed is None else seed + episode
            obs, info = env.reset(seed=episode_seed)
            
            if hasattr(self.policy, 'reset'):
                self.policy.reset()
            
            episode_reward = 0.0
            episode_steps = 0
            success = False
            contact_history = []
            
            for step in range(self.max_episode_steps):
                action = self.policy.select_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                
                num_contacts = info.get("num_contacts", 0)
                contact_history.append([1.0 if i < num_contacts else 0.0 for i in range(5)])
                
                if terminated or truncated:
                    success = terminated
                    break
            
            episodes.append({
                "success": success,
                "episode_steps": episode_steps,
                "num_contacts": info.get("num_contacts", 0),
                "final_contacts": info.get("num_contacts", 0),
                "contact_history": contact_history,
                "episode_reward": float(episode_reward),
            })
        
        env.close()
        
        # Compute metrics
        metrics_calc = EvaluationMetrics(success_threshold=3)
        metrics = metrics_calc.compute_aggregate_metrics(episodes, max_steps=self.max_episode_steps)
        
        return {
            "episodes": episodes,
            "metrics": metrics,
            "noise_levels": {
                "observation_noise_std": observation_noise_std,
                "dynamics_noise_std": dynamics_noise_std,
            },
        }
    
    def run_robustness_sweep(
        self,
        observation_noise_levels: List[float],
        dynamics_noise_levels: List[float],
        num_episodes: int = 20,
        seed: Optional[int] = None
    ) -> Dict:
        """
        Run robustness tests across multiple noise levels.
        
        Args:
            observation_noise_levels: List of observation noise standard deviations
            dynamics_noise_levels: List of dynamics noise standard deviations
            num_episodes: Number of episodes per noise level
            seed: Random seed
            
        Returns:
            Dictionary with results for all noise levels
        """
        results = {}
        
        # Baseline (no noise)
        print("Running baseline evaluation (no noise)...")
        baseline = self.evaluate_with_noise(
            observation_noise_std=0.0,
            dynamics_noise_std=0.0,
            num_episodes=num_episodes,
            seed=seed
        )
        results["baseline"] = baseline
        
        # Test observation noise
        print("\nTesting observation noise...")
        obs_noise_results = {}
        for obs_noise in observation_noise_levels:
            if obs_noise > 0.0:
                print(f"  Observation noise std: {obs_noise:.3f}")
                result = self.evaluate_with_noise(
                    observation_noise_std=obs_noise,
                    dynamics_noise_std=0.0,
                    num_episodes=num_episodes,
                    seed=seed
                )
                obs_noise_results[obs_noise] = result
        results["observation_noise"] = obs_noise_results
        
        # Test dynamics noise
        print("\nTesting dynamics noise...")
        dyn_noise_results = {}
        for dyn_noise in dynamics_noise_levels:
            if dyn_noise > 0.0:
                print(f"  Dynamics noise std: {dyn_noise:.3f}")
                result = self.evaluate_with_noise(
                    observation_noise_std=0.0,
                    dynamics_noise_std=dyn_noise,
                    num_episodes=num_episodes,
                    seed=seed
                )
                dyn_noise_results[dyn_noise] = result
        results["dynamics_noise"] = dyn_noise_results
        
        # Test combined noise
        print("\nTesting combined noise...")
        combined_results = {}
        for obs_noise in observation_noise_levels[:3]:  # Limit combinations
            for dyn_noise in dynamics_noise_levels[:3]:
                if obs_noise > 0.0 or dyn_noise > 0.0:
                    print(f"  Combined: obs={obs_noise:.3f}, dyn={dyn_noise:.3f}")
                    result = self.evaluate_with_noise(
                        observation_noise_std=obs_noise,
                        dynamics_noise_std=dyn_noise,
                        num_episodes=num_episodes,
                        seed=seed
                    )
                    combined_results[f"obs_{obs_noise:.3f}_dyn_{dyn_noise:.3f}"] = result
        results["combined_noise"] = combined_results
        
        return results
