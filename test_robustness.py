"""
Test script for robustness stress tests.

This script validates that robustness tests work correctly and
measures performance degradation under noise.
"""

import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from experiments import CurriculumConfig
from evaluation import RobustnessTester
from evaluation.robustness_analysis import analyze_robustness_results, plot_robustness_results, print_robustness_report
from policies import RandomPolicy
from envs import DexterousManipulationEnv


def test_robustness_system():
    """Test robustness testing system."""
    print("=" * 60)
    print("Robustness Stress Tests")
    print("=" * 60)
    
    # Create evaluation configuration (use hard for more realistic degradation)
    eval_config = CurriculumConfig.hard()
    
    # Create policy
    env = DexterousManipulationEnv(curriculum_config=eval_config)
    policy = RandomPolicy(env.action_space, seed=42)
    env.close()
    
    # Create robustness tester
    tester = RobustnessTester(
        policy=policy,
        eval_config=eval_config,
        reward_type="dense",
        max_episode_steps=200
    )
    
    # Define noise levels to test
    observation_noise_levels = [0.0, 0.01, 0.05, 0.10, 0.20]
    dynamics_noise_levels = [0.0, 0.05, 0.10, 0.20, 0.30]
    
    print(f"\nNoise levels to test:")
    print(f"  Observation noise: {observation_noise_levels}")
    print(f"  Dynamics noise: {dynamics_noise_levels}")
    
    # Run robustness sweep
    print("\n" + "=" * 60)
    print("Running robustness tests...")
    print("=" * 60)
    
    results = tester.run_robustness_sweep(
        observation_noise_levels=observation_noise_levels,
        dynamics_noise_levels=dynamics_noise_levels,
        num_episodes=15,  # Reduced for faster testing
        seed=42
    )
    
    # Analyze results
    print("\n" + "=" * 60)
    print("Analyzing results...")
    print("=" * 60)
    
    analysis = analyze_robustness_results(results)
    
    # Print report
    print_robustness_report(analysis)
    
    # Validate degradation is controlled
    print("\n" + "=" * 60)
    print("Degradation Validation")
    print("=" * 60)
    
    baseline_success = analysis["baseline"]["success_rate"]
    print(f"Baseline success rate: {baseline_success:.1%}")
    
    # Check observation noise degradation
    max_obs_degradation = 0.0
    for noise_level, data in analysis["observation_noise_degradation"].items():
        degradation_pct = data["degradation_percentage"]
        max_obs_degradation = max(max_obs_degradation, degradation_pct)
        print(f"  Observation noise {noise_level:.3f}: {degradation_pct:.1f}% degradation")
    
    # Check dynamics noise degradation
    max_dyn_degradation = 0.0
    for noise_level, data in analysis["dynamics_noise_degradation"].items():
        degradation_pct = data["degradation_percentage"]
        max_dyn_degradation = max(max_dyn_degradation, degradation_pct)
        print(f"  Dynamics noise {noise_level:.3f}: {degradation_pct:.1f}% degradation")
    
    print(f"\nMaximum degradation:")
    print(f"  Observation noise: {max_obs_degradation:.1f}%")
    print(f"  Dynamics noise: {max_dyn_degradation:.1f}%")
    
    # Validate that degradation is reasonable (not catastrophic)
    # For a robust system, degradation should be gradual, not sudden
    print(f"\nValidation:")
    if max_obs_degradation < 50.0 and max_dyn_degradation < 50.0:
        print(f"  [PASS] Degradation is controlled (< 50% for tested noise levels)")
    else:
        print(f"  [WARNING] High degradation detected (> 50%)")
        print(f"           This may indicate sensitivity to noise")
    
    # Save results
    output_dir = Path("logs")
    output_dir.mkdir(exist_ok=True)
    
    # Convert results to JSON-serializable format
    def convert_to_json(obj):
        if isinstance(obj, (np.integer, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_json(item) for item in obj)
        return obj
    
    json_results = {
        "results": convert_to_json(results),
        "analysis": convert_to_json(analysis),
    }
    
    results_path = output_dir / "robustness_results.json"
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Plot results
    plot_path = output_dir / "robustness_analysis.png"
    plot_robustness_results(results, analysis, str(plot_path))
    
    print("\n" + "=" * 60)
    print("Robustness tests completed!")
    print("=" * 60)
    
    return results, analysis


if __name__ == "__main__":
    test_robustness_system()
