"""
Analysis and visualization of robustness test results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List


def analyze_robustness_results(results: Dict) -> Dict:
    """
    Analyze robustness test results and compute performance degradation.
    
    Args:
        results: Dictionary with robustness test results
        
    Returns:
        Dictionary with analysis results
    """
    baseline_metrics = results["baseline"]["metrics"]
    baseline_success = baseline_metrics["grasp_success_rate"]
    
    analysis = {
        "baseline": {
            "success_rate": baseline_success,
            "mean_episode_length": baseline_metrics["mean_episode_length"],
        },
        "observation_noise_degradation": {},
        "dynamics_noise_degradation": {},
        "combined_noise_degradation": {},
    }
    
    # Analyze observation noise
    for noise_level, result in results["observation_noise"].items():
        metrics = result["metrics"]
        success_rate = metrics["grasp_success_rate"]
        degradation = baseline_success - success_rate
        degradation_pct = (degradation / baseline_success * 100) if baseline_success > 0 else 0.0
        
        analysis["observation_noise_degradation"][noise_level] = {
            "success_rate": success_rate,
            "degradation": degradation,
            "degradation_percentage": degradation_pct,
            "mean_episode_length": metrics["mean_episode_length"],
        }
    
    # Analyze dynamics noise
    for noise_level, result in results["dynamics_noise"].items():
        metrics = result["metrics"]
        success_rate = metrics["grasp_success_rate"]
        degradation = baseline_success - success_rate
        degradation_pct = (degradation / baseline_success * 100) if baseline_success > 0 else 0.0
        
        analysis["dynamics_noise_degradation"][noise_level] = {
            "success_rate": success_rate,
            "degradation": degradation,
            "degradation_percentage": degradation_pct,
            "mean_episode_length": metrics["mean_episode_length"],
        }
    
    # Analyze combined noise
    for key, result in results["combined_noise"].items():
        metrics = result["metrics"]
        success_rate = metrics["grasp_success_rate"]
        degradation = baseline_success - success_rate
        degradation_pct = (degradation / baseline_success * 100) if baseline_success > 0 else 0.0
        
        analysis["combined_noise_degradation"][key] = {
            "success_rate": success_rate,
            "degradation": degradation,
            "degradation_percentage": degradation_pct,
            "mean_episode_length": metrics["mean_episode_length"],
        }
    
    return analysis


def plot_robustness_results(results: Dict, analysis: Dict, output_path: str = "logs/robustness_analysis.png"):
    """
    Plot robustness test results.
    
    Args:
        results: Dictionary with robustness test results
        analysis: Dictionary with analysis results
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    baseline_success = analysis["baseline"]["success_rate"]
    
    # Plot 1: Observation noise degradation
    ax1 = axes[0, 0]
    obs_noise_levels = sorted(analysis["observation_noise_degradation"].keys())
    obs_success_rates = [analysis["observation_noise_degradation"][level]["success_rate"] 
                        for level in obs_noise_levels]
    obs_degradations = [analysis["observation_noise_degradation"][level]["degradation_percentage"]
                       for level in obs_noise_levels]
    
    ax1.plot(obs_noise_levels, obs_success_rates, 'o-', label="Success Rate", linewidth=2)
    ax1.axhline(baseline_success, color='r', linestyle='--', label="Baseline", alpha=0.7)
    ax1.set_xlabel("Observation Noise Std")
    ax1.set_ylabel("Success Rate")
    ax1.set_title("Robustness to Observation Noise")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.1])
    
    # Plot 2: Dynamics noise degradation
    ax2 = axes[0, 1]
    dyn_noise_levels = sorted(analysis["dynamics_noise_degradation"].keys())
    dyn_success_rates = [analysis["dynamics_noise_degradation"][level]["success_rate"]
                        for level in dyn_noise_levels]
    
    ax2.plot(dyn_noise_levels, dyn_success_rates, 's-', label="Success Rate", linewidth=2, color='orange')
    ax2.axhline(baseline_success, color='r', linestyle='--', label="Baseline", alpha=0.7)
    ax2.set_xlabel("Dynamics Noise Std")
    ax2.set_ylabel("Success Rate")
    ax2.set_title("Robustness to Dynamics Noise")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.1])
    
    # Plot 3: Degradation percentage
    ax3 = axes[1, 0]
    ax3.plot(obs_noise_levels, obs_degradations, 'o-', label="Observation Noise", linewidth=2)
    dyn_degradations = [analysis["dynamics_noise_degradation"][level]["degradation_percentage"]
                       for level in dyn_noise_levels]
    ax3.plot(dyn_noise_levels, dyn_degradations, 's-', label="Dynamics Noise", linewidth=2, color='orange')
    ax3.set_xlabel("Noise Std")
    ax3.set_ylabel("Performance Degradation (%)")
    ax3.set_title("Performance Degradation vs Noise Level")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Combined noise heatmap (if available)
    ax4 = axes[1, 1]
    if analysis["combined_noise_degradation"]:
        # Extract noise levels
        obs_levels = sorted(set([float(k.split('_')[1]) for k in analysis["combined_noise_degradation"].keys()]))
        dyn_levels = sorted(set([float(k.split('_')[3]) for k in analysis["combined_noise_degradation"].keys()]))
        
        # Create degradation matrix
        degradation_matrix = np.zeros((len(dyn_levels), len(obs_levels)))
        for key, data in analysis["combined_noise_degradation"].items():
            parts = key.split('_')
            obs_idx = obs_levels.index(float(parts[1]))
            dyn_idx = dyn_levels.index(float(parts[3]))
            degradation_matrix[dyn_idx, obs_idx] = data["degradation_percentage"]
        
        im = ax4.imshow(degradation_matrix, aspect='auto', cmap='Reds', origin='lower')
        ax4.set_xticks(range(len(obs_levels)))
        ax4.set_xticklabels([f"{l:.2f}" for l in obs_levels])
        ax4.set_yticks(range(len(dyn_levels)))
        ax4.set_yticklabels([f"{l:.2f}" for l in dyn_levels])
        ax4.set_xlabel("Observation Noise Std")
        ax4.set_ylabel("Dynamics Noise Std")
        ax4.set_title("Combined Noise Degradation (%)")
        plt.colorbar(im, ax=ax4)
    else:
        ax4.text(0.5, 0.5, "No combined noise data", 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title("Combined Noise (Not Available)")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Robustness analysis plot saved to {output_path}")


def print_robustness_report(analysis: Dict):
    """
    Print formatted robustness analysis report.
    
    Args:
        analysis: Dictionary with analysis results
    """
    print("\n" + "=" * 60)
    print("Robustness Analysis Report")
    print("=" * 60)
    
    baseline = analysis["baseline"]
    print(f"\nBaseline Performance (No Noise):")
    print(f"  Success rate: {baseline['success_rate']:.1%}")
    print(f"  Mean episode length: {baseline['mean_episode_length']:.1f} steps")
    
    print(f"\nObservation Noise Degradation:")
    for noise_level in sorted(analysis["observation_noise_degradation"].keys()):
        data = analysis["observation_noise_degradation"][noise_level]
        print(f"  Noise std {noise_level:.3f}:")
        print(f"    Success rate: {data['success_rate']:.1%}")
        print(f"    Degradation: {data['degradation']:.1%} ({data['degradation_percentage']:.1f}%)")
    
    print(f"\nDynamics Noise Degradation:")
    for noise_level in sorted(analysis["dynamics_noise_degradation"].keys()):
        data = analysis["dynamics_noise_degradation"][noise_level]
        print(f"  Noise std {noise_level:.3f}:")
        print(f"    Success rate: {data['success_rate']:.1%}")
        print(f"    Degradation: {data['degradation']:.1%} ({data['degradation_percentage']:.1f}%)")
    
    print("=" * 60)
