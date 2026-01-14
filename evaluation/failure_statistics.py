"""
Statistical analysis and visualization of failure modes.

This module provides tools to quantify, analyze, and visualize
failure modes to identify improvement opportunities.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from evaluation.failure_taxonomy import FailureMode
from evaluation.failure_logger import FailureLogger


def load_failure_logs(filepath: str) -> List[Dict]:
    """
    Load failure episodes from JSON file.
    
    Args:
        filepath: Path to failure log JSON file
        
    Returns:
        List of failure episode dictionaries
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data.get("episodes", [])


def compute_failure_distribution(episodes: List[Dict]) -> Dict[str, Dict]:
    """
    Compute distribution statistics for failure modes.
    
    Args:
        episodes: List of failure episode dictionaries
        
    Returns:
        Dictionary with distribution statistics
    """
    if not episodes:
        return {}
    
    # Count failures by mode
    mode_counts = defaultdict(int)
    mode_episodes = defaultdict(list)
    
    for episode in episodes:
        mode = episode.get("failure_mode")
        if mode:
            mode_counts[mode] += 1
            mode_episodes[mode].append(episode)
    
    total = len(episodes)
    
    # Compute statistics per mode
    distribution = {}
    for mode, count in mode_counts.items():
        mode_eps = mode_episodes[mode]
        
        # Episode length statistics
        episode_lengths = [ep["episode_steps"] for ep in mode_eps]
        
        # Contact statistics
        final_contacts = [ep.get("final_contacts", ep.get("num_contacts", 0)) for ep in mode_eps]
        
        # Object property statistics
        object_sizes = [ep["object_properties"]["size"] for ep in mode_eps]
        object_masses = [ep["object_properties"]["mass"] for ep in mode_eps]
        friction_coeffs = [ep["object_properties"]["friction_coefficient"] for ep in mode_eps]
        
        distribution[mode] = {
            "count": count,
            "frequency": count / total if total > 0 else 0.0,
            "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "std_episode_length": float(np.std(episode_lengths)) if episode_lengths else 0.0,
            "mean_final_contacts": float(np.mean(final_contacts)) if final_contacts else 0.0,
            "mean_object_size": float(np.mean(object_sizes)) if object_sizes else 0.0,
            "std_object_size": float(np.std(object_sizes)) if object_sizes else 0.0,
            "mean_object_mass": float(np.mean(object_masses)) if object_masses else 0.0,
            "std_object_mass": float(np.std(object_masses)) if object_masses else 0.0,
            "mean_friction": float(np.mean(friction_coeffs)) if friction_coeffs else 0.0,
            "std_friction": float(np.std(friction_coeffs)) if friction_coeffs else 0.0,
        }
    
    return distribution


def compute_correlations(episodes: List[Dict]) -> Dict[str, Dict]:
    """
    Compute simple correlations between object properties and failure modes.
    
    Args:
        episodes: List of failure episode dictionaries
        
    Returns:
        Dictionary with correlation analysis
    """
    if not episodes:
        return {}
    
    # Group by failure mode
    mode_data = defaultdict(lambda: {
        "sizes": [],
        "masses": [],
        "frictions": [],
        "episode_lengths": [],
        "contacts": [],
    })
    
    for episode in episodes:
        mode = episode.get("failure_mode")
        if mode:
            props = episode["object_properties"]
            mode_data[mode]["sizes"].append(props["size"])
            mode_data[mode]["masses"].append(props["mass"])
            mode_data[mode]["frictions"].append(props["friction_coefficient"])
            mode_data[mode]["episode_lengths"].append(episode["episode_steps"])
            mode_data[mode]["contacts"].append(episode.get("final_contacts", episode.get("num_contacts", 0)))
    
    # Compute correlations
    correlations = {}
    for mode, data in mode_data.items():
        if len(data["sizes"]) < 2:
            continue
        
        correlations[mode] = {
            "object_size": {
                "mean": float(np.mean(data["sizes"])),
                "std": float(np.std(data["sizes"])),
                "min": float(np.min(data["sizes"])),
                "max": float(np.max(data["sizes"])),
            },
            "object_mass": {
                "mean": float(np.mean(data["masses"])),
                "std": float(np.std(data["masses"])),
                "min": float(np.min(data["masses"])),
                "max": float(np.max(data["masses"])),
            },
            "friction": {
                "mean": float(np.mean(data["frictions"])),
                "std": float(np.std(data["frictions"])),
                "min": float(np.min(data["frictions"])),
                "max": float(np.max(data["frictions"])),
            },
            "episode_length": {
                "mean": float(np.mean(data["episode_lengths"])),
                "std": float(np.std(data["episode_lengths"])),
            },
            "contacts": {
                "mean": float(np.mean(data["contacts"])),
                "std": float(np.std(data["contacts"])),
            },
        }
    
    return correlations


def plot_failure_distribution(
    episodes: List[Dict],
    output_path: str = "logs/failure_distribution.png"
) -> None:
    """
    Plot failure mode distribution.
    
    Args:
        episodes: List of failure episode dictionaries
        output_path: Path to save the plot
    """
    distribution = compute_failure_distribution(episodes)
    
    if not distribution:
        print("No failure episodes to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    modes = list(distribution.keys())
    counts = [distribution[mode]["count"] for mode in modes]
    frequencies = [distribution[mode]["frequency"] for mode in modes]
    
    # Plot 1: Pie chart of failure distribution
    ax1 = axes[0, 0]
    ax1.pie(counts, labels=modes, autopct='%1.1f%%', startangle=90)
    ax1.set_title("Failure Mode Distribution (Pie Chart)")
    
    # Plot 2: Bar chart of failure counts
    ax2 = axes[0, 1]
    bars = ax2.bar(modes, counts, color='steelblue', alpha=0.7)
    ax2.set_xlabel("Failure Mode")
    ax2.set_ylabel("Count")
    ax2.set_title("Failure Mode Distribution (Bar Chart)")
    ax2.tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # Plot 3: Episode length by failure mode
    ax3 = axes[1, 0]
    episode_lengths = [distribution[mode]["mean_episode_length"] for mode in modes]
    episode_stds = [distribution[mode]["std_episode_length"] for mode in modes]
    ax3.bar(modes, episode_lengths, yerr=episode_stds, color='coral', alpha=0.7, capsize=5)
    ax3.set_xlabel("Failure Mode")
    ax3.set_ylabel("Mean Episode Length (steps)")
    ax3.set_title("Episode Length by Failure Mode")
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Final contacts by failure mode
    ax4 = axes[1, 1]
    final_contacts = [distribution[mode]["mean_final_contacts"] for mode in modes]
    ax4.bar(modes, final_contacts, color='lightgreen', alpha=0.7)
    ax4.set_xlabel("Failure Mode")
    ax4.set_ylabel("Mean Final Contacts")
    ax4.set_title("Final Contacts by Failure Mode")
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_ylim([0, 5])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Failure distribution plot saved to {output_path}")


def plot_failure_correlations(
    episodes: List[Dict],
    output_path: str = "logs/failure_correlations.png"
) -> None:
    """
    Plot correlations between object properties and failure modes.
    
    Args:
        episodes: List of failure episode dictionaries
        output_path: Path to save the plot
    """
    correlations = compute_correlations(episodes)
    
    if not correlations:
        print("No correlation data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    modes = list(correlations.keys())
    
    # Plot 1: Object size by failure mode
    ax1 = axes[0, 0]
    sizes = [correlations[mode]["object_size"]["mean"] for mode in modes]
    size_stds = [correlations[mode]["object_size"]["std"] for mode in modes]
    ax1.bar(modes, sizes, yerr=size_stds, color='skyblue', alpha=0.7, capsize=5)
    ax1.set_xlabel("Failure Mode")
    ax1.set_ylabel("Mean Object Size (m)")
    ax1.set_title("Object Size vs Failure Mode")
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Object mass by failure mode
    ax2 = axes[0, 1]
    masses = [correlations[mode]["object_mass"]["mean"] for mode in modes]
    mass_stds = [correlations[mode]["object_mass"]["std"] for mode in modes]
    ax2.bar(modes, masses, yerr=mass_stds, color='orange', alpha=0.7, capsize=5)
    ax2.set_xlabel("Failure Mode")
    ax2.set_ylabel("Mean Object Mass (kg)")
    ax2.set_title("Object Mass vs Failure Mode")
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Friction by failure mode
    ax3 = axes[1, 0]
    frictions = [correlations[mode]["friction"]["mean"] for mode in modes]
    friction_stds = [correlations[mode]["friction"]["std"] for mode in modes]
    ax3.bar(modes, frictions, yerr=friction_stds, color='lightcoral', alpha=0.7, capsize=5)
    ax3.set_xlabel("Failure Mode")
    ax3.set_ylabel("Mean Friction Coefficient")
    ax3.set_title("Friction vs Failure Mode")
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Scatter plot: size vs mass colored by failure mode
    ax4 = axes[1, 1]
    colors = plt.cm.Set3(np.linspace(0, 1, len(modes)))
    for i, mode in enumerate(modes):
        mode_episodes = [ep for ep in episodes if ep.get("failure_mode") == mode]
        sizes = [ep["object_properties"]["size"] for ep in mode_episodes]
        masses = [ep["object_properties"]["mass"] for ep in mode_episodes]
        ax4.scatter(sizes, masses, label=mode, alpha=0.6, color=colors[i], s=50)
    
    ax4.set_xlabel("Object Size (m)")
    ax4.set_ylabel("Object Mass (kg)")
    ax4.set_title("Object Properties by Failure Mode")
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Failure correlations plot saved to {output_path}")


def generate_failure_report(
    episodes: List[Dict],
    output_dir: str = "logs"
) -> Dict:
    """
    Generate comprehensive failure analysis report.
    
    Args:
        episodes: List of failure episode dictionaries
        output_dir: Directory to save reports and plots
        
    Returns:
        Dictionary with analysis results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Failure Mode Statistical Analysis")
    print("=" * 60)
    
    if not episodes:
        print("No failure episodes to analyze")
        return {}
    
    # Compute distribution
    print("\n1. Computing failure distribution...")
    distribution = compute_failure_distribution(episodes)
    
    print(f"\n   Total failure episodes: {len(episodes)}")
    print(f"   Failure modes identified: {len(distribution)}")
    
    # Print distribution summary
    print("\n2. Failure Mode Distribution:")
    sorted_modes = sorted(distribution.items(), key=lambda x: x[1]["count"], reverse=True)
    for mode, stats in sorted_modes:
        print(f"   {mode}:")
        print(f"     Count: {stats['count']} ({stats['frequency']:.1%})")
        print(f"     Mean episode length: {stats['mean_episode_length']:.1f} ± {stats['std_episode_length']:.1f}")
        print(f"     Mean final contacts: {stats['mean_final_contacts']:.2f}")
    
    # Compute correlations
    print("\n3. Computing correlations...")
    correlations = compute_correlations(episodes)
    
    print("\n4. Object Property Correlations:")
    for mode, corr_data in correlations.items():
        print(f"   {mode}:")
        print(f"     Object size: {corr_data['object_size']['mean']:.4f} ± {corr_data['object_size']['std']:.4f}")
        print(f"     Object mass: {corr_data['object_mass']['mean']:.4f} ± {corr_data['object_mass']['std']:.4f}")
        print(f"     Friction: {corr_data['friction']['mean']:.3f} ± {corr_data['friction']['std']:.3f}")
    
    # Generate plots
    print("\n5. Generating visualizations...")
    plot_failure_distribution(episodes, str(output_path / "failure_distribution.png"))
    plot_failure_correlations(episodes, str(output_path / "failure_correlations.png"))
    
    # Save analysis results
    analysis_results = {
        "total_episodes": len(episodes),
        "distribution": distribution,
        "correlations": correlations,
    }
    
    results_path = output_path / "failure_analysis.json"
    with open(results_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\n6. Analysis results saved to {results_path}")
    
    # Print insights
    print("\n" + "=" * 60)
    print("Key Insights")
    print("=" * 60)
    
    if distribution:
        most_common = max(distribution.items(), key=lambda x: x[1]["count"])
        print(f"\nMost common failure mode: {most_common[0]} ({most_common[1]['frequency']:.1%})")
        print(f"  This suggests focusing improvement efforts on: {most_common[0]}")
        
        # Find correlations
        if correlations:
            print(f"\nProperty correlations:")
            for mode, corr_data in correlations.items():
                size_mean = corr_data['object_size']['mean']
                mass_mean = corr_data['object_mass']['mean']
                friction_mean = corr_data['friction']['mean']
                print(f"  {mode}:")
                print(f"    Tends to occur with: size={size_mean:.4f}m, mass={mass_mean:.4f}kg, friction={friction_mean:.3f}")
    
    print("=" * 60)
    
    return analysis_results
