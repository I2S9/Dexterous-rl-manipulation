"""
Run component ablation study.

This script executes the complete component ablation study,
testing all combinations of curriculum learning and dense reward shaping.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.component_ablation import (
    run_component_ablation,
    compute_ablation_statistics,
    print_ablation_report
)
from evaluation.plot_component_ablation import plot_component_ablation, load_component_ablation_results
from experiments import load_config, load_named_config
import argparse


def main():
    """Run component ablation study."""
    parser = argparse.ArgumentParser(description="Run component ablation study")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment configuration JSON file (default: use default config)"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Name of predefined configuration (e.g., 'default', 'quick_test')"
    )
    args = parser.parse_args()
    
    # Load configuration
    if args.config_name:
        exp_config = load_named_config(args.config_name)
    elif args.config:
        exp_config = load_config(args.config)
    else:
        exp_config = load_config()  # Use default
    
    ablation_config = exp_config.component_ablation
    
    print("=" * 80)
    print("Component Ablation Study")
    print("=" * 80)
    print(f"Experiment: {exp_config.experiment_name}")
    if exp_config.description:
        print(f"Description: {exp_config.description}")
    print("\nThis study tests all combinations of:")
    print("  - Curriculum learning (on/off)")
    print("  - Dense reward shaping (on/off)")
    print("\nConfigurations tested:")
    print("  1. Baseline: Curriculum + Dense Reward")
    print("  2. No Curriculum: Fixed difficulty + Dense Reward")
    print("  3. No Dense Reward: Curriculum + Sparse Reward")
    print("  4. Minimal: Fixed difficulty + Sparse Reward")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Episodes per run: {ablation_config.num_episodes}")
    print(f"  Max episode steps: {ablation_config.max_episode_steps}")
    print(f"  Seeds: {ablation_config.seeds}")
    print(f"  Learning rate: {ablation_config.learning_rate}")
    print("=" * 80)
    
    # Run ablation study using configuration
    all_results = run_component_ablation(
        num_episodes=ablation_config.num_episodes,
        max_episode_steps=ablation_config.max_episode_steps,
        seeds=ablation_config.seeds,
        learning_rate=ablation_config.learning_rate,
        curriculum_scheduler_config=ablation_config.curriculum_scheduler,
        output_dir=exp_config.output_dir
    )
    
    # Compute statistics
    stats = compute_ablation_statistics(all_results)
    
    # Print report
    print_ablation_report(all_results, stats)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    try:
        results_dict = load_component_ablation_results()
        plot_component_ablation(results_dict)
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
    
    print("\n" + "=" * 80)
    print("Component Ablation Study Complete")
    print("=" * 80)
    print("\nResults saved to: logs/component_ablation_results.json")
    print("Plots saved to: logs/component_ablation.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
