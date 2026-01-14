"""
Main script for analyzing failure episodes.

This script loads logged failure episodes and generates
comprehensive statistical analysis and visualizations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.failure_statistics import (
    load_failure_logs,
    generate_failure_report,
    compute_failure_distribution,
    compute_correlations,
)


def main():
    """Run failure analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze failure episodes")
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/failures/failure_episodes.json",
        help="Path to failure log JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs",
        help="Directory to save analysis results"
    )
    
    args = parser.parse_args()
    
    # Load failure episodes
    print(f"Loading failure episodes from {args.log_file}...")
    try:
        episodes = load_failure_logs(args.log_file)
        print(f"Loaded {len(episodes)} failure episodes")
    except FileNotFoundError:
        print(f"Error: File not found: {args.log_file}")
        print("Please run evaluation with failure logging first.")
        return
    
    if not episodes:
        print("No failure episodes found in log file")
        return
    
    # Generate comprehensive report
    analysis_results = generate_failure_report(episodes, output_dir=args.output_dir)
    
    print("\nAnalysis complete!")
    print(f"Results saved to {args.output_dir}/")
    print("  - failure_distribution.png")
    print("  - failure_correlations.png")
    print("  - failure_analysis.json")


if __name__ == "__main__":
    main()
