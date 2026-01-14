"""
Test script for failure statistics and visualization.

This script validates that failure statistics are computed correctly
and that visualizations are generated properly.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.failure_statistics import (
    load_failure_logs,
    compute_failure_distribution,
    compute_correlations,
    plot_failure_distribution,
    plot_failure_correlations,
    generate_failure_report,
)


def test_load_and_analyze():
    """Test loading and analyzing failure logs."""
    print("=" * 60)
    print("Test: Load and Analyze Failure Logs")
    print("=" * 60)
    
    # Try to load existing failure logs
    log_path = Path("logs/failures/failure_episodes.json")
    
    if not log_path.exists():
        print(f"  [NOTE] No existing failure logs found at {log_path}")
        print("         Run evaluation/run_failure_logging.py first to generate logs")
        return True
    
    # Load episodes
    episodes = load_failure_logs(str(log_path))
    print(f"  Loaded {len(episodes)} failure episodes")
    
    if not episodes:
        print("  [NOTE] No episodes in log file")
        return True
    
    # Compute distribution
    distribution = compute_failure_distribution(episodes)
    print(f"\n  Failure distribution:")
    for mode, stats in sorted(distribution.items(), key=lambda x: x[1]["count"], reverse=True):
        print(f"    {mode}: {stats['count']} ({stats['frequency']:.1%})")
    
    # Compute correlations
    correlations = compute_correlations(episodes)
    print(f"\n  Correlations computed for {len(correlations)} failure modes")
    
    # Validate results
    assert len(distribution) > 0, "Should have distribution data"
    assert all("count" in stats for stats in distribution.values()), "Distribution should have counts"
    assert all("frequency" in stats for stats in distribution.values()), "Distribution should have frequencies"
    
    print("\n  [PASS] Load and analyze test passed")
    return True


def test_visualization():
    """Test visualization generation."""
    print("\n" + "=" * 60)
    print("Test: Visualization Generation")
    print("=" * 60)
    
    log_path = Path("logs/failures/failure_episodes.json")
    
    if not log_path.exists():
        print(f"  [NOTE] No existing failure logs found")
        print("         Skipping visualization test")
        return True
    
    episodes = load_failure_logs(str(log_path))
    
    if not episodes:
        print("  [NOTE] No episodes to visualize")
        return True
    
    # Generate plots
    print("  Generating distribution plot...")
    plot_failure_distribution(episodes, "logs/test_failure_distribution.png")
    
    print("  Generating correlations plot...")
    plot_failure_correlations(episodes, "logs/test_failure_correlations.png")
    
    # Check files were created
    dist_path = Path("logs/test_failure_distribution.png")
    corr_path = Path("logs/test_failure_correlations.png")
    
    assert dist_path.exists(), "Distribution plot should be created"
    assert corr_path.exists(), "Correlations plot should be created"
    
    print(f"  [PASS] Visualizations generated successfully")
    return True


def test_comprehensive_report():
    """Test comprehensive report generation."""
    print("\n" + "=" * 60)
    print("Test: Comprehensive Report Generation")
    print("=" * 60)
    
    log_path = Path("logs/failures/failure_episodes.json")
    
    if not log_path.exists():
        print(f"  [NOTE] No existing failure logs found")
        print("         Skipping report test")
        return True
    
    episodes = load_failure_logs(str(log_path))
    
    if not episodes:
        print("  [NOTE] No episodes to analyze")
        return True
    
    # Generate comprehensive report
    print("  Generating comprehensive report...")
    results = generate_failure_report(episodes, output_dir="logs")
    
    # Validate results
    assert "total_episodes" in results, "Results should have total_episodes"
    assert "distribution" in results, "Results should have distribution"
    assert "correlations" in results, "Results should have correlations"
    
    # Check output files
    analysis_path = Path("logs/failure_analysis.json")
    assert analysis_path.exists(), "Analysis JSON should be created"
    
    dist_plot_path = Path("logs/failure_distribution.png")
    corr_plot_path = Path("logs/failure_correlations.png")
    
    assert dist_plot_path.exists(), "Distribution plot should be created"
    assert corr_plot_path.exists(), "Correlations plot should be created"
    
    print(f"  [PASS] Comprehensive report generated successfully")
    return True


def test_interpretability():
    """Test that results are interpretable."""
    print("\n" + "=" * 60)
    print("Test: Results Interpretability")
    print("=" * 60)
    
    log_path = Path("logs/failures/failure_episodes.json")
    
    if not log_path.exists():
        print(f"  [NOTE] No existing failure logs found")
        return True
    
    episodes = load_failure_logs(str(log_path))
    
    if not episodes:
        print("  [NOTE] No episodes to analyze")
        return True
    
    distribution = compute_failure_distribution(episodes)
    correlations = compute_correlations(episodes)
    
    # Check interpretability
    print("  Checking interpretability...")
    
    # Distribution should be clear
    print(f"    Distribution: {len(distribution)} failure modes identified")
    for mode, stats in sorted(distribution.items(), key=lambda x: x[1]["count"], reverse=True):
        print(f"      {mode}: {stats['count']} episodes ({stats['frequency']:.1%})")
    
    # Correlations should show patterns
    if correlations:
        print(f"    Correlations: {len(correlations)} modes with correlation data")
        for mode, corr_data in correlations.items():
            print(f"      {mode}:")
            print(f"        Size: {corr_data['object_size']['mean']:.4f}m")
            print(f"        Mass: {corr_data['object_mass']['mean']:.4f}kg")
            print(f"        Friction: {corr_data['friction']['mean']:.3f}")
    
    # Results should be interpretable
    assert len(distribution) > 0, "Should identify failure modes"
    assert all(stats["frequency"] >= 0 and stats["frequency"] <= 1 for stats in distribution.values()), \
        "Frequencies should be valid"
    
    print(f"  [PASS] Results are interpretable")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Failure Statistics Tests")
    print("=" * 60)
    
    tests = [
        ("Load and Analyze", test_load_and_analyze),
        ("Visualization", test_visualization),
        ("Comprehensive Report", test_comprehensive_report),
        ("Interpretability", test_interpretability),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\nERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    main()
