"""
Example script for failure logging.

This script demonstrates how to use the failure logging system
to capture and analyze failure episodes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments import CurriculumConfig
from evaluation import (
    HeldOutObjectSet,
    Evaluator,
    FailureLogger,
    print_failure_statistics,
    analyze_failure_modes,
)
from policies import RandomPolicy
from envs import DexterousManipulationEnv


def main():
    """Run failure logging example."""
    print("=" * 60)
    print("Failure Logging Example")
    print("=" * 60)
    
    # Step 1: Setup
    print("\n1. Setting up evaluation...")
    train_config = CurriculumConfig.hard()  # Use hard config for more failures
    heldout_set = HeldOutObjectSet(train_config, num_heldout_objects=10, seed=42)
    
    env = DexterousManipulationEnv(curriculum_config=train_config)
    policy = RandomPolicy(env.action_space, seed=42)
    env.close()
    
    # Step 2: Create failure logger
    print("2. Creating failure logger...")
    failure_logger = FailureLogger(
        log_dir="logs/failures",
        save_full_trajectories=True,
        success_threshold=3
    )
    print(f"   Logger will save to: {failure_logger.log_dir}")
    print(f"   Full trajectories: {failure_logger.save_full_trajectories}")
    
    # Step 3: Create evaluator with logger
    print("3. Creating evaluator with failure logger...")
    evaluator = Evaluator(
        policy=policy,
        heldout_set=heldout_set,
        reward_type="dense",
        failure_logger=failure_logger
    )
    print("   Evaluator configured to log failures automatically")
    
    # Step 4: Run evaluation
    print("\n4. Running evaluation...")
    print("   (Failures will be automatically logged)")
    results = evaluator.evaluate_heldout_set(
        num_episodes_per_object=5,
        seed=42
    )
    
    # Step 5: Analyze logged failures
    print("\n5. Analyzing logged failures...")
    num_logged = len(failure_logger.logged_episodes)
    print(f"   Logged {num_logged} failure episodes")
    
    if num_logged > 0:
        # Get statistics
        stats = failure_logger.get_statistics()
        print(f"\n   Failure statistics:")
        print(f"     Total episodes: {stats['total_episodes']}")
        print(f"     Failure mode distribution:")
        for mode, count in stats['failure_mode_counts'].items():
            print(f"       {mode}: {count}")
        
        # Analyze failure modes
        episodes_for_analysis = [
            {
                "success": False,
                "episode_steps": ep["episode_steps"],
                "num_contacts": ep["num_contacts"],
                "final_contacts": ep["final_contacts"],
                "contact_history": ep["contact_history"],
            }
            for ep in failure_logger.logged_episodes
        ]
        
        analysis = analyze_failure_modes(episodes_for_analysis)
        print_failure_statistics(analysis)
        
        # Show example logged episode
        print("\n6. Example logged episode structure:")
        example = failure_logger.logged_episodes[0]
        print(f"   Episode ID: {example['episode_id']}")
        print(f"   Failure mode: {example['failure_mode']}")
        print(f"   Episode steps: {example['episode_steps']}")
        print(f"   States recorded: {len(example.get('states', []))}")
        print(f"   Actions recorded: {len(example.get('actions', []))}")
        print(f"   Contacts recorded: {len(example.get('contact_history', []))}")
        print(f"   Metadata keys: {list(example.get('metadata', {}).keys())}")
    else:
        print("\n   [NOTE] No failures logged")
        print("          All episodes succeeded (or evaluation was too short)")
        print("          Try with harder configuration or more episodes")
    
    # Step 7: Save logged failures
    print("\n7. Saving logged failures...")
    log_path = failure_logger.save("failure_episodes.json")
    print(f"   Saved to: {log_path}")
    
    # Step 8: Demonstrate reproducibility
    print("\n8. Reproducibility information:")
    if num_logged > 0:
        example = failure_logger.logged_episodes[0]
        metadata = example.get("metadata", {})
        print(f"   Example episode can be reproduced with:")
        print(f"     Seed: {metadata.get('seed')}")
        print(f"     Object size: {metadata.get('object_size')}")
        print(f"     Object mass: {metadata.get('object_mass')}")
        print(f"     Friction: {metadata.get('friction_coefficient')}")
        print(f"   All states, actions, and contacts are saved for offline analysis")
    
    print("\n" + "=" * 60)
    print("Failure Logging Complete")
    print("=" * 60)
    print("\nKey features:")
    print("  - Automatic failure detection and logging")
    print("  - Full state/action/contact trajectories")
    print("  - Failure mode classification")
    print("  - Metadata for reproducibility")
    print("  - Offline analysis capability")
    print("=" * 60)


if __name__ == "__main__":
    main()
