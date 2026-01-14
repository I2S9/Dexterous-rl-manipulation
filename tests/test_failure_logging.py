"""
Test script for failure logging system.

This script validates that failure episodes are correctly logged
with states, actions, contacts, and failure mode classification.
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments import CurriculumConfig
from evaluation import (
    HeldOutObjectSet,
    Evaluator,
    FailureLogger,
    EpisodeRecorder,
)
from policies import RandomPolicy
from envs import DexterousManipulationEnv


def test_failure_logging():
    """Test failure logging functionality."""
    print("=" * 60)
    print("Failure Logging Test")
    print("=" * 60)
    
    # Create evaluation setup
    train_config = CurriculumConfig.hard()  # Use hard for more failures
    heldout_set = HeldOutObjectSet(train_config, num_heldout_objects=5, seed=42)
    
    env = DexterousManipulationEnv(curriculum_config=train_config)
    policy = RandomPolicy(env.action_space, seed=42)
    env.close()
    
    # Create failure logger
    failure_logger = FailureLogger(
        log_dir="logs/failures",
        save_full_trajectories=True
    )
    
    # Create evaluator with failure logger
    evaluator = Evaluator(
        policy=policy,
        heldout_set=heldout_set,
        reward_type="dense",
        failure_logger=failure_logger
    )
    
    print("\nRunning evaluation with failure logging...")
    print("(Only failures will be logged)")
    
    # Run evaluation on held-out set
    results = evaluator.evaluate_heldout_set(
        num_episodes_per_object=3,
        seed=42
    )
    
    # Check that failures were logged
    print(f"\nLogged {len(failure_logger.logged_episodes)} failure episodes")
    
    if failure_logger.logged_episodes:
        # Check first logged episode
        first_episode = failure_logger.logged_episodes[0]
        
        print(f"\nFirst logged episode:")
        print(f"  Episode ID: {first_episode['episode_id']}")
        print(f"  Failure mode: {first_episode['failure_mode']}")
        print(f"  Confidence: {first_episode['failure_confidence']}")
        print(f"  Episode steps: {first_episode['episode_steps']}")
        print(f"  Has states: {first_episode.get('states') is not None}")
        print(f"  Has actions: {first_episode.get('actions') is not None}")
        print(f"  Has contacts: {len(first_episode.get('contact_history', [])) > 0}")
        print(f"  Has metadata: {bool(first_episode.get('metadata'))}")
        
        # Validate logged data
        assert first_episode['failure_mode'] is not None, "Failure mode should be classified"
        assert first_episode.get('states') is not None, "States should be logged"
        assert first_episode.get('actions') is not None, "Actions should be logged"
        assert len(first_episode.get('contact_history', [])) > 0, "Contacts should be logged"
        assert first_episode.get('metadata') is not None, "Metadata should be logged"
        
        # Check reproducibility metadata
        metadata = first_episode.get('metadata', {})
        print(f"\n  Metadata for reproducibility:")
        print(f"    Seed: {metadata.get('seed')}")
        print(f"    Object size: {metadata.get('object_size')}")
        print(f"    Object mass: {metadata.get('object_mass')}")
        print(f"    Friction: {metadata.get('friction_coefficient')}")
        
        assert 'seed' in metadata, "Seed should be in metadata for reproducibility"
        assert 'object_size' in metadata, "Object properties should be in metadata"
    else:
        print("\n[NOTE] No failures logged (all episodes succeeded)")
        print("       This is expected if policy performs well")
    
    # Save logged failures
    log_path = failure_logger.save("test_failures.json")
    print(f"\nFailures saved to: {log_path}")
    
    # Get statistics
    stats = failure_logger.get_statistics()
    print(f"\nFailure logging statistics:")
    print(f"  Total logged episodes: {stats.get('total_episodes', 0)}")
    print(f"  Failure mode counts: {stats.get('failure_mode_counts', {})}")
    
    # Test loading
    print("\nTesting load functionality...")
    new_logger = FailureLogger()
    loaded_data = new_logger.load(str(log_path))
    
    print(f"  Loaded {len(new_logger.logged_episodes)} episodes")
    assert len(new_logger.logged_episodes) == len(failure_logger.logged_episodes), \
        "Loaded episodes should match saved episodes"
    
    print("\n[PASS] Failure logging test passed")
    return True


def test_episode_recorder():
    """Test EpisodeRecorder functionality."""
    print("\n" + "=" * 60)
    print("Test: Episode Recorder")
    print("=" * 60)
    
    recorder = EpisodeRecorder(record_states=True, record_actions=True)
    
    # Simulate recording steps
    for i in range(5):
        state = np.random.randn(45)
        action = np.random.randn(15)
        contacts = [1.0, 1.0, 0.0, 0.0, 0.0]
        recorder.record_step(state=state, action=action, contacts=contacts)
    
    recorder.set_metadata(seed=42, test_param=1.0)
    
    data = recorder.get_recorded_data()
    
    print(f"  Recorded {len(data['states'])} states")
    print(f"  Recorded {len(data['actions'])} actions")
    print(f"  Recorded {len(data['contacts'])} contact arrays")
    print(f"  Metadata: {data['metadata']}")
    
    assert len(data['states']) == 5, "Should have 5 states"
    assert len(data['actions']) == 5, "Should have 5 actions"
    assert len(data['contacts']) == 5, "Should have 5 contact arrays"
    assert data['metadata']['seed'] == 42, "Metadata should be preserved"
    
    print("\n[PASS] Episode recorder test passed")
    return True


def test_reproducibility():
    """Test that logged episodes can be reproduced."""
    print("\n" + "=" * 60)
    print("Test: Reproducibility")
    print("=" * 60)
    
    # Create and log an episode
    train_config = CurriculumConfig.hard()
    heldout_set = HeldOutObjectSet(train_config, num_heldout_objects=1, seed=42)
    
    env = DexterousManipulationEnv(curriculum_config=train_config)
    policy = RandomPolicy(env.action_space, seed=42)
    env.close()
    
    failure_logger = FailureLogger()
    evaluator = Evaluator(
        policy=policy,
        heldout_set=heldout_set,
        failure_logger=failure_logger
    )
    
    # Run one episode
    eval_config = heldout_set.get_eval_config(0)
    result = evaluator.evaluate_episode(eval_config, seed=123)
    
    if not result["success"] and failure_logger.logged_episodes:
        logged_ep = failure_logger.logged_episodes[0]
        metadata = logged_ep.get("metadata", {})
        
        print(f"  Logged episode metadata:")
        print(f"    Seed: {metadata.get('seed')}")
        print(f"    Object properties: {metadata.get('object_size')}, {metadata.get('object_mass')}")
        
        # Save and verify
        log_path = failure_logger.save("reproducibility_test.json")
        
        # Load and verify metadata is preserved
        new_logger = FailureLogger()
        new_logger.load(str(log_path))
        
        loaded_ep = new_logger.logged_episodes[0]
        loaded_metadata = loaded_ep.get("metadata", {})
        
        assert loaded_metadata.get('seed') == metadata.get('seed'), \
            "Seed should be preserved for reproducibility"
        
        print(f"  [PASS] Metadata preserved for reproducibility")
        print(f"         Can reproduce episode with seed={metadata.get('seed')}")
    else:
        print(f"  [NOTE] Episode succeeded, no failure to log")
    
    print("\n[PASS] Reproducibility test passed")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Failure Logging System Tests")
    print("=" * 60)
    
    tests = [
        ("Failure Logging", test_failure_logging),
        ("Episode Recorder", test_episode_recorder),
        ("Reproducibility", test_reproducibility),
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
