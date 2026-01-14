"""
Failure episode logging system.

This module provides logging functionality for failure episodes,
capturing states, actions, contacts, and associating them with
failure mode classifications for offline analysis.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from evaluation.failure_taxonomy import FailureClassifier, FailureMode


class FailureLogger:
    """
    Logger for failure episodes with full traceability.
    
    Captures states, actions, contacts, and metadata for each
    failure episode, enabling offline analysis and reproduction.
    """
    
    def __init__(
        self,
        log_dir: str = "logs/failures",
        save_full_trajectories: bool = True,
        success_threshold: int = 3
    ):
        """
        Initialize failure logger.
        
        Args:
            log_dir: Directory to save failure logs
            save_full_trajectories: Whether to save full state/action trajectories
            success_threshold: Success threshold for contacts
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_full_trajectories = save_full_trajectories
        self.classifier = FailureClassifier(success_threshold=success_threshold)
        
        # Storage for logged episodes
        self.logged_episodes: List[Dict] = []
        self.episode_counter = 0
    
    def log_episode(
        self,
        episode_data: Dict,
        states: Optional[List[np.ndarray]] = None,
        actions: Optional[List[np.ndarray]] = None,
        contacts: Optional[List[List[float]]] = None,
        metadata: Optional[Dict] = None,
        max_steps: int = 200
    ) -> Dict:
        """
        Log a failure episode with full traceability.
        
        Args:
            episode_data: Dictionary with episode information
            states: List of state observations (if None, not logged)
            actions: List of actions taken (if None, not logged)
            contacts: List of contact arrays per step (if None, derived from episode_data)
            metadata: Additional metadata (seeds, configs, etc.)
            max_steps: Maximum steps per episode
            
        Returns:
            Dictionary with logged episode information
        """
        # Classify failure mode
        failure_mode, confidence = self.classifier.classify(episode_data, max_steps=max_steps)
        
        # Prepare contact history
        if contacts is None:
            contact_history = episode_data.get("contact_history", [])
        else:
            contact_history = contacts
        
        # Create logged episode entry
        logged_episode = {
            "episode_id": self.episode_counter,
            "timestamp": datetime.now().isoformat(),
            "success": episode_data.get("success", False),
            "failure_mode": failure_mode.value if failure_mode else None,
            "failure_confidence": confidence,
            "episode_steps": episode_data.get("episode_steps", 0),
            "episode_reward": episode_data.get("episode_reward", 0.0),
            "num_contacts": episode_data.get("num_contacts", 0),
            "final_contacts": episode_data.get("final_contacts", episode_data.get("num_contacts", 0)),
            "contact_history": [
                [float(c) for c in contacts] for contacts in contact_history
            ] if contact_history else [],
            "object_properties": {
                "size": episode_data.get("object_size", 0.0),
                "mass": episode_data.get("object_mass", 0.0),
                "friction_coefficient": episode_data.get("friction_coefficient", 0.0),
            },
            "metadata": metadata or {},
        }
        
        # Add full trajectories if requested and available
        if self.save_full_trajectories:
            if states is not None:
                logged_episode["states"] = [
                    state.tolist() if isinstance(state, np.ndarray) else state
                    for state in states
                ]
            
            if actions is not None:
                logged_episode["actions"] = [
                    action.tolist() if isinstance(action, np.ndarray) else action
                    for action in actions
                ]
        
        self.logged_episodes.append(logged_episode)
        self.episode_counter += 1
        
        return logged_episode
    
    def save(self, filename: Optional[str] = None) -> Path:
        """
        Save logged failure episodes to JSON file.
        
        Args:
            filename: Optional custom filename (default: failures_YYYYMMDD_HHMMSS.json)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"failures_{timestamp}.json"
        
        filepath = self.log_dir / filename
        
        # Prepare data for JSON serialization
        data = {
            "metadata": {
                "total_episodes": len(self.logged_episodes),
                "failure_modes": self._get_failure_mode_counts(),
                "logged_at": datetime.now().isoformat(),
            },
            "episodes": self._convert_to_json_serializable(self.logged_episodes),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def _get_failure_mode_counts(self) -> Dict[str, int]:
        """Get counts of failure modes in logged episodes."""
        counts = {}
        for episode in self.logged_episodes:
            mode = episode.get("failure_mode")
            if mode:
                counts[mode] = counts.get(mode, 0) + 1
        return counts
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """Convert numpy types and other objects to JSON-serializable format."""
        if isinstance(obj, (np.integer, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._convert_to_json_serializable(item) for item in obj]
        return obj
    
    def load(self, filepath: str) -> Dict:
        """
        Load failure episodes from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Dictionary with loaded data
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.logged_episodes = data.get("episodes", [])
        if self.logged_episodes:
            self.episode_counter = max(ep["episode_id"] for ep in self.logged_episodes) + 1
        else:
            self.episode_counter = 0
        
        return data
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about logged failure episodes.
        
        Returns:
            Dictionary with statistics
        """
        if not self.logged_episodes:
            return {}
        
        failure_modes = [ep.get("failure_mode") for ep in self.logged_episodes if ep.get("failure_mode")]
        mode_counts = {}
        for mode in failure_modes:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        episode_lengths = [ep["episode_steps"] for ep in self.logged_episodes]
        rewards = [ep["episode_reward"] for ep in self.logged_episodes]
        
        return {
            "total_episodes": len(self.logged_episodes),
            "failure_mode_counts": mode_counts,
            "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        }
    
    def reset(self):
        """Reset logger state."""
        self.logged_episodes = []
        self.episode_counter = 0


class EpisodeRecorder:
    """
    Helper class to record episode trajectories during evaluation.
    """
    
    def __init__(self, record_states: bool = True, record_actions: bool = True):
        """
        Initialize episode recorder.
        
        Args:
            record_states: Whether to record states
            record_actions: Whether to record actions
        """
        self.record_states = record_states
        self.record_actions = record_actions
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.contacts: List[List[float]] = []
        self.metadata: Dict = {}
    
    def record_step(
        self,
        state: Optional[np.ndarray] = None,
        action: Optional[np.ndarray] = None,
        contacts: Optional[List[float]] = None
    ):
        """
        Record a single step.
        
        Args:
            state: Current state observation
            action: Action taken
            contacts: Contact array for this step
        """
        if self.record_states and state is not None:
            self.states.append(state.copy())
        
        if self.record_actions and action is not None:
            self.actions.append(action.copy())
        
        if contacts is not None:
            self.contacts.append([float(c) for c in contacts])
    
    def set_metadata(self, **kwargs):
        """Set metadata for the episode."""
        self.metadata.update(kwargs)
    
    def get_recorded_data(self) -> Dict:
        """
        Get all recorded data.
        
        Returns:
            Dictionary with recorded states, actions, contacts, metadata
        """
        return {
            "states": self.states if self.record_states else None,
            "actions": self.actions if self.record_actions else None,
            "contacts": self.contacts,
            "metadata": self.metadata,
        }
    
    def reset(self):
        """Reset recorder for new episode."""
        self.states = []
        self.actions = []
        self.contacts = []
        self.metadata = {}
