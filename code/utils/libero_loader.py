import h5py
import numpy as np
from typing import List, Tuple, Dict, Any
import random
import os
from pathlib import Path

class LIBEROLoader:
    """
    Loads LIBERO episodes from HDF5 files with proper train/test split.
    Handles 50 demonstrations per task with random frame sampling.
    
    IMPORTANT: Uses DETERMINISTIC splits for reproducibility.
    """
    
    def __init__(self, base_path: str = "/data1/ma1/Ishaq/ump-vla/data/libero"):
        self.base_path = base_path
        
    def load_task(self, suite: str, task_id: int) -> Dict[str, Any]:
        """
        Load a specific task's HDF5 file.
        
        Args:
            suite: "libero_spatial", "libero_object", "libero_goal"
            task_id: 0-9 for each suite
            
        Returns:
            Dict with keys: episodes, task_name, instruction, total_frames
        """
        # Get list of files in the suite directory
        suite_dir = Path(self.base_path) / suite
        hdf5_files = sorted([f for f in os.listdir(suite_dir) if f.endswith('.hdf5')])
        
        if task_id >= len(hdf5_files):
            raise ValueError(f"task_id {task_id} out of range. Found {len(hdf5_files)} tasks in {suite}")
        
        task_file = suite_dir / hdf5_files[task_id]
        
        # Extract task name from filename (remove _demo.hdf5 suffix)
        task_name = hdf5_files[task_id].replace('_demo.hdf5', '')
        
        # Get instruction from LIBERO benchmark if possible
        instruction = self._get_instruction_from_benchmark(suite, task_id, task_name)
        
        with h5py.File(task_file, 'r') as f:
            episodes = []
            # Get all demo keys and sort them numerically
            demo_keys = sorted(f['data'].keys(), key=lambda x: int(x.split('_')[1]))
            num_episodes = len(demo_keys)
            
            for demo_key in demo_keys:
                demo = f[f"data/{demo_key}"]
                ep_id = int(demo_key.split('_')[1])
                
                # Get images - note: files have 128x128, but we'll use as-is
                # If 256x256 is needed, resize can be added
                agentview_images = demo['obs/agentview_rgb'][:]  # (T, 128, 128, 3) or (T, H, W, 3)
                
                episode = {
                    "episode_id": ep_id,
                    "actions": demo['actions'][:],  # (T, 7)
                    "agentview_images": agentview_images,  # (T, H, W, 3)
                    "instruction": instruction,
                    "task_name": task_name,
                    "length": len(demo['actions'])
                }
                episodes.append(episode)
            
            return {
                "episodes": episodes,
                "task_name": task_name,
                "instruction": instruction,
                "num_episodes": num_episodes
            }
    
    def _get_instruction_from_benchmark(self, suite: str, task_id: int, task_name: str) -> str:
        """
        Try to get instruction from LIBERO benchmark, fallback to task name.
        """
        try:
            from libero.libero import benchmark
            benchmark_dict = benchmark.get_benchmark_dict()
            if suite in benchmark_dict:
                task_suite = benchmark_dict[suite]()
                if task_id < len(task_suite.tasks):
                    task = task_suite.get_task(task_id)
                    return task.language
        except Exception as e:
            # Fallback: convert task name to instruction
            pass
        
        # Fallback: convert task name to readable instruction
        # Replace underscores with spaces and capitalize
        instruction = task_name.replace('_', ' ').title()
        return instruction
    
    def split_episodes(self, episodes: List[Dict], train_ratio: float = 0.7, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
        """
        Splits episodes into train/test sets DETERMINISTICALLY.
        
        CRITICAL: Uses fixed seed for reproducibility. Same episodes will be in
        train/test across all scripts.
        
        Args:
            episodes: List of all episodes for a task
            train_ratio: Fraction of episodes to use for training (0.7 = 35/50 episodes)
            seed: Random seed for reproducibility (default: 42)
            
        Returns:
            (train_episodes, test_episodes)
        """
        # Create local RNG with fixed seed for reproducibility
        rng = random.Random(seed)
        
        n_train = int(len(episodes) * train_ratio)
        
        # Get indices and shuffle with fixed seed
        indices = list(range(len(episodes)))
        rng.shuffle(indices)
        
        train_indices = set(indices[:n_train])
        
        train_episodes = [ep for i, ep in enumerate(episodes) if i in train_indices]
        test_episodes = [ep for i, ep in enumerate(episodes) if i not in train_indices]
        
        return train_episodes, test_episodes
    
    def get_split_indices(self, num_episodes: int, train_ratio: float = 0.7, seed: int = 42) -> Tuple[List[int], List[int]]:
        """
        Get train/test indices without loading episodes.
        Useful for verification that splits match.
        
        Args:
            num_episodes: Total number of episodes (usually 50)
            train_ratio: Fraction for training
            seed: Random seed
            
        Returns:
            (train_indices, test_indices)
        """
        rng = random.Random(seed)
        
        n_train = int(num_episodes * train_ratio)
        indices = list(range(num_episodes))
        rng.shuffle(indices)
        
        train_indices = sorted(indices[:n_train])
        test_indices = sorted(indices[n_train:])
        
        return train_indices, test_indices
    
    def sample_random_frame(self, episode: Dict, rng: random.Random = None) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Sample a random frame from an episode for ZOO query.
        Avoids static frames (where action is near-zero).
        
        Args:
            episode: Episode dict from load_task
            rng: Optional random.Random instance for reproducibility
            
        Returns:
            (image, action, instruction)
        """
        if rng is None:
            rng = random.Random()
        
        # Find non-static frames (where max(action[:3]) > 0.01)
        non_static_indices = [
            i for i, action in enumerate(episode["actions"])
            if np.max(np.abs(action[:3])) > 0.01
        ]
        
        if not non_static_indices:
            # Fallback: use middle frame
            frame_idx = len(episode["actions"]) // 2
        else:
            frame_idx = rng.choice(non_static_indices)
        
        image = episode["agentview_images"][frame_idx]
        action = episode["actions"][frame_idx]
        instruction = episode["instruction"]
        
        return image, action, instruction
    
    def sample_frame_at_index(self, episode: Dict, frame_idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Get a specific frame from an episode.
        
        Args:
            episode: Episode dict
            frame_idx: Frame index to retrieve
            
        Returns:
            (image, action, instruction)
        """
        frame_idx = min(frame_idx, len(episode["actions"]) - 1)
        frame_idx = max(frame_idx, 0)
        
        image = episode["agentview_images"][frame_idx]
        action = episode["actions"][frame_idx]
        instruction = episode["instruction"]
        
        return image, action, instruction
    
    def get_evenly_spaced_frames(self, episode: Dict, num_frames: int = 10, 
                                  skip_start: int = 5, skip_end: int = 5) -> List[int]:
        """
        Get evenly spaced frame indices, avoiding start/end static frames.
        
        Args:
            episode: Episode dict
            num_frames: Number of frames to sample
            skip_start: Frames to skip at start
            skip_end: Frames to skip at end
            
        Returns:
            List of frame indices
        """
        total = len(episode["actions"])
        start = min(skip_start, total // 10)
        end = max(total - skip_end, total - total // 10)
        
        if end <= start:
            # Very short episode, use middle
            return [total // 2]
        
        indices = np.linspace(start, end - 1, num_frames, dtype=int)
        return list(indices)
