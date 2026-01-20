#!/usr/bin/env python3
"""
Extract and visualize episodes from a LIBERO HDF5 dataset file.

For each episode, this script saves:
1. agentview_rgb (top-down camera) as MP4 video
2. eye_in_hand_rgb (wrist camera) as MP4 video
3. Actions and states as CSV files
"""

import argparse
import os
import h5py
import numpy as np
import pandas as pd
import imageio
from pathlib import Path
from tqdm import tqdm


def extract_episodes(hdf5_path, output_dir):
    """
    Extract all episodes from a LIBERO HDF5 file and save videos and CSVs.
    
    Args:
        hdf5_path: Path to the HDF5 file
        output_dir: Directory to save extracted episodes
    """
    # Create output directory structure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    videos_dir = output_dir / "videos"
    csvs_dir = output_dir / "csvs"
    videos_dir.mkdir(exist_ok=True)
    csvs_dir.mkdir(exist_ok=True)
    
    # Open HDF5 file
    print(f"Opening HDF5 file: {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        data = f['data']
        episode_keys = sorted(data.keys(), key=lambda x: int(x.split('_')[1]))
        num_episodes = len(episode_keys)
        
        print(f"Found {num_episodes} episodes")
        
        # Process each episode
        for episode_key in tqdm(episode_keys, desc="Processing episodes"):
            episode_num = episode_key.split('_')[1]
            demo = data[episode_key]
            
            # Extract data
            obs = demo['obs']
            agentview_rgb = obs['agentview_rgb'][:]  # Top-down camera
            eye_in_hand_rgb = obs['eye_in_hand_rgb'][:]  # Wrist camera
            actions = demo['actions'][:]
            states = demo['states'][:]
            robot_states = demo['robot_states'][:]
            
            num_frames = len(actions)
            print(f"\nEpisode {episode_num}: {num_frames} frames")
            
            # Save videos
            # Convert uint8 if needed (images should already be in 0-255 range)
            agentview_video_path = videos_dir / f"episode_{episode_num}_agentview.mp4"
            eye_in_hand_video_path = videos_dir / f"episode_{episode_num}_eye_in_hand.mp4"
            
            # Ensure images are uint8
            agentview_rgb_uint8 = agentview_rgb.astype(np.uint8)
            eye_in_hand_rgb_uint8 = eye_in_hand_rgb.astype(np.uint8)
            
            # Write videos using imageio (fps=30 is typical for robot demonstrations)
            print(f"  Saving agentview video: {agentview_video_path}")
            writer = imageio.get_writer(str(agentview_video_path), fps=30)
            for frame in agentview_rgb_uint8:
                writer.append_data(frame)
            writer.close()
            
            print(f"  Saving eye_in_hand video: {eye_in_hand_video_path}")
            writer = imageio.get_writer(str(eye_in_hand_video_path), fps=30)
            for frame in eye_in_hand_rgb_uint8:
                writer.append_data(frame)
            writer.close()
            
            # Save CSV files
            # Create DataFrame for actions
            action_df = pd.DataFrame(
                actions,
                columns=[f'action_{i}' for i in range(actions.shape[1])]
            )
            action_df.insert(0, 'frame', range(num_frames))
            action_csv_path = csvs_dir / f"episode_{episode_num}_actions.csv"
            action_df.to_csv(action_csv_path, index=False)
            print(f"  Saved actions CSV: {action_csv_path}")
            
            # Create DataFrame for states
            state_df = pd.DataFrame(
                states,
                columns=[f'state_{i}' for i in range(states.shape[1])]
            )
            state_df.insert(0, 'frame', range(num_frames))
            state_csv_path = csvs_dir / f"episode_{episode_num}_states.csv"
            state_df.to_csv(state_csv_path, index=False)
            print(f"  Saved states CSV: {state_csv_path}")
            
            # Create DataFrame for robot_states
            robot_state_df = pd.DataFrame(
                robot_states,
                columns=[f'robot_state_{i}' for i in range(robot_states.shape[1])]
            )
            robot_state_df.insert(0, 'frame', range(num_frames))
            robot_state_csv_path = csvs_dir / f"episode_{episode_num}_robot_states.csv"
            robot_state_df.to_csv(robot_state_csv_path, index=False)
            print(f"  Saved robot_states CSV: {robot_state_csv_path}")
            
            # Also create a combined CSV with actions and states
            combined_df = pd.DataFrame({
                'frame': range(num_frames),
            })
            # Add actions
            for i in range(actions.shape[1]):
                combined_df[f'action_{i}'] = actions[:, i]
            # Add states
            for i in range(states.shape[1]):
                combined_df[f'state_{i}'] = states[:, i]
            # Add robot_states
            for i in range(robot_states.shape[1]):
                combined_df[f'robot_state_{i}'] = robot_states[:, i]
            
            combined_csv_path = csvs_dir / f"episode_{episode_num}_combined.csv"
            combined_df.to_csv(combined_csv_path, index=False)
            print(f"  Saved combined CSV: {combined_csv_path}")
    
    print(f"\n✓ Extraction complete!")
    print(f"  Videos saved to: {videos_dir}")
    print(f"  CSVs saved to: {csvs_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract episodes from LIBERO HDF5 dataset"
    )
    parser.add_argument(
        '--hdf5_path',
        type=str,
        required=True,
        help='Path to the HDF5 file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: same as HDF5 file with _extracted suffix)'
    )
    
    args = parser.parse_args()
    
    # Set default output directory if not provided
    if args.output_dir is None:
        hdf5_path = Path(args.hdf5_path)
        args.output_dir = hdf5_path.parent / f"{hdf5_path.stem}_extracted"
    
    extract_episodes(args.hdf5_path, args.output_dir)


if __name__ == "__main__":
    main()

