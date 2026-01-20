#!/usr/bin/env python3
"""
SE(3) ZOO Adversarial Patch Testing Script - Direction 2 Aligned

Evaluates a trained adversarial patch on held-out episodes.
Uses DETERMINISTIC splits matching training for fair evaluation.

Direction 2 Focus:
- Primary metric: DEVIATION from clean prediction (not target distance)
- Cumulative deviation: Proxy for closed-loop trajectory impact
- Per-component analysis: Position, rotation, gripper separately

Key Metrics (Direction 2):

Frame-Level Metrics:
1. Average Deviation: Mean SE(3) distance between patched and clean predictions
2. Deviation Rate: % of frames with deviation > threshold
3. Per-Component: Position, rotation, gripper deviations separately

Trajectory-Level Metrics (NEW - for sequential attack evaluation):
4. Cumulative Drift Threshold (CDT): Binary success if cumulative pos drift > threshold
5. Time-to-Failure (TTF): Minimum frames needed to exceed drift threshold
6. Sustained Deviation Rate (SDR): Fraction of N-frame windows with mean deviation > threshold
7. Task Failure Proxy (TFP): Estimated task failure based on cumulative drift vs task scale

These trajectory-level metrics capture the SEQUENTIAL, COMPOUNDING nature of the attack
that single-frame metrics miss.
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import random

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

# Set headless rendering before importing robosuite/libero
os.environ['MUJOCO_GL'] = 'osmesa'

from openvla_action_extractor import OpenVLAActionExtractor
from utils.libero_loader import LIBEROLoader
from utils.se3_distance import se3_distance, normalized_se3_distance, position_only_distance


def parse_args():
    parser = argparse.ArgumentParser(
        description='SE(3) ZOO Adversarial Patch Testing (Direction 2: Deviation Analysis)'
    )

    # Patch configuration
    parser.add_argument('--patch_path', type=str, required=True,
                        help='Path to trained patch (.npy file)')

    # Task configuration
    parser.add_argument('--suite', type=str, default='libero_spatial',
                        choices=['libero_spatial', 'libero_object', 'libero_goal', 'libero_10'],
                        help='LIBERO task suite')
    parser.add_argument('--task_id', type=int, default=0,
                        help='Task ID within the suite (0-9)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Train ratio (must match training!)')
    parser.add_argument('--frames_per_episode', type=int, default=10,
                        help='Frames to sample per episode')

    # Deviation thresholds (Direction 2 metrics)
    parser.add_argument('--deviation_threshold', type=float, default=0.3,
                        help='SE(3) distance threshold for "significant" deviation')

    # Trajectory-level metric thresholds (RECALIBRATED for realistic attack evaluation)
    parser.add_argument('--drift_threshold', type=float, default=0.05,
                        help='Cumulative position drift threshold for CDT (meters) - 5cm is significant for manipulation')
    parser.add_argument('--task_scale', type=float, default=0.1,
                        help='Typical task movement range for TFP calculation (meters)')
    parser.add_argument('--sdr_windows', type=str, default='3,5,10',
                        help='Comma-separated window sizes for SDR calculation (smaller windows for short episodes)')

    # Patch placement (must match training!)
    parser.add_argument('--patch_x', type=int, default=48,
                        help='Patch X position (must match training!)')
    parser.add_argument('--patch_y', type=int, default=48,
                        help='Patch Y position (must match training!)')

    # Model configuration
    parser.add_argument('--model_path', type=str,
                        default='/data1/ma1/Ishaq/ump-vla/checkpoints/openvla-7b',
                        help='Path to OpenVLA checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for model inference')

    # Output configuration
    parser.add_argument('--output_dir', type=str,
                        default='/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (must match training!)')

    return parser.parse_args()


def compute_actual_trajectory_drift(episode_results: list) -> dict:
    """
    Compute TRUE position drift (where robot ends up vs. clean).

    CRITICAL: This computes ||Σ Δ[t]|| (norm of sum), NOT Σ ||Δ[t]|| (sum of norms).

    The difference matters:
    - Sum of norms: Counts all deviations, even if they oscillate and cancel out
    - Norm of sum: Measures ACTUAL displacement between clean and patched trajectories

    Example:
        Oscillating attack: Δ = [+0.05, -0.05, +0.05, -0.05]
            Sum of norms = 0.20 (looks effective)
            Norm of sum = 0.00 (actually no effect!)

        Consistent attack: Δ = [+0.02, +0.02, +0.02, +0.02]
            Sum of norms = 0.08
            Norm of sum = 0.08 (true drift!)

    Args:
        episode_results: List of per-frame result dicts with 'clean_pred', 'patched_pred'

    Returns:
        dict: Contains actual trajectory drift metrics
    """
    if not episode_results:
        return {
            "final_drift": 0.0,
            "max_drift": 0.0,
            "drift_trajectory": [],
            "final_displacement": [0.0, 0.0, 0.0],
            "drift_consistency": 0.0,
            "sum_of_norms": 0.0
        }

    # Cumulative positions (simulating robot trajectory)
    clean_position = np.array([0.0, 0.0, 0.0])
    patched_position = np.array([0.0, 0.0, 0.0])

    drift_over_time = []
    sum_of_norms = 0.0  # For comparison (the OLD metric)

    for r in episode_results:
        clean_pred = np.array(r["clean_pred"][:3])
        patched_pred = np.array(r["patched_pred"][:3])

        clean_position += clean_pred
        patched_position += patched_pred

        # Per-frame deviation (OLD metric component)
        frame_deviation = np.linalg.norm(patched_pred - clean_pred)
        sum_of_norms += frame_deviation

        # Actual displacement at this timestep (NEW metric)
        displacement = patched_position - clean_position
        current_drift = np.linalg.norm(displacement)
        drift_over_time.append(float(current_drift))

    final_displacement = patched_position - clean_position
    final_drift = np.linalg.norm(final_displacement)
    max_drift = max(drift_over_time) if drift_over_time else 0.0

    # Drift consistency: ratio of actual drift to sum of norms
    # 1.0 = perfectly consistent (all deviations same direction)
    # < 0.5 = significant oscillation (attack partially cancels itself)
    # ~0 = pure oscillation (no net effect)
    drift_consistency = final_drift / (sum_of_norms + 1e-8)

    return {
        "final_drift": float(final_drift),
        "max_drift": float(max_drift),
        "drift_trajectory": drift_over_time,
        "final_displacement": final_displacement.tolist(),
        "drift_consistency": float(drift_consistency),
        "sum_of_norms": float(sum_of_norms)  # Keep for comparison
    }


def compute_trajectory_metrics(episode_results: list, drift_threshold: float,
                                task_scale: float, deviation_threshold: float,
                                sdr_windows: list) -> dict:
    """
    Compute trajectory-level metrics for an episode.

    UPDATED: Now uses ACTUAL trajectory drift (norm of sum) instead of
    summed deviation (sum of norms) for CDT and TFP calculations.

    These metrics capture the SEQUENTIAL, COMPOUNDING nature of adversarial attacks
    that single-frame metrics miss.

    Args:
        episode_results: List of per-frame result dicts with 'pos_deviation', 'deviation'
        drift_threshold: Threshold for CDT (cumulative drift threshold) in meters
        task_scale: Typical task movement range for TFP calculation
        deviation_threshold: Threshold for "significant" deviation
        sdr_windows: List of window sizes for SDR calculation

    Returns:
        Dict containing:
        - cdt_success: Binary, True if ACTUAL pos drift > drift_threshold
        - ttf_frames: Time-to-failure (frames needed to exceed drift_threshold), None if never
        - sdr: Dict of SDR values for each window size
        - tfp_score: Task failure proxy score (actual_drift / task_scale)
        - actual_drift: The TRUE trajectory drift metrics
        - legacy_cumulative: Old sum-of-norms metric (for comparison)
    """
    if not episode_results:
        return {
            "cdt_success": False,
            "ttf_frames": None,
            "sdr": {w: 0.0 for w in sdr_windows},
            "tfp_score": 0.0,
            "cumulative_pos_drift": 0.0,
            "actual_drift": {
                "final_drift": 0.0,
                "max_drift": 0.0,
                "drift_consistency": 0.0
            }
        }

    # Compute ACTUAL trajectory drift (the correct metric)
    actual_drift = compute_actual_trajectory_drift(episode_results)

    # Legacy metrics for comparison
    pos_deviations = [r["pos_deviation"] for r in episode_results]
    deviations = [r["deviation"] for r in episode_results]
    n_frames = len(pos_deviations)
    legacy_cumulative = sum(pos_deviations)  # Old metric (sum of norms)

    # 1. Cumulative Drift Threshold (CDT) - NOW USES ACTUAL DRIFT
    # Binary success: Does ACTUAL position drift exceed threshold?
    cdt_success = actual_drift["final_drift"] > drift_threshold

    # 2. Time-to-Failure (TTF) - NOW USES ACTUAL DRIFT
    # Minimum number of frames to exceed drift threshold
    ttf_frames = None
    for i, drift in enumerate(actual_drift["drift_trajectory"]):
        if drift > drift_threshold:
            ttf_frames = i + 1  # 1-indexed frame count
            break

    # 3. Sustained Deviation Rate (SDR)
    # Fraction of N-frame windows where mean deviation > threshold
    sdr = {}
    for window_size in sdr_windows:
        if window_size > n_frames:
            # Can't compute SDR for window larger than episode
            sdr[window_size] = None
        else:
            n_windows = n_frames - window_size + 1
            windows_above_threshold = 0
            for start in range(n_windows):
                window_mean = np.mean(deviations[start:start + window_size])
                if window_mean > deviation_threshold:
                    windows_above_threshold += 1
            sdr[window_size] = windows_above_threshold / n_windows

    # 4. Task Failure Proxy (TFP) - NOW USES ACTUAL DRIFT
    # Ratio of ACTUAL drift to typical task movement
    # TFP > 1.0 suggests high probability of task failure
    tfp_score = actual_drift["final_drift"] / task_scale if task_scale > 0 else 0.0

    return {
        "cdt_success": bool(cdt_success),
        "ttf_frames": ttf_frames,
        "sdr": {int(k): float(v) if v is not None else None for k, v in sdr.items()},
        "tfp_score": float(tfp_score),
        "cumulative_pos_drift": float(legacy_cumulative),  # Keep legacy for comparison
        "actual_drift": {
            "final_drift": float(actual_drift["final_drift"]),
            "max_drift": float(actual_drift["max_drift"]),
            "drift_consistency": float(actual_drift["drift_consistency"]),
            "drift_trajectory": actual_drift["drift_trajectory"]
        }
    }


def compute_aggregate_trajectory_metrics(all_episode_trajectory_metrics: list,
                                         sdr_windows: list) -> dict:
    """
    Aggregate trajectory metrics across all episodes.

    UPDATED: Now includes actual drift metrics (norm of sum) alongside
    legacy metrics (sum of norms) for comparison.

    Args:
        all_episode_trajectory_metrics: List of trajectory metric dicts per episode
        sdr_windows: Window sizes used for SDR

    Returns:
        Aggregated metrics with means and success rates
    """
    n_episodes = len(all_episode_trajectory_metrics)
    if n_episodes == 0:
        return {}

    # CDT success rate (fraction of episodes that exceed drift threshold)
    cdt_successes = [m["cdt_success"] for m in all_episode_trajectory_metrics]
    cdt_success_rate = sum(cdt_successes) / n_episodes

    # TTF statistics (for episodes that reached threshold)
    ttf_values = [m["ttf_frames"] for m in all_episode_trajectory_metrics if m["ttf_frames"] is not None]
    if ttf_values:
        ttf_mean = np.mean(ttf_values)
        ttf_min = min(ttf_values)
        ttf_max = max(ttf_values)
        ttf_success_rate = len(ttf_values) / n_episodes
    else:
        ttf_mean = None
        ttf_min = None
        ttf_max = None
        ttf_success_rate = 0.0

    # SDR averages per window size
    sdr_means = {}
    for w in sdr_windows:
        values = [m["sdr"][w] for m in all_episode_trajectory_metrics if m["sdr"].get(w) is not None]
        sdr_means[w] = float(np.mean(values)) if values else None

    # TFP statistics (now based on actual drift)
    tfp_scores = [m["tfp_score"] for m in all_episode_trajectory_metrics]
    tfp_mean = np.mean(tfp_scores)
    tfp_max = max(tfp_scores)
    tfp_above_1 = sum(1 for t in tfp_scores if t > 1.0) / n_episodes  # Fraction with TFP > 1

    # Legacy cumulative drift statistics (sum of norms - OLD metric)
    legacy_drift_values = [m["cumulative_pos_drift"] for m in all_episode_trajectory_metrics]
    legacy_drift_mean = np.mean(legacy_drift_values)
    legacy_drift_max = max(legacy_drift_values)

    # ACTUAL drift statistics (norm of sum - NEW metric)
    actual_drift_values = [m["actual_drift"]["final_drift"] for m in all_episode_trajectory_metrics]
    actual_drift_mean = np.mean(actual_drift_values)
    actual_drift_max = max(actual_drift_values)

    # Drift consistency (how consistent is the drift direction?)
    consistency_values = [m["actual_drift"]["drift_consistency"] for m in all_episode_trajectory_metrics]
    consistency_mean = np.mean(consistency_values)

    return {
        "cdt": {
            "success_rate": float(cdt_success_rate),
            "n_successes": int(sum(cdt_successes)),
            "n_episodes": n_episodes
        },
        "ttf": {
            "mean_frames": float(ttf_mean) if ttf_mean is not None else None,
            "min_frames": int(ttf_min) if ttf_min is not None else None,
            "max_frames": int(ttf_max) if ttf_max is not None else None,
            "success_rate": float(ttf_success_rate)
        },
        "sdr": {int(k): v for k, v in sdr_means.items()},
        "tfp": {
            "mean_score": float(tfp_mean),
            "max_score": float(tfp_max),
            "above_1_rate": float(tfp_above_1)
        },
        # NEW: Actual drift (norm of sum) - the CORRECT metric
        "actual_drift": {
            "mean": float(actual_drift_mean),
            "max": float(actual_drift_max),
            "consistency_mean": float(consistency_mean)
        },
        # OLD: Legacy cumulative drift (sum of norms) - for comparison only
        "legacy_cumulative_drift": {
            "mean": float(legacy_drift_mean),
            "max": float(legacy_drift_max)
        }
    }


def apply_patch(image: np.ndarray, patch: np.ndarray, position: tuple) -> np.ndarray:
    """Apply patch to image at specified position."""
    patched = image.copy()
    x, y = position
    h, w = patch.shape[:2]
    img_h, img_w = image.shape[:2]

    # Clip position to valid range
    max_x = img_w - w
    max_y = img_h - h
    x = np.clip(x, 0, max(0, max_x))
    y = np.clip(y, 0, max(0, max_y))

    # Handle edge case where patch is larger than image
    if h > img_h or w > img_w:
        scale = min(img_h / h, img_w / w) * 0.9
        new_h, new_w = int(h * scale), int(w * scale)
        patch_resized = patch[:new_h, :new_w]
        patched[y:y+new_h, x:x+new_w] = (patch_resized * 255).astype(np.uint8)
    else:
        patched[y:y+h, x:x+w] = (patch * 255).astype(np.uint8)

    return patched


def main():
    args = parse_args()

    # Parse SDR window sizes
    sdr_windows = [int(w.strip()) for w in args.sdr_windows.split(',')]

    # Set random seeds (must match training!)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create output directories
    output_dir = Path(args.output_dir)
    results_dir = output_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Extract run_id from patch path
    patch_filename = Path(args.patch_path).stem
    run_id = patch_filename.replace('_patch', '')

    patch_position = (args.patch_x, args.patch_y)

    print("=" * 80)
    print("DIRECTION 2: DEVIATION ANALYSIS TESTING")
    print("=" * 80)
    print(f"Focus: Measuring deviation from clean predictions")
    print(f"Goal: Validate patch effect for closed-loop trajectory attack")
    print("=" * 80)
    print(f"Run ID:           {run_id}")
    print(f"Start time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Patch:            {args.patch_path}")
    print(f"Patch position:   {patch_position}")
    print(f"Suite:            {args.suite}")
    print(f"Task ID:          {args.task_id}")
    print(f"Train ratio:      {args.train_ratio}")
    print(f"Frames/episode:   {args.frames_per_episode}")
    print(f"Deviation thresh: {args.deviation_threshold}")
    print(f"Drift threshold:  {args.drift_threshold}m (for CDT)")
    print(f"Task scale:       {args.task_scale}m (for TFP)")
    print(f"SDR windows:      {sdr_windows}")
    print(f"Seed:             {args.seed}")
    print("=" * 80)

    # Load patch
    print(f"\n[1/4] Loading adversarial patch...")
    patch = np.load(args.patch_path)
    print(f"  Shape:          {patch.shape}")
    print(f"  Range:          [{patch.min():.3f}, {patch.max():.3f}]")

    # Load LIBERO data with SAME split as training
    print(f"\n[2/4] Loading LIBERO data...")
    loader = LIBEROLoader()
    task_data = loader.load_task(args.suite, args.task_id)

    # Use SAME deterministic split as training
    train_episodes, test_episodes = loader.split_episodes(
        task_data["episodes"],
        train_ratio=args.train_ratio,
        seed=args.seed
    )

    # Verify split indices
    train_indices, test_indices = loader.get_split_indices(
        len(task_data["episodes"]),
        train_ratio=args.train_ratio,
        seed=args.seed
    )

    print(f"  Task:           {task_data['task_name']}")
    print(f"  Instruction:    {task_data['instruction']}")
    print(f"  Total episodes: {task_data['num_episodes']}")
    print(f"  Test episodes:  {len(test_episodes)} (indices: {test_indices[:5]}...)")

    # Load OpenVLA model
    print(f"\n[3/4] Loading OpenVLA model...")
    model = OpenVLAActionExtractor(
        model_path=args.model_path,
        device=args.device
    )

    # Evaluate
    print(f"\n[4/4] Evaluating patch on {len(test_episodes)} test episodes...")
    print("=" * 80)

    start_time = time.time()

    all_results = []
    episode_metrics = []

    for ep_idx, episode in enumerate(test_episodes):
        ep_results = []
        instruction = episode["instruction"]

        # Get evenly spaced frame indices
        frame_indices = loader.get_evenly_spaced_frames(
            episode,
            num_frames=args.frames_per_episode
        )

        for frame_idx in frame_indices:
            image, gt_action, _ = loader.sample_frame_at_index(episode, frame_idx)

            # 1. Get CLEAN prediction (no patch) - this is our baseline
            clean_pred = model.get_action_vector(image, instruction)
            if isinstance(clean_pred, torch.Tensor):
                clean_pred = clean_pred.cpu().numpy()
            clean_pred = np.array(clean_pred).flatten()

            # 2. Get PATCHED prediction
            patched_image = apply_patch(image, patch, patch_position)
            patched_pred = model.get_action_vector(patched_image, instruction)
            if isinstance(patched_pred, torch.Tensor):
                patched_pred = patched_pred.cpu().numpy()
            patched_pred = np.array(patched_pred).flatten()

            # ============== DIRECTION 2 METRICS ==============

            # A. Deviation from clean (PRIMARY METRIC)
            deviation = se3_distance(patched_pred, clean_pred)

            # B. Component-wise deviations
            pos_dev = np.linalg.norm(patched_pred[:3] - clean_pred[:3])
            rot_dev = np.linalg.norm(patched_pred[3:6] - clean_pred[3:6])
            grip_dev = np.abs(patched_pred[6] - clean_pred[6])

            # C. Is this a "significant" deviation?
            is_significant = deviation > args.deviation_threshold

            # D. Baseline errors (for context, not primary)
            gt_error_clean = se3_distance(clean_pred, gt_action)
            gt_error_patched = se3_distance(patched_pred, gt_action)

            result = {
                "episode_id": ep_idx,
                "episode_original_id": episode.get("episode_id", ep_idx),
                "frame_idx": int(frame_idx),
                # Raw predictions
                "gt_action": gt_action.tolist(),
                "clean_pred": clean_pred.tolist(),
                "patched_pred": patched_pred.tolist(),
                # DIRECTION 2 PRIMARY METRICS
                "deviation": float(deviation),
                "pos_deviation": float(pos_dev),
                "rot_deviation": float(rot_dev),
                "grip_deviation": float(grip_dev),
                "is_significant": bool(is_significant),
                # Baseline context
                "gt_error_clean": float(gt_error_clean),
                "gt_error_patched": float(gt_error_patched),
            }
            ep_results.append(result)

        # Episode summary - DIRECTION 2 FOCUS
        ep_deviations = [r["deviation"] for r in ep_results]
        ep_pos_devs = [r["pos_deviation"] for r in ep_results]
        ep_rot_devs = [r["rot_deviation"] for r in ep_results]
        ep_grip_devs = [r["grip_deviation"] for r in ep_results]
        ep_significant = [r["is_significant"] for r in ep_results]

        # Cumulative deviation (proxy for closed-loop trajectory error)
        cumulative_deviation = sum(ep_deviations)
        cumulative_pos = sum(ep_pos_devs)

        # Compute TRAJECTORY-LEVEL METRICS for this episode
        traj_metrics = compute_trajectory_metrics(
            episode_results=ep_results,
            drift_threshold=args.drift_threshold,
            task_scale=args.task_scale,
            deviation_threshold=args.deviation_threshold,
            sdr_windows=sdr_windows
        )

        episode_metrics.append({
            "episode_id": ep_idx,
            "original_id": episode.get("episode_id", ep_idx),
            "num_frames": len(ep_results),
            # Per-frame averages
            "avg_deviation": float(np.mean(ep_deviations)),
            "std_deviation": float(np.std(ep_deviations)),
            "avg_pos_deviation": float(np.mean(ep_pos_devs)),
            "avg_rot_deviation": float(np.mean(ep_rot_devs)),
            "avg_grip_deviation": float(np.mean(ep_grip_devs)),
            "deviation_rate": float(np.mean(ep_significant)),
            # Cumulative (Direction 2 key metric)
            "cumulative_deviation": float(cumulative_deviation),
            "cumulative_pos_deviation": float(cumulative_pos),
            # TRAJECTORY-LEVEL METRICS
            "trajectory_metrics": traj_metrics
        })

        all_results.extend(ep_results)

        # Log progress
        if (ep_idx + 1) % 3 == 0 or ep_idx == 0:
            print(f"[TEST] Episode {ep_idx+1:3d}/{len(test_episodes)} | "
                  f"Avg Dev: {np.mean(ep_deviations):.4f} | "
                  f"Cumul: {cumulative_deviation:.4f} | "
                  f"Rate: {np.mean(ep_significant)*100:5.1f}% | "
                  f"Pos: {np.mean(ep_pos_devs):.4f}")

    testing_time = time.time() - start_time

    # ============== AGGREGATE METRICS (DIRECTION 2) ==============

    # Primary: Deviation metrics
    all_deviations = [r["deviation"] for r in all_results]
    all_pos_devs = [r["pos_deviation"] for r in all_results]
    all_rot_devs = [r["rot_deviation"] for r in all_results]
    all_grip_devs = [r["grip_deviation"] for r in all_results]
    all_significant = [r["is_significant"] for r in all_results]

    avg_deviation = np.mean(all_deviations)
    std_deviation = np.std(all_deviations)
    deviation_rate = np.mean(all_significant)

    avg_pos_dev = np.mean(all_pos_devs)
    avg_rot_dev = np.mean(all_rot_devs)
    avg_grip_dev = np.mean(all_grip_devs)

    # Cumulative across all test frames (proxy for total trajectory error)
    total_cumulative_deviation = sum(all_deviations)
    total_cumulative_pos = sum(all_pos_devs)

    # Per-episode cumulative (for closed-loop analysis)
    episode_cumulatives = [em["cumulative_deviation"] for em in episode_metrics]
    avg_episode_cumulative = np.mean(episode_cumulatives)

    # Baseline context
    avg_gt_error_clean = np.mean([r["gt_error_clean"] for r in all_results])
    avg_gt_error_patched = np.mean([r["gt_error_patched"] for r in all_results])

    # ============== TRAJECTORY-LEVEL AGGREGATE METRICS ==============
    all_traj_metrics = [em["trajectory_metrics"] for em in episode_metrics]
    aggregate_traj = compute_aggregate_trajectory_metrics(all_traj_metrics, sdr_windows)

    print("=" * 80)
    print(f"\nTesting Complete!")
    print(f"  Total time:     {testing_time:.1f} seconds ({testing_time/60:.1f} min)")
    print(f"  Episodes:       {len(test_episodes)}")
    print(f"  Frames/episode: {args.frames_per_episode}")
    print(f"  Total frames:   {len(all_results)}")

    # Print detailed summary - DIRECTION 2 FOCUS
    print("\n" + "=" * 80)
    print("DIRECTION 2 TESTING RESULTS: DEVIATION ANALYSIS")
    print("=" * 80)

    print(f"\n[PRIMARY METRICS: DEVIATION FROM CLEAN PREDICTION]")
    print(f"  Average Deviation:     {avg_deviation:.4f} +/- {std_deviation:.4f}")
    print(f"  Deviation Rate:        {deviation_rate*100:5.1f}% (threshold: {args.deviation_threshold})")
    print(f"  Max Deviation:         {max(all_deviations):.4f}")
    print(f"  Min Deviation:         {min(all_deviations):.4f}")

    print(f"\n[COMPONENT-WISE DEVIATIONS]")
    print(f"  Position Deviation:    {avg_pos_dev:.4f} (meters)")
    print(f"  Rotation Deviation:    {avg_rot_dev:.4f} (radians)")
    print(f"  Gripper Deviation:     {avg_grip_dev:.4f} (0-2 scale)")

    print(f"\n[CUMULATIVE METRICS (Proxy for Closed-Loop Impact)]")
    print(f"  Avg Episode Cumulative: {avg_episode_cumulative:.4f}")
    print(f"  Total Cumulative:       {total_cumulative_deviation:.4f}")
    print(f"  Cumulative Position:    {total_cumulative_pos:.4f} meters")

    print(f"\n[BASELINE CONTEXT (for reference)]")
    print(f"  Clean Model GT Error:   {avg_gt_error_clean:.4f}")
    print(f"  Patched Model GT Error: {avg_gt_error_patched:.4f}")
    print(f"  GT Error Change:        {avg_gt_error_patched - avg_gt_error_clean:+.4f}")

    print("=" * 80)

    # ============== TRAJECTORY-LEVEL METRICS OUTPUT ==============
    print("\n" + "=" * 80)
    print("TRAJECTORY-LEVEL METRICS (Sequential Attack Evaluation)")
    print("=" * 80)

    print(f"\n[CDT - Cumulative Drift Threshold (threshold: {args.drift_threshold}m)]")
    print(f"  Success Rate:     {aggregate_traj['cdt']['success_rate']*100:.1f}%")
    print(f"  Episodes Success: {aggregate_traj['cdt']['n_successes']}/{aggregate_traj['cdt']['n_episodes']}")

    print(f"\n[TTF - Time-to-Failure (frames to exceed {args.drift_threshold}m drift)]")
    if aggregate_traj['ttf']['mean_frames'] is not None:
        print(f"  Mean TTF:         {aggregate_traj['ttf']['mean_frames']:.1f} frames")
        print(f"  Min TTF:          {aggregate_traj['ttf']['min_frames']} frames")
        print(f"  Max TTF:          {aggregate_traj['ttf']['max_frames']} frames")
        print(f"  TTF Success Rate: {aggregate_traj['ttf']['success_rate']*100:.1f}%")
    else:
        print(f"  No episodes reached drift threshold")

    print(f"\n[SDR - Sustained Deviation Rate (mean dev > {args.deviation_threshold} per window)]")
    for w, sdr_val in aggregate_traj['sdr'].items():
        if sdr_val is not None:
            print(f"  Window {w:3d} frames: {sdr_val*100:.1f}%")
        else:
            print(f"  Window {w:3d} frames: N/A (window > episode length)")

    print(f"\n[TFP - Task Failure Proxy (actual_drift / task_scale={args.task_scale}m)]")
    print(f"  Mean TFP Score:   {aggregate_traj['tfp']['mean_score']:.2f}x task scale")
    print(f"  Max TFP Score:    {aggregate_traj['tfp']['max_score']:.2f}x task scale")
    print(f"  TFP > 1.0 Rate:   {aggregate_traj['tfp']['above_1_rate']*100:.1f}% of episodes")

    print(f"\n[ACTUAL TRAJECTORY DRIFT (||Σ Δ|| - the CORRECT metric)]")
    print(f"  Mean Actual Drift:    {aggregate_traj['actual_drift']['mean']:.4f}m")
    print(f"  Max Actual Drift:     {aggregate_traj['actual_drift']['max']:.4f}m")
    print(f"  Drift Consistency:    {aggregate_traj['actual_drift']['consistency_mean']:.2f}")
    print(f"    (1.0 = all deviations same direction, <0.5 = significant oscillation)")

    print(f"\n[LEGACY CUMULATIVE DRIFT (Σ ||Δ|| - for comparison only)]")
    print(f"  Mean Legacy Drift:    {aggregate_traj['legacy_cumulative_drift']['mean']:.4f}m")
    print(f"  Max Legacy Drift:     {aggregate_traj['legacy_cumulative_drift']['max']:.4f}m")

    # Alert if there's significant metric discrepancy (indicating oscillation)
    if aggregate_traj['actual_drift']['consistency_mean'] < 0.5:
        print(f"\n  ⚠️  WARNING: Low drift consistency ({aggregate_traj['actual_drift']['consistency_mean']:.2f})")
        print(f"      Attack may be oscillating - deviations canceling out!")
        print(f"      Actual drift ({aggregate_traj['actual_drift']['mean']:.4f}m) << Legacy ({aggregate_traj['legacy_cumulative_drift']['mean']:.4f}m)")

    print("=" * 80)

    # Direction 2 success evaluation
    print("\n[DIRECTION 2 OVERALL ASSESSMENT]")
    if avg_deviation >= 0.5:
        print(f"  STRONG DEVIATION: {avg_deviation:.4f} >= 0.5")
        print(f"  -> High per-frame deviation suggests significant closed-loop impact")
    elif avg_deviation >= 0.3:
        print(f"  MODERATE DEVIATION: {avg_deviation:.4f} >= 0.3")
        print(f"  -> Moderate per-frame deviation may cause task failure over long episodes")
    else:
        print(f"  WEAK DEVIATION: {avg_deviation:.4f} < 0.3")
        print(f"  -> Low per-frame deviation may not cause sufficient trajectory drift")

    # Cumulative interpretation
    print(f"\n  Cumulative Position Drift: {total_cumulative_pos:.4f}m over {len(all_results)} frames")
    frames_per_ep = len(all_results) / len(test_episodes)
    expected_drift_100_frames = (total_cumulative_pos / len(all_results)) * 100
    print(f"  Projected Drift (100 frames): {expected_drift_100_frames:.4f}m")
    if expected_drift_100_frames >= 0.5:
        print(f"  -> Significant trajectory drift expected in closed-loop execution")
    elif expected_drift_100_frames >= 0.2:
        print(f"  -> Moderate trajectory drift may cause task failure")
    else:
        print(f"  -> Low trajectory drift, may need larger patch or more queries")

    print("=" * 80)

    # Save comprehensive results
    summary = {
        "run_id": run_id,
        "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S'),
        "direction": "Direction 2: Deviation Analysis for Closed-Loop Attack",
        "patch_path": str(args.patch_path),
        "patch_position": patch_position,
        "args": vars(args),
        "task_info": {
            "task_name": task_data['task_name'],
            "instruction": task_data['instruction'],
            "total_episodes": task_data['num_episodes'],
            "test_episodes": len(test_episodes),
            "test_indices": test_indices,
            "frames_per_episode": args.frames_per_episode,
            "total_frames": len(all_results)
        },
        "testing_time_seconds": testing_time,
        "metrics": {
            # Frame-level metrics
            "deviation": {
                "average": float(avg_deviation),
                "std": float(std_deviation),
                "max": float(max(all_deviations)),
                "min": float(min(all_deviations)),
                "rate": float(deviation_rate)
            },
            "components": {
                "position": float(avg_pos_dev),
                "rotation": float(avg_rot_dev),
                "gripper": float(avg_grip_dev)
            },
            "cumulative": {
                "avg_per_episode": float(avg_episode_cumulative),
                "total": float(total_cumulative_deviation),
                "total_position": float(total_cumulative_pos),
                "projected_100_frames": float(expected_drift_100_frames)
            },
            "baseline": {
                "avg_gt_error_clean": float(avg_gt_error_clean),
                "avg_gt_error_patched": float(avg_gt_error_patched),
                "error_change": float(avg_gt_error_patched - avg_gt_error_clean)
            },
            # TRAJECTORY-LEVEL METRICS (NEW)
            "trajectory": aggregate_traj
        },
        "episode_metrics": episode_metrics
    }

    # Save summary
    results_path = results_dir / f"{run_id}_testing.json"
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SAVED] Testing summary: {results_path}")

    # Save detailed per-frame results
    detailed_path = results_dir / f"{run_id}_detailed.json"
    with open(detailed_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"[SAVED] Detailed results: {detailed_path}")

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return summary


if __name__ == "__main__":
    main()
