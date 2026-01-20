#!/usr/bin/env python3
"""
SE(3) ZOO Adversarial Patch Training Script - Direction 2 Aligned

Trains an adversarial patch using Zero-Order Optimization to MAXIMIZE DEVIATION
from the model's clean predictions.

Direction 2 Goal: Create patches that cause consistent per-frame deviations,
which accumulate in closed-loop execution to cause task failure.

Key Design Decisions:
1. Uses ZOO V2 (maximize deviation) - NO adversarial target needed
2. Best patch selected by DEVIATION metric, not loss
3. Mini-batch gradient estimation for stability
4. Reproducible with proper seed passing
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
from attacks.zoo_optimizer_v2 import ZOOSOptimizerV2
from utils.libero_loader import LIBEROLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description='SE(3) ZOO Adversarial Patch Training (Direction 2: Maximize Deviation)'
    )

    # Task configuration
    parser.add_argument('--suite', type=str, default='libero_spatial',
                        choices=['libero_spatial', 'libero_object', 'libero_goal', 'libero_10'],
                        help='LIBERO task suite')
    parser.add_argument('--task_id', type=int, default=0,
                        help='Task ID within the suite (0-9)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Fraction of episodes for training (default: 0.7 = 35/50)')

    # Optimization parameters
    parser.add_argument('--queries', type=int, default=200,
                        help='Query budget for ZOO optimization')
    parser.add_argument('--mini_batch_size', type=int, default=3,
                        help='Number of frames to average for gradient estimation')
    parser.add_argument('--patch_size', type=int, default=32,
                        help='Size of adversarial patch (default: 32x32)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for patch optimization')
    parser.add_argument('--perturbation_scale', type=float, default=0.1,
                        help='Scale of random perturbations for gradient estimation')

    # Deviation thresholds
    parser.add_argument('--deviation_threshold', type=float, default=0.3,
                        help='SE(3) distance threshold for "significant" deviation')
    parser.add_argument('--early_stop_threshold', type=float, default=90.0,
                        help='Deviation rate threshold for early stopping (%)')
    parser.add_argument('--early_stop_patience', type=int, default=20,
                        help='Number of consecutive steps above threshold for early stop')

    # Component weights (Direction 2: position matters most for trajectory drift)
    parser.add_argument('--position_weight', type=float, default=1.0,
                        help='Weight on position deviation in loss')
    parser.add_argument('--rotation_weight', type=float, default=1.0,
                        help='Weight on rotation deviation in loss')
    parser.add_argument('--gripper_weight', type=float, default=5.0,
                        help='Weight on gripper change in loss')

    # Patch placement
    parser.add_argument('--patch_x', type=int, default=48,
                        help='Patch X position (default: 48 for 128x128 images)')
    parser.add_argument('--patch_y', type=int, default=48,
                        help='Patch Y position (default: 48 for 128x128 images)')

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
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Custom experiment name (default: auto-generated)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create output directories
    output_dir = Path(args.output_dir)
    patches_dir = output_dir / 'patches'
    results_dir = output_dir / 'results'
    logs_dir = output_dir / 'logs'
    patches_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Generate experiment name
    if args.experiment_name:
        exp_name = args.experiment_name
    else:
        exp_name = f"dir2_{args.suite}_task{args.task_id}_q{args.queries}"

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id = f"{exp_name}_{timestamp}"

    print("=" * 80)
    print("DIRECTION 2: MAXIMIZE DEVIATION ATTACK")
    print("=" * 80)
    print(f"Goal: Maximize per-frame deviation for closed-loop trajectory attack")
    print("=" * 80)
    print(f"Run ID:           {run_id}")
    print(f"Start time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Suite:            {args.suite}")
    print(f"Task ID:          {args.task_id}")
    print(f"Train ratio:      {args.train_ratio} ({int(args.train_ratio*50)}/50 episodes)")
    print(f"Query budget:     {args.queries}")
    print(f"Mini-batch size:  {args.mini_batch_size}")
    print(f"Patch size:       {args.patch_size}x{args.patch_size}")
    print(f"Patch position:   ({args.patch_x}, {args.patch_y})")
    print(f"Learning rate:    {args.lr}")
    print(f"Perturbation σ:   {args.perturbation_scale}")
    print(f"Deviation thresh: {args.deviation_threshold}")
    print(f"Weights:          pos={args.position_weight}, rot={args.rotation_weight}, grip={args.gripper_weight}")
    print(f"Device:           {args.device}")
    print(f"Seed:             {args.seed}")
    print("=" * 80)

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU:              {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("WARNING: CUDA not available, using CPU")

    # Load LIBERO data
    print(f"\n[1/3] Loading LIBERO data...")
    loader = LIBEROLoader()
    task_data = loader.load_task(args.suite, args.task_id)

    # Use DETERMINISTIC split with seed
    train_episodes, test_episodes = loader.split_episodes(
        task_data["episodes"],
        train_ratio=args.train_ratio,
        seed=args.seed
    )

    # Get and print split indices for verification
    train_indices, test_indices = loader.get_split_indices(
        len(task_data["episodes"]),
        train_ratio=args.train_ratio,
        seed=args.seed
    )

    print(f"  Task:           {task_data['task_name']}")
    print(f"  Instruction:    {task_data['instruction']}")
    print(f"  Total episodes: {task_data['num_episodes']}")
    print(f"  Train episodes: {len(train_episodes)} (indices: {train_indices[:5]}...)")
    print(f"  Test episodes:  {len(test_episodes)} (indices: {test_indices[:5]}...)")

    # Load OpenVLA model
    print(f"\n[2/3] Loading OpenVLA model...")
    model = OpenVLAActionExtractor(
        model_path=args.model_path,
        device=args.device
    )

    # Initialize optimizer (V2 - maximize deviation)
    print(f"\n[3/3] Initializing ZOO V2 Optimizer...")
    optimizer = ZOOSOptimizerV2(
        model=model,
        patch_size=args.patch_size,
        learning_rate=args.lr,
        perturbation_scale=args.perturbation_scale,
        query_budget=args.queries,
        mini_batch_size=args.mini_batch_size,
        early_stop_threshold=args.early_stop_threshold,
        early_stop_patience=args.early_stop_patience,
        deviation_threshold=args.deviation_threshold,
        position_weight=args.position_weight,
        rotation_weight=args.rotation_weight,
        gripper_weight=args.gripper_weight,
        seed=args.seed  # FIXED: Now passing seed to optimizer
    )

    # Train patch
    patch_position = (args.patch_x, args.patch_y)
    start_time = time.time()
    results = optimizer.train(train_episodes, patch_position=patch_position)
    training_time = time.time() - start_time

    print("=" * 80)
    print(f"Training Complete!")
    print(f"  Total time:         {training_time:.1f} seconds ({training_time/60:.1f} min)")
    print(f"  Total queries:      {results['total_queries']}")
    print(f"  Best avg deviation: {results['best_avg_deviation']:.4f}")
    print("=" * 80)

    # Save patch
    patch_path = patches_dir / f"{run_id}_patch.npy"
    np.save(patch_path, results['best_patch'])
    print(f"\n[SAVED] Best patch: {patch_path}")

    # Save final patch as image for visualization
    from PIL import Image
    patch_img = (results['best_patch'] * 255).astype(np.uint8)
    patch_img_path = patches_dir / f"{run_id}_patch.png"
    Image.fromarray(patch_img).save(patch_img_path)
    print(f"[SAVED] Patch image: {patch_img_path}")

    # Compute training statistics
    history = results['training_history']
    deviations = [h['avg_deviation'] for h in history]
    deviation_rates = [h['deviation_rate'] for h in history]

    training_stats = {
        "deviation": {
            "initial": deviations[0] if deviations else 0,
            "final": np.mean(deviations[-10:]) if len(deviations) >= 10 else np.mean(deviations) if deviations else 0,
            "max": max(deviations) if deviations else 0,
            "mean": np.mean(deviations) if deviations else 0,
            "std": np.std(deviations) if deviations else 0,
            "best_rolling_avg": results['best_avg_deviation']
        },
        "deviation_rate": {
            "initial": deviation_rates[0] if deviation_rates else 0,
            "final": np.mean(deviation_rates[-10:]) if len(deviation_rates) >= 10 else np.mean(deviation_rates) if deviation_rates else 0,
            "max": max(deviation_rates) if deviation_rates else 0,
            "mean": np.mean(deviation_rates) if deviation_rates else 0
        }
    }

    # Save comprehensive training results
    training_results = {
        "run_id": run_id,
        "timestamp": timestamp,
        "direction": "Direction 2: Maximize Deviation for Closed-Loop Attack",
        "args": vars(args),
        "task_info": {
            "task_name": task_data['task_name'],
            "instruction": task_data['instruction'],
            "total_episodes": task_data['num_episodes'],
            "train_episodes": len(train_episodes),
            "test_episodes": len(test_episodes),
            "train_indices": train_indices,
            "test_indices": test_indices
        },
        "training_summary": {
            "training_time_seconds": training_time,
            "total_queries": results['total_queries'],
            "best_avg_deviation": results['best_avg_deviation']
        },
        "training_stats": training_stats,
        "patch_info": {
            "path": str(patch_path),
            "shape": list(results['best_patch'].shape),
            "position": patch_position,
            "range": [float(results['best_patch'].min()), float(results['best_patch'].max())]
        },
        "query_history": [
            {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in h.items()}
            for h in history
        ]
    }

    results_path = results_dir / f"{run_id}_training.json"
    with open(results_path, 'w') as f:
        json.dump(training_results, f, indent=2)
    print(f"[SAVED] Training results: {results_path}")

    # Print final summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY (Direction 2: Maximize Deviation)")
    print("=" * 80)
    print(f"Run ID:              {run_id}")
    print(f"Suite/Task:          {args.suite} / task_{args.task_id}")
    print(f"Train episodes:      {len(train_episodes)}")
    print(f"Queries used:        {results['total_queries']}")
    print(f"Training time:       {training_time:.1f}s ({training_time/60:.1f} min)")
    print("-" * 80)
    print(f"DEVIATION METRICS (key for Direction 2):")
    print(f"  Best avg deviation:  {results['best_avg_deviation']:.4f}")
    print(f"  Initial deviation:   {training_stats['deviation']['initial']:.4f}")
    print(f"  Final deviation:     {training_stats['deviation']['final']:.4f}")
    print(f"  Deviation rate:      {training_stats['deviation_rate']['final']*100:.1f}%")
    print("-" * 80)
    print(f"Patch saved:         {patch_path}")
    print(f"Results saved:       {results_path}")
    print("=" * 80)

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated() / 1e9
        print(f"Final GPU memory: {final_memory:.1f} GB")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Return path to patch for testing script
    return str(patch_path), run_id


if __name__ == "__main__":
    main()
