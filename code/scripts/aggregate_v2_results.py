#!/usr/bin/env python3
"""
Aggregate V2 Evaluation Results

Collects and summarizes results from all 10 LIBERO Spatial tasks
after running the V2 evaluation SLURM jobs.

Usage:
    python code/scripts/aggregate_v2_results.py --experiment_prefix v2_eval

This script will:
1. Find all testing JSON files matching the prefix
2. Extract key metrics from each task
3. Generate a summary report
4. Save aggregated results to a JSON file
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate V2 evaluation results')
    parser.add_argument('--output_dir', type=str,
                        default='/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/',
                        help='Directory containing results')
    parser.add_argument('--experiment_prefix', type=str, default='v2_eval',
                        help='Prefix for experiment names to aggregate')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save aggregated results (default: auto-generated)')
    return parser.parse_args()


def load_testing_results(results_dir: Path, prefix: str) -> list:
    """Load all testing results matching the prefix."""
    results = []

    for f in results_dir.glob(f"{prefix}*_testing.json"):
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                results.append({
                    'file': str(f),
                    'data': data
                })
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")

    return sorted(results, key=lambda x: x['data'].get('args', {}).get('task_id', -1))


def extract_key_metrics(result: dict) -> dict:
    """Extract key metrics from a testing result."""
    data = result['data']
    args = data.get('args', {})
    metrics = data.get('metrics', {})
    traj = metrics.get('trajectory', {})

    return {
        'task_id': args.get('task_id', -1),
        'task_name': data.get('task_info', {}).get('task_name', 'Unknown'),

        # Frame-level metrics
        'avg_deviation': metrics.get('deviation', {}).get('average', 0),
        'deviation_rate': metrics.get('deviation', {}).get('rate', 0),

        # Component breakdown
        'position_dev': metrics.get('components', {}).get('position', 0),
        'rotation_dev': metrics.get('components', {}).get('rotation', 0),
        'gripper_dev': metrics.get('components', {}).get('gripper', 0),

        # ACTUAL trajectory drift (the correct metric)
        'actual_drift_mean': traj.get('actual_drift', {}).get('mean', 0),
        'actual_drift_max': traj.get('actual_drift', {}).get('max', 0),
        'drift_consistency': traj.get('actual_drift', {}).get('consistency_mean', 0),

        # Legacy metric for comparison
        'legacy_drift_mean': traj.get('legacy_cumulative_drift', {}).get('mean', 0),

        # CDT and TFP (now using actual drift)
        'cdt_success_rate': traj.get('cdt', {}).get('success_rate', 0),
        'tfp_mean': traj.get('tfp', {}).get('mean_score', 0),
        'tfp_above_1_rate': traj.get('tfp', {}).get('above_1_rate', 0),

        # TTF
        'ttf_mean': traj.get('ttf', {}).get('mean_frames'),
        'ttf_success_rate': traj.get('ttf', {}).get('success_rate', 0),

        # SDR
        'sdr': traj.get('sdr', {}),

        # Timing
        'testing_time': data.get('testing_time_seconds', 0),
        'patch_path': data.get('patch_path', '')
    }


def print_summary_table(all_metrics: list):
    """Print a summary table of all results."""

    print("\n" + "=" * 120)
    print("V2 EVALUATION SUMMARY - ALL TASKS")
    print("=" * 120)

    # Header
    print(f"\n{'Task':>5} | {'Avg Dev':>8} | {'Dev Rate':>8} | {'Pos Dev':>8} | {'Grip Dev':>8} | "
          f"{'Actual Drift':>12} | {'Drift Cons':>10} | {'CDT Rate':>8} | {'TFP Mean':>8}")
    print("-" * 120)

    # Data rows
    for m in all_metrics:
        print(f"{m['task_id']:>5} | {m['avg_deviation']:>8.4f} | {m['deviation_rate']*100:>7.1f}% | "
              f"{m['position_dev']:>8.4f} | {m['gripper_dev']:>8.4f} | "
              f"{m['actual_drift_mean']:>12.4f} | {m['drift_consistency']:>10.2f} | "
              f"{m['cdt_success_rate']*100:>7.1f}% | {m['tfp_mean']:>8.2f}")

    print("-" * 120)

    # Aggregates
    n = len(all_metrics)
    if n > 0:
        avg_dev = np.mean([m['avg_deviation'] for m in all_metrics])
        avg_rate = np.mean([m['deviation_rate'] for m in all_metrics])
        avg_pos = np.mean([m['position_dev'] for m in all_metrics])
        avg_grip = np.mean([m['gripper_dev'] for m in all_metrics])
        avg_actual = np.mean([m['actual_drift_mean'] for m in all_metrics])
        avg_cons = np.mean([m['drift_consistency'] for m in all_metrics])
        avg_cdt = np.mean([m['cdt_success_rate'] for m in all_metrics])
        avg_tfp = np.mean([m['tfp_mean'] for m in all_metrics])

        print(f"{'MEAN':>5} | {avg_dev:>8.4f} | {avg_rate*100:>7.1f}% | "
              f"{avg_pos:>8.4f} | {avg_grip:>8.4f} | "
              f"{avg_actual:>12.4f} | {avg_cons:>10.2f} | "
              f"{avg_cdt*100:>7.1f}% | {avg_tfp:>8.2f}")

    print("=" * 120)

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    if n > 0:
        # Check gripper vs position ratio
        avg_grip_ratio = np.mean([m['gripper_dev'] / (m['avg_deviation'] + 1e-8) for m in all_metrics])
        avg_pos_ratio = np.mean([m['position_dev'] / (m['avg_deviation'] + 1e-8) for m in all_metrics])

        print(f"\n1. Component Contribution to Total Deviation:")
        print(f"   Position: {avg_pos_ratio*100:.1f}% | Gripper: {avg_grip_ratio*100:.1f}%")

        if avg_grip_ratio > 0.5:
            print(f"   WARNING: Gripper still dominates! Consider using --position_only mode.")
        else:
            print(f"   GOOD: Position is the primary contributor (gripper de-prioritized).")

        # Check drift consistency
        print(f"\n2. Drift Consistency (attack direction stability):")
        print(f"   Mean: {avg_cons:.2f}")
        if avg_cons < 0.5:
            print(f"   WARNING: Low consistency - attack oscillates! Actual drift << Legacy drift.")
            print(f"   Actual: {avg_actual:.4f}m vs Legacy: {np.mean([m['legacy_drift_mean'] for m in all_metrics]):.4f}m")
        else:
            print(f"   GOOD: Attack produces consistent directional drift.")

        # Check CDT success
        print(f"\n3. Attack Success Rate (CDT @ 0.05m threshold):")
        print(f"   Mean CDT Success: {avg_cdt*100:.1f}%")
        if avg_cdt > 0.5:
            print(f"   GOOD: Attack exceeds drift threshold in majority of episodes.")
        else:
            print(f"   NEEDS IMPROVEMENT: Attack rarely exceeds drift threshold.")

        # Check TFP
        print(f"\n4. Task Failure Proxy (TFP):")
        print(f"   Mean TFP: {avg_tfp:.2f}x task scale")
        tfp_above_1 = np.mean([m['tfp_above_1_rate'] for m in all_metrics])
        print(f"   TFP > 1.0 rate: {tfp_above_1*100:.1f}%")
        if avg_tfp > 1.0:
            print(f"   GOOD: Attack causes significant task-relevant drift.")
        elif avg_tfp > 0.5:
            print(f"   MODERATE: Attack causes meaningful drift but may not always fail tasks.")
        else:
            print(f"   WEAK: Attack drift is small relative to task scale.")

    print("\n" + "=" * 80)


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    results_dir = output_dir / 'results'

    print("=" * 80)
    print("V2 RESULTS AGGREGATOR")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Experiment prefix: {args.experiment_prefix}")
    print("=" * 80)

    # Load results
    results = load_testing_results(results_dir, args.experiment_prefix)

    if not results:
        print(f"\nERROR: No results found matching prefix '{args.experiment_prefix}'")
        print(f"Looking in: {results_dir}")
        print("\nAvailable files:")
        for f in results_dir.glob("*.json"):
            print(f"  {f.name}")
        return

    print(f"\nFound {len(results)} result files")

    # Extract metrics
    all_metrics = [extract_key_metrics(r) for r in results]

    # Print summary
    print_summary_table(all_metrics)

    # Save aggregated results
    if args.save_path:
        save_path = Path(args.save_path)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = output_dir / 'experiments' / 'v2_evaluation' / f'aggregated_results_{timestamp}.json'

    save_path.parent.mkdir(parents=True, exist_ok=True)

    aggregated = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'experiment_prefix': args.experiment_prefix,
        'num_tasks': len(all_metrics),
        'per_task_metrics': all_metrics,
        'summary': {
            'avg_deviation': float(np.mean([m['avg_deviation'] for m in all_metrics])),
            'avg_position_dev': float(np.mean([m['position_dev'] for m in all_metrics])),
            'avg_gripper_dev': float(np.mean([m['gripper_dev'] for m in all_metrics])),
            'avg_actual_drift': float(np.mean([m['actual_drift_mean'] for m in all_metrics])),
            'avg_drift_consistency': float(np.mean([m['drift_consistency'] for m in all_metrics])),
            'avg_cdt_success_rate': float(np.mean([m['cdt_success_rate'] for m in all_metrics])),
            'avg_tfp_mean': float(np.mean([m['tfp_mean'] for m in all_metrics])),
            'avg_tfp_above_1_rate': float(np.mean([m['tfp_above_1_rate'] for m in all_metrics]))
        }
    }

    with open(save_path, 'w') as f:
        json.dump(aggregated, f, indent=2)

    print(f"\n[SAVED] Aggregated results: {save_path}")
    print(f"\nRun complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
