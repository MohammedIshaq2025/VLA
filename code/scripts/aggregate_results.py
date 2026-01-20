#!/usr/bin/env python3
"""
Results Aggregation and Analysis Script for Direction 2 Experiments

This script aggregates results from multiple experiment runs and generates
comprehensive analysis reports including:
- Per-task performance summary
- Per-query-budget analysis
- Trajectory-level metric aggregation
- Statistical comparisons
- Publication-ready tables

Usage:
    python aggregate_results.py --exp_dir /path/to/experiment/dir
    python aggregate_results.py --results_file /path/to/all_results.json
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Aggregate and analyze Direction 2 experiment results'
    )

    parser.add_argument('--exp_dir', type=str, default=None,
                        help='Experiment directory containing all_results.json')
    parser.add_argument('--results_file', type=str, default=None,
                        help='Direct path to results JSON file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for analysis (default: same as exp_dir)')

    return parser.parse_args()


def load_results(exp_dir: Optional[str] = None,
                 results_file: Optional[str] = None) -> Dict[str, Any]:
    """Load results from file or directory."""

    if results_file:
        with open(results_file, 'r') as f:
            return json.load(f)
    elif exp_dir:
        results_path = Path(exp_dir) / 'all_results.json'
        if results_path.exists():
            with open(results_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"No all_results.json found in {exp_dir}")
    else:
        raise ValueError("Must provide either exp_dir or results_file")


def compute_per_task_summary(results: List[Dict]) -> Dict[int, Dict]:
    """Compute summary statistics per task across all query budgets."""

    by_task = {}
    for r in results:
        task_id = r['task_id']
        if task_id not in by_task:
            by_task[task_id] = []
        by_task[task_id].append(r)

    summaries = {}
    for task_id, task_results in by_task.items():
        # Get best result (highest avg deviation) for this task
        best_result = max(task_results, key=lambda x: x['frame_metrics']['avg_deviation'])

        # Compute statistics across query budgets
        avg_devs = [r['frame_metrics']['avg_deviation'] for r in task_results]
        cdt_rates = [r['trajectory_metrics']['cdt_success_rate'] for r in task_results]
        tfp_scores = [r['trajectory_metrics']['tfp_mean_score'] for r in task_results]
        drifts = [r['trajectory_metrics']['drift_mean'] for r in task_results]

        summaries[task_id] = {
            'n_runs': len(task_results),
            'best_queries': best_result['queries'],
            'best_avg_deviation': best_result['frame_metrics']['avg_deviation'],
            'best_cdt_rate': best_result['trajectory_metrics']['cdt_success_rate'],
            'best_tfp_score': best_result['trajectory_metrics']['tfp_mean_score'],
            'avg_deviation': {
                'mean': float(np.mean(avg_devs)),
                'std': float(np.std(avg_devs)),
                'min': float(np.min(avg_devs)),
                'max': float(np.max(avg_devs))
            },
            'cdt_rate': {
                'mean': float(np.mean(cdt_rates)),
                'std': float(np.std(cdt_rates))
            },
            'tfp_score': {
                'mean': float(np.mean(tfp_scores)),
                'std': float(np.std(tfp_scores))
            },
            'drift': {
                'mean': float(np.mean(drifts)),
                'std': float(np.std(drifts))
            }
        }

    return summaries


def compute_per_query_summary(results: List[Dict]) -> Dict[int, Dict]:
    """Compute summary statistics per query budget across all tasks."""

    by_queries = {}
    for r in results:
        queries = r['queries']
        if queries not in by_queries:
            by_queries[queries] = []
        by_queries[queries].append(r)

    summaries = {}
    for queries, query_results in by_queries.items():
        # Frame-level metrics
        avg_devs = [r['frame_metrics']['avg_deviation'] for r in query_results]
        dev_rates = [r['frame_metrics']['deviation_rate'] for r in query_results]
        pos_devs = [r['frame_metrics']['position_deviation'] for r in query_results]

        # Trajectory-level metrics
        cdt_rates = [r['trajectory_metrics']['cdt_success_rate'] for r in query_results]
        tfp_scores = [r['trajectory_metrics']['tfp_mean_score'] for r in query_results]
        tfp_above_1 = [r['trajectory_metrics']['tfp_above_1_rate'] for r in query_results]
        drifts = [r['trajectory_metrics']['drift_mean'] for r in query_results]

        # TTF (filter None values)
        ttf_values = [r['trajectory_metrics']['ttf_mean_frames']
                      for r in query_results
                      if r['trajectory_metrics']['ttf_mean_frames'] is not None]

        # SDR metrics
        sdr_10 = [r['trajectory_metrics']['sdr_10'] for r in query_results]
        sdr_25 = [r['trajectory_metrics']['sdr_25'] for r in query_results]
        sdr_50 = [r['trajectory_metrics']['sdr_50'] for r in query_results]

        summaries[queries] = {
            'n_tasks': len(query_results),
            'frame_metrics': {
                'avg_deviation': {
                    'mean': float(np.mean(avg_devs)),
                    'std': float(np.std(avg_devs)),
                    'min': float(np.min(avg_devs)),
                    'max': float(np.max(avg_devs))
                },
                'deviation_rate': {
                    'mean': float(np.mean(dev_rates)),
                    'std': float(np.std(dev_rates))
                },
                'position_deviation': {
                    'mean': float(np.mean(pos_devs)),
                    'std': float(np.std(pos_devs))
                }
            },
            'trajectory_metrics': {
                'cdt_success_rate': {
                    'mean': float(np.mean(cdt_rates)),
                    'std': float(np.std(cdt_rates))
                },
                'tfp_score': {
                    'mean': float(np.mean(tfp_scores)),
                    'std': float(np.std(tfp_scores))
                },
                'tfp_above_1_rate': {
                    'mean': float(np.mean(tfp_above_1)),
                    'std': float(np.std(tfp_above_1))
                },
                'drift': {
                    'mean': float(np.mean(drifts)),
                    'std': float(np.std(drifts))
                },
                'ttf': {
                    'mean': float(np.mean(ttf_values)) if ttf_values else None,
                    'std': float(np.std(ttf_values)) if ttf_values else None,
                    'n_valid': len(ttf_values)
                },
                'sdr': {
                    '10': {'mean': float(np.mean(sdr_10)), 'std': float(np.std(sdr_10))},
                    '25': {'mean': float(np.mean(sdr_25)), 'std': float(np.std(sdr_25))},
                    '50': {'mean': float(np.mean(sdr_50)), 'std': float(np.std(sdr_50))}
                }
            }
        }

    return summaries


def compute_overall_summary(results: List[Dict]) -> Dict[str, Any]:
    """Compute overall summary across all experiments."""

    # All frame-level metrics
    avg_devs = [r['frame_metrics']['avg_deviation'] for r in results]
    dev_rates = [r['frame_metrics']['deviation_rate'] for r in results]
    pos_devs = [r['frame_metrics']['position_deviation'] for r in results]

    # All trajectory-level metrics
    cdt_rates = [r['trajectory_metrics']['cdt_success_rate'] for r in results]
    tfp_scores = [r['trajectory_metrics']['tfp_mean_score'] for r in results]
    tfp_above_1 = [r['trajectory_metrics']['tfp_above_1_rate'] for r in results]
    drifts = [r['trajectory_metrics']['drift_mean'] for r in results]

    return {
        'n_total_runs': len(results),
        'n_tasks': len(set(r['task_id'] for r in results)),
        'n_query_budgets': len(set(r['queries'] for r in results)),
        'frame_metrics': {
            'avg_deviation': {
                'mean': float(np.mean(avg_devs)),
                'std': float(np.std(avg_devs)),
                'median': float(np.median(avg_devs)),
                'min': float(np.min(avg_devs)),
                'max': float(np.max(avg_devs))
            },
            'deviation_rate': {
                'mean': float(np.mean(dev_rates)),
                'std': float(np.std(dev_rates))
            },
            'position_deviation': {
                'mean': float(np.mean(pos_devs)),
                'std': float(np.std(pos_devs)),
                'projected_100_frames': float(np.mean(pos_devs) * 100)
            }
        },
        'trajectory_metrics': {
            'cdt_success_rate': {
                'mean': float(np.mean(cdt_rates)),
                'std': float(np.std(cdt_rates))
            },
            'tfp_score': {
                'mean': float(np.mean(tfp_scores)),
                'std': float(np.std(tfp_scores))
            },
            'tfp_above_1_rate': {
                'mean': float(np.mean(tfp_above_1)),
                'std': float(np.std(tfp_above_1))
            },
            'drift': {
                'mean': float(np.mean(drifts)),
                'std': float(np.std(drifts))
            }
        }
    }


def generate_latex_table(per_query: Dict[int, Dict], output_path: Path) -> None:
    """Generate LaTeX table for paper."""

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Direction 2 Attack Results on LIBERO-Spatial (10 tasks)}",
        r"\label{tab:dir2_results}",
        r"\begin{tabular}{l|ccc|ccc}",
        r"\toprule",
        r"Queries & \multicolumn{3}{c|}{Frame-Level} & \multicolumn{3}{c}{Trajectory-Level} \\",
        r"        & Avg Dev & Dev Rate & Pos Dev & CDT Rate & TFP Score & TFP>1 \\",
        r"\midrule"
    ]

    for queries in sorted(per_query.keys()):
        s = per_query[queries]
        fm = s['frame_metrics']
        tm = s['trajectory_metrics']

        line = f"{queries} & "
        line += f"{fm['avg_deviation']['mean']:.3f}$\\pm${fm['avg_deviation']['std']:.3f} & "
        line += f"{fm['deviation_rate']['mean']*100:.1f}\\% & "
        line += f"{fm['position_deviation']['mean']:.4f} & "
        line += f"{tm['cdt_success_rate']['mean']*100:.1f}\\% & "
        line += f"{tm['tfp_score']['mean']:.2f} & "
        line += f"{tm['tfp_above_1_rate']['mean']*100:.1f}\\% \\\\"
        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def generate_markdown_report(data: Dict, per_task: Dict, per_query: Dict,
                              overall: Dict, output_path: Path) -> None:
    """Generate comprehensive markdown report."""

    lines = [
        "# Direction 2: Full Evaluation Results",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Experiment**: {data.get('experiment_name', 'Unknown')}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"- **Total Runs**: {overall['n_total_runs']}",
        f"- **Tasks Evaluated**: {overall['n_tasks']}",
        f"- **Query Budgets**: {overall['n_query_budgets']}",
        "",
        "### Key Findings",
        "",
        f"| Metric | Mean | Std |",
        f"|--------|------|-----|",
        f"| Average Deviation | {overall['frame_metrics']['avg_deviation']['mean']:.4f} | {overall['frame_metrics']['avg_deviation']['std']:.4f} |",
        f"| Deviation Rate | {overall['frame_metrics']['deviation_rate']['mean']*100:.1f}% | {overall['frame_metrics']['deviation_rate']['std']*100:.1f}% |",
        f"| CDT Success Rate | {overall['trajectory_metrics']['cdt_success_rate']['mean']*100:.1f}% | {overall['trajectory_metrics']['cdt_success_rate']['std']*100:.1f}% |",
        f"| TFP Score | {overall['trajectory_metrics']['tfp_score']['mean']:.2f}x | {overall['trajectory_metrics']['tfp_score']['std']:.2f}x |",
        f"| TFP > 1.0 Rate | {overall['trajectory_metrics']['tfp_above_1_rate']['mean']*100:.1f}% | {overall['trajectory_metrics']['tfp_above_1_rate']['std']*100:.1f}% |",
        "",
        f"**Projected 100-frame Position Drift**: {overall['frame_metrics']['position_deviation']['projected_100_frames']:.2f}m",
        "",
        "---",
        "",
        "## Results by Query Budget",
        ""
    ]

    # Per-query table
    lines.append("| Queries | Avg Dev | Dev Rate | Pos Dev | CDT Rate | TFP Score | TFP>1 | Mean Drift |")
    lines.append("|---------|---------|----------|---------|----------|-----------|-------|------------|")

    for queries in sorted(per_query.keys()):
        s = per_query[queries]
        fm = s['frame_metrics']
        tm = s['trajectory_metrics']
        lines.append(
            f"| {queries} | "
            f"{fm['avg_deviation']['mean']:.4f} | "
            f"{fm['deviation_rate']['mean']*100:.1f}% | "
            f"{fm['position_deviation']['mean']:.4f} | "
            f"{tm['cdt_success_rate']['mean']*100:.1f}% | "
            f"{tm['tfp_score']['mean']:.2f}x | "
            f"{tm['tfp_above_1_rate']['mean']*100:.1f}% | "
            f"{tm['drift']['mean']:.4f}m |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Results by Task",
        ""
    ])

    # Per-task table
    lines.append("| Task | Best Queries | Best Avg Dev | Best CDT | Best TFP | Mean Drift |")
    lines.append("|------|--------------|--------------|----------|----------|------------|")

    for task_id in sorted(per_task.keys()):
        s = per_task[task_id]
        lines.append(
            f"| {task_id} | "
            f"{s['best_queries']} | "
            f"{s['best_avg_deviation']:.4f} | "
            f"{s['best_cdt_rate']*100:.1f}% | "
            f"{s['best_tfp_score']:.2f}x | "
            f"{s['drift']['mean']:.4f}m |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Trajectory-Level Metrics Explained",
        "",
        "### CDT (Cumulative Drift Threshold)",
        "- Binary success: Does cumulative position drift exceed threshold (0.2m)?",
        "- Higher is better for attack effectiveness",
        "",
        "### TTF (Time-to-Failure)",
        "- Minimum frames needed to exceed drift threshold",
        "- Lower is better (faster attack)",
        "",
        "### SDR (Sustained Deviation Rate)",
        "- Fraction of N-frame windows with mean deviation > threshold",
        "- Higher is better (consistent attack effect)",
        "",
        "### TFP (Task Failure Proxy)",
        "- Ratio of cumulative drift to typical task scale (0.5m)",
        "- TFP > 1.0 suggests high probability of task failure",
        "",
        "---",
        "",
        f"*Report generated by aggregate_results.py*"
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    args = parse_args()

    # Determine paths
    if args.results_file:
        data = load_results(results_file=args.results_file)
        output_dir = Path(args.output_dir) if args.output_dir else Path(args.results_file).parent
    elif args.exp_dir:
        data = load_results(exp_dir=args.exp_dir)
        output_dir = Path(args.output_dir) if args.output_dir else Path(args.exp_dir)
    else:
        print("ERROR: Must provide --exp_dir or --results_file")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DIRECTION 2: RESULTS AGGREGATION")
    print("=" * 80)

    results = data.get('results', [])
    if not results:
        print("ERROR: No results found in data")
        sys.exit(1)

    print(f"Loaded {len(results)} experiment results")

    # Compute summaries
    print("\n[1/4] Computing per-task summary...")
    per_task = compute_per_task_summary(results)

    print("[2/4] Computing per-query summary...")
    per_query = compute_per_query_summary(results)

    print("[3/4] Computing overall summary...")
    overall = compute_overall_summary(results)

    # Save analysis
    print("[4/4] Generating reports...")

    # JSON analysis
    analysis = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'source_experiment': data.get('experiment_name', 'Unknown'),
        'overall': overall,
        'per_task': {str(k): v for k, v in per_task.items()},
        'per_query': {str(k): v for k, v in per_query.items()}
    }

    analysis_path = output_dir / 'analysis.json'
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"  Saved: {analysis_path}")

    # LaTeX table
    latex_path = output_dir / 'results_table.tex'
    generate_latex_table(per_query, latex_path)
    print(f"  Saved: {latex_path}")

    # Markdown report
    report_path = output_dir / 'RESULTS_REPORT.md'
    generate_markdown_report(data, per_task, per_query, overall, report_path)
    print(f"  Saved: {report_path}")

    # Print summary to console
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nOverall (n={overall['n_total_runs']} runs):")
    print(f"  Avg Deviation: {overall['frame_metrics']['avg_deviation']['mean']:.4f} +/- {overall['frame_metrics']['avg_deviation']['std']:.4f}")
    print(f"  Dev Rate:      {overall['frame_metrics']['deviation_rate']['mean']*100:.1f}%")
    print(f"  CDT Rate:      {overall['trajectory_metrics']['cdt_success_rate']['mean']*100:.1f}%")
    print(f"  TFP Score:     {overall['trajectory_metrics']['tfp_score']['mean']:.2f}x")
    print(f"  TFP > 1.0:     {overall['trajectory_metrics']['tfp_above_1_rate']['mean']*100:.1f}%")

    print("\nPer Query Budget:")
    for q in sorted(per_query.keys()):
        s = per_query[q]
        print(f"  {q} queries: Avg Dev={s['frame_metrics']['avg_deviation']['mean']:.4f}, "
              f"TFP={s['trajectory_metrics']['tfp_score']['mean']:.2f}x")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
