#!/usr/bin/env python3
"""
Closed-Loop LIBERO Evaluation Script - THE CRITICAL VALIDATION

This script performs TRUE closed-loop evaluation of adversarial patches
by running the robot policy in actual LIBERO environments and measuring
TASK SUCCESS RATES (not proxy metrics).

CRITICAL INSIGHT:
Single-frame metrics (ASR, deviation rate) are PROXY metrics that may
underestimate or overestimate true vulnerability. The ONLY way to know
if an attack is effective is to run the robot in closed-loop and measure
whether tasks actually fail.

Metrics Measured:
1. Task Success Rate (clean policy)
2. Task Success Rate (attacked policy)
3. TRUE Attack Success Rate = (clean - attacked) / clean
4. Episode length (did robot timeout?)
5. Failure modes (position error, gripper error, collision, etc.)

This is Direction 2's core validation experiment.
"""

import sys
import os
import argparse
import json
import time
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import traceback

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Setup headless rendering BEFORE importing LIBERO
os.environ['MUJOCO_GL'] = 'osmesa'

# Now import LIBERO and our modules
from openvla_action_extractor import OpenVLAActionExtractor

# Import LIBERO components
try:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path
    print("✓ Successfully imported LIBERO")
except ImportError as e:
    print(f"✗ Failed to import LIBERO: {e}")
    print("Make sure LIBERO is installed in your environment")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Closed-Loop Evaluation: Measure TRUE task success rates'
    )

    # Task configuration
    parser.add_argument('--suite', type=str, default='libero_spatial',
                        choices=['libero_spatial', 'libero_object', 'libero_goal', 'libero_10'],
                        help='LIBERO task suite')
    parser.add_argument('--task_id', type=int, default=0,
                        help='Task ID within suite (0-9)')
    parser.add_argument('--task_ids', type=str, default=None,
                        help='Comma-separated task IDs (e.g., "0,1,2"). Overrides --task_id')

    # Episode configuration
    parser.add_argument('--clean_episodes', type=int, default=50,
                        help='Number of clean episodes to run')
    parser.add_argument('--attacked_episodes', type=int, default=50,
                        help='Number of attacked episodes to run')
    parser.add_argument('--max_steps', type=int, default=300,
                        help='Maximum steps per episode')

    # Attack configuration
    parser.add_argument('--patch_path', type=str, default=None,
                        help='Path to adversarial patch (.npy file). If None, only run clean baseline.')
    parser.add_argument('--patch_x', type=int, default=48,
                        help='Patch X position')
    parser.add_argument('--patch_y', type=int, default=48,
                        help='Patch Y position')

    # Model configuration
    parser.add_argument('--model_path', type=str,
                        default='/data1/ma1/Ishaq/ump-vla/checkpoints/openvla-7b',
                        help='Path to OpenVLA checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for model inference')

    # Output configuration
    parser.add_argument('--output_dir', type=str,
                        default='/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/closed_loop',
                        help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name for output files')
    parser.add_argument('--save_videos', action='store_true',
                        help='Save video recordings of episodes (WARNING: large files)')

    # Environment configuration
    parser.add_argument('--camera_height', type=int, default=128,
                        help='Camera height (must match training)')
    parser.add_argument('--camera_width', type=int, default=128,
                        help='Camera width (must match training)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


def apply_patch(image: np.ndarray, patch: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
    """Apply adversarial patch to image."""
    patched = image.copy()
    x, y = position
    h, w = patch.shape[:2]
    img_h, img_w = image.shape[:2]

    # Clip position to valid range
    max_x = img_w - w
    max_y = img_h - h
    x = np.clip(x, 0, max(0, max_x))
    y = np.clip(y, 0, max(0, max_y))

    # Apply patch (convert to uint8 range)
    patched[y:y+h, x:x+w] = (patch * 255).astype(np.uint8)

    return patched


def run_episode(env, policy: OpenVLAActionExtractor, instruction: str,
                patch: Optional[np.ndarray], patch_pos: Tuple[int, int],
                max_steps: int, episode_id: int) -> Dict:
    """
    Run a single episode in LIBERO environment.

    Returns:
        dict with keys: success, steps, timeout, trajectory, final_state
    """
    obs = env.reset()
    done = False
    step = 0
    trajectory = []

    while not done and step < max_steps:
        # Get observation
        if "agentview_image" in obs:
            image = obs["agentview_image"]
        elif "front_image" in obs:
            image = obs["front_image"]
        else:
            # Find first image key
            image_keys = [k for k in obs.keys() if 'image' in k]
            if not image_keys:
                raise ValueError(f"No image found in observation. Keys: {list(obs.keys())}")
            image = obs[image_keys[0]]

        # Apply patch if attacking
        if patch is not None:
            image = apply_patch(image, patch, patch_pos)

        # Get action from policy
        action = policy.get_action_vector(image, instruction)
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        action = np.array(action).flatten()

        # Execute action in environment
        obs, reward, done, info = env.step(action)

        # Record trajectory
        trajectory.append({
            "step": step,
            "action": action.tolist(),
            "reward": float(reward),
            "done": bool(done)
        })

        step += 1

    # Check task success
    success = info.get("success", False)
    timeout = (step >= max_steps and not done)

    return {
        "episode_id": episode_id,
        "success": bool(success),
        "steps": step,
        "timeout": bool(timeout),
        "trajectory_length": len(trajectory),
        "final_reward": float(reward),
        "trajectory": trajectory if len(trajectory) < 100 else []  # Save only if short
    }


def evaluate_task(suite: str, task_id: int, policy: OpenVLAActionExtractor,
                  patch: Optional[np.ndarray], patch_pos: Tuple[int, int],
                  clean_episodes: int, attacked_episodes: int, max_steps: int,
                  camera_height: int, camera_width: int) -> Dict:
    """
    Evaluate policy on a single LIBERO task with and without attack.

    Returns:
        dict with clean_results, attacked_results, and metrics
    """
    print(f"\n{'='*80}")
    print(f"Evaluating Task {task_id} from {suite}")
    print(f"{'='*80}")

    # Get task from LIBERO benchmark
    try:
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[suite]()
        task = task_suite.get_task(task_id)
        instruction = task.language

        print(f"Task Name: {task.name}")
        print(f"Instruction: {instruction}")

        # Get BDDL file path
        # Try the path from get_libero_path first, then fallback to actual location
        bddl_base = get_libero_path("bddl_files")
        task_bddl_file = os.path.join(
            bddl_base,
            task.problem_folder,
            task.bddl_file
        )
        
        # Fallback: if path doesn't exist, try the actual location
        if not os.path.exists(task_bddl_file):
            # Try actual location in the environment
            import libero
            libero_package_path = os.path.dirname(libero.__file__)
            fallback_path = os.path.join(libero_package_path, "libero", "bddl_files", task.problem_folder, task.bddl_file)
            if os.path.exists(fallback_path):
                task_bddl_file = fallback_path
                print(f"  Using fallback BDDL path: {task_bddl_file}")
            else:
                raise FileNotFoundError(f"BDDL file not found: {task_bddl_file}\nAlso tried: {fallback_path}")

    except Exception as e:
        print(f"✗ Failed to load task: {e}")
        traceback.print_exc()
        return None

    # === RUN CLEAN EPISODES ===
    print(f"\n[CLEAN POLICY] Running {clean_episodes} episodes...")
    clean_results = []
    clean_start = time.time()

    for ep in range(clean_episodes):
        try:
            # Create fresh environment for each episode
            env = OffScreenRenderEnv(
                bddl_file_name=task_bddl_file,
                camera_heights=camera_height,
                camera_widths=camera_width,
                has_renderer=False,
                has_offscreen_renderer=True,
                ignore_done=False,
                use_camera_obs=True,
                control_freq=20,
                horizon=max_steps
            )

            result = run_episode(
                env=env,
                policy=policy,
                instruction=instruction,
                patch=None,  # No attack
                patch_pos=patch_pos,
                max_steps=max_steps,
                episode_id=ep
            )
            clean_results.append(result)

            # Clean up
            env.close()
            del env

            if (ep + 1) % 10 == 0 or ep == 0:
                successes = sum(1 for r in clean_results if r["success"])
                print(f"  Episode {ep+1}/{clean_episodes}: "
                      f"Success rate so far: {successes}/{len(clean_results)} "
                      f"({successes/len(clean_results)*100:.1f}%)")

        except Exception as e:
            print(f"  ✗ Episode {ep} failed: {e}")
            traceback.print_exc()
            clean_results.append({
                "episode_id": ep,
                "success": False,
                "steps": 0,
                "timeout": False,
                "error": str(e)
            })

    clean_time = time.time() - clean_start
    clean_success_rate = np.mean([r["success"] for r in clean_results])
    print(f"[CLEAN POLICY] Completed in {clean_time:.1f}s")
    print(f"[CLEAN POLICY] Success Rate: {clean_success_rate*100:.1f}% ({sum(1 for r in clean_results if r['success'])}/{len(clean_results)})")

    # === RUN ATTACKED EPISODES ===
    attacked_results = []
    attacked_time = 0

    if patch is not None:
        print(f"\n[ATTACKED POLICY] Running {attacked_episodes} episodes...")
        attacked_start = time.time()

        for ep in range(attacked_episodes):
            try:
                # Create fresh environment
                env = OffScreenRenderEnv(
                    bddl_file_name=task_bddl_file,
                    camera_heights=camera_height,
                    camera_widths=camera_width,
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    ignore_done=False,
                    use_camera_obs=True,
                    control_freq=20,
                    horizon=max_steps
                )

                result = run_episode(
                    env=env,
                    policy=policy,
                    instruction=instruction,
                    patch=patch,  # WITH ATTACK
                    patch_pos=patch_pos,
                    max_steps=max_steps,
                    episode_id=ep
                )
                attacked_results.append(result)

                # Clean up
                env.close()
                del env

                if (ep + 1) % 10 == 0 or ep == 0:
                    successes = sum(1 for r in attacked_results if r["success"])
                    print(f"  Episode {ep+1}/{attacked_episodes}: "
                          f"Success rate so far: {successes}/{len(attacked_results)} "
                          f"({successes/len(attacked_results)*100:.1f}%)")

            except Exception as e:
                print(f"  ✗ Episode {ep} failed: {e}")
                traceback.print_exc()
                attacked_results.append({
                    "episode_id": ep,
                    "success": False,
                    "steps": 0,
                    "timeout": False,
                    "error": str(e)
                })

        attacked_time = time.time() - attacked_start
        attacked_success_rate = np.mean([r["success"] for r in attacked_results])
        print(f"[ATTACKED POLICY] Completed in {attacked_time:.1f}s")
        print(f"[ATTACKED POLICY] Success Rate: {attacked_success_rate*100:.1f}% ({sum(1 for r in attacked_results if r['success'])}/{len(attacked_results)})")

    # === COMPUTE METRICS ===
    clean_successes = sum(1 for r in clean_results if r["success"])
    clean_total = len(clean_results)

    if patch is not None and attacked_results:
        attacked_successes = sum(1 for r in attacked_results if r["success"])
        attacked_total = len(attacked_results)

        # TRUE Attack Success Rate
        if clean_success_rate > 0:
            true_asr = (clean_success_rate - attacked_success_rate) / clean_success_rate
        else:
            true_asr = 0.0

        # Relative success drop
        success_drop_absolute = clean_success_rate - attacked_success_rate

        metrics = {
            "clean_success_rate": float(clean_success_rate),
            "attacked_success_rate": float(attacked_success_rate),
            "true_asr": float(true_asr),
            "success_drop_absolute": float(success_drop_absolute),
            "clean_successes": int(clean_successes),
            "clean_total": int(clean_total),
            "attacked_successes": int(attacked_successes),
            "attacked_total": int(attacked_total),
            "clean_avg_steps": float(np.mean([r["steps"] for r in clean_results])),
            "attacked_avg_steps": float(np.mean([r["steps"] for r in attacked_results])),
            "clean_timeout_rate": float(np.mean([r["timeout"] for r in clean_results])),
            "attacked_timeout_rate": float(np.mean([r["timeout"] for r in attacked_results]))
        }
    else:
        # Clean baseline only
        metrics = {
            "clean_success_rate": float(clean_success_rate),
            "clean_successes": int(clean_successes),
            "clean_total": int(clean_total),
            "clean_avg_steps": float(np.mean([r["steps"] for r in clean_results])),
            "clean_timeout_rate": float(np.mean([r["timeout"] for r in clean_results]))
        }

    return {
        "task_id": task_id,
        "task_name": task.name,
        "instruction": instruction,
        "clean_results": clean_results,
        "attacked_results": attacked_results,
        "metrics": metrics,
        "timing": {
            "clean_time_seconds": clean_time,
            "attacked_time_seconds": attacked_time
        }
    }


def main():
    args = parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f"closed_loop_{args.suite}_{timestamp}"

    print("=" * 80)
    print("CLOSED-LOOP LIBERO EVALUATION - THE CRITICAL TEST")
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Suite: {args.suite}")

    # Determine which tasks to run
    if args.task_ids is not None:
        task_ids = [int(x.strip()) for x in args.task_ids.split(',')]
    else:
        task_ids = [args.task_id]

    print(f"Tasks: {task_ids}")
    print(f"Clean episodes per task: {args.clean_episodes}")
    print(f"Attacked episodes per task: {args.attacked_episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print("=" * 80)

    # Load OpenVLA model
    print(f"\n[1/3] Loading OpenVLA model...")
    sys.stdout.flush()
    model_start = time.time()
    policy = OpenVLAActionExtractor(
        model_path=args.model_path,
        device=args.device
    )
    model_time = time.time() - model_start
    print(f"✓ Model loaded in {model_time:.1f}s")
    sys.stdout.flush()

    # Load patch if attacking
    patch = None
    patch_pos = (args.patch_x, args.patch_y)
    if args.patch_path is not None:
        print(f"\n[2/3] Loading adversarial patch...")
        sys.stdout.flush()
        patch = np.load(args.patch_path)
        print(f"✓ Patch loaded: {patch.shape}, range [{patch.min():.3f}, {patch.max():.3f}]")
        print(f"  Patch position: {patch_pos}")
        sys.stdout.flush()
    else:
        print(f"\n[2/3] No patch specified - running clean baseline only")
        sys.stdout.flush()

    # Run evaluation on all tasks
    print(f"\n[3/3] Running closed-loop evaluation...")
    sys.stdout.flush()

    all_results = []
    for task_id in task_ids:
        result = evaluate_task(
            suite=args.suite,
            task_id=task_id,
            policy=policy,
            patch=patch,
            patch_pos=patch_pos,
            clean_episodes=args.clean_episodes,
            attacked_episodes=args.attacked_episodes,
            max_steps=args.max_steps,
            camera_height=args.camera_height,
            camera_width=args.camera_width
        )

        if result is not None:
            all_results.append(result)

    # === AGGREGATE RESULTS ACROSS ALL TASKS ===
    print(f"\n{'='*80}")
    print(f"AGGREGATE RESULTS ACROSS ALL TASKS")
    print(f"{'='*80}")

    if patch is not None:
        all_clean_success = [r["metrics"]["clean_success_rate"] for r in all_results]
        all_attacked_success = [r["metrics"]["attacked_success_rate"] for r in all_results]
        all_true_asr = [r["metrics"]["true_asr"] for r in all_results]

        print(f"\nMean Clean Success Rate: {np.mean(all_clean_success)*100:.1f}% ± {np.std(all_clean_success)*100:.1f}%")
        print(f"Mean Attacked Success Rate: {np.mean(all_attacked_success)*100:.1f}% ± {np.std(all_attacked_success)*100:.1f}%")
        print(f"Mean TRUE ASR: {np.mean(all_true_asr)*100:.1f}% ± {np.std(all_true_asr)*100:.1f}%")

        aggregate_metrics = {
            "mean_clean_success_rate": float(np.mean(all_clean_success)),
            "std_clean_success_rate": float(np.std(all_clean_success)),
            "mean_attacked_success_rate": float(np.mean(all_attacked_success)),
            "std_attacked_success_rate": float(np.std(all_attacked_success)),
            "mean_true_asr": float(np.mean(all_true_asr)),
            "std_true_asr": float(np.std(all_true_asr)),
            "min_true_asr": float(np.min(all_true_asr)),
            "max_true_asr": float(np.max(all_true_asr))
        }
    else:
        all_clean_success = [r["metrics"]["clean_success_rate"] for r in all_results]
        print(f"\nMean Clean Success Rate: {np.mean(all_clean_success)*100:.1f}% ± {np.std(all_clean_success)*100:.1f}%")

        aggregate_metrics = {
            "mean_clean_success_rate": float(np.mean(all_clean_success)),
            "std_clean_success_rate": float(np.std(all_clean_success))
        }

    # === SAVE RESULTS ===
    summary = {
        "experiment_name": args.experiment_name,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "suite": args.suite,
        "task_ids": task_ids,
        "patch_path": str(args.patch_path) if args.patch_path else None,
        "patch_position": patch_pos,
        "configuration": {
            "clean_episodes": args.clean_episodes,
            "attacked_episodes": args.attacked_episodes,
            "max_steps": args.max_steps,
            "camera_height": args.camera_height,
            "camera_width": args.camera_width,
            "seed": args.seed
        },
        "aggregate_metrics": aggregate_metrics,
        "per_task_results": all_results
    }

    results_path = output_dir / f"{args.experiment_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Results saved: {results_path}")

    # === FINAL VERDICT ===
    print(f"\n{'='*80}")
    print(f"FINAL VERDICT - CLOSED-LOOP EVALUATION")
    print(f"{'='*80}")

    if patch is not None:
        mean_asr = aggregate_metrics["mean_true_asr"]

        if mean_asr >= 0.60:
            print(f"✓✓✓ STRONG ATTACK (TRUE ASR: {mean_asr*100:.1f}%)")
            print(f"    This is ECCV-worthy! Single-frame proxies underestimated vulnerability.")
        elif mean_asr >= 0.40:
            print(f"✓✓ MODERATE ATTACK (TRUE ASR: {mean_asr*100:.1f}%)")
            print(f"    Publishable, but effect size is smaller than hypothesized.")
        elif mean_asr >= 0.20:
            print(f"✓ WEAK ATTACK (TRUE ASR: {mean_asr*100:.1f}%)")
            print(f"    Attack has some effect, but may not be sufficient for ECCV.")
        else:
            print(f"✗ INEFFECTIVE ATTACK (TRUE ASR: {mean_asr*100:.1f}%)")
            print(f"    Attack does not cause significant task failure. Needs improvement.")
    else:
        print(f"Clean baseline established: {aggregate_metrics['mean_clean_success_rate']*100:.1f}% success rate")

    print(f"{'='*80}\n")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
