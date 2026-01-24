#!/usr/bin/env python3
"""
Closed-Loop Evaluation for VLA Frequency Attack
Runs episodes in LIBERO and applies frequency attack to each observation frame.

V3-ITER200: Attack applied AFTER center crop (ABLATION: 200 iterations instead of 100)
    - Preprocessing: rotation → JPEG → resize → center crop → ATTACK
    - This ensures perturbations are optimized for the final input
"""

import torch
import numpy as np
import sys
import os
import json
import yaml
import tensorflow as tf
from PIL import Image
from typing import Dict, List

sys.path.insert(0, '/data1/ma1/Ishaq/VLA_Frequency_Attack')

from experiments.frequency_attack import FrequencyConstrainedCWAttack


# ============================================================================
# Preprocessing functions (EXACTLY matching official OpenVLA)
# ============================================================================

def resize_image(img, resize_size):
    """EXACTLY matches official OpenVLA preprocessing."""
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_libero_image(obs, resize_size=224):
    """EXACTLY matches official OpenVLA preprocessing."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # 180 degree rotation
    img = resize_image(img, resize_size)
    return img


def crop_and_resize(image, crop_scale, batch_size):
    """EXACTLY matches official OpenVLA code."""
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [height_offsets, width_offsets, height_offsets + new_heights, width_offsets + new_widths],
        axis=1,
    )
    image = tf.image.crop_and_resize(
        image, bounding_boxes, tf.range(batch_size), [image.shape[1], image.shape[2]]
    )
    if expanded_dims:
        image = image[0]
    return image


def apply_center_crop(image, crop_scale=0.9):
    """Apply center crop to PIL Image."""
    batch_size = 1
    image = tf.convert_to_tensor(np.array(image))
    orig_dtype = image.dtype
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = crop_and_resize(image, crop_scale, batch_size)
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)
    image = Image.fromarray(image.numpy())
    return image.convert("RGB")


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension) from [0,1] to [-1,+1].
    EXACTLY matches official OpenVLA code.
    """
    action = np.array(action) if not isinstance(action, np.ndarray) else action.copy()
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension).
    EXACTLY matches official OpenVLA code.
    """
    action = action.copy()
    action[..., -1] = action[..., -1] * -1.0
    return action


def get_libero_dummy_action():
    """Get dummy/no-op action for wait period."""
    return [0, 0, 0, 0, 0, 0, -1]


def evaluate_closed_loop(
    model,
    processor,
    env,
    task_description: str,
    attack_config: Dict,
    initial_states: List = None,
    num_episodes: int = 10,
    max_steps_per_episode: int = 200,
    num_steps_wait: int = 10,
    device: str = "cuda:0",
    center_crop: bool = True,
    crop_scale: float = 0.9,
    unnorm_key: str = "libero_spatial"
):
    """
    Run closed-loop evaluation with attack.

    Args:
        model: OpenVLA model
        processor: OpenVLA processor
        env: LIBERO environment
        task_description: Task instruction text
        attack_config: Dict with keys:
            - 'enabled': bool, whether to apply attack
            - 'freq_ratio': float, frequency ratio (0.25, 0.5, 1.0)
            - 'epsilon': float, L∞ bound
            - 'alpha': float, step size
            - 'num_iterations': int
            - 'kappa': float, CW margin
        num_episodes: Number of episodes to run
        max_steps_per_episode: Max steps per episode
        device: CUDA device

    Returns:
        Dict with results
    """

    # Initialize attack if enabled
    attack = None
    if attack_config['enabled']:
        attack = FrequencyConstrainedCWAttack(
            model=model,
            processor=processor,
            epsilon=attack_config['epsilon'],
            alpha=attack_config['alpha'],
            num_iterations=attack_config['num_iterations'],
            freq_ratio=attack_config['freq_ratio'],
            kappa=attack_config['kappa'],
            device=device
        )

    # Results tracking
    episode_results = []
    total_success = 0
    total_steps = 0
    total_bfr = []
    total_purity = []

    # Use correct prompt format (matching official OpenVLA)
    instruction = f"In: What action should the robot take to {task_description.lower()}?\nOut:"

    for episode_idx in range(num_episodes):
        print(f"\n[Episode {episode_idx+1}/{num_episodes}]")

        # Reset environment with fixed initial state (matching baseline_evaluation.py)
        env.reset()
        if initial_states is not None and episode_idx < len(initial_states):
            obs = env.set_init_state(initial_states[episode_idx])
        else:
            env.seed(42 + episode_idx)
            obs = env.reset()

        episode_success = False
        episode_steps = 0
        episode_bfr = []
        episode_purity = []

        # Wait period with dummy actions (matching baseline_evaluation.py)
        for _ in range(num_steps_wait):
            obs, _, done, _ = env.step(get_libero_dummy_action())
            if done:
                break

        for step in range(max_steps_per_episode):
            # Get observation image with CORRECT preprocessing
            # This applies: 180° rotation, JPEG encode/decode, resize with lanczos3
            img_array = get_libero_image(obs, resize_size=224)

            # Convert to PIL and apply center crop FIRST (before attack)
            # This ensures attack targets the EXACT image the model will see
            img_pil = Image.fromarray(img_array)
            if center_crop:
                img_pil = apply_center_crop(img_pil, crop_scale)

            # Convert to tensor for attack (if enabled)
            if attack:
                # Convert cropped PIL to float tensor [0,1]
                img_array_cropped = np.array(img_pil)
                img_tensor = torch.from_numpy(img_array_cropped).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(device)

                # Apply attack on the FINAL cropped image
                adv_pixel_values, attack_info = attack.attack(img_tensor, instruction, verbose=False)

                # Track metrics
                episode_bfr.append(attack_info['bin_flip_rate'])
                episode_purity.append(attack_info['frequency_purity'])

                # Convert adversarial pixel_values back to PIL image for processor
                # adv_pixel_values is (1, 6, 224, 224) in normalized space
                # We need to reconstruct the original image from the SigLIP path (first 3 channels)
                # Denormalize: x = (x_norm * 0.5) + 0.5
                adv_siglip = adv_pixel_values[:, :3, :, :]  # (1, 3, 224, 224)
                adv_img_tensor = adv_siglip[0].cpu() * 0.5 + 0.5  # (3, 224, 224) in [0,1]
                adv_img_array = (adv_img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                img_pil = Image.fromarray(adv_img_array)
            # else: img_pil already has the clean cropped image from above

            # Get action from OpenVLA
            inputs = processor(instruction, img_pil).to(device, dtype=torch.bfloat16)
            action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

            # Convert to numpy if needed
            if hasattr(action, 'cpu'):
                action = action.cpu().numpy()

            # CRITICAL: Apply gripper post-processing (normalize + invert)
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)

            # Step environment
            obs, reward, done, info = env.step(action.tolist())
            episode_steps += 1

            if done:
                break

        # Check success using env.check_success() (matching baseline_evaluation.py)
        episode_success = env.check_success()

        # Record episode results
        total_success += 1 if episode_success else 0
        total_steps += episode_steps

        if attack:
            total_bfr.extend(episode_bfr)
            total_purity.extend(episode_purity)

        episode_results.append({
            'episode': episode_idx + 1,
            'success': bool(episode_success),  # Convert numpy bool to Python bool
            'steps': episode_steps,
            'avg_bfr': float(np.mean(episode_bfr)) if episode_bfr else 0.0,
            'avg_purity': float(np.mean(episode_purity)) if episode_purity else 0.0
        })

        print(f"  Success: {episode_success}, Steps: {episode_steps}")
        if attack:
            print(f"  Avg BFR: {np.mean(episode_bfr)*100:.1f}%, Avg Purity: {np.mean(episode_purity)*100:.1f}%")

    # Aggregate results
    success_rate = total_success / num_episodes
    avg_steps = total_steps / num_episodes

    results = {
        'attack_config': attack_config,
        'num_episodes': num_episodes,
        'success_rate': success_rate,
        'avg_steps_per_episode': avg_steps,
        'episode_results': episode_results
    }

    if attack:
        results['avg_bin_flip_rate'] = float(np.mean(total_bfr))
        results['avg_frequency_purity'] = float(np.mean(total_purity))

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Success Rate: {success_rate*100:.1f}% ({total_success}/{num_episodes})")
    print(f"Avg Steps: {avg_steps:.1f}")
    if attack:
        print(f"Avg BFR: {np.mean(total_bfr)*100:.1f}%")
        print(f"Avg Purity: {np.mean(total_purity)*100:.1f}%")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', type=str, default='baseline', choices=['baseline', 'full', 'low', 'mid', 'high'])
    parser.add_argument('--task_id', type=int, default=0, help='Task ID from LIBERO-Spatial (0-9)')
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--output', type=str, default='results/closed_loop')
    args = parser.parse_args()

    # Set random seeds (matching baseline_evaluation.py SEED=7)
    SEED = 7
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("="*80)
    print("Closed-Loop Evaluation (V3 - Attack After Center Crop)")
    print("="*80)
    print(f"  Random seed: {SEED}")

    # Setup LIBERO config FIRST (before any libero imports)
    LIBERO_ROOT = "/data1/ma1/envs/upa-vla/lib/python3.10/site-packages/libero/libero"
    config = {
        "benchmark_root": LIBERO_ROOT,
        "bddl_files": os.path.join(LIBERO_ROOT, "bddl_files"),
        "init_states": os.path.join(LIBERO_ROOT, "init_files"),
        "datasets": os.path.join(LIBERO_ROOT, "../datasets"),
        "assets": os.path.join(LIBERO_ROOT, "assets"),
    }
    libero_config_path = os.path.expanduser("~/.libero")
    os.makedirs(libero_config_path, exist_ok=True)
    config_file = os.path.join(libero_config_path, "config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    print(f"[*] LIBERO config written to: {config_file}")

    # Load OpenVLA - STANDARD checkpoint (NOT OFT!)
    print("\nLoading OpenVLA...")
    from transformers import AutoModelForVision2Seq, AutoProcessor

    # CRITICAL: Use standard checkpoint, NOT OFT
    MODEL_PATH = '/data1/ma1/Ishaq/ump-vla/checkpoints/openvla-7b-finetuned-libero-spatial'
    UNNORM_KEY = 'libero_spatial'
    CENTER_CROP = True
    CROP_SCALE = 0.9

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).to('cuda:0')

    # Load dataset statistics for unnormalization
    stats_path = os.path.join(MODEL_PATH, "dataset_statistics.json")
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            model.norm_stats = json.load(f)
        print(f"[*] Loaded dataset statistics, unnorm_key: {UNNORM_KEY}")

    print("✓ OpenVLA fine-tuned model loaded (STANDARD checkpoint)")
    print(f"  Model path: {MODEL_PATH}")
    print(f"  Center crop: {CENTER_CROP} (scale={CROP_SCALE})")
    print(f"  Gripper post-processing: normalize + invert")

    # Add OpenVLA to path for libero_utils
    sys.path.insert(0, '/data1/ma1/Ishaq/VLA_Frequency_Attack/openvla')

    # Create LIBERO environment
    print("\nCreating LIBERO environment...")
    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import get_libero_env

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict['libero_spatial']()
    task = task_suite.get_task(args.task_id)
    print(f"✓ Task {args.task_id}: {task.name}")
    print(f"  Description: {task.language}")

    # Use official libero_utils to create environment
    env, _ = get_libero_env(task, "openvla", resolution=256)
    env.seed(SEED)  # Use same seed as baseline_evaluation.py
    print("✓ Environment created")

    # Configure attack
    attack_configs = {
        'baseline': {'enabled': False},
        'full': {'enabled': True, 'freq_ratio': 1.0, 'epsilon': 32/255, 'alpha': 4/255, 'num_iterations': 100, 'kappa': 30.0},
        'low': {'enabled': True, 'freq_ratio': 0.35, 'epsilon': 32/255, 'alpha': 4/255, 'num_iterations': 100, 'kappa': 30.0},
        'mid': {'enabled': True, 'freq_ratio': 0.5, 'epsilon': 32/255, 'alpha': 4/255, 'num_iterations': 100, 'kappa': 30.0},
        'high': {'enabled': True, 'freq_ratio': -0.35, 'epsilon': 32/255, 'alpha': 4/255, 'num_iterations': 100, 'kappa': 30.0},  # Negative for high-pass
    }

    attack_config = attack_configs[args.attack]
    print(f"\nAttack: {args.attack}")
    print(f"Config: {attack_config}")

    # Get initial states for reproducibility (matching baseline_evaluation.py)
    initial_states = task_suite.get_task_init_states(args.task_id)
    print(f"  Using {len(initial_states)} fixed initial states")

    # Run evaluation with CORRECT parameters
    results = evaluate_closed_loop(
        model=model,
        processor=processor,
        env=env,
        task_description=task.language,
        attack_config=attack_config,
        initial_states=initial_states,
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        num_steps_wait=10,  # Match baseline_evaluation.py
        device='cuda:0',
        center_crop=CENTER_CROP,
        crop_scale=CROP_SCALE,
        unnorm_key=UNNORM_KEY
    )

    # Add task metadata to results
    results['task_id'] = args.task_id
    results['task_name'] = task.name
    results['task_description'] = task.language
    results['preprocessing'] = {
        'center_crop': CENTER_CROP,
        'crop_scale': CROP_SCALE,
        'rotation': '180_degrees',
        'jpeg_encode_decode': True,
        'gripper_postprocessing': 'normalize + invert',
        'attack_position': 'after_center_crop'  # V3: attack on final image
    }

    # Save results
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(args.output, f'{args.attack}_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    env.close()
