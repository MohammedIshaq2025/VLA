#!/usr/bin/env python3
"""
Hyperparameter Tuning for Frequency-Constrained Attack
Tests different combinations to maximize bin flip rate while maintaining frequency purity.
"""

import torch
import numpy as np
import sys
from PIL import Image
import json

sys.path.insert(0, '/data1/ma1/Ishaq/VLA_Frequency_Attack')

from experiments.frequency_attack import FrequencyConstrainedCWAttack

print("=" * 80)
print("Hyperparameter Tuning for Frequency-Constrained Attack")
print("=" * 80)

# Load model
print("\nLoading OpenVLA...")
from transformers import AutoModelForVision2Seq, AutoProcessor

MODEL_PATH = "/data1/ma1/Ishaq/ump-vla/checkpoints/openvla-7b/"

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
).to("cuda:0")

print("Model loaded successfully")

# Create test images
print("\nCreating test images...")
np.random.seed(42)
test_images = []
for i in range(3):
    img_np = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to("cuda:0")
    test_images.append(img_tensor)

print(f"Created {len(test_images)} test images")

instruction = "In: What action should the robot take to pick up the cup?\nOut:"

# Hyperparameter grid
configs = [
    # Baseline
    {"epsilon": 16/255, "alpha": 2/255, "num_iterations": 50, "kappa": 5.0, "name": "baseline"},

    # More iterations
    {"epsilon": 16/255, "alpha": 2/255, "num_iterations": 100, "kappa": 5.0, "name": "iter100"},
    {"epsilon": 16/255, "alpha": 2/255, "num_iterations": 200, "kappa": 5.0, "name": "iter200"},

    # Larger epsilon
    {"epsilon": 32/255, "alpha": 2/255, "num_iterations": 100, "kappa": 5.0, "name": "eps32"},
    {"epsilon": 64/255, "alpha": 2/255, "num_iterations": 100, "kappa": 5.0, "name": "eps64"},

    # Larger step size
    {"epsilon": 16/255, "alpha": 4/255, "num_iterations": 100, "kappa": 5.0, "name": "alpha4"},
    {"epsilon": 32/255, "alpha": 4/255, "num_iterations": 100, "kappa": 5.0, "name": "eps32_alpha4"},

    # Different kappa
    {"epsilon": 16/255, "alpha": 2/255, "num_iterations": 100, "kappa": 10.0, "name": "kappa10"},
    {"epsilon": 16/255, "alpha": 2/255, "num_iterations": 100, "kappa": 0.0, "name": "kappa0"},
]

results = []

# Test each config on low-frequency attack (r=0.25)
freq_ratio = 0.25

print(f"\n{'='*80}")
print(f"Testing {len(configs)} hyperparameter configurations")
print(f"Frequency ratio: {freq_ratio} (low-frequency)")
print(f"{'='*80}\n")

for config_idx, config in enumerate(configs):
    print(f"\n[Config {config_idx+1}/{len(configs)}] {config['name']}")
    print(f"  epsilon={config['epsilon']:.4f}, alpha={config['alpha']:.4f}, "
          f"iters={config['num_iterations']}, kappa={config['kappa']}")

    # Create attack with this config
    attack = FrequencyConstrainedCWAttack(
        model=model,
        processor=processor,
        epsilon=config['epsilon'],
        alpha=config['alpha'],
        num_iterations=config['num_iterations'],
        freq_ratio=freq_ratio,
        kappa=config['kappa'],
        device="cuda:0"
    )

    # Test on all images
    bfr_list = []
    purity_list = []
    loss_list = []
    iters_list = []

    for img_idx, img in enumerate(test_images):
        x_adv, info = attack.attack(img, instruction, verbose=False)

        bfr_list.append(info['bin_flip_rate'])
        purity_list.append(info['frequency_purity'])
        loss_list.append(info['final_cw_loss'])
        iters_list.append(info['iterations'])

        print(f"    Image {img_idx+1}: BFR={info['bin_flip_rate']*100:.1f}%, "
              f"Purity={info['frequency_purity']*100:.1f}%, "
              f"Loss={info['final_cw_loss']:.2f}")

    # Aggregate stats
    avg_bfr = np.mean(bfr_list)
    avg_purity = np.mean(purity_list)
    avg_loss = np.mean(loss_list)
    avg_iters = np.mean(iters_list)

    result = {
        'config': config,
        'avg_bfr': avg_bfr,
        'avg_purity': avg_purity,
        'avg_loss': avg_loss,
        'avg_iters': avg_iters,
        'bfr_list': bfr_list,
        'purity_list': purity_list
    }
    results.append(result)

    print(f"  [AVERAGE] BFR={avg_bfr*100:.1f}%, Purity={avg_purity*100:.1f}%, "
          f"Loss={avg_loss:.2f}, Iters={avg_iters:.1f}")

    # Check if this meets criteria
    if avg_bfr >= 0.57 and avg_purity >= 0.70:
        print(f"  [OK] Meets criteria (BFR≥57%, Purity≥70%)")
    else:
        if avg_bfr < 0.57:
            print(f"  [X] BFR too low ({avg_bfr*100:.1f}% < 57%)")
        if avg_purity < 0.70:
            print(f"  [X] Purity too low ({avg_purity*100:.1f}% < 70%)")

# Sort by BFR (descending)
results_sorted = sorted(results, key=lambda x: x['avg_bfr'], reverse=True)

# Print summary table
print(f"\n{'='*80}")
print(f"SUMMARY: Hyperparameter Tuning Results")
print(f"{'='*80}")
print(f"{'Config':<20} {'BFR':<10} {'Purity':<10} {'Loss':<10} {'Iters':<10} {'Status'}")
print(f"{'-'*80}")

for result in results_sorted:
    config_name = result['config']['name']
    avg_bfr = result['avg_bfr']
    avg_purity = result['avg_purity']
    avg_loss = result['avg_loss']
    avg_iters = result['avg_iters']

    status = "PASS" if avg_bfr >= 0.57 and avg_purity >= 0.70 else "FAIL"

    print(f"{config_name:<20} {avg_bfr*100:>6.1f}%   {avg_purity*100:>6.1f}%   "
          f"{avg_loss:>8.2f}  {avg_iters:>8.1f}   {status}")

print(f"{'='*80}")

# Find best config
best_result = results_sorted[0]
print(f"\nBEST CONFIG: {best_result['config']['name']}")
print(f"  Average BFR: {best_result['avg_bfr']*100:.1f}%")
print(f"  Average Purity: {best_result['avg_purity']*100:.1f}%")
print(f"  Config details:")
for key, value in best_result['config'].items():
    if key != 'name':
        print(f"    {key}: {value}")

# Save results to JSON
output_path = "/data1/ma1/Ishaq/VLA_Frequency_Attack/results/hyperparameter_tuning.json"
with open(output_path, 'w') as f:
    json.dump(results_sorted, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x)

print(f"\nResults saved to: {output_path}")

print(f"\n{'='*80}")
print("Hyperparameter Tuning Complete")
print(f"{'='*80}")
