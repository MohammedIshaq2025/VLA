#!/usr/bin/env python3
"""
Phase 0: Environment Verification
Tests that OpenVLA loads correctly and we can access action tokens.
"""

import torch
import sys
import os

# Add project to path
sys.path.insert(0, '/data1/ma1/Ishaq/VLA_Frequency_Attack')

print("=" * 80)
print("Phase 0: Environment Verification")
print("=" * 80)

# ==============================================================================
# Test 0.1: Verify OpenVLA loads correctly
# ==============================================================================
print("\n[0.1] Loading OpenVLA...")
try:
    from transformers import AutoModelForVision2Seq, AutoProcessor

    MODEL_PATH = "/data1/ma1/Ishaq/ump-vla/checkpoints/openvla-7b/"

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).to("cuda:0")

    # Get vocab size from tokenizer instead of config
    vocab_size = len(processor.tokenizer)
    print(f"✓ Model loaded successfully")
    print(f"✓ Vocab size (from tokenizer): {vocab_size}")

    # Verify vocab size is expected value
    assert vocab_size >= 32000, f"Unexpected vocab size: {vocab_size}"
    print(f"✓ Vocab size verified (got {vocab_size})")

    VOCAB_SIZE = vocab_size

except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# ==============================================================================
# Test 0.2: Verify predict_action works
# ==============================================================================
print("\n[0.2] Testing predict_action...")
try:
    from PIL import Image
    import numpy as np

    # Create dummy image
    dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    prompt = "In: What action should the robot take to pick up the cup?\nOut:"

    # Predict action
    inputs = processor(prompt, dummy_img).to("cuda:0", dtype=torch.bfloat16)
    with torch.no_grad():
        action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

    print(f"✓ predict_action works")
    print(f"✓ Action shape: {action.shape}")
    print(f"✓ Action values: {action}")

    assert action.shape == (7,), f"Expected shape (7,), got {action.shape}"
    print(f"✓ Action shape verified")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==============================================================================
# Test 0.3: Verify forward pass returns logits
# ==============================================================================
print("\n[0.3] Testing forward pass...")
try:
    inputs = processor(prompt, dummy_img, return_tensors="pt").to("cuda:0", dtype=torch.bfloat16)

    with torch.no_grad():
        outputs = model(**inputs)

    print(f"✓ Forward pass works")
    print(f"✓ Logits shape: {outputs.logits.shape}")
    print(f"✓ Has loss attribute: {hasattr(outputs, 'loss')}")

    # Verify logits shape (may be padded to multiple of 64)
    batch_size, seq_len, logits_vocab_size = outputs.logits.shape
    print(f"✓ Logits vocab dimension: {logits_vocab_size} (tokenizer: {VOCAB_SIZE})")

    # The action tokens are the LAST 256 of logits vocab
    # Update VOCAB_SIZE to use the actual logits dimension
    VOCAB_SIZE = logits_vocab_size
    print(f"✓ Using logits vocab size for action token indexing")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==============================================================================
# Test 0.4: Verify action token vocabulary indices
# ==============================================================================
print("\n[0.4] Testing action token indices...")
try:
    # Generate action tokens
    gen_output = model.generate(
        **inputs,
        max_new_tokens=7,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id
    )

    # Extract last 7 tokens
    action_tokens = gen_output[0, -7:]

    print(f"✓ Generated action tokens: {action_tokens.tolist()}")

    # Calculate expected action token range
    action_start_idx = VOCAB_SIZE - 256
    action_end_idx = VOCAB_SIZE - 1

    print(f"✓ Expected action token range: [{action_start_idx}, {action_end_idx}]")

    # Verify all tokens are in action range
    for i, token_id in enumerate(action_tokens):
        token_id = token_id.item()
        if not (action_start_idx <= token_id <= action_end_idx):
            print(f"✗ Token {i} (ID={token_id}) NOT in action range!")
            sys.exit(1)

    print(f"✓ All action tokens in expected range")

    # Convert to bin indices
    clean_bins = action_tokens - action_start_idx
    print(f"✓ Bin indices: {clean_bins.tolist()}")

    assert all(0 <= b < 256 for b in clean_bins), "Bin indices out of range [0, 255]"
    print(f"✓ All bin indices in valid range [0, 255]")

    print(f"\n✓ ACTION TOKEN OFFSET: {action_start_idx} (use this in attack code)")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "=" * 80)
print("Phase 0 Verification: ALL TESTS PASSED ✓")
print("=" * 80)
print(f"Vocab size: {VOCAB_SIZE}")
print(f"Action token range: [{VOCAB_SIZE - 256}, {VOCAB_SIZE - 1}]")
print(f"Action token offset: {VOCAB_SIZE - 256}")
print("=" * 80)
