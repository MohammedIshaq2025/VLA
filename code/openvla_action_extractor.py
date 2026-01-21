"""
OpenVLA Action Extraction Hook
Wrapper around OpenVLA that extracts the 7D action vector for SE(3) manifold optimization.
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from typing import Optional
import sys


class OpenVLAActionExtractor:
    """
    Wrapper around OpenVLA that extracts continuous 7D action vectors.
    This is CRITICAL for SE(3) manifold optimization.
    """
    
    def __init__(self, model_path: str = "/data1/ma1/Ishaq/ump-vla/checkpoints/openvla-7b",
                 device: Optional[str] = None,
                 unnorm_key: Optional[str] = None):
        """
        Initialize OpenVLA model for action extraction.

        Args:
            model_path: Path to OpenVLA checkpoint
            device: Device to use (default: cuda:0 if available, else cpu)
            unnorm_key: Dataset-specific unnormalization key (default: "bridge_orig")
                       For LIBERO: use "libero_spatial_no_noops", "libero_object_no_noops", etc.
        """
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_path = model_path

        print(f"[OpenVLA] Loading model from {model_path} on {device}...")
        sys.stdout.flush()

        # Load processor
        print(f"[OpenVLA] Step 1/4: Loading processor...")
        sys.stdout.flush()
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print(f"[OpenVLA] ✓ Processor loaded")
        sys.stdout.flush()

        # Load model
        print(f"[OpenVLA] Step 2/4: Loading model weights...")
        sys.stdout.flush()
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
            low_cpu_mem_usage=True,
            load_in_4bit=False
        )
        print(f"[OpenVLA] ✓ Model weights loaded")
        sys.stdout.flush()

        # Move to device
        if device.startswith("cuda"):
            print(f"[OpenVLA] Step 3/4: Moving model to {device}...")
            sys.stdout.flush()
            self.model = self.model.to(device)
            print(f"[OpenVLA] ✓ Model moved to {device}")
            sys.stdout.flush()

        # Set unnorm_key for action denormalization
        # CRITICAL: Must match the dataset/robot for correct action scaling
        self.unnorm_key = unnorm_key if unnorm_key is not None else "bridge_orig"

        print(f"[OpenVLA] Step 4/4: Getting action dimension...")
        print(f"[OpenVLA] Using unnorm_key: {self.unnorm_key}")
        sys.stdout.flush()
        try:
            action_dim = self.model.get_action_dim(self.unnorm_key)
            print(f"[OpenVLA] ✓ Model loaded successfully")
            print(f"[OpenVLA] Action dim: {action_dim}")
        except Exception as e:
            print(f"[OpenVLA] ⚠ Warning: Could not get action dim: {e}")
            print(f"[OpenVLA] ✓ Model loaded successfully (action dim check skipped)")
        sys.stdout.flush()
        
    def get_action_vector(self, image: np.ndarray, instruction: str) -> np.ndarray:
        """
        Get 7D action vector from image + instruction.
        
        Args:
            image: np.ndarray (H, W, 3) RGB uint8 in range [0, 255], or PIL Image
            instruction: Task instruction string (without prompt formatting)
            
        Returns:
            np.ndarray (7,) - 7D action vector [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            # Ensure uint8 and correct shape
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            image = Image.fromarray(image).convert("RGB")
        
        # Build prompt (OpenVLA format)
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        
        # Process inputs
        inputs = self.processor(prompt, image)
        
        # Move to device and convert dtype
        # Note: input_ids must remain Long/Int, only pixel_values should be bfloat16
        processed_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if k == "pixel_values" and self.device.startswith("cuda"):
                    # Only pixel_values should be bfloat16
                    processed_inputs[k] = v.to(self.device, dtype=torch.bfloat16)
                else:
                    # input_ids and other tensors should remain their original dtype
                    processed_inputs[k] = v.to(self.device)
            else:
                processed_inputs[k] = v
        inputs = processed_inputs
        
        # Get action using model's predict_action method
        # This handles all tokenization, decoding, and unnormalization
        with torch.no_grad():
            action = self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
        
        # Return as numpy array (7,)
        return action
    
    def set_unnorm_key(self, unnorm_key: str):
        """
        Set the normalization key for action denormalization.

        Args:
            unnorm_key: Dataset name for unnormalization statistics
        """
        self.unnorm_key = unnorm_key
        print(f"[OpenVLA] Unnorm key set to: {unnorm_key}")

    def set_unnorm_key_for_libero(self, suite: str):
        """
        Automatically set the correct unnorm_key for a LIBERO suite.

        Args:
            suite: LIBERO suite name (libero_spatial, libero_object, libero_goal, libero_10)
        """
        # Map LIBERO suite names to their unnormalization keys
        # Based on OpenVLA-OFT implementation
        libero_unnorm_keys = {
            "libero_spatial": "libero_spatial_no_noops",
            "libero_object": "libero_object_no_noops",
            "libero_goal": "libero_goal_no_noops",
            "libero_10": "libero_10_no_noops"
        }

        if suite in libero_unnorm_keys:
            self.unnorm_key = libero_unnorm_keys[suite]
            print(f"[OpenVLA] Unnorm key set to: {self.unnorm_key} (for {suite})")
        else:
            print(f"[OpenVLA] ⚠ Warning: Unknown LIBERO suite '{suite}', keeping unnorm_key={self.unnorm_key}")
            print(f"[OpenVLA]   Valid suites: {list(libero_unnorm_keys.keys())}")


