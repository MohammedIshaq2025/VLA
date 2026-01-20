"""
OpenVLA Action Extraction Hook
Wrapper around OpenVLA that extracts the 7D action vector for SE(3) manifold optimization.
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from typing import Optional


class OpenVLAActionExtractor:
    """
    Wrapper around OpenVLA that extracts continuous 7D action vectors.
    This is CRITICAL for SE(3) manifold optimization.
    """
    
    def __init__(self, model_path: str = "/data1/ma1/Ishaq/ump-vla/checkpoints/openvla-7b", device: Optional[str] = None):
        """
        Initialize OpenVLA model for action extraction.
        
        Args:
            model_path: Path to OpenVLA checkpoint
            device: Device to use (default: cuda:0 if available, else cpu)
        """
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_path = model_path
        
        print(f"[OpenVLA] Loading model from {model_path} on {device}...")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Load model
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
            low_cpu_mem_usage=True,
            load_in_4bit=False
        )
        
        if device.startswith("cuda"):
            self.model = self.model.to(device)
        
        # Default unnorm_key for action denormalization
        self.unnorm_key = "bridge_orig"
        
        print(f"[OpenVLA] Model loaded successfully")
        print(f"[OpenVLA] Action dim: {self.model.get_action_dim(self.unnorm_key)}")
        
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


