"""
Base Configuration System for VLA Frequency Attack Research
Phase 0 - Section 0.5.3

Provides dataclass-based configurations for:
- Model settings
- Attack parameters
- Dataset paths
- Experiment tracking

Usage:
    from configs.base_config import ModelConfig, AttackConfig, ExperimentConfig

    model_cfg = ModelConfig()
    attack_cfg = AttackConfig(epsilon=16/255, steps=100)
    exp_cfg = ExperimentConfig(name="my_experiment")
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for VLA model"""

    # Model identification
    name: str = "openvla-7b"
    model_id: str = "openvla/openvla-7b"

    # Paths
    cache_dir: str = "/data1/ma1/Ishaq/VLA_Frequency_Attack/cache"

    # Model loading
    device: str = "cuda"
    torch_dtype: str = "float16"  # "float32", "float16", "bfloat16"
    device_map: str = "auto"
    low_cpu_mem_usage: bool = True

    # Inference
    batch_size: int = 1
    max_length: int = 512

    def __post_init__(self):
        """Validate configuration"""
        assert self.torch_dtype in ["float32", "float16", "bfloat16"]
        assert self.device in ["cuda", "cpu"]


@dataclass
class AttackConfig:
    """Configuration for adversarial attacks"""

    # Basic attack parameters
    epsilon: float = 8/255  # Maximum perturbation magnitude
    steps: int = 50  # Number of optimization steps
    alpha: Optional[float] = None  # Step size (will be epsilon/steps if None)

    # Frequency domain parameters
    freq_ratio: float = 0.25  # Ratio of frequency components to perturb
    freq_band: str = "low"  # "low", "mid", "high", "all"
    dct_block_size: int = 8  # Block size for block-wise DCT (0 for full image)

    # Attack type
    attack_type: str = "pgd"  # "fgsm", "pgd", "frequency_pgd", "spectrum_simulation"
    targeted: bool = False  # Targeted vs untargeted attack
    target_action: Optional[List[float]] = None  # Target action for targeted attacks

    # Optimization
    norm: str = "inf"  # "inf", "2", "1"
    loss_type: str = "ce"  # "ce" (cross-entropy), "mse", "action_divergence"

    # Early stopping
    early_stop: bool = True
    early_stop_threshold: float = 0.95  # Stop if success rate > threshold

    def __post_init__(self):
        """Validate and set derived parameters"""
        if self.alpha is None:
            self.alpha = self.epsilon / self.steps

        assert 0 < self.epsilon <= 1.0
        assert self.steps > 0
        assert self.freq_ratio > 0 and self.freq_ratio <= 1.0
        assert self.freq_band in ["low", "mid", "high", "all"]
        assert self.attack_type in ["fgsm", "pgd", "frequency_pgd", "spectrum_simulation"]
        assert self.norm in ["inf", "2", "1"]


@dataclass
class DataConfig:
    """Configuration for datasets"""

    # Paths
    project_root: str = "/data1/ma1/Ishaq/VLA_Frequency_Attack"
    libero_path: str = "/data1/ma1/Ishaq/VLA_Frequency_Attack/data/libero"
    test_images_path: str = "/data1/ma1/Ishaq/VLA_Frequency_Attack/data/test_images"
    jigsaw_path: str = "/data1/ma1/Ishaq/VLA_Frequency_Attack/data/jigsaw"

    # LIBERO settings
    libero_suite: str = "libero_spatial"  # "libero_spatial", "libero_object", "libero_goal"
    num_tasks: Optional[int] = None  # Number of tasks to use (None for all)
    num_demos_per_task: int = 10  # Number of demonstrations per task

    # Data loading
    batch_size: int = 1
    num_workers: int = 4
    shuffle: bool = False

    # Image preprocessing
    image_size: int = 224  # Input size for vision encoder
    normalize: bool = True
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    def __post_init__(self):
        """Validate paths"""
        assert self.libero_suite in ["libero_spatial", "libero_object", "libero_goal"]
        assert self.image_size > 0
        assert len(self.mean) == 3 and len(self.std) == 3


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking and logging"""

    # Experiment identification
    name: str = "frequency_attack_exp"
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Paths
    output_dir: str = "/data1/ma1/Ishaq/VLA_Frequency_Attack/experiments"
    log_dir: str = "/data1/ma1/Ishaq/VLA_Frequency_Attack/experiments/logs"
    results_dir: str = "/data1/ma1/Ishaq/VLA_Frequency_Attack/experiments/results"
    viz_dir: str = "/data1/ma1/Ishaq/VLA_Frequency_Attack/experiments/visualizations"

    # Logging
    log_interval: int = 10  # Log every N steps
    save_images: bool = True  # Save adversarial examples
    save_frequency: int = 50  # Save every N examples

    # Experiment tracking (Weights & Biases)
    use_wandb: bool = False
    wandb_project: str = "vla-frequency-attack"
    wandb_entity: Optional[str] = None

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Saving
    save_checkpoint: bool = True
    checkpoint_frequency: int = 100

    def __post_init__(self):
        """Create directories if they don't exist"""
        # Only create directories if we have write permission
        try:
            for dir_path in [self.output_dir, self.log_dir, self.results_dir, self.viz_dir]:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            # Skip directory creation if on read-only filesystem or no permissions
            # Directories will be created during actual deployment on cluster
            pass


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""

    # Evaluation settings
    num_episodes: int = 50  # Number of episodes per task
    max_steps_per_episode: int = 500  # Maximum steps per episode
    success_threshold: float = 0.9  # Threshold for success

    # Metrics to compute
    compute_success_rate: bool = True
    compute_action_divergence: bool = True
    compute_trajectory_similarity: bool = True

    # Video recording
    record_video: bool = True
    video_frequency: int = 10  # Record every N episodes

    # Comparison
    compare_clean_vs_adv: bool = True
    compare_freq_bands: bool = True  # Compare different frequency bands


@dataclass
class FrequencyAnalysisConfig:
    """Configuration for frequency analysis experiments"""

    # DCT parameters
    dct_norm: str = "ortho"  # "ortho" or None
    block_based: bool = False  # Block-wise DCT (like JPEG)
    block_size: int = 8

    # Frequency bands to analyze
    bands: List[str] = field(default_factory=lambda: ["low", "mid", "high"])

    # Low frequency: top-left corner
    low_freq_ratio: float = 0.25  # 25% of coefficients

    # Mid frequency: diagonal band
    mid_freq_ratio: float = 0.5  # 50% of coefficients

    # High frequency: bottom-right corner
    high_freq_ratio: float = 1.0  # Remaining coefficients

    # Analysis settings
    visualize_spectrum: bool = True
    compute_energy_distribution: bool = True
    test_band_importance: bool = True  # Test which bands are most important


# Utility function to create default configs
def get_default_configs() -> Dict[str, Any]:
    """Get dictionary of all default configurations"""
    return {
        "model": ModelConfig(),
        "attack": AttackConfig(),
        "data": DataConfig(),
        "experiment": ExperimentConfig(),
        "evaluation": EvaluationConfig(),
        "frequency": FrequencyAnalysisConfig(),
    }


# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("Base Configuration System Test")
    print("="*70)

    # Create configs
    configs = get_default_configs()

    # Print each config
    for name, config in configs.items():
        print(f"\n{name.upper()} CONFIG:")
        print("-"*70)
        for field_name in config.__dataclass_fields__:
            value = getattr(config, field_name)
            print(f"  {field_name}: {value}")

    print("\n" + "="*70)
    print("Configuration system working correctly!")
    print("="*70)

    # Example: Create custom attack config
    print("\nExample: Custom Attack Configuration")
    print("-"*70)

    custom_attack = AttackConfig(
        epsilon=16/255,
        steps=100,
        freq_ratio=0.5,
        freq_band="low",
        attack_type="frequency_pgd"
    )

    print(f"Epsilon: {custom_attack.epsilon:.4f}")
    print(f"Alpha (auto-computed): {custom_attack.alpha:.4f}")
    print(f"Steps: {custom_attack.steps}")
    print(f"Frequency band: {custom_attack.freq_band}")

    print("\n✅ Configuration system ready for use!")
