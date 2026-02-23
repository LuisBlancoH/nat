"""
Configuration management for NAT.

Loads YAML configs and provides a simple namespace-style access.
"""

import yaml
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class NATConfig:
    """Configuration for the Nested Adaptive Transformer."""

    # Model
    base_model_name: str = "Qwen/Qwen2.5-1.5B"
    rank: int = 32
    d_hidden: int = 256

    # Adaptation
    adapt_every_n: int = 32
    lr_clamp: float = 0.1
    fast_weight_max_norm: float = 10.0

    # Consolidation
    beta: float = 0.999
    session_reset_alpha: float = 0.5

    # Training - Phase 1
    lr_phase1: float = 3e-4
    num_episodes_p1: int = 50000
    batch_size: int = 4
    seq_len: int = 2048
    truncated_bptt: int = 16
    grad_clip: float = 1.0
    weight_decay: float = 0.01

    # Training - Phase 2
    lr_phase2: float = 3e-4
    num_episodes_p2: int = 30000
    improvement_weight: float = 0.1
    num_problems_per_episode: int = 8
    adapt_problems_p2: int = 5
    batch_size_p2: int = 2

    # Training - Phase 3
    lr_phase3: float = 1e-4
    num_runs_p3: int = 500
    sessions_per_domain_p3: int = 20
    forgetting_test_sessions_p3: int = 5
    p3_truncate_sessions: int = 4

    # Device
    device: str = "auto"
    base_dtype: str = "bfloat16"

    # Performance / device-specific
    gradient_checkpointing: bool = False
    compile_model: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    empty_cache_every: int = 0         # 0 = disabled
    tf32_matmul: bool = False          # A100 TF32 tensor cores
    cudnn_benchmark: bool = False      # cuDNN auto-tuner
    cuda_amp: bool = False             # torch.amp autocast for frozen layers

    # Logging
    wandb_project: str = "nat"
    wandb_entity: Optional[str] = None
    log_every: int = 50

    # Saving
    save_dir: str = "checkpoints"
    save_every: int = 1000
    save_path: str = "checkpoints/phase1.pt"

    # Derived (set in __post_init__)
    lr: float = field(default=3e-4, init=False)
    num_episodes: int = field(default=50000, init=False)

    def __post_init__(self):
        # Resolve device
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # Resolve dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(self.base_dtype, torch.bfloat16)

        # Set aliases used by training scripts
        self.lr = float(self.lr_phase1)
        self.num_episodes = int(self.num_episodes_p1)

        # Apply CUDA-specific global settings
        if self.device == "cuda":
            if self.tf32_matmul:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            if self.cudnn_benchmark:
                torch.backends.cudnn.benchmark = True

        # Create save directory
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: str) -> "NATConfig":
        """Load config from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        # Coerce YAML values to declared dataclass types (some YAML
        # parsers return strings for scientific notation like 3e-4).
        filtered = {}
        for k, v in data.items():
            if k not in cls.__dataclass_fields__:
                continue
            ft = cls.__dataclass_fields__[k].type
            if ft is float and isinstance(v, str):
                v = float(v)
            elif ft is int and isinstance(v, str):
                v = int(v)
            elif ft is bool and isinstance(v, str):
                v = v.lower() in ("true", "1", "yes")
            filtered[k] = v
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Convert config to a dictionary (for wandb logging)."""
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and k != "torch_dtype"
        }
