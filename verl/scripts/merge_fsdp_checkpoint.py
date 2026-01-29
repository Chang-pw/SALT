"""
Merge FSDP sharded checkpoint into a single HuggingFace model.

Usage:
    python scripts/merge_fsdp_checkpoint.py \
        --ckpt_dir checkpoints/grpo_math-train/deepseek-qwen1.5b_gradient/global_step_100/actor \
        --output_dir /path/to/merged_model

Or use the built-in verl merger:
    python -m verl.model_merger.merge \
        --backend fsdp \
        --local_dir checkpoints/.../actor \
        --target_dir /path/to/merged_model
"""

import argparse
import os
import sys

# Add verl to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelMergerConfig:
    operation: str = "merge"
    backend: str = "fsdp"
    local_dir: str = ""
    target_dir: str = ""
    test_hf_dir: Optional[str] = None
    hf_upload: bool = False
    hf_upload_repo: Optional[str] = None
    trust_remote_code: bool = False
    hf_model_config_path: Optional[str] = None


def merge_checkpoint(ckpt_dir: str, output_dir: str):
    """Merge FSDP checkpoint using verl's FSDPModelMerger."""
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    # Create config
    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=ckpt_dir,
        target_dir=output_dir,
    )

    # Check huggingface dir exists
    hf_dir = os.path.join(ckpt_dir, "huggingface")
    if not os.path.exists(hf_dir):
        raise FileNotFoundError(f"huggingface config not found in {hf_dir}")

    # Set hf_dir for tokenizer/config loading
    config.hf_dir = hf_dir
    config.hf_model_config_path = hf_dir

    print(f"Merging checkpoint from: {ckpt_dir}")
    print(f"Output directory: {output_dir}")

    # Create merger and run
    merger = FSDPModelMerger(config)
    merger.merge_and_save()

    print(f"\nDone! Merged model saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Merge FSDP checkpoint to HuggingFace format")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Path to actor checkpoint directory (e.g., .../global_step_100/actor)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to save merged HuggingFace model")
    args = parser.parse_args()

    merge_checkpoint(args.ckpt_dir, args.output_dir)


if __name__ == "__main__":
    main()
