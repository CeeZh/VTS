"""Convert a LlamaFactory (transformers 5.x) checkpoint to be compatible with
vLLM / ms-swift (transformers 4.x).

The main issues fixed:
  1. rope_parameters -> rope_scaling (transformers 5.x renamed this)
  2. rope_theta moved back to text_config level
  3. use_cache set to true for inference
  4. Extraneous top-level fields removed
  5. Preprocessor / tokenizer / chat_template files restored from original model
  6. generation_config.json deduplicated eos_token_id
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def fix_config(checkpoint_dir: str, output_dir: str):
    """Fix config.json: rope_parameters -> rope_scaling, use_cache, etc.

    Use this for models where the reference HF config has the same nested
    `text_config` layout as the trained checkpoint (e.g. Qwen3-VL). For models
    whose reference config uses a flat (transformers-4.x) layout (e.g.
    Qwen2.5-VL), pass --copy-config-from-original instead.
    """
    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        config = json.load(f)

    text_config = config["text_config"]

    # Fix rope_parameters -> rope_scaling
    if "rope_parameters" in text_config:
        rope_params = text_config.pop("rope_parameters")
        rope_theta = rope_params.pop("rope_theta", None)
        if rope_theta is not None:
            text_config["rope_theta"] = rope_theta
        text_config["rope_scaling"] = rope_params

    # Fix use_cache for inference
    text_config["use_cache"] = True
    config["use_cache"] = True

    # Remove extraneous top-level fields from transformers 5.x
    for key in ["bos_token_id", "eos_token_id", "pad_token_id", "hidden_size", "dtype"]:
        config.pop(key, None)

    # Remove fields not present in original config
    text_config.pop("pad_token_id", None)
    text_config.pop("dtype", None)
    if "vision_config" in config:
        config["vision_config"].pop("dtype", None)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print("[OK] config.json fixed (rope_scaling, use_cache, removed extraneous fields)")


def fix_generation_config(checkpoint_dir: str, output_dir: str):
    """Fix generation_config.json: deduplicate eos_token_id."""
    src = os.path.join(checkpoint_dir, "generation_config.json")
    if not os.path.exists(src):
        print("[SKIP] generation_config.json not found in checkpoint")
        return

    with open(src) as f:
        gen_config = json.load(f)

    if "eos_token_id" in gen_config and isinstance(gen_config["eos_token_id"], list):
        gen_config["eos_token_id"] = list(dict.fromkeys(gen_config["eos_token_id"]))

    with open(os.path.join(output_dir, "generation_config.json"), "w") as f:
        json.dump(gen_config, f, indent=2)
        f.write("\n")

    print("[OK] generation_config.json fixed (deduplicated eos_token_id)")


def copy_from_original(original_model_dir: str, output_dir: str, filenames: list[str]):
    """Copy files from the original model directory."""
    for fname in filenames:
        src = os.path.join(original_model_dir, fname)
        if os.path.exists(src):
            # Resolve symlinks (HF cache uses symlinks to blobs)
            real_src = os.path.realpath(src)
            shutil.copy2(real_src, os.path.join(output_dir, fname))
            print(f"[OK] {fname} copied from original model")
        else:
            print(f"[SKIP] {fname} not found in original model")


def copy_from_checkpoint(checkpoint_dir: str, output_dir: str, filenames: list[str]):
    """Copy files from the checkpoint directory."""
    for fname in filenames:
        src = os.path.join(checkpoint_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))
            print(f"[OK] {fname} copied from checkpoint")
        else:
            print(f"[SKIP] {fname} not found in checkpoint")


def symlink_weights(checkpoint_dir: str, output_dir: str):
    """Symlink the large model weights file instead of copying."""
    weight_src = os.path.join(checkpoint_dir, "model.safetensors")
    weight_dst = os.path.join(output_dir, "model.safetensors")
    if not os.path.exists(weight_src):
        raise FileNotFoundError(f"model.safetensors not found in {checkpoint_dir}")
    if os.path.exists(weight_dst) or os.path.islink(weight_dst):
        os.remove(weight_dst)
    os.symlink(os.path.abspath(weight_src), weight_dst)
    print(f"[OK] model.safetensors symlinked ({os.path.getsize(weight_src) / 1e9:.1f} GB)")


def verify(output_dir: str, strict_qwen3: bool = True):
    """Verify the converted checkpoint loads correctly."""
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(output_dir, trust_remote_code=True)
    if strict_qwen3:
        assert config.text_config.rope_scaling is not None, "rope_scaling is None"
        assert "mrope_section" in config.text_config.rope_scaling, "mrope_section missing"
        assert config.text_config.rope_theta == 5000000.0, f"rope_theta wrong: {config.text_config.rope_theta}"
        assert config.text_config.use_cache is True, "use_cache should be True"

    tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
    assert tokenizer.eos_token_id == 151645, f"eos_token_id wrong: {tokenizer.eos_token_id}"

    print("[OK] All verification checks passed!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LlamaFactory checkpoint (transformers 5.x) to vLLM-compatible format"
    )
    parser.add_argument("--checkpoint-dir", required=True, help="Path to LlamaFactory checkpoint")
    parser.add_argument("--original-model-dir", required=True, help="Path to original HF model directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for converted checkpoint")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification step")
    parser.add_argument(
        "--copy-config-from-original",
        action="store_true",
        help="Copy config.json from --original-model-dir verbatim instead of "
             "fixing the checkpoint's. Use for Qwen2.5-VL where the reference "
             "config has a flat (transformers-4.x) layout that differs from "
             "the trained checkpoint's nested layout.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Symlink weights
    symlink_weights(args.checkpoint_dir, args.output_dir)

    # Step 2: config.json
    if args.copy_config_from_original:
        copy_from_original(args.original_model_dir, args.output_dir, ["config.json"])
    else:
        fix_config(args.checkpoint_dir, args.output_dir)

    # Step 3: Copy preprocessor configs from original model
    copy_from_original(args.original_model_dir, args.output_dir, [
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ])
    # Also keep checkpoint's processor_config.json
    copy_from_checkpoint(args.checkpoint_dir, args.output_dir, [
        "processor_config.json",
    ])

    # Step 4: Tokenizer files
    copy_from_original(args.original_model_dir, args.output_dir, [
        "tokenizer_config.json",
        "merges.txt",
        "vocab.json",
    ])
    copy_from_checkpoint(args.checkpoint_dir, args.output_dir, [
        "tokenizer.json",
    ])

    # Step 5: Fix generation_config.json
    fix_generation_config(args.checkpoint_dir, args.output_dir)

    # Step 6: Copy chat_template.json from original
    copy_from_original(args.original_model_dir, args.output_dir, [
        "chat_template.json",
    ])

    print(f"\nConversion complete! Output: {args.output_dir}")

    # Step 7: Verify
    if not args.no_verify:
        print("\nRunning verification...")
        verify(args.output_dir, strict_qwen3=not args.copy_config_from_original)


if __name__ == "__main__":
    main()


'''
python rl/convert_checkpoint.py \
    --checkpoint-dir /mnt/arc/cezhang/projects/LlamaFactory/datagen/output/checkpoints/checkpoint-1600 \
    --original-model-dir /mnt/arc/cezhang/xdg_dirs/cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
    --output-dir /mnt/arc/cezhang/projects/LlamaFactory/datagen/output/checkpoints/checkpoint-1600-converted
'''


'''
python rl/convert_checkpoint.py \
    --checkpoint-dir /mnt/arc/cezhang/projects/LlamaFactory/datagen_lvhaystack/output/checkpoints/checkpoint-800 \
    --original-model-dir /mnt/arc/cezhang/xdg_dirs/cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
    --output-dir /mnt/arc/cezhang/projects/LlamaFactory/datagen_lvhaystack/output/checkpoints/checkpoint-800-converted
'''

'''
python rl/convert_checkpoint.py \
    --checkpoint-dir /mnt/arc/cezhang/projects/LlamaFactory/datagen_youtube_40/output/checkpoints/checkpoint-759 \
    --original-model-dir /mnt/arc/cezhang/xdg_dirs/cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
    --output-dir /mnt/arc/cezhang/projects/LlamaFactory/datagen_youtube_40/output/checkpoints/checkpoint-759-converted
'''


'''
python rl/scripts/convert_checkpoint.py \
    --checkpoint-dir /mnt/arc/cezhang/projects/LlamaFactory/datagen_3source_40/output/checkpoints/checkpoint-1260 \
    --original-model-dir /mnt/arc/cezhang/xdg_dirs/cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
    --output-dir /mnt/arc/cezhang/projects/LlamaFactory/datagen_3source_40/output/checkpoints/checkpoint-1260-converted
'''

'''
python rl/scripts/convert_checkpoint.py \
    --checkpoint-dir /mnt/arc/cezhang/projects/LlamaFactory/datagen_crop_video_flat/output/checkpoints/checkpoint-524 \
    --original-model-dir /mnt/arc/cezhang/xdg_dirs/cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
    --output-dir /mnt/arc/cezhang/projects/LlamaFactory/datagen_crop_video_flat/output/checkpoints/checkpoint-524-converted
'''

'''

huggingface-cli upload ceezh/vtemp \
  /mnt/arc/cezhang/projects/LlamaFactory/datagen_lvhaystack/output/checkpoints/checkpoint-800-converted \
  sft-lvhaystack

'''


'''
python rl/scripts/convert_checkpoint.py \
    --checkpoint-dir /mnt/arc/cezhang/projects/LlamaFactory/datagen_zoom_in/output/checkpoints/checkpoint-1052 \
    --original-model-dir /mnt/arc/cezhang/xdg_dirs/cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
    --output-dir /mnt/arc/cezhang/projects/LlamaFactory/datagen_zoom_in/output/checkpoints/checkpoint-1052-converted
'''

# Qwen2.5-VL: reference config is flat (transformers 4.x), so use --copy-config-from-original.
'''
python rl/scripts/convert_checkpoint.py \
    --checkpoint-dir /mnt/arc/cezhang/projects/LlamaFactory/datagen_3source_40/output_qwen2.5/checkpoints/checkpoint-1260 \
    --original-model-dir /mnt/arc/cezhang/xdg_dirs/cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
    --output-dir /mnt/arc/cezhang/projects/LlamaFactory/datagen_3source_40/output_qwen2.5/checkpoints/checkpoint-1260-converted \
    --copy-config-from-original
'''