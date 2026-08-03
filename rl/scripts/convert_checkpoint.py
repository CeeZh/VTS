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
    """Symlink the model weights instead of copying.

    Handles both single-file (`model.safetensors`) and sharded
    (`model-00001-of-0000N.safetensors` + `model.safetensors.index.json`)
    checkpoints.
    """
    single = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(single):
        dst = os.path.join(output_dir, "model.safetensors")
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(single), dst)
        print(f"[OK] model.safetensors symlinked ({os.path.getsize(single) / 1e9:.1f} GB)")
        return

    # Sharded checkpoint: symlink every shard + copy the index.
    shards = sorted(Path(checkpoint_dir).glob("model-*-of-*.safetensors"))
    index = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if not shards or not os.path.exists(index):
        raise FileNotFoundError(
            f"No model.safetensors or sharded weights (+index) found in {checkpoint_dir}"
        )
    total = 0
    for shard in shards:
        dst = os.path.join(output_dir, shard.name)
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(shard), dst)
        total += shard.stat().st_size
    shutil.copy2(index, os.path.join(output_dir, "model.safetensors.index.json"))
    print(f"[OK] {len(shards)} weight shards symlinked + index copied "
          f"({total / 1e9:.1f} GB)")


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
        "added_tokens.json",
        "special_tokens_map.json",
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
# Qwen3-VL (single-file or sharded LLaMA-Factory checkpoint):
python rl/scripts/convert_checkpoint.py \
    --checkpoint-dir output/vts_qwen3_vl_8b \
    --original-model-dir ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/<snapshot> \
    --output-dir output/vts_qwen3_vl_8b-converted

# Qwen2.5-VL: reference config is flat (transformers 4.x), so add --copy-config-from-original.
python rl/scripts/convert_checkpoint.py \
    --checkpoint-dir output/vts_qwen2_5_vl_7b \
    --original-model-dir ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/<snapshot> \
    --output-dir output/vts_qwen2_5_vl_7b-converted \
    --copy-config-from-original
'''