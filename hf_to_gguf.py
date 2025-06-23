import argparse
import os
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import safetensors.torch


def save_metadata(model, output_dir):
    config = model.config
    metadata = {
        "gguf_version": "2",
        "name": config.name_or_path,
        "description": getattr(
            config, "summary", "Model converted from HuggingFace format"
        ),
        "context_length": getattr(config, "max_position_embeddings", 2048),
        "embedding_size": getattr(config, "hidden_size", 4096),
        "is_quantized": False,
        "precision": 1.0,
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("✅ Saved metadata to meta.json")


def save_model_tensors(model, output_dir):
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    safetensors.torch.save_file(model.state_dict(), safetensors_path)
    print(f"✅ Saved tensors to {safetensors_path}")


def save_tokenizer(tokenizer, output_dir):
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Saved tokenizer files to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model ID or local path")
    parser.add_argument(
        "--output-dir", default=".", help="Directory to save output files"
    )
    parser.add_argument(
        "--no-tokenizer", action="store_true", help="Skip saving tokenizer files"
    )
    parser.add_argument(
        "--local", action="store_true", help="Force loading model from a local path"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model)
    load_kwargs = {"device_map": "auto", "torch_dtype": torch.float16}

    if args.local or model_path.exists():
        print(f"📂 Loading model from local path: {args.model}")
    else:
        print(f"🌐 Downloading model from Hugging Face Hub: {args.model}")

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    save_metadata(model, output_dir)
    save_model_tensors(model, output_dir)

    if not args.no_tokenizer:
        save_tokenizer(tokenizer, output_dir)

    print("🎉 Export complete!")


if __name__ == "__main__":
    main()
