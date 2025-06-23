import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_adapter(base_model_path, adapter_path, output_path, use_safetensors=True):
    print(f"📦 Loading base model from: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

    print(f"🔧 Loading LoRA adapter from: {adapter_path}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)

    print("🔗 Merging adapter weights using merge_and_unload...")
    merged_model = peft_model.merge_and_unload()

    print(f"💾 Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path, safe_serialization=use_safetensors)

    print("💾 Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)

    print("✅ Merge complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, help="Base model ID or path")
    parser.add_argument("--adapter", required=True, help="PEFT adapter path")
    parser.add_argument(
        "--output", required=True, help="Output directory for merged model"
    )
    parser.add_argument(
        "--no-safetensors", action="store_true", help="Disable saving with safetensors"
    )
    args = parser.parse_args()

    merge_adapter(
        args.base_model,
        args.adapter,
        args.output,
        use_safetensors=not args.no_safetensors,
    )


if __name__ == "__main__":
    main()
