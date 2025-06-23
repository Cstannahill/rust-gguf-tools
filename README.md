# 🧠 rust-gguf-tools

A modular toolkit for creating, quantizing, inspecting, and validating [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) files from Hugging Face models and adapters.

---

## 📦 Tools Overview

| Tool            | Purpose                                             |
| --------------- | --------------------------------------------------- |
| `gguf-writer`   | Writes GGUF from `meta.json` + `tensors.json`       |
| `quantize-rs`   | Applies Q4_0 or Q5_1 quantization to float32 GGUF   |
| `gguf-inspect`  | Dumps metadata and tensors from a `.gguf` file      |
| `gguf-validate` | Validates tensor decode logic for quantized GGUF    |
| `hf_to_gguf.py` | Converts a HF model (or adapter) to GGUF-ready JSON |
| `merge.py`      | Merges LoRA adapter into base model                 |

---

## 🔁 Common Usage Flow

```bash
# Step 1: Convert Hugging Face model (or fine-tuned adapter)
python hf_to_gguf.py \
  --model mistralai/Mistral-7B-v0.1 \
  --output-dir ./output/mistral

# OR convert local fine-tuned adapter
python hf_to_gguf.py \
  --model ./checkpoints/my-lora \
  --output-dir ./output/my-lora \
  --local

# Step 2 (optional): Merge LoRA adapter into base model
python merge.py \
  --base-model mistralai/Mistral-7B-v0.1 \
  --adapter ./checkpoints/my-lora \
  --output ./merged-model

# Step 3: Write GGUF file
cargo run --release -p gguf-writer -- \
  -m ./output/mistral/meta.json \
  -t ./output/mistral/tensors.json \
  -o test.gguf

# Step 4: Quantize the GGUF
cargo run --release -p quantize-rs -- \
  -i test.gguf -o test_q4.gguf -f Q4_0

# Step 5: Inspect the result
cargo run --release -p gguf-inspect -- test_q4.gguf

# Step 6: Validate it works
cargo run --release -p gguf-validate -- test_q4.gguf
```

---

## 🧪 Python Scripts

### huggingface_to_gguf.py

| Arg              | Description                                   |
| ---------------- | --------------------------------------------- |
| `--model`        | HF model ID or local path                     |
| `--output-dir`   | Where to store `meta.json` and `tensors.json` |
| `--local`        | Force local loading (default: auto-detect)    |
| `--no-tokenizer` | Skip tokenizer saving                         |

### merge_adapter.py

| Arg            | Description                                                 |
| -------------- | ----------------------------------------------------------- |
| `--base-model` | Base HF model ID or local path                              |
| `--adapter`    | Path to LoRA adapter (dir with `adapter_model.safetensors`) |
| `--output`     | Output directory for merged HF model                        |

---

## ✅ Output Compatibility

The final `.gguf` files can be used in:

- **Ollama**
- **ggml / llama.cpp**
- **Any tool accepting quantized GGUF models**

---

## 🧰 Notes

- All tensor data is extracted as float32, with support for float16/bfloat16 downcast
- GGUF metadata is inferred from `model.config`
- Quantized output supports Q4_0 and Q5_1 (more formats coming soon!)
