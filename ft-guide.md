# Fine-Tuning Workflow Guide for GGUF Tooling

This guide outlines the general workflow for preparing, fine-tuning, quantizing, and inspecting a model using the GGUF tooling suite provided by this repository. It assumes familiarity with HuggingFace Transformers, safetensors, and quantization concepts.

---

## 📁 Recommended Output Structure

```
project-dir/
├── merged-model/                     # Output from merging or fine-tuning HuggingFace models
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── generation_config.json
│   ├── model-00001-of-00004.safetensors
│   ├── ...
│   └── model.safetensors.index.json
│
├── metadata.json                    # Metadata extracted or manually edited (see below)
├── config_promoted.json            # Auto-extracted metadata from config.json (optional)
├── model.gguf                      # Float32 GGUF conversion
├── model-q4.gguf                   # Quantized GGUF (Q4_0)
├── model-q5.gguf                   # Quantized GGUF (Q5_1)
```

---

## ⚙️ Step-by-Step Usage

### 1. Fine-Tune or Merge Your Model

- Use HuggingFace Transformers to fine-tune or merge adapters.
- Ensure you export the full model using `save_pretrained(output_dir)` with `safe_serialization=True`.

### 2. Convert to GGUF Float Format

Use the writer CLI to convert from `safetensors` to `.gguf`:

```sh
cargo run --release -p gguf-writer \
  --metadata metadata.json \
  --safetensors merged-model/model-00001-of-00004.safetensors \
  --output model.gguf \
  --config merged-model/config.json
```

You can create `metadata.json` manually or extract it from the original model config using our:

```sh
hf_config_to_gguf.rs → Vec<(String, GGUFValue)>
```

### 3. Quantize the GGUF

Run the quantizer to reduce file size and enable faster inference:

```sh
cargo run --release -p quantize-rs \
  --input model.gguf \
  --output model-q4.gguf \
  --format Q4_0
```

Supported formats:

- Q4_0 → smaller size, lower precision
- Q5_1 → better quality, larger

Quantization process automatically injects metadata:

- `quantized = true`
- `quantization_format = "Q4_0"`
- `precision = 1.0` (can be updated later to reflect true loss)

### 4. Inspect the Model

Use the inspector to verify tensor metadata, shapes, and file size:

```sh
cargo run --release -p gguf-inspect model-q4.gguf
```

Sample output:

```
magic: GGUF
version: 2
tensor count: 195
metadata count: 9
...
Top tensors by size:
  - lm_head.weight | [32064, 3072] | type_id: 100
...
File Size:         2.87 GB
Total tensor memory: 3.82 GB (approx)
```

---

## 🧠 Optional Enhancements

- **Add Training Metadata**:

  - `fine_tuned_from`, `training_steps`, `lr`, `dataset`

- **Inject Tokenizer Metadata**:

  - `pad_token_id`, `bos_token_id`, `eos_token_id`, `vocab_size`

- **Compare Original vs Quantized**:

  - Use loss functions or metrics to validate precision

You can patch metadata directly into `metadata.json` or inject during quantization via the CLI.

---

## 📌 Best Practices

- Always inspect `.gguf` files post-quantization.
- Check that key fields like `is_quantized`, `precision`, `name`, `embedding_size`, etc. are accurate.
- Use `config.json` + `tokenizer.json` if preserving tokenizer fidelity.
- Consider versioning your output files clearly (e.g., `model-q4-2024-06-23.gguf`).

---

For integration help, advanced loss metric tools, or dataset evaluation scaffolding, see future modules in this repo.
