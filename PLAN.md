# GGUF Tooling: Extended Quantization + Metadata Plan

This plan outlines the features, modules, and steps needed to enrich our Rust-based GGUF tooling with more robust quantization options, metadata preservation, and diagnostic utilities. The focus is on:

- 🧠 Preserving full model context
- 🔎 Injecting meaningful metadata
- 🧪 Tracking quantization quality
- 🛠️ Supporting flexible quantization formats

---

## ✅ GOAL

To ensure that all GGUF outputs retain the original HuggingFace model metadata, track quantization quality, and support a variety of quantization formats with validation.

---

## 📂 PHASE 1: Metadata Preservation

### 🔧 `hf_config_to_gguf.rs`

**Purpose:** Extract architectural and tokenizer metadata from `config.json` and convert into GGUF metadata entries.

#### Keys to Promote:

- Architecture:

  - `hidden_size`
  - `intermediate_size`
  - `num_attention_heads`
  - `num_hidden_layers`
  - `layer_norm_eps`
  - `tie_word_embeddings`
  - `kv_cache`
  - `rotary_dim`

- Tokenizer (if applicable):

  - `vocab_size`
  - `pad_token_id`, `bos_token_id`, `eos_token_id`

- Training context:

  - `fine_tuned_from`
  - `fine_tune_dataset`
  - `training_steps`
  - `learning_rate`

#### Output:

```rust
Vec<(String, GGUFValue)> // key-value metadata to append to GGUF
```

### 🛠 Add injection step to writer pipeline

- Load the `config.json`
- Convert to GGUF metadata using `hf_config_to_gguf`
- Inject into `gguf_metadata` during write

---

## 📂 PHASE 2: Quantization Enrichment

### 🎯 Add support for:

- `Q3_K`
- `Q6_K`
- `Q8_0`

### 🧮 `quantization_loss.rs`

- Compare float32 tensor and quantized tensor (dequantized to f32)
- Output MSE and max absolute difference per tensor

```rust
struct QuantizationLoss {
    tensor_name: String,
    mse: f32,
    max_error: f32,
}
```

### 🧾 Metadata Injection:

- `quantized = true`
- `quantization_format = "Q4_0" | "Q5_1" | ...`
- `original_dtype = "f32"`
- `quantization_loss_mse = <avg>`
- `quantizer = "quantize-rs v0.1.0"`

---

## 📂 PHASE 3: Validation + Regression Diffing

### 🧪 `gguf_diff.rs`

- Compare two GGUF metadata maps
- Highlight added, removed, or changed keys
- Optionally validate tensor type ID consistency

```sh
gguf-diff original.gguf quantized.gguf
```

### ✅ Checks to assert:

- All pre-quantization metadata keys still exist post-quantization
- Quantization metadata added
- Tensor type IDs shifted to correct range (Q4: 100, Q5: 101, etc.)

---

## 📂 PHASE 4: Benchmarking and Evaluation CLI

### `benchmarks/compare.rs`

Run multiple quantizations side-by-side:

- Output size
- Average quantization loss
- Optional inference benchmark hook

```sh
quantize-bench --model newtest.gguf --modes Q4_0 Q5_1 Q6_K Q8_0
```

---

## 🔮 FUTURE IDEAS

- Save tensor-wise loss as separate file
- GUI or Web-based visual diff tool
- LLM-based prompt summarizer of quantization tradeoffs

---

## 🔚 Final Notes

This plan will be implemented incrementally but tested thoroughly after each phase. Metadata fidelity, quantization quality, and tooling consistency are top priorities.

---

Next: `METADATA_KEYS.md` — a formal reference of all expected keys for model introspection + validation.
