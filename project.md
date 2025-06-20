# 🦙 Rust GGUF Tooling Project — Quantization & Conversion Utilities

## 📌 Project Overview

This project aims to **rebuild two core utilities from the `llama.cpp` ecosystem** in **Rust**:

1. **Model Quantizer** – Converts high-precision transformer model weights (e.g. from Hugging Face safetensors) to quantized formats like `Q4_0`, `Q5_K`, etc.
2. **GGUF Converter** – Packages model weights and metadata into a valid `.gguf` format file for local inference with engines like `llama.cpp` or `Ollama`.

These utilities are **not** intended to run inference — they are part of the preprocessing and model preparation pipeline, especially following a fine-tuning step done using Hugging Face's `transformers`.

---

## ✅ Motivation & Justification

### Why Rust?

- ✅ **Modern build system** (`cargo`) avoids typical CMake/Python/Windows compiler hell.
- ✅ **Cross-platform** binary generation and reproducibility.
- ✅ **High performance**, SIMD-capable, but memory-safe.
- ✅ Well-suited for CLI tools with precise binary I/O, math, and file conversion logic.

### Why Replace llama.cpp Tools?

- You’re currently only using the **quantization and conversion** CLI tools (not inference).
- llama.cpp is:
  - Written in C++ with manual memory management.
  - Often difficult to build on Windows (especially when linked to Python bindings).
- You want more **modular**, **maintainable**, and **extensible** tools that integrate well with your own platform and workflows.

---

## 🛠 Goals

### 1. `quantize-rs`

> 🧠 CLI utility that loads HF safetensors or PyTorch `.bin` weights and rewrites them into a quantized version compatible with GGUF.

- [ ] Support Q4_0, Q4_K, Q5_0, Q5_K, Q8_0 formats (priority: Q4_0/Q5_K).
- [ ] Compatible with weights produced from HF training (BFloat16, Float16, Float32).
- [ ] Output can be directly fed into GGUF writer or another inference engine.

### 2. `gguf-writer`

> 📦 CLI tool that reads quantized model weights, tokenizer config, and metadata, then writes a `.gguf` file following llama.cpp's [GGUF spec](https://github.com/ggerganov/llama.cpp/blob/master/docs/gguf.md).

- [ ] Parse tokenizer JSON or vocab files.
- [ ] Add appropriate metadata blocks (model type, architecture, tokenizer, quantization scheme, etc.).
- [ ] Validate output against llama.cpp's GGUF loader.
- [ ] Fully cross-platform and usable as a standalone tool.

---

## 🔍 Current Workflow Context

You are currently using:

- 🧠 Hugging Face `transformers` for training/fine-tuning.
- 🧪 External llama.cpp tools (`quantize.cc`, `convert-llama-gguf`) for preparing models for use in:
  - `llama.cpp` for inference
  - `Ollama` for local deployment

Problems encountered:

- Compilation issues on Windows with C++ tools and Python bindings.
- Inability to easily extend or automate quantization/formatting logic.
- Desire for deeper understanding and control of the GGUF pipeline.

---

## 💡 Implementation Plan

### Phase 1 – Bootstrap & Scaffolding

- [ ] Set up `cargo` workspaces for both `quantize-rs` and `gguf-writer`.
- [ ] Create CLI structure using `clap` or `argp`.
- [ ] Add logging and clear status output.

### Phase 2 – Quantization Core

- [ ] Load model from `safetensors` (via `safetensors` crate).
- [ ] Implement scalar/vector quantization routines.
- [ ] Support batched input and model inspection/debug view.

### Phase 3 – GGUF Writer

- [ ] Implement GGUF header + tensor block encoding.
- [ ] Ingest tokenizer + metadata files.
- [ ] Output `.gguf` file and validate against llama.cpp parser.

### Phase 4 – Integration and Polish

- [ ] Add tests against reference models.
- [ ] Benchmark performance against llama.cpp’s tooling.
- [ ] Optional: provide Python bindings via `pyo3`.

---

## 📦 Dependencies and Libraries (Tentative)

| Purpose            | Crate                                                                                                    |
| ------------------ | -------------------------------------------------------------------------------------------------------- |
| CLI Parsing        | `clap`, `argp`, or `structopt`                                                                           |
| File IO            | `tokio`, `std::fs`, `memmap2`                                                                            |
| Matrix Ops / Math  | `ndarray`, `nalgebra`, `simdeez` (if needed)                                                             |
| Safetensors Loader | [`safetensors`](https://crates.io/crates/safetensors)                                                    |
| Logging            | `tracing`, `env_logger`, or `log`                                                                        |
| GGUF Spec          | Custom – based on [llama.cpp GGUF spec](https://github.com/ggerganov/llama.cpp/blob/master/docs/gguf.md) |

---

## 🧠 Future Expansion Ideas

- Support direct `.pt` and `.bin` parsing.
- Add full tokenizer conversion tools (from HF JSON to GGUF).
- GGUF schema validation library.
- Create an API-first service for quantization + conversion for web-based UIs.

---

## 🗃️ Project Layout Proposal
