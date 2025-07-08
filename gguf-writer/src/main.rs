use clap::Parser;
use log::info;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::Path;

use byteorder::{LittleEndian, WriteBytesExt};
use gguf_core::types::{GGUFValue, GGUFTensor};
use gguf_core::writer::write_gguf_file;
use safetensors::tensor::Dtype;
use safetensors::SafeTensors as SafeTensorFile;
use serde::Deserialize;

mod hf_config_to_gguf;
use hf_config_to_gguf::convert_config_to_metadata;

/// ------------------------------
/// CLI
/// ------------------------------
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Optional metadata JSON file
    #[arg(short, long)]
    metadata: Option<String>,

    /// Output GGUF file
    #[arg(short, long)]
    output: String,

    /// Tensor definitions in JSON
    #[arg(short, long)]
    tensors: Option<String>,

    /// Tensors in safetensors format
    #[arg(short = 's', long)]
    safetensors: Option<String>,

    /// HuggingFace `config.json`
    #[arg(long)]
    config: Option<String>,
}

/// ------------------------------
/// TensorDef - only for JSON tensors
/// ------------------------------
#[derive(Debug, Deserialize)]
struct TensorDef {
    name: String,
    #[serde(rename = "type")]
    type_id: u32,
    dims: Vec<u64>,
    values: Vec<f32>,
}

/// ------------------------------
/// Metadata helpers
/// ------------------------------
fn parse_metadata(raw: BTreeMap<String, serde_json::Value>) -> BTreeMap<String, GGUFValue> {
    let mut out = BTreeMap::new();
    for (k, v) in raw {
        let val = match v {
            serde_json::Value::String(s) => GGUFValue::String(s),
            serde_json::Value::Number(n) => {
                if let Some(u) = n.as_u64() {
                    GGUFValue::U64(u)
                } else if let Some(i) = n.as_i64() {
                    GGUFValue::I64(i)
                } else if let Some(f) = n.as_f64() {
                    GGUFValue::F64(f)
                } else {
                    eprintln!("⚠️  Unsupported number for key {k}");
                    continue;
                }
            }
            serde_json::Value::Bool(b) => GGUFValue::Bool(b),
            _ => {
                eprintln!("⚠️  Skipping unsupported metadata key {k}");
                continue;
            }
        };
        out.insert(k, val);
    }
    out
}

fn parse_metadata_file(path: &str) -> io::Result<BTreeMap<String, GGUFValue>> {
    let file = File::open(path)?;
    let raw: BTreeMap<String, serde_json::Value> = serde_json::from_reader(file)?;
    Ok(parse_metadata(raw))
}

fn build_default_metadata(
    cfg_path: &Option<String>,
    is_quantized: bool,
    quant_fmt: &str,
) -> io::Result<BTreeMap<String, GGUFValue>> {
    let mut meta = BTreeMap::new();

    // —— Core defaults ——
    meta.insert("gguf_version".into(), GGUFValue::String("2".into()));
    meta.insert(
        "description".into(),
        GGUFValue::String("Model converted from HuggingFace format".into()),
    );
    meta.insert("precision".into(), GGUFValue::F64(1.0));
    meta.insert("is_quantized".into(), GGUFValue::Bool(is_quantized));
    if is_quantized {
        meta.insert(
            "quantization_format".into(),
            GGUFValue::String(quant_fmt.into()),
        );
    }

    // —— Promote fields from HF config if provided ——
    if let Some(p) = cfg_path {
        let mut buf = Vec::new();
        File::open(p)?.read_to_end(&mut buf)?;
        let cfg: serde_json::Value = serde_json::from_slice(&buf)?;

        if let Some(u) = cfg["max_position_embeddings"].as_u64() {
            meta.insert("context_length".into(), GGUFValue::U64(u));
        }
        if let Some(u) = cfg["hidden_size"].as_u64() {
            meta.insert("embedding_size".into(), GGUFValue::U64(u));
        }
        if let Some(name) = cfg["architectures"]
            .get(0)
            .and_then(|v| v.as_str())
            .map(str::to_owned)
        {
            meta.insert("name".into(), GGUFValue::String(name));
        }

        // merge any extra keys via helper
        if let Ok(extra) = convert_config_to_metadata(p) {
            for (k, v) in extra {
                meta.entry(k).or_insert(v);
            }
        }
    }

    Ok(meta)
}

/// ------------------------------
/// Tensor loaders
/// ------------------------------
fn load_tensors_from_json(path: &str) -> io::Result<Vec<GGUFTensor>> {
    let file = File::open(path)?;
    let defs: Vec<TensorDef> = serde_json::from_reader(file)?;

    Ok(defs
        .into_iter()
        .map(|def| {
            let mut buf = Vec::with_capacity(def.values.len() * 4);
            for v in def.values {
                buf.write_f32::<LittleEndian>(v).unwrap();
            }
            GGUFTensor {
                name: def.name,
                type_id: def.type_id,
                dims: def.dims,
                offset: 0,
                values: buf,
            }
        })
        .collect())
}

fn load_tensors_from_safetensors(path: &str) -> io::Result<(Vec<GGUFTensor>, bool)> {
    let data = fs::read(path)?;
    let st = SafeTensorFile::deserialize(&data)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let mut out = Vec::new();
    let mut all_f32 = true;

    for (name, tv) in st.tensors() {
        let dims = tv.shape().iter().map(|&d| d as u64).collect::<Vec<_>>();
        let bytes = match tv.dtype() {
            Dtype::F32 => tv.data().to_vec(),
            Dtype::F16 | Dtype::BF16 => {
                all_f32 = false;
                // convert to f32 little-endian
                tv.data()
                    .chunks_exact(2)
                    .flat_map(|c| {
                        let [lo, hi] = [c[0], c[1]];
                        let bits = match tv.dtype() {
                            Dtype::F16 => {
                                use half::f16;
                                f16::from_le_bytes([lo, hi]).to_f32().to_bits()
                            }
                            Dtype::BF16 => ((hi as u32) << 24) | ((lo as u32) << 16),
                            _ => unreachable!(),
                        };
                        bits.to_le_bytes()
                    })
                    .collect()
            }
            other => {
                eprintln!("⚠️  Unsupported dtype {:?} for {}", other, name);
                continue;
            }
        };

        out.push(GGUFTensor {
            name: name.to_string(),
            type_id: 0,
            dims,
            offset: 0,
            values: bytes,
        });
    }
    Ok((out, all_f32))
}

/// ------------------------------
/// main
/// ------------------------------
fn main() -> io::Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    info!(
        "Writer invoked → metadata: {:?}, output: {}",
        cli.metadata, cli.output
    );

    // crude heuristic: when we load from safetensors we haven't quantised yet
    let is_quantized = cli.safetensors.is_none();
    let quant_fmt = if is_quantized { "UNKNOWN" } else { "NA" };

    // -------- metadata ------------
    let mut metadata: BTreeMap<String, GGUFValue> = if let Some(path) = &cli.metadata {
        parse_metadata_file(path)?
    } else {
        build_default_metadata(&cli.config, is_quantized, quant_fmt)?
    };

    // -------- tensors -------------
    let (tensors, _native_f32) = if let Some(safe) = &cli.safetensors {
        info!("📦  Loading tensors from safetensors: {safe}");
        load_tensors_from_safetensors(safe)?
    } else if let Some(json) = &cli.tensors {
        info!("📦  Loading tensors from JSON: {json}");
        (load_tensors_from_json(json)?, true)
    } else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "No tensor input provided",
        ));
    };

    // -------- write ---------------
    write_gguf_file(&cli.output, &metadata, &tensors)?;
    println!("✅ GGUF file written to '{}'", cli.output);
    Ok(())
}
