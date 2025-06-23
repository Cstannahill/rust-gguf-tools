use clap::Parser;
use log::info;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self};

use byteorder::{LittleEndian, WriteBytesExt};
use gguf_core::types::{GGUFValue, GGUFTensor};
use gguf_core::writer::write_gguf_file;
use safetensors::tensor::Dtype;
use safetensors::SafeTensors as SafeTensorFile;
use serde::Deserialize;

mod hf_config_to_gguf;
use hf_config_to_gguf::convert_config_to_metadata;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Path to metadata JSON
    #[arg(short, long)]
    metadata: String,

    /// Output GGUF file path
    #[arg(short, long)]
    output: String,

    /// Optional path to tensor definitions in JSON
    #[arg(short, long)]
    tensors: Option<String>,

    /// Optional path to tensors in safetensors format
    #[arg(short = 's', long)]
    safetensors: Option<String>,

    /// Optional path to HuggingFace config.json
    #[arg(long)]
    config: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TensorDef {
    name: String,
    #[serde(rename = "type")]
    type_id: u32,
    dims: Vec<u64>,
    values: Vec<f32>,
}

fn parse_metadata(raw_metadata: BTreeMap<String, serde_json::Value>) -> BTreeMap<String, GGUFValue> {
    let mut metadata = BTreeMap::new();

    for (key, val) in raw_metadata {
        let parsed = match val {
            serde_json::Value::String(s) => GGUFValue::String(s),
            serde_json::Value::Number(n) => {
                if let Some(u) = n.as_u64() {
                    GGUFValue::U64(u)
                } else if let Some(i) = n.as_i64() {
                    GGUFValue::I64(i)
                } else if let Some(f) = n.as_f64() {
                    GGUFValue::F64(f)
                } else {
                    eprintln!("⚠️ Skipping unsupported number: {n}");
                    continue;
                }
            }
            serde_json::Value::Bool(b) => GGUFValue::Bool(b),
            _ => {
                eprintln!("⚠️ Skipping unsupported metadata key: {key}");
                continue;
            }
        };
        metadata.insert(key, parsed);
    }
    metadata
}

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
    let safetensors = SafeTensorFile::deserialize(&data)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let mut tensors = Vec::new();
    let mut used_float32 = true;

    for (name, tensor_view) in safetensors.tensors() {
        let dtype = tensor_view.dtype();
        let dims = tensor_view.shape().iter().map(|&x| x as u64).collect::<Vec<_>>();

        let tensor_data = match dtype {
            Dtype::F32 => tensor_view.data().to_vec(),
            Dtype::F16 => {
                used_float32 = false;
                use half::f16;
                tensor_view
                    .data()
                    .chunks_exact(2)
                    .flat_map(|chunk| {
                        let bytes = [chunk[0], chunk[1]];
                        let half = f16::from_le_bytes(bytes);
                        half.to_f32().to_le_bytes()
                    })
                    .collect::<Vec<u8>>()
            }
            Dtype::BF16 => {
                used_float32 = false;
                tensor_view
                    .data()
                    .chunks_exact(2)
                    .flat_map(|chunk| {
                        let b1 = chunk[0] as u32;
                        let b2 = chunk[1] as u32;
                        let u32_val = (b2 << 24) | (b1 << 16);
                        f32::from_bits(u32_val).to_le_bytes()
                    })
                    .collect::<Vec<u8>>()
            }
            _ => {
                eprintln!("⚠️ Unsupported dtype for tensor '{}': {:?}", name, dtype);
                continue;
            }
        };

        tensors.push(GGUFTensor {
            name: name.to_string(),
            type_id: 0,
            dims,
            offset: 0,
            values: tensor_data,
        });
    }

    Ok((tensors, used_float32))
}

fn main() -> io::Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    info!("Metadata path: {}", cli.metadata);
    info!("Output path: {}", cli.output);

    let meta_file = File::open(&cli.metadata)?;
    let raw_metadata: BTreeMap<String, serde_json::Value> = serde_json::from_reader(meta_file)?;
    let mut metadata = parse_metadata(raw_metadata);

    // Add HF config promoted fields
    if let Some(config_path) = &cli.config {
        match convert_config_to_metadata(config_path) {
            Ok(extra) => {
                for (k, v) in extra {
                    metadata.insert(k, v);
                }
            }
            Err(e) => eprintln!("⚠️ Failed to load config metadata: {e}"),
        }
    }

    let (tensors, _is_native_f32) = if let Some(safetensors_path) = &cli.safetensors {
        info!("📦 Loading tensors from safetensors: {}", safetensors_path);
        load_tensors_from_safetensors(safetensors_path)?
    } else if let Some(tensor_path) = &cli.tensors {
        info!("📦 Loading tensors from JSON: {}", tensor_path);
        (load_tensors_from_json(tensor_path)?, true)
    } else {
        eprintln!("❌ No tensor input provided.");
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "No tensor input provided"));
    };

    write_gguf_file(&cli.output, &metadata, &tensors)?;
    println!("✅ GGUF file written to: {}", cli.output);
    Ok(())
}
