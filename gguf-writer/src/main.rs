use clap::Parser;
use log::info;
use std::collections::BTreeMap;
use std::fs::File;
use std::io;

use gguf_core::types::{GGUFValue, GGUFTensor};
use gguf_core::writer::write_gguf_file;

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
}

#[derive(Debug, serde::Deserialize)]
struct TensorDef {
    name: String,
    #[serde(rename = "type")]
    type_id: u32,
    dims: Vec<u64>,
    values: Vec<f32>,
}

fn main() -> io::Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    info!("Metadata path: {}", cli.metadata);
    info!("Output path: {}", cli.output);

    // === Load metadata JSON ===
    let meta_file = File::open(&cli.metadata)?;
    let raw_metadata: BTreeMap<String, serde_json::Value> = serde_json::from_reader(meta_file)?;
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

    // === Load tensors or fall back to dummy ===
    let tensors: Vec<GGUFTensor> = if let Some(tensor_path) = &cli.tensors {
        let file = File::open(tensor_path)?;
        let defs: Vec<TensorDef> = serde_json::from_reader(file)?;        defs.into_iter()
            .map(|def| GGUFTensor {
                name: def.name,
                type_id: def.type_id,
                dims: def.dims,
                offset: 0, // This will be set by the writer
                values: def.values,
            })
            .collect()
    } else {        vec![
            GGUFTensor {
                name: "dummy_tensor_1".into(),
                type_id: 0,
                dims: vec![3],
                offset: 0,
                values: vec![1.0, 2.0, 3.0],
            },
            GGUFTensor {
                name: "dummy_tensor_2".into(),
                type_id: 0,
                dims: vec![2, 2],
                offset: 0,
                values: vec![4.0, 5.0, 6.0, 7.0],
            },
        ]
    };

    match write_gguf_file(&cli.output, &metadata, &tensors) {
        Ok(_) => {
            println!("✅ GGUF file written to: {}", cli.output);
            Ok(())
        }
        Err(e) => {
            eprintln!("❌ Failed to write GGUF: {e}");
            Err(e)
        }
    }
}
