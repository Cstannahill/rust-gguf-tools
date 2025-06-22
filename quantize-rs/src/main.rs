use std::path::PathBuf;

use byteorder::{LittleEndian};
use byteorder::WriteBytesExt;

use clap::Parser;

use gguf_core::reader::read_gguf_file;
use gguf_core::types::{GGUFValue, GGUFTensor};
use gguf_core::writer::write_gguf_file;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Input GGUF file path
    #[arg(short, long)]
    input: PathBuf,

    /// Output GGUF file path
    #[arg(short, long)]
    output: PathBuf,

    /// Quantization format (e.g., Q4_0, Q5_1)
    #[arg(short, long)]
    format: String,
}

#[derive(Debug)]
enum QuantizationType {
    Q4_0,
    Q5_1,
    Unknown(String),
}

impl QuantizationType {
    fn as_str(&self) -> &'static str {
        match self {
            QuantizationType::Q4_0 => "Q4_0",
            QuantizationType::Q5_1 => "Q5_1",
            QuantizationType::Unknown(_) => "Unknown",
        }
    }
}

impl From<String> for QuantizationType {
    fn from(s: String) -> Self {
        match s.to_lowercase().as_str() {
            "q4_0" => QuantizationType::Q4_0,
            "q5_1" => QuantizationType::Q5_1,
            _ => QuantizationType::Unknown(s),
        }
    }
}

// === Q4_0 ===
fn quantize_tensor_q4_0(tensor: &[f32]) -> Vec<u8> {
    const BLOCK_SIZE: usize = 32;
    let mut out = Vec::new();

    for chunk in tensor.chunks(BLOCK_SIZE) {
        let min = chunk.iter().copied().fold(f32::INFINITY, f32::min);
        let max = chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = (max - min).max(1e-6) / 15.0;
        let zero = min;

        let values: Vec<u8> = chunk
            .iter()
            .map(|v| ((*v - zero) / scale).round().clamp(0.0, 15.0) as u8)
            .collect();

        out.write_f32::<LittleEndian>(scale).unwrap();
        out.write_f32::<LittleEndian>(zero).unwrap();

        for pair in values.chunks(2) {
            let byte = if pair.len() == 2 {
                (pair[0] & 0x0F) | ((pair[1] & 0x0F) << 4)
            } else {
                pair[0] & 0x0F
            };
            out.push(byte);
        }
    }

    out
}

// === Q5_1 ===
fn quantize_tensor_q5_1(tensor: &[f32]) -> Vec<u8> {
    const BLOCK_SIZE: usize = 32;
    let mut out = Vec::new();

    for chunk in tensor.chunks(BLOCK_SIZE) {
        let min = chunk.iter().copied().fold(f32::INFINITY, f32::min);
        let max = chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = (max - min).max(1e-6) / 31.0;
        let zero = min;

        let levels: Vec<u8> = chunk
            .iter()
            .map(|v| (((*v - zero) / scale).round().clamp(0.0, 31.0)) as u8)
            .collect();

        let mut packed = Vec::with_capacity(20);
        let mut buffer = 0u64;
        let mut bits = 0;

        for &val in &levels {
            buffer |= (val as u64) << bits;
            bits += 5;

            while bits >= 8 {
                packed.push((buffer & 0xFF) as u8);
                buffer >>= 8;
                bits -= 8;
            }
        }

        if bits > 0 {
            packed.push(buffer as u8);
        }

        out.write_f32::<LittleEndian>(scale).unwrap();
        out.write_f32::<LittleEndian>(zero).unwrap();
        out.extend_from_slice(&packed);
    }

    out
}

// === Dispatch ===
fn quantize_tensor(tensor: &[f32], format: &QuantizationType) -> (Vec<f32>, u32) {
    match format {
        QuantizationType::Q4_0 => {
            let bytes = quantize_tensor_q4_0(tensor);
            let f32_compat: Vec<f32> = bytes.iter().map(|b| *b as f32).collect();
            (f32_compat, 100)
        }
        QuantizationType::Q5_1 => {
            let bytes = quantize_tensor_q5_1(tensor);
            let f32_compat: Vec<f32> = bytes.iter().map(|b| *b as f32).collect();
            (f32_compat, 101)
        }
        QuantizationType::Unknown(s) => panic!("Unsupported quantization format: {s}"),
    }
}

// === Main ===
fn main() -> std::io::Result<()> {
    let cli = Cli::parse();
    let format = QuantizationType::from(cli.format.clone());

    if matches!(format, QuantizationType::Unknown(_)) {
        eprintln!("❌ Unsupported quantization format: {}", cli.format);
        std::process::exit(1);
    }

    let (mut metadata, tensors) = read_gguf_file(&cli.input)?;

    let quantized: Vec<GGUFTensor> = tensors
        .into_iter()
        .map(|t| {
            let (values, new_type_id) = quantize_tensor(&t.values, &format);
            GGUFTensor {
                name: t.name,
                type_id: new_type_id,
                dims: t.dims,
                offset: 0,
                values,
            }
        })
        .collect();

    metadata.insert("quantized".into(), GGUFValue::Bool(true));
    metadata.insert(
        "quantization_format".into(),
        GGUFValue::String(format.as_str().into()),
    );

    write_gguf_file(&cli.output, &metadata, &quantized)?;
    println!("✅ Wrote quantized GGUF to {}", cli.output.display());
    Ok(())
}
