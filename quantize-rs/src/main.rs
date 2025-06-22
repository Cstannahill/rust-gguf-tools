use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, BufWriter, Write};
use std::path::PathBuf;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use clap::Parser;
use serde::{Deserialize, Serialize};

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

impl From<String> for QuantizationType {
    fn from(s: String) -> Self {
        match s.to_lowercase().as_str() {
            "q4_0" => QuantizationType::Q4_0,
            "q5_1" => QuantizationType::Q5_1,
            _ => QuantizationType::Unknown(s),
        }
    }
}

/// Placeholder: Fake quantization that just casts f32s to u8s (lossy!)
fn quantize_tensor(tensor: &[f32], _format: &QuantizationType) -> Vec<u8> {
    tensor.iter().map(|v| (*v as u8)).collect()
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();
    let format = QuantizationType::from(cli.format);

    if matches!(format, QuantizationType::Unknown(_)) {
        eprintln!("❌ Unsupported quantization format: {}", cli.format);
        std::process::exit(1);
    }

    let mut input = BufReader::new(File::open(&cli.input)?);

    // === Read GGUF Header ===
    let mut magic = [0u8; 4];
    input.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        panic!("Not a GGUF file");
    }

    let version = input.read_u32::<LittleEndian>()?;
    let tensor_count = input.read_u64::<LittleEndian>()?;
    let metadata_count = input.read_u64::<LittleEndian>()?;

    println!("📥 Loaded GGUF v{version} with {tensor_count} tensors");

    // === Skip Metadata ===
    for _ in 0..metadata_count {
        let key_len = input.read_u64::<LittleEndian>()?;
        input.seek(SeekFrom::Current(key_len as i64))?;
        let value_type = input.read_u8()?;
        match value_type {
            1 => {
                let len = input.read_u64::<LittleEndian>()?;
                input.seek(SeekFrom::Current(len as i64))?;
            }
            9 | 11 | 12 => {
                input.seek(SeekFrom::Current(8))?;
            }
            10 => {
                input.seek(SeekFrom::Current(1))?;
            }
            _ => {}
        }
    }

    let mut quantized_tensors = Vec::new();

    // === Read + Quantize Tensors ===
    for _ in 0..tensor_count {
        let name_len = input.read_u64::<LittleEndian>()?;
        let mut name_bytes = vec![0u8; name_len as usize];
        input.read_exact(&mut name_bytes)?;
        let name = String::from_utf8_lossy(&name_bytes).to_string();

        let type_id = input.read_u32::<LittleEndian>()?;
        let ndim = input.read_u32::<LittleEndian>()?;
        let mut dims = Vec::new();
        for _ in 0..ndim {
            dims.push(input.read_u64::<LittleEndian>()?);
        }
        let offset = input.read_u64::<LittleEndian>()?;

        let mut raw_file = File::open(&cli.input)?;
        raw_file.seek(SeekFrom::Start(offset))?;

        let total_elems: u64 = dims.iter().product();
        let mut float_data = Vec::new();
        for _ in 0..total_elems {
            float_data.push(raw_file.read_f32::<LittleEndian>()?);
        }

        let quantized = quantize_tensor(&float_data, &format);

        quantized_tensors.push((name, type_id, dims, quantized));
    }

    // === Write New GGUF (quantized) ===
    let mut out = BufWriter::new(File::create(&cli.output)?);
    out.write_all(b"GGUF")?;
    out.write_u32::<LittleEndian>(version)?;
    out.write_u64::<LittleEndian>(quantized_tensors.len() as u64)?;
    out.write_u64::<LittleEndian>(0)?; // skip metadata for now

    // tensor headers
    let mut offset_placeholders = Vec::new();
    for (name, type_id, dims, _) in &quantized_tensors {
        out.write_u64::<LittleEndian>(name.len() as u64)?;
        out.write_all(name.as_bytes())?;
        out.write_u32::<LittleEndian>(*type_id)?; // reuse original type_id for now
        out.write_u32::<LittleEndian>(dims.len() as u32)?;
        for d in dims {
            out.write_u64::<LittleEndian>(*d)?;
        }
        offset_placeholders.push(out.seek(SeekFrom::Current(0))?);
        out.write_u64::<LittleEndian>(0)?; // offset placeholder
    }

    for (i, (_, _, _, data)) in quantized_tensors.iter().enumerate() {
        let data_offset = out.seek(SeekFrom::Current(0))?;
        out.write_all(data)?;

        let cur = out.seek(SeekFrom::Current(0))?;
        out.seek(SeekFrom::Start(offset_placeholders[i]))?;
        out.write_u64::<LittleEndian>(data_offset)?;
        out.seek(SeekFrom::Start(cur))?;
    }

    out.flush()?;
    println!("✅ Wrote quantized GGUF to {}", cli.output.display());
    Ok(())
}
