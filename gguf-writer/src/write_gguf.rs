use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write, Seek, SeekFrom};
use byteorder::{LittleEndian, WriteBytesExt};
use serde_json::Value;

/// Write a GGUF file with metadata loaded from JSON, plus a dummy tensor
pub fn write_gguf_with_tensor(metadata_path: &str, output_path: &str) -> std::io::Result<()> {
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);

    // Load metadata
    let json_file = File::open(metadata_path)?;
    let metadata: BTreeMap<String, Value> = serde_json::from_reader(json_file)?;

    // === HEADER ===
    writer.write_all(b"GGUF")?;
    writer.write_u32::<LittleEndian>(2)?; // version
    writer.write_u64::<LittleEndian>(1)?; // tensor count
    writer.write_u64::<LittleEndian>(metadata.len() as u64)?; // metadata count

    // === METADATA ===
    for (key, value) in &metadata {
        let key_bytes = key.as_bytes();
        writer.write_u64::<LittleEndian>(key_bytes.len() as u64)?;
        writer.write_all(key_bytes)?;

        match value {
            Value::String(s) => {
                writer.write_u8(1)?;
                let bytes = s.as_bytes();
                writer.write_u64::<LittleEndian>(bytes.len() as u64)?;
                writer.write_all(bytes)?;
            }
            Value::Number(n) => {
                if n.is_u64() {
                    writer.write_u8(9)?;
                    writer.write_u64::<LittleEndian>(n.as_u64().unwrap())?;
                } else if n.is_i64() {
                    writer.write_u8(11)?;
                    writer.write_i64::<LittleEndian>(n.as_i64().unwrap())?;
                } else if let Some(f) = n.as_f64() {
                    writer.write_u8(12)?;
                    writer.write_f64::<LittleEndian>(f)?;
                }
            }
            Value::Bool(b) => {
                writer.write_u8(10)?;
                writer.write_u8(if *b { 1 } else { 0 })?;
            }
            _ => {
                eprintln!("⚠️  Skipping unsupported metadata key: {}", key);
            }
        }
    }

    // === TENSOR HEADER ===
    let tensor_name = "dummy_tensor";
    writer.write_u64::<LittleEndian>(tensor_name.len() as u64)?;
    writer.write_all(tensor_name.as_bytes())?;

    writer.write_u32::<LittleEndian>(0)?; // GGML type: F32
    writer.write_u32::<LittleEndian>(1)?; // n_dims
    writer.write_u64::<LittleEndian>(3)?; // dim[0] = 3
    writer.write_u64::<LittleEndian>(0)?; // offset placeholder (filled below)

    // === TENSOR DATA ===
    let tensor_offset = writer.seek(SeekFrom::Current(0))?;

    writer.write_f32::<LittleEndian>(1.0)?;
    writer.write_f32::<LittleEndian>(2.0)?;
    writer.write_f32::<LittleEndian>(3.0)?;

    // Go back and patch offset
    writer.seek(SeekFrom::Start(
        4 + 4 + 8 + 8 + // GGUF header
        metadata.iter().map(|(k, v)| {
            8 + k.len() as u64 + 1 + match v {
                Value::String(s) => 8 + s.len() as u64,
                Value::Number(_) => 8,
                Value::Bool(_) => 1,
                _ => 0
            }
        }).sum::<u64>() +
        8 + tensor_name.len() as u64 + 4 + 4 + 8 // tensor header pre-offset
    ))?;
    writer.write_u64::<LittleEndian>(tensor_offset)?;

    writer.flush()?;
    Ok(())
}