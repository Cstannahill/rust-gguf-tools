use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write, Seek, SeekFrom};
use byteorder::{LittleEndian, WriteBytesExt};
use serde_json::Value;

/// Tensor specification
// struct TensorSpec {
//     name: String,
//     type_id: u32,
//     dims: Vec<u64>,
//     values: Vec<f32>,
// }

#[derive(Debug, serde::Deserialize)]
struct TensorDef {
    name: String,
    #[serde(rename = "type")]
    type_id: u32,
    dims: Vec<u64>,
    values: Vec<f32>,
}

/// Write a GGUF file with metadata and multiple dummy tensors
pub fn write_gguf_with_tensor(metadata_path: &str, tensor_path: Option<&str>, output_path: &str) -> std::io::Result<()> {

    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);

    // Load metadata
    let json_file = File::open(metadata_path)?;
    let metadata: BTreeMap<String, Value> = serde_json::from_reader(json_file)?;

    // Dummy tensor list
    let tensors: Vec<TensorDef> = if let Some(tensor_path) = tensor_path {
        let tensor_file = File::open(tensor_path)?;
        serde_json::from_reader(tensor_file)?
    } else {
        vec![
            TensorDef {
                name: "dummy_tensor_1".to_string(),
                type_id: 0,
                dims: vec![3],
                values: vec![1.0, 2.0, 3.0],
            },
            TensorDef {
                name: "dummy_tensor_2".to_string(),
                type_id: 0,
                dims: vec![2, 2],
                values: vec![4.0, 5.0, 6.0, 7.0],
            },
        ]
    };

    // === GGUF HEADER ===
    writer.write_all(b"GGUF")?;
    writer.write_u32::<LittleEndian>(2)?; // version
    writer.write_u64::<LittleEndian>(tensors.len() as u64)?; // tensor count
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

    // === TENSOR HEADERS ===
    let mut offset_positions = Vec::new();
    for tensor in &tensors {
        writer.write_u64::<LittleEndian>(tensor.name.len() as u64)?;
        writer.write_all(tensor.name.as_bytes())?;
        writer.write_u32::<LittleEndian>(tensor.type_id)?;
        writer.write_u32::<LittleEndian>(tensor.dims.len() as u32)?;
        for d in &tensor.dims {
            writer.write_u64::<LittleEndian>(*d)?;
        }
        offset_positions.push(writer.seek(SeekFrom::Current(0))?);
        writer.write_u64::<LittleEndian>(0)?; // offset placeholder
    }

    // === TENSOR DATA + BACKPATCH ===
    for (i, tensor) in tensors.iter().enumerate() {
        let data_offset = writer.seek(SeekFrom::Current(0))?;

        for v in &tensor.values {
            writer.write_f32::<LittleEndian>(*v)?;
        }

        let current_pos = writer.seek(SeekFrom::Current(0))?;
        writer.seek(SeekFrom::Start(offset_positions[i]))?;
        writer.write_u64::<LittleEndian>(data_offset)?;
        writer.seek(SeekFrom::Start(current_pos))?;
    }

    writer.flush()?;
    Ok(())
}