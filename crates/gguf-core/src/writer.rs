use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use byteorder::{LittleEndian, WriteBytesExt};

use crate::types::{GGUFValue, GGUFTensor};

/// Write a GGUF file with metadata and tensors
pub fn write_gguf_file<P: AsRef<std::path::Path>>(
    path: P,
    metadata: &BTreeMap<String, GGUFValue>,
    tensors: &[GGUFTensor],
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // === HEADER ===
    writer.write_all(b"GGUF")?;
    writer.write_u32::<LittleEndian>(2)?; // version
    writer.write_u64::<LittleEndian>(tensors.len() as u64)?;
    writer.write_u64::<LittleEndian>(metadata.len() as u64)?;

    // === METADATA ===
    for (key, value) in metadata {
        writer.write_u64::<LittleEndian>(key.len() as u64)?;
        writer.write_all(key.as_bytes())?;

        match value {
            GGUFValue::String(s) => {
                writer.write_u8(1)?; // type
                writer.write_u64::<LittleEndian>(s.len() as u64)?;
                writer.write_all(s.as_bytes())?;
            }
            GGUFValue::Bool(b) => {
                writer.write_u8(10)?;
                writer.write_u8(if *b { 1 } else { 0 })?;
            }
            GGUFValue::U64(v) => {
                writer.write_u8(9)?;
                writer.write_u64::<LittleEndian>(*v)?;
            }
            GGUFValue::I64(v) => {
                writer.write_u8(11)?;
                writer.write_i64::<LittleEndian>(*v)?;
            }
            GGUFValue::F64(v) => {
                writer.write_u8(12)?;
                writer.write_f64::<LittleEndian>(*v)?;
            }
            GGUFValue::F32(v) => {
                writer.write_u8(14)?;
                writer.write_f32::<LittleEndian>(*v)?;
            }
            GGUFValue::U8(v) => {
                writer.write_u8(2)?;
                writer.write_u8(*v)?;
            }
            GGUFValue::I8(v) => {
                writer.write_u8(3)?;
                writer.write_i8(*v)?;
            }
            GGUFValue::U16(v) => {
                writer.write_u8(4)?;
                writer.write_u16::<LittleEndian>(*v)?;
            }
            GGUFValue::I16(v) => {
                writer.write_u8(5)?;
                writer.write_i16::<LittleEndian>(*v)?;
            }
            GGUFValue::U32(v) => {
                writer.write_u8(6)?;
                writer.write_u32::<LittleEndian>(*v)?;
            }
            GGUFValue::I32(v) => {
                writer.write_u8(7)?;
                writer.write_i32::<LittleEndian>(*v)?;
            }
            GGUFValue::StringArray(arr) => {
                writer.write_u8(13)?;
                writer.write_u64::<LittleEndian>(arr.len() as u64)?;
                for s in arr {
                    writer.write_u64::<LittleEndian>(s.len() as u64)?;
                    writer.write_all(s.as_bytes())?;
                }
            }
            GGUFValue::Binary(data) => {
                writer.write_u8(15)?;
                writer.write_u64::<LittleEndian>(data.len() as u64)?;
                writer.write_all(data)?;
            }
            GGUFValue::Unknown(type_id) => {
                // For unknown types, we just write the type_id without any data
                writer.write_u8(*type_id)?;
            }
        }
    }

    // === TENSOR HEADERS ===
    let mut offset_positions = Vec::new();
    for tensor in tensors {
        writer.write_u64::<LittleEndian>(tensor.name.len() as u64)?;
        writer.write_all(tensor.name.as_bytes())?;
        writer.write_u32::<LittleEndian>(tensor.type_id)?;
        writer.write_u32::<LittleEndian>(tensor.dims.len() as u32)?;
        for &dim in &tensor.dims {
            writer.write_u64::<LittleEndian>(dim)?;
        }

        // Save the position where the offset lives so we can backpatch later
        offset_positions.push(writer.seek(SeekFrom::Current(0))?);
        writer.write_u64::<LittleEndian>(0)?; // offset placeholder
    }

    // === TENSOR DATA & PATCH OFFSETS ===
    for (i, tensor) in tensors.iter().enumerate() {
        let data_offset = writer.seek(SeekFrom::Current(0))?;

        for v in &tensor.values {
            writer.write_f32::<LittleEndian>(*v)?;
        }

        // backpatch
        let return_pos = writer.seek(SeekFrom::Current(0))?;
        writer.seek(SeekFrom::Start(offset_positions[i]))?;
        writer.write_u64::<LittleEndian>(data_offset)?;
        writer.seek(SeekFrom::Start(return_pos))?;
    }

    writer.flush()?;
    Ok(())
}
