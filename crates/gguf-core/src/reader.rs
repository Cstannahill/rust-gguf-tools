use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use byteorder::{LittleEndian, ReadBytesExt};

use crate::types::{GGUFValue, GGUFValueType, GGUFTensor};

/// Reads a GGUF file and returns metadata and tensors
pub fn read_gguf_file<P: AsRef<std::path::Path>>(
    path: P,
) -> io::Result<(BTreeMap<String, GGUFValue>, Vec<GGUFTensor>)> {
    let mut file = File::open(&path)?;
    let mut reader = BufReader::new(&file);

    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Missing GGUF header"));
    }

    let _version = reader.read_u32::<LittleEndian>()?;
    let tensor_count = reader.read_u64::<LittleEndian>()?;
    let metadata_count = reader.read_u64::<LittleEndian>()?;

    let mut metadata = BTreeMap::new();

    for _ in 0..metadata_count {
        let key_len = reader.read_u64::<LittleEndian>()?;
        let mut key_bytes = vec![0u8; key_len as usize];
        reader.read_exact(&mut key_bytes)?;
        let key = String::from_utf8_lossy(&key_bytes).to_string();

        let type_byte = reader.read_u8()?;
        let parsed = match GGUFValueType::from_u8(type_byte) {
            GGUFValueType::String => {
                let len = reader.read_u64::<LittleEndian>()?;
                let mut buf = vec![0u8; len as usize];
                reader.read_exact(&mut buf)?;
                Some(GGUFValue::String(String::from_utf8_lossy(&buf).to_string()))
            }
            GGUFValueType::Bool => Some(GGUFValue::Bool(reader.read_u8()? != 0)),
            GGUFValueType::U64 => Some(GGUFValue::U64(reader.read_u64::<LittleEndian>()?)),
            GGUFValueType::I64 => Some(GGUFValue::I64(reader.read_i64::<LittleEndian>()?)),
            GGUFValueType::F64 => Some(GGUFValue::F64(reader.read_f64::<LittleEndian>()?)),
            GGUFValueType::F32 => Some(GGUFValue::F32(reader.read_f32::<LittleEndian>()?)),
            GGUFValueType::U8 => Some(GGUFValue::U8(reader.read_u8()?)),
            GGUFValueType::I8 => Some(GGUFValue::I8(reader.read_i8()?)),
            GGUFValueType::U16 => Some(GGUFValue::U16(reader.read_u16::<LittleEndian>()?)),
            GGUFValueType::I16 => Some(GGUFValue::I16(reader.read_i16::<LittleEndian>()?)),
            GGUFValueType::U32 => Some(GGUFValue::U32(reader.read_u32::<LittleEndian>()?)),
            GGUFValueType::I32 => Some(GGUFValue::I32(reader.read_i32::<LittleEndian>()?)),
            GGUFValueType::StringArray => {
                let count = reader.read_u64::<LittleEndian>()?;
                let mut items = Vec::with_capacity(count as usize);
                for _ in 0..count {
                    let len = reader.read_u64::<LittleEndian>()?;
                    let mut buf = vec![0u8; len as usize];
                    reader.read_exact(&mut buf)?;
                    items.push(String::from_utf8_lossy(&buf).to_string());
                }
                Some(GGUFValue::StringArray(items))
            }
            GGUFValueType::Binary => {
                let len = reader.read_u64::<LittleEndian>()?;
                let mut buf = vec![0u8; len as usize];
                reader.read_exact(&mut buf)?;
                Some(GGUFValue::Binary(buf))
            }
            GGUFValueType::Array => {
                eprintln!("⚠️ Skipping unsupported metadata type Array for key: {key}");
                None
            }
            GGUFValueType::Unknown(t) => {
                eprintln!("⚠️ Skipping unsupported metadata type {t} for key: {key}");
                None
            }
        };

        if let Some(val) = parsed {
            metadata.insert(key, val);
        }
    }

    // === TENSOR HEADERS ===
    let mut tensors = Vec::new();
    let mut tensor_headers = Vec::new();
    for _ in 0..tensor_count {
        let name_len = reader.read_u64::<LittleEndian>()?;
        let mut name_bytes = vec![0u8; name_len as usize];
        reader.read_exact(&mut name_bytes)?;
        let name = String::from_utf8_lossy(&name_bytes).to_string();

        let type_id = reader.read_u32::<LittleEndian>()?;
        let ndim = reader.read_u32::<LittleEndian>()?;
        let mut dims = Vec::with_capacity(ndim as usize);
        for _ in 0..ndim {
            dims.push(reader.read_u64::<LittleEndian>()?);
        }

        let offset = reader.read_u64::<LittleEndian>()?;

        tensor_headers.push((name, type_id, dims, offset));
    }

    // === TENSOR BLOBS ===
    let file_len = file.metadata()?.len();
    for i in 0..tensor_headers.len() {
        let (name, type_id, dims, offset) = &tensor_headers[i];
        let end = if i + 1 < tensor_headers.len() {
            tensor_headers[i + 1].3
        } else {
            file_len
        };

        let size = end - offset;
        file.seek(SeekFrom::Start(*offset))?;
        let mut values = vec![0u8; size as usize];
        file.read_exact(&mut values)?;

        tensors.push(GGUFTensor {
            name: name.clone(),
            type_id: *type_id,
            dims: dims.clone(),
            offset: *offset,
            values,
        });
    }

    Ok((metadata, tensors))
}
