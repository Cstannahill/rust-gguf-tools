use std::io::{self, Cursor, Read};
use byteorder::{LittleEndian, ReadBytesExt};

#[derive(Debug)]
pub enum DecodeError {
    InvalidScale,
    InvalidBlock,
    UnexpectedEOF,
    Io(io::Error),
}

impl From<io::Error> for DecodeError {
    fn from(e: io::Error) -> Self {
        DecodeError::Io(e)
    }
}
// Add this to gguf-core/src/decoder.rs

pub fn try_decode_f32(bytes: &[u8], dims: &[u64]) -> Result<Vec<f32>, DecodeError> {
    let expected_len = dims.iter().product::<u64>() as usize;
    if bytes.len() % 4 != 0 {
        return Err(DecodeError::InvalidBlock);
    }
    if bytes.len() < expected_len * 4 {
        return Err(DecodeError::UnexpectedEOF);
    }
    let mut floats = Vec::with_capacity(expected_len);
    for chunk in bytes.chunks_exact(4) {
        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        floats.push(val);
    }
    Ok(floats)
}
pub fn try_decode_q4_0(bytes: &[u8], dims: &[u64]) -> Result<Vec<f32>, DecodeError> {
    let expected_len = dims.iter().product::<u64>() as usize;
    let mut cursor = Cursor::new(bytes);
    let mut decoded = Vec::with_capacity(expected_len);

    while decoded.len() < expected_len {
        if (cursor.position() as usize) + 8 > bytes.len() {
            return Err(DecodeError::UnexpectedEOF);
        }

        let scale = cursor.read_f32::<LittleEndian>()?;
        let zero = cursor.read_f32::<LittleEndian>()?;

        if !scale.is_finite() || scale == 0.0 {
            return Err(DecodeError::InvalidScale);
        }

        // Read up to 16 packed bytes (max 32 values)
        let remaining = bytes.len() - cursor.position() as usize;
        let packed_len = remaining.min(16);
        let mut packed = vec![0u8; packed_len];
        cursor.read_exact(&mut packed)?;

        for byte in packed {
            if decoded.len() >= expected_len {
                break;
            }
            let lo = byte & 0x0F;
            decoded.push(scale * lo as f32 + zero);

            if decoded.len() >= expected_len {
                break;
            }
            let hi = (byte >> 4) & 0x0F;
            decoded.push(scale * hi as f32 + zero);
        }
    }

    Ok(decoded)
}

pub fn try_decode_q5_1(bytes: &[u8], dims: &[u64]) -> Result<Vec<f32>, DecodeError> {
    let expected_len = dims.iter().product::<u64>() as usize;
    let mut cursor = Cursor::new(bytes);
    let mut decoded = Vec::with_capacity(expected_len);

    while decoded.len() < expected_len {
        if (cursor.position() as usize) + 8 > bytes.len() {
            return Err(DecodeError::UnexpectedEOF);
        }

        let scale = cursor.read_f32::<LittleEndian>()?;
        let zero = cursor.read_f32::<LittleEndian>()?;

        if !scale.is_finite() || scale == 0.0 {
            return Err(DecodeError::InvalidScale);
        }

        // Read up to 20 bytes of 5-bit values (max 32 values)
        let remaining = bytes.len() - cursor.position() as usize;
        let packed_len = remaining.min(20);
        let mut packed = vec![0u8; packed_len];
        cursor.read_exact(&mut packed)?;

        let mut acc: u64 = 0;
        let mut bits = 0;
        let mut values = Vec::with_capacity(32);

        for byte in packed {
            acc |= (byte as u64) << bits;
            bits += 8;

            while bits >= 5 {
                let val = (acc & 0x1F) as u8;
                values.push(val);
                acc >>= 5;
                bits -= 5;

                if values.len() >= 32 {
                    break;
                }
            }
        }

        for val in values {
            decoded.push(scale * val as f32 + zero);
            if decoded.len() >= expected_len {
                break;
            }
        }
    }

    Ok(decoded)
}
