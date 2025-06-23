use std::env;
use std::io;

use gguf_core::decoder::{try_decode_q4_0, try_decode_q5_1, try_decode_f32, DecodeError};
use gguf_core::reader::read_gguf_file;
use gguf_core::types::GGUFValue;

fn main() -> io::Result<()> {
    let path = env::args().nth(1).expect("Usage: gguf-validate <file.gguf>");
    println!("🧪 Validating GGUF file: {path}\n");

    let (metadata, tensors) = read_gguf_file(&path)?;

    let format = metadata
        .get("quantization_format")
        .and_then(|v| match v {
            GGUFValue::String(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_else(|| "Unknown".to_string());

    println!("Format: {}", format);
    println!("Tensors: {}\n", tensors.len());

    let mut errors = 0;

    for tensor in tensors {
        println!("→ tensor '{}':", tensor.name);
        println!("   type_id: {}", tensor.type_id);
        println!("   dims: {:?}", tensor.dims);

        let result = match tensor.type_id {
            0 => try_decode_f32(&tensor.values, &tensor.dims),
            100 => try_decode_q4_0(&tensor.values, &tensor.dims),
            101 => try_decode_q5_1(&tensor.values, &tensor.dims),
            _ => {
                println!("   ⚠ Unsupported tensor type — skipping validation.\n");
                continue;
            }
        };

        match result {
            Ok(decoded) => {
                println!("   ✅ Decoded successfully ({} floats)\n", decoded.len());
            }
            Err(e) => {
                println!(
                    "   ❌ Decode error: {}\n",
                    match e {
                        DecodeError::InvalidScale => "InvalidScale".to_string(),
                        DecodeError::InvalidBlock => "InvalidBlock".to_string(),
                        DecodeError::UnexpectedEOF => "UnexpectedEOF".to_string(),
                        DecodeError::Io(ioe) => format!("Io({})", ioe),
                    }
                );
                errors += 1;
            }
        }
    }

    println!("========================================");
    if errors == 0 {
        println!("✅ Validation passed!");
        Ok(())
    } else {
        println!("❌ Validation failed: {} issue(s) found.", errors);
        std::process::exit(1);
    }
}
