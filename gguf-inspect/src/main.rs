use std::env;
use std::fs;
use std::io;

use gguf_core::reader::read_gguf_file;
use gguf_core::types::GGUFValue;

fn main() -> io::Result<()> {
    let path = env::args().nth(1).expect("Usage: gguf-inspect <file.gguf>");
    println!("magic: GGUF");

    let (metadata, tensors) = read_gguf_file(&path)?;
    let file_size = fs::metadata(&path)?.len();

    // Print general metadata
    let version = metadata
        .get("gguf_version")
        .and_then(|v| match v {
            GGUFValue::String(s) => Some(s.clone()),
            GGUFValue::U64(n) => Some(n.to_string()),
            _ => None,
        })
        .unwrap_or_else(|| "unknown".to_string());

    println!("version: {}", version);
    println!("tensor count: {}", tensors.len());
    println!("metadata count: {}", metadata.len());

    for (i, (key, value)) in metadata.iter().enumerate() {
        println!("  {}. {} => {:?}", i, key, value);
    }

    // Count tensor types
    use std::collections::HashMap;
    let mut type_counts: HashMap<u32, usize> = HashMap::new();
    let mut top_tensors = Vec::new();
    let mut total_memory: u64 = 0;

    for t in &tensors {
        let num_elements: u64 = t.dims.iter().product();
        let per_elem_size = match t.type_id {
            0 => 4,     // f32
            100 => 1,   // Q4_0 (approx)
            101 => 2,   // Q5_1 (approx)
            _ => 4,     // fallback
        };

        let tensor_size = num_elements * per_elem_size;
        total_memory += tensor_size;

        *type_counts.entry(t.type_id).or_insert(0) += 1;

        // Save top tensor info for summary
        top_tensors.push((tensor_size, t.name.clone(), t.dims.clone(), t.type_id));
    }

    top_tensors.sort_by(|a, b| b.0.cmp(&a.0));
    let top_display = top_tensors.iter().take(6);

    println!("\n--- Tensor Table Overview ---");

    for (type_id, count) in &type_counts {
        println!("Types: {} ({})", type_id, count);
    }

    println!("\nTop tensors by size:");
    for (_, name, dims, type_id) in top_display {
        println!("  - {:<40} | {:?} | type_id: {}", name, dims, type_id);
    }

    println!(
        "\nFile Size:         {:.2} GB",
        file_size as f64 / 1e9
    );
    println!(
        "Total tensor memory: {:.2} GB (approx)",
        total_memory as f64 / 1e9
    );

    Ok(())
}
