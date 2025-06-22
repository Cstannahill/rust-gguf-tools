use std::env;
use std::io;
use gguf_core::reader::read_gguf_file;
use gguf_core::types::GGUFValue;

fn main() -> io::Result<()> {
    let path = env::args().nth(1).expect("Usage: gguf-inspect <file.gguf>");

    let (metadata, tensors) = read_gguf_file(&path)?;

    println!("magic: GGUF");
    println!("version: 2"); // currently hardcoded; can extract if needed
    println!("tensor count: {}", tensors.len());
    println!("metadata count: {}", metadata.len());

    for (i, (key, value)) in metadata.iter().enumerate() {
        match value {
            GGUFValue::String(s) => println!("  {i}. {key} => \"{s}\""),
            GGUFValue::Bool(b) => println!("  {i}. {key} => {}", b),
            GGUFValue::U64(v) => println!("  {i}. {key} => {}", v),
            GGUFValue::I64(v) => println!("  {i}. {key} => {}", v),
            GGUFValue::F64(v) => println!("  {i}. {key} => {}", v),
            GGUFValue::F32(v) => println!("  {i}. {key} => {}", v),
            GGUFValue::U8(v) => println!("  {i}. {key} => {}", v),
            GGUFValue::I8(v) => println!("  {i}. {key} => {}", v),
            GGUFValue::U16(v) => println!("  {i}. {key} => {}", v),
            GGUFValue::I16(v) => println!("  {i}. {key} => {}", v),
            GGUFValue::U32(v) => println!("  {i}. {key} => {}", v),
            GGUFValue::I32(v) => println!("  {i}. {key} => {}", v),
            GGUFValue::StringArray(arr) => println!("  {i}. {key} => {:?}", arr),
            GGUFValue::Binary(data) => println!("  {i}. {key} => <{} bytes binary>", data.len()),
            GGUFValue::Unknown(t) => println!("  {i}. {key} => (unknown type {t})"),
        }
    }

    println!("\n--- Tensor Table ---");

    for tensor in tensors {
        println!("tensor: {}", tensor.name);
        println!("  type: {}", tensor.type_id);
        println!("  dims: {:?}", tensor.dims);
        println!("  offset: {}", tensor.offset);
        println!("  values: {:?}", tensor.values);
    }

    Ok(())
}
