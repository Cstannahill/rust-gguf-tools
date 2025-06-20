use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use byteorder::{LittleEndian, ReadBytesExt};

fn main() -> io::Result<()> {
    let path = std::env::args().nth(1).expect("Usage: gguf-inspect <file.gguf>");
    let file = File::open(&path)?;
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        panic!("Invalid GGUF file: magic header not found");
    }

    let version = reader.read_u32::<LittleEndian>()?;
    let tensor_count = reader.read_u64::<LittleEndian>()?;
    let metadata_count = reader.read_u64::<LittleEndian>()?;

    println!("magic: GGUF");
    println!("version: {}", version);
    println!("tensor count: {}", tensor_count);
    println!("metadata count: {}", metadata_count);

    for i in 0..metadata_count {
        let key_len = reader.read_u64::<LittleEndian>()?;
        let mut key_bytes = vec![0u8; key_len as usize];
        reader.read_exact(&mut key_bytes)?;
        let key = String::from_utf8_lossy(&key_bytes);

        let value_type = reader.read_u8()?;
        match value_type {
            1 => {
                let str_len = reader.read_u64::<LittleEndian>()?;
                let mut str_bytes = vec![0u8; str_len as usize];
                reader.read_exact(&mut str_bytes)?;
                let value = String::from_utf8_lossy(&str_bytes);
                println!("  {i}. {key} => \"{value}\"");
            }
            9 => {
                let val = reader.read_u64::<LittleEndian>()?;
                println!("  {i}. {key} => {}", val);
            }
            10 => {
                let val = reader.read_u8()? != 0;
                println!("  {i}. {key} => {}", val);
            }
            11 => {
                let val = reader.read_i64::<LittleEndian>()?;
                println!("  {i}. {key} => {}", val);
            }
            12 => {
                let val = reader.read_f64::<LittleEndian>()?;
                println!("  {i}. {key} => {}", val);
            }
            _ => {
                println!("  {i}. {key} => (unhandled type {value_type})");
            }
        }
    }

    println!("\n--- Tensor Table ---");

    for _ in 0..tensor_count {
        let name_len = reader.read_u64::<LittleEndian>()?;
        let mut name_bytes = vec![0u8; name_len as usize];
        reader.read_exact(&mut name_bytes)?;
        let name = String::from_utf8_lossy(&name_bytes);

        let type_id = reader.read_u32::<LittleEndian>()?;
        let ndim = reader.read_u32::<LittleEndian>()?;
        let mut dims = Vec::with_capacity(ndim as usize);
        for _ in 0..ndim {
            dims.push(reader.read_u64::<LittleEndian>()?);
        }

        let offset = reader.read_u64::<LittleEndian>()?;

        println!("tensor: {name}");
        println!("  type: {type_id}");
        println!("  dims: {:?}", dims);
        println!("  offset: {offset}");

        // Go read the tensor data
        let mut file = File::open(&path)?;
        file.seek(SeekFrom::Start(offset))?;       
        let mut values = Vec::new();
        for _ in 0..dims.iter().product::<u64>() {
            values.push(file.read_f32::<LittleEndian>()?);
        }
        println!("  values: {:?}", values);
    }

    Ok(())
}