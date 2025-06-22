use serde::Deserialize;

/// Enum for all core GGUF metadata scalar value types
#[derive(Debug, Clone, PartialEq)]
pub enum GGUFValue {
    String(String),
    Bool(bool),
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    StringArray(Vec<String>),
    Binary(Vec<u8>),
    Unknown(u8), // fallback
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GGUFValueType {
    String,
    Array,
    Bool,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F32,
    F64,
    StringArray,
    Binary,
    Unknown(u8), // required for fallback handling
}

impl GGUFValueType {
    pub fn from_u8(n: u8) -> Self {
        match n {
            1 => GGUFValueType::String,
            2 => GGUFValueType::Array,
            3 => GGUFValueType::U8,
            4 => GGUFValueType::I8,
            5 => GGUFValueType::U16,
            6 => GGUFValueType::I16,
            7 => GGUFValueType::U32,
            8 => GGUFValueType::I32,
            9 => GGUFValueType::U64,
            10 => GGUFValueType::Bool,
            11 => GGUFValueType::I64,
            12 => GGUFValueType::F64,
            13 => GGUFValueType::F32,
            14 => GGUFValueType::StringArray,
            15 => GGUFValueType::Binary,
            _ => GGUFValueType::Unknown(n),
        }
    }

    pub fn to_u8(self) -> u8 {
        match self {
            GGUFValueType::String => 1,
            GGUFValueType::Array => 2,
            GGUFValueType::U8 => 3,
            GGUFValueType::I8 => 4,
            GGUFValueType::U16 => 5,
            GGUFValueType::I16 => 6,
            GGUFValueType::U32 => 7,
            GGUFValueType::I32 => 8,
            GGUFValueType::U64 => 9,
            GGUFValueType::Bool => 10,
            GGUFValueType::I64 => 11,
            GGUFValueType::F64 => 12,
            GGUFValueType::F32 => 13,
            GGUFValueType::StringArray => 14,
            GGUFValueType::Binary => 15,
            GGUFValueType::Unknown(n) => n,
        }
    }
}


/// Minimal tensor definition for writing (JSON-based)
#[derive(Debug, Deserialize, Clone)]
pub struct TensorDef {
    pub name: String,
    #[serde(rename = "type")]
    pub type_id: u32,
    pub dims: Vec<u64>,
    pub values: Vec<f32>,
}

/// Optional tensor representation used by readers
#[derive(Debug, Clone)]
pub struct GGUFTensor {
    pub name: String,
    pub type_id: u32,
    pub dims: Vec<u64>,
    pub offset: u64,
    pub values: Vec<u8>,
}
