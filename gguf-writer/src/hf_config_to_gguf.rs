use std::fs::File;
use std::io::{self, BufReader};
use std::path::Path;
use std::collections::VecDeque;

use serde_json::Value;
use gguf_core::types::GGUFValue;

pub fn convert_config_to_metadata<P: AsRef<Path>>(config_path: P) -> io::Result<Vec<(String, GGUFValue)>> {
    let file = File::open(config_path)?;
    let reader = BufReader::new(file);
    let json: Value = serde_json::from_reader(reader)?;

    let mut out = VecDeque::new();

    // Safe promote helper
    let mut promote = |key: &str, convert: fn(&Value) -> Option<GGUFValue>| {
        if let Some(v) = json.get(key).and_then(convert) {
            out.push_back((key.to_string(), v));
        }
    };

    // === Architecture ===
    promote("hidden_size", to_u64);
    promote("intermediate_size", to_u64);
    promote("num_attention_heads", to_u64);
    promote("num_hidden_layers", to_u64);
    promote("layer_norm_eps", to_f64);
    promote("tie_word_embeddings", to_bool);
    promote("kv_cache", to_bool);
    promote("rotary_dim", to_u64);

    // === Tokenizer ===
    promote("vocab_size", to_u64);
    promote("pad_token_id", to_u64);
    promote("bos_token_id", to_u64);
    promote("eos_token_id", to_u64);
    promote("unk_token", to_string);
    promote("cls_token", to_string);
    promote("sep_token", to_string);
    promote("mask_token", to_string);
    promote("add_prefix_space", to_bool);

    // === Fine-tuning Context ===
    promote("fine_tuned_from", to_string);
    promote("fine_tune_dataset", to_string);
    promote("training_steps", to_u64);
    promote("learning_rate", to_f64);

    Ok(out.into())
}

fn to_u64(val: &Value) -> Option<GGUFValue> {
    val.as_u64().map(GGUFValue::U64)
}

fn to_f64(val: &Value) -> Option<GGUFValue> {
    val.as_f64().map(GGUFValue::F64)
}

fn to_bool(val: &Value) -> Option<GGUFValue> {
    val.as_bool().map(GGUFValue::Bool)
}

fn to_string(val: &Value) -> Option<GGUFValue> {
    val.as_str().map(|s| GGUFValue::String(s.to_string()))
}
