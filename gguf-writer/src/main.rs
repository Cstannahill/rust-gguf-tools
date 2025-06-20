use clap::Parser;
use log::info;
mod write_gguf;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Path to metadata JSON
    #[arg(short, long)]
    metadata: String,

    /// Output GGUF file path
    #[arg(short, long)]
    output: String,
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    info!("Metadata path: {}", cli.metadata);
    info!("Output path: {}", cli.output);

    match write_gguf::write_gguf_with_tensor(&cli.metadata, &cli.output){
        Ok(_) => println!("✅ GGUF file written to: {}", cli.output),
        Err(e) => eprintln!("❌ Error writing GGUF file: {e}"),
    }
}
