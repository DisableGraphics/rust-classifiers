use std::{env::args, error::Error, fs::File, hash::{DefaultHasher, Hash, Hasher}};
use tokenizers::tokenizer::{Tokenizer, Encoding};
use csv::Writer;

// Hash function to map token -> index in fixed-size vector
fn hash_token(token: &str, vector_size: usize) -> usize {
    let mut hasher = DefaultHasher::new();
    token.hash(&mut hasher);
    (hasher.finish() as usize) % vector_size
}

fn main() -> Result<(), Box<dyn Error>> {
    // 1️⃣ Load the tokenizer
    let tokenizer = Tokenizer::from_file("tokenizer.json").unwrap();

    // 2️⃣ Read input CSV (first arg = input file)
    let input_file = args().nth(1).expect("Provide input CSV file as first argument");
    let output_file = args().nth(2).expect("Provide output CSV file as second argument");
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(input_file)?;

    let mut headlines: Vec<(String, u8)> = Vec::new();
    for result in rdr.records() {
        let record = result?;
        let text = record.get(0).unwrap_or("").to_string();
        let label: u8 = record.get(1).unwrap_or("0").parse().unwrap_or(0);
        headlines.push((text, label));
    }

    // 3️⃣ Prepare CSV writer
    let file = File::create(output_file)?;
    let mut wtr = Writer::from_writer(file);

    // Feature hashing vector size
    let vector_size = args()
		.nth(3)
		.and_then(|s| Some(s.parse())).unwrap_or(Ok(512)).unwrap_or(512);

    // 4️⃣ Write header row (f0, f1, ... fN-1, label)
    let mut header: Vec<String> = (0..vector_size).map(|i| format!("f{}", i)).collect();
    header.push("label".to_string());
    wtr.write_record(&header)?;

    // 5️⃣ Process each headline
    for (headline, label) in &headlines {
        let encoding: Encoding = tokenizer.encode(headline as &str, true).unwrap();
        let mut vector = vec![0u8; vector_size];

        // Hash each token into vector
        for id in encoding.get_ids() {
            let token = tokenizer.id_to_token(*id).unwrap_or_default();
            let idx = hash_token(&token, vector_size);
            vector[idx] = 1; // binary presence
        }

        // Convert vector + label to strings
        let mut row: Vec<String> = vector.iter().map(|v| v.to_string()).collect();
        row.push(label.to_string());
        wtr.write_record(&row)?;
    }

    wtr.flush()?;
    println!("✅ Feature-hashed CSV saved");

    Ok(())
}
