#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
struct Instant {
    start: f64,
}

#[cfg(target_arch = "wasm32")]
impl Instant {
    fn now() -> Self {
        Instant {
            start: js_sys::Date::now(),
        }
    }

    fn elapsed(&self) -> std::time::Duration {
        let elapsed_ms = js_sys::Date::now() - self.start;
        std::time::Duration::from_secs_f64(elapsed_ms as f64 / 1000.0)
    }
}

// Route print!/println!/eprint!/eprintln! through process.stdout/stderr on wasm32.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
extern "C" {
    #[wasm_bindgen::prelude::wasm_bindgen(js_namespace = globalThis, js_name = "process.stdout.write")]
    fn stdout_write(s: &str);

    #[wasm_bindgen::prelude::wasm_bindgen(js_namespace = globalThis, js_name = "process.stderr.write")]
    fn stderr_write(s: &str);
}

#[cfg(target_arch = "wasm32")]
macro_rules! print {
    ($($arg:tt)*) => {{
        stdout_write(&format!($($arg)*));
    }};
}

#[cfg(target_arch = "wasm32")]
macro_rules! println {
    ($($arg:tt)*) => {{
        stdout_write(&format!($($arg)*));
        stdout_write("\n");
    }};
}

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

// ======================================================================
// CONST

const SHARD_BYTES: usize = 1024;

// ======================================================================
// MAIN

fn main() {
    #[cfg(debug_assertions)]
    {
        eprintln!("Warning: Running in debug mode! Please run like this instead: cargo run --release --example quick-comparison");
    }

    println!("                           µs (init)   µs (encode)   µs (decode)");
    println!("                           ---------   -----------   -----------");

    for count in [8, 16, 32, 64, 128, 256, 512, 1024, 4 * 1024, 32 * 1024] {
        println!("\n{}:{} ({} kiB)", count, count, SHARD_BYTES / 1024);
        test_reed_solomon_simd(count);
        #[cfg(not(target_arch = "wasm32"))]
        test_reed_solomon_16(count);
        #[cfg(not(target_arch = "wasm32"))]
        test_reed_solomon_novelpoly(count);
        if count <= 128 {
            #[cfg(not(target_arch = "wasm32"))]
            test_reed_solomon_erasure_8(count);
            #[cfg(not(target_arch = "wasm32"))]
            test_leopard_codec(count);
        }
        if count <= 512 {
            #[cfg(not(target_arch = "wasm32"))]
            test_reed_solomon_erasure_16(count);
        }
    }
}

// ======================================================================
// reed-solomon-simd

fn test_reed_solomon_simd(count: usize) {
    // INIT

    let start = Instant::now();

    // This table is only used in decoding.
    reed_solomon_simd::engine::tables::get_log_walsh();

    // This initializes the remaining needed tables.
    reed_solomon_simd::engine::DefaultEngine::new();

    let elapsed = start.elapsed();
    print!("> reed-solomon-simd        {:9}", elapsed.as_micros());

    // CREATE ORIGINAL

    let mut original = vec![vec![0u8; SHARD_BYTES]; count];
    let mut rng = ChaCha8Rng::from_seed([0; 32]);
    for original in &mut original {
        rng.fill::<[u8]>(original);
    }

    // ENCODE

    let start = Instant::now();
    let recovery = reed_solomon_simd::encode(count, count, &original).unwrap();
    let elapsed = start.elapsed();
    print!("{:14}", elapsed.as_micros());

    // PREPARE DECODE

    let decoder_recovery: Vec<_> = recovery.iter().enumerate().collect();

    // DECODE

    let start = Instant::now();
    let restored = reed_solomon_simd::decode(count, count, [(0, ""); 0], decoder_recovery).unwrap();
    let elapsed = start.elapsed();
    println!("{:14}", elapsed.as_micros());

    // CHECK

    for i in 0..count {
        assert_eq!(restored[&i], original[i]);
    }
}

// ======================================================================
// reed-solomon-16

#[cfg(not(target_arch = "wasm32"))]
fn test_reed_solomon_16(count: usize) {
    // INIT

    let start = Instant::now();
    // This initializes all the needed tables.
    reed_solomon_16::engine::DefaultEngine::new();
    let elapsed = start.elapsed();
    print!("> reed-solomon-16          {:9}", elapsed.as_micros());

    // CREATE ORIGINAL

    let mut original = vec![vec![0u8; SHARD_BYTES]; count];
    let mut rng = ChaCha8Rng::from_seed([0; 32]);
    for original in &mut original {
        rng.fill::<[u8]>(original);
    }

    // ENCODE

    let start = Instant::now();
    let recovery = reed_solomon_16::encode(count, count, &original).unwrap();
    let elapsed = start.elapsed();
    print!("{:14}", elapsed.as_micros());

    // PREPARE DECODE

    let decoder_recovery: Vec<_> = recovery.iter().enumerate().collect();

    // DECODE

    let start = Instant::now();
    let restored = reed_solomon_16::decode(count, count, [(0, ""); 0], decoder_recovery).unwrap();
    let elapsed = start.elapsed();
    println!("{:14}", elapsed.as_micros());

    // CHECK

    for i in 0..count {
        assert_eq!(restored[&i], original[i]);
    }
}

// ======================================================================
// reed-solomon-erasure

#[cfg(not(target_arch = "wasm32"))]
fn test_reed_solomon_erasure_8(count: usize) {
    // INIT

    let start = Instant::now();
    let r = reed_solomon_erasure::galois_8::ReedSolomon::new(count, count).unwrap();
    let elapsed = start.elapsed();
    print!("> reed-solomon-erasure/8   {:9}", elapsed.as_micros());

    // CREATE ORIGINAL

    let mut original = vec![vec![0u8; SHARD_BYTES]; count];
    let mut rng = ChaCha8Rng::from_seed([0; 32]);
    for shard in &mut original {
        rng.fill::<[u8]>(shard);
    }

    // ENCODE

    let mut recovery = vec![vec![0; SHARD_BYTES]; count];

    let start = Instant::now();
    r.encode_sep(&original, &mut recovery).unwrap();
    let elapsed = start.elapsed();
    print!("{:14}", elapsed.as_micros());

    // PREPARE DECODE

    let mut decoder_shards = Vec::with_capacity(2 * count);
    for _ in 0..count {
        decoder_shards.push(None);
    }
    for i in 0..count {
        decoder_shards.push(Some(recovery[i].clone()));
    }

    // DECODE

    let start = Instant::now();
    r.reconstruct(&mut decoder_shards).unwrap();
    let elapsed = start.elapsed();
    println!("{:14}", elapsed.as_micros());

    // CHECK

    for i in 0..count {
        assert_eq!(decoder_shards[i].as_ref(), Some(&original[i]));
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn test_reed_solomon_erasure_16(count: usize) {
    // INIT

    let start = Instant::now();
    let r = reed_solomon_erasure::galois_16::ReedSolomon::new(count, count).unwrap();
    let elapsed = start.elapsed();
    print!("> reed-solomon-erasure/16  {:9}", elapsed.as_micros());

    // CREATE ORIGINAL

    let mut original = vec![vec![[0u8; 2]; SHARD_BYTES / 2]; count];
    let mut rng = ChaCha8Rng::from_seed([0; 32]);
    for shard in &mut original {
        for element in shard.iter_mut() {
            element[0] = rng.random();
            element[1] = rng.random();
        }
    }

    // ENCODE

    let mut recovery = vec![vec![[0; 2]; SHARD_BYTES / 2]; count];

    let start = Instant::now();
    r.encode_sep(&original, &mut recovery).unwrap();
    let elapsed = start.elapsed();
    print!("{:14}", elapsed.as_micros());

    // PREPARE DECODE

    let mut decoder_shards = Vec::with_capacity(2 * count);
    for _ in 0..count {
        decoder_shards.push(None);
    }
    for i in 0..count {
        decoder_shards.push(Some(recovery[i].clone()));
    }

    // DECODE

    let start = Instant::now();
    r.reconstruct(&mut decoder_shards).unwrap();
    let elapsed = start.elapsed();
    println!("{:14}", elapsed.as_micros());

    // CHECK

    for i in 0..count {
        assert_eq!(decoder_shards[i].as_ref(), Some(&original[i]));
    }
}

// ======================================================================
// reed-solomon-novelpoly

#[cfg(not(target_arch = "wasm32"))]
fn test_reed_solomon_novelpoly(count: usize) {
    // INIT

    let start = Instant::now();
    let r = reed_solomon_novelpoly::CodeParams::derive_parameters(2 * count, count)
        .unwrap()
        .make_encoder();
    let elapsed = start.elapsed();
    print!("> reed-solomon-novelpoly   {:9}", elapsed.as_micros());

    // CREATE ORIGINAL

    let mut original = vec![0u8; count * SHARD_BYTES];
    let mut rng = ChaCha8Rng::from_seed([0; 32]);
    rng.fill::<[u8]>(&mut original);

    // ENCODE

    let start = Instant::now();
    let encoded = r
        .encode::<reed_solomon_novelpoly::WrappedShard>(&original)
        .unwrap();
    let elapsed = start.elapsed();
    print!("{:14}", elapsed.as_micros());

    // PREPARE DECODE

    let mut decoder_shards = Vec::with_capacity(2 * count);
    for _ in 0..count {
        decoder_shards.push(None);
    }
    for i in 0..count {
        decoder_shards.push(Some(encoded[count + i].clone()));
    }

    // DECODE

    let start = Instant::now();
    let reconstructed = r.reconstruct(decoder_shards).unwrap();
    let elapsed = start.elapsed();
    println!("{:14}", elapsed.as_micros());

    // CHECK

    assert_eq!(reconstructed, original);
}

// ======================================================================
// leopard-codec

#[cfg(not(target_arch = "wasm32"))]
fn test_leopard_codec(count: usize) {
    // INIT

    // I don't see a way to ask the library to initialize the tables.
    // Also the `leopard_codec::lut` module is private, so we can't do it directly.

    print!("> leopard-codec                    ?");

    // CREATE ORIGINAL

    let mut original = vec![vec![0u8; SHARD_BYTES]; count];
    let mut rng = ChaCha8Rng::from_seed([0; 32]);
    for shard in &mut original {
        rng.fill::<[u8]>(shard);
    }

    let mut recovery = vec![vec![0u8; SHARD_BYTES]; count];

    let mut all_shards: Vec<&mut Vec<u8>> =
        original.iter_mut().chain(recovery.iter_mut()).collect();

    // ENCODE

    let start = Instant::now();
    leopard_codec::encode(&mut all_shards, count).unwrap();

    let elapsed = start.elapsed();
    print!("{:14}", elapsed.as_micros());

    // PREPARE DECODE

    let mut restored = vec![Vec::<u8>::new(); count];
    let mut restored_and_recovery: Vec<&mut Vec<u8>> =
        restored.iter_mut().chain(recovery.iter_mut()).collect();

    // DECODE

    let start = Instant::now();
    leopard_codec::reconstruct(&mut restored_and_recovery, count).unwrap();

    let elapsed = start.elapsed();
    println!("{:14}", elapsed.as_micros());

    // CHECK
    assert_eq!(restored, original);
}
