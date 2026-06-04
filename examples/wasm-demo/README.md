# WASM Demo

This example compiles `reed-solomon-simd` to WebAssembly and loads it in a browser page to demonstrate encoding and decoding.

## Prerequisites

- [Rust](https://rustup.rs/) (stable)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)

## Build

```bash
cd examples/wasm-demo
wasm-pack build --target web --out-dir pkg
```

## Run

Serve the `examples/wasm-demo` directory over HTTP (browsers block `file://` for ES modules):

```bash
cd examples/wasm-demo
python3 -m http.server 8080
# Open http://localhost:8080/index.html
```

The page provides configurable inputs for:

- **Originals** — number of original shards (default 512)
- **Recoveries** — number of recovery shards (default 102)
- **Shard bytes** — size of each shard in bytes, must be even and ≥ 2 (default 1024)
- **Lost** — how many original shards to randomly drop before decoding (default 2)

Click **Run** to encode, simulate loss, decode, and verify. If `lost` exceeds the number of recovery shards, decoding will fail with a "not enough shards" error.
