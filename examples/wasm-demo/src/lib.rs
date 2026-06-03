use js_sys::Uint8Array;
use reed_solomon_simd::{ReedSolomonDecoder, ReedSolomonEncoder};
use wasm_bindgen::prelude::*;

/// Encode original shards and return recovery shards.
#[wasm_bindgen]
pub fn encode(original_shards: &js_sys::Array, recovery_count: usize) -> Result<js_sys::Array, JsValue> {
    let original_count = original_shards.length() as usize;
    if original_count == 0 {
        return Err(JsValue::from_str("original_shards must not be empty"));
    }

    let first = original_shards.get(0);
    let shard_bytes = if let Some(arr) = first.dyn_ref::<Uint8Array>() {
        arr.length() as usize
    } else {
        return Err(JsValue::from_str("elements of original_shards must be Uint8Array"));
    };
    if shard_bytes == 0 {
        return Err(JsValue::from_str("shard size must be greater than zero"));
    }

    let mut encoder = ReedSolomonEncoder::new(original_count, recovery_count, shard_bytes)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    for i in 0..original_count {
        let val = original_shards.get(i as u32);
        let arr = val.dyn_ref::<Uint8Array>()
            .ok_or_else(|| JsValue::from_str(&format!("shard[{}] is not a Uint8Array", i)))?;
        encoder.add_original_shard(arr.to_vec())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    let result = encoder.encode()
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let recovery = js_sys::Array::new();
    for (i, shard) in result.recovery_iter().enumerate() {
        recovery.set(i as u32, Uint8Array::from(shard).into());
    }

    Ok(recovery)
}

/// Decode from available shards and return restored originals as [index, Uint8Array] pairs.
#[wasm_bindgen]
pub fn decode(
    original_count: usize,
    recovery_count: usize,
    shard_bytes: usize,
    originals: &js_sys::Array,
    recoveries: &js_sys::Array,
) -> Result<js_sys::Array, JsValue> {
    let mut decoder = ReedSolomonDecoder::new(original_count, recovery_count, shard_bytes)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    for i in 0..originals.length() {
        let val = originals.get(i);
        let pair = val.dyn_ref::<js_sys::Array>()
            .ok_or_else(|| JsValue::from_str(&format!("originals[{}] is not an array", i)))?;
        let index = pair.get(0).as_f64().unwrap() as usize;
        let data = pair.get(1).dyn_ref::<Uint8Array>().unwrap().to_vec();
        decoder.add_original_shard(index, data)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    for i in 0..recoveries.length() {
        let val = recoveries.get(i);
        let pair = val.dyn_ref::<js_sys::Array>()
            .ok_or_else(|| JsValue::from_str(&format!("recoveries[{}] is not an array", i)))?;
        let index = pair.get(0).as_f64().unwrap() as usize;
        let data = pair.get(1).dyn_ref::<Uint8Array>().unwrap().to_vec();
        decoder.add_recovery_shard(index, data)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    let result = decoder.decode()
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let restored = js_sys::Array::new();
    for (j, (index, shard)) in result.restored_original_iter().enumerate() {
        let pair = js_sys::Array::new();
        pair.set(0, JsValue::from_f64(index as f64));
        pair.set(1, Uint8Array::from(shard).into());
        restored.set(j as u32, pair.into());
    }

    Ok(restored)
}

/// Check if a shard combination is supported.
#[wasm_bindgen]
pub fn supports(original_count: usize, recovery_count: usize) -> bool {
    ReedSolomonEncoder::supports(original_count, recovery_count)
}
