#!/usr/bin/env node
// Isolated cache benchmark: measures only update() + materialize() cost
// per simulated decoder step. Does NOT run a model — just synthetic
// [1, num_kv_heads, seq_len, head_dim] tensors sized like Gemma 4-E2B.
//
// Writes results as JSON to stdout so the runner can diff old vs new.

import { TurboQuantCache, DynamicCache } from '../src/cache_utils.js';
import { Tensor } from '../src/utils/tensor.js';

const LAYERS = 4; // 8 entries (4 key + 4 value) per decoder step
const NUM_KV_HEADS = 4;
const HEAD_DIM = 128; // Gemma 4-ish, power of 2 so rotation applies
const SEQ_LENS = [128, 256, 512, 1024];

function makePresentTensor(seqLen) {
  const data = new Float32Array(NUM_KV_HEADS * seqLen * HEAD_DIM);
  let flat = 0;
  for (let h = 0; h < NUM_KV_HEADS; ++h) {
    for (let s = 0; s < seqLen; ++s) {
      for (let d = 0; d < HEAD_DIM; ++d) {
        // Position-indexed values: stable across tensors of different lengths.
        const k = h * 10000 + s * 131 + d;
        data[flat++] = Math.sin(k / 11) * 0.5 + Math.cos(k / 13) * 0.3;
      }
    }
  }
  return new Tensor('float32', data, [1, NUM_KV_HEADS, seqLen, HEAD_DIM]);
}

function buildFeeds(seqLen) {
  const feeds = {};
  for (let l = 0; l < LAYERS; ++l) {
    feeds[`present.${l}.key`] = makePresentTensor(seqLen);
    feeds[`present.${l}.value`] = makePresentTensor(seqLen);
  }
  return feeds;
}

async function simulateGeneration(CacheClass, config, seqLen) {
  // Simulate "prefill then decode" — one fresh cache, then seqLen-1 single-token
  // steps. Measures update + materialize cost per step, as ORT would do.
  let cache = new CacheClass(config);
  let updateMs = 0;
  let materializeMs = 0;
  let firstUpdateMs = 0;

  // Prefill (full sequence arrives as one large present.*).
  let t0 = performance.now();
  cache = await cache.update(buildFeeds(seqLen));
  let t1 = performance.now();
  updateMs += t1 - t0;
  firstUpdateMs = t1 - t0;

  t0 = performance.now();
  cache.materialize();
  t1 = performance.now();
  materializeMs += t1 - t0;

  // "Decode" — 32 additional single-token steps.
  const DECODE_STEPS = 32;
  for (let step = 1; step <= DECODE_STEPS; ++step) {
    const totalLen = seqLen + step;
    t0 = performance.now();
    cache = await cache.update(buildFeeds(totalLen));
    t1 = performance.now();
    updateMs += t1 - t0;

    t0 = performance.now();
    cache.materialize();
    t1 = performance.now();
    materializeMs += t1 - t0;
  }

  const stats = cache.getStats();
  const decodeSteps = DECODE_STEPS;
  return {
    prefill_update_ms: firstUpdateMs,
    decode_update_total_ms: updateMs - firstUpdateMs,
    decode_update_per_step_ms: (updateMs - firstUpdateMs) / decodeSteps,
    decode_materialize_total_ms: materializeMs - (materializeMs - (materializeMs - materializeMs / (decodeSteps + 1))),
    decode_per_step_ms: ((updateMs - firstUpdateMs) + (materializeMs * decodeSteps / (decodeSteps + 1))) / decodeSteps,
    total_ms: updateMs + materializeMs,
    compression_ratio: stats.dense_bytes && stats.packed_bytes ? stats.dense_bytes / stats.packed_bytes : 1,
    packed_bytes: stats.packed_bytes,
    dense_bytes: stats.dense_bytes,
    compressed_blocks: stats.compressed_blocks ?? null,
  };
}

async function run() {
  const configs = [
    { label: 'Safe Default', cfg: { b_key: 4, b_value: 8, residual_length: 64 } },
    { label: 'Key Heavy',    cfg: { b_key: 3, b_value: 8, residual_length: 64 } },
    { label: 'Mid Compression', cfg: { b_key: 4, b_value: 8, residual_length: 48 } },
    // New-only configs (silently fall through to safe defaults on OLD code):
    { label: 'LongCtx sigma', cfg: { b_key: 3, b_value: 4, residual_length: 128, eviction_batch: 128, quantization: 'sigma', sigma_k: 2.5 } },
    { label: 'Aggressive sigma', cfg: { b_key: 3, b_value: 3, residual_length: 64, eviction_batch: 64, quantization: 'sigma', sigma_k: 2.5 } },
  ];

  const rows = [];
  for (const { label, cfg } of configs) {
    for (const seqLen of SEQ_LENS) {
      // Warm up (discard first measurement)
      await simulateGeneration(TurboQuantCache, cfg, 32);
      const result = await simulateGeneration(TurboQuantCache, cfg, seqLen);
      rows.push({ config: label, seq_len: seqLen, ...result });
    }
  }

  // Also run a pure DynamicCache baseline.
  for (const seqLen of SEQ_LENS) {
    await simulateGeneration(DynamicCache, {}, 32);
    const result = await simulateGeneration(DynamicCache, {}, seqLen);
    rows.push({ config: 'DynamicCache', seq_len: seqLen, ...result });
  }

  console.log(JSON.stringify({ layers: LAYERS, num_kv_heads: NUM_KV_HEADS, head_dim: HEAD_DIM, rows }, null, 2));
}

await run();
