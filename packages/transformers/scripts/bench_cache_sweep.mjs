#!/usr/bin/env node
// Sweeps seq length and TurboQuant residual-window parameters on the synthetic
// cache benchmark so crossover points are easy to compare in one JSON blob.

import { TurboQuantCache, DynamicCache } from '../src/cache_utils.js';
import { Tensor } from '../src/utils/tensor.js';

const LAYERS = Number(process.env.LAYERS ?? 4);
const NUM_KV_HEADS = Number(process.env.NUM_KV_HEADS ?? 4);
const HEAD_DIM = Number(process.env.HEAD_DIM ?? 128);
const SEQ_LENS = (process.env.SEQ_LENS ?? '128,512,2048,8192,16384')
  .split(',')
  .map(x => Number(x.trim()))
  .filter(Number.isFinite);
const DECODE_STEPS = Number(process.env.DECODE_STEPS ?? 32);

const CONFIGS = [
  { label: 'DynamicCache', type: 'dynamic', cfg: {} },
  { label: 'TurboQuant-64tail', type: 'turboquant', cfg: { b_key: 3, b_value: 3, residual_length: 64, eviction_batch: 64, quantization: 'sigma', sigma_k: 2.5 } },
  { label: 'TurboQuant-128tail', type: 'turboquant', cfg: { b_key: 3, b_value: 3, residual_length: 128, eviction_batch: 128, quantization: 'sigma', sigma_k: 2.5 } },
  { label: 'TurboQuant-256tail', type: 'turboquant', cfg: { b_key: 3, b_value: 3, residual_length: 256, eviction_batch: 256, quantization: 'sigma', sigma_k: 2.5 } },
  { label: 'TurboQuant-64tail-fine', type: 'turboquant', cfg: { b_key: 3, b_value: 3, residual_length: 64, eviction_batch: 16, quantization: 'sigma', sigma_k: 2.5 } },
];

function makePresentTensor(seqLen) {
  const data = new Float32Array(NUM_KV_HEADS * seqLen * HEAD_DIM);
  let flat = 0;
  for (let h = 0; h < NUM_KV_HEADS; ++h) {
    for (let s = 0; s < seqLen; ++s) {
      for (let d = 0; d < HEAD_DIM; ++d) {
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

async function simulate(CacheClass, config, seqLen) {
  let cache = new CacheClass(config);
  let updateMs = 0;
  let materializeMs = 0;
  let prefillUpdateMs = 0;
  let prefillMaterializeMs = 0;

  let started = performance.now();
  cache = await cache.update(buildFeeds(seqLen));
  let ended = performance.now();
  updateMs += ended - started;
  prefillUpdateMs = ended - started;

  started = performance.now();
  let materialized = cache.materialize();
  ended = performance.now();
  materializeMs += ended - started;
  prefillMaterializeMs = ended - started;
  for (const tensor of Object.values(materialized)) tensor?.dispose?.();

  for (let step = 1; step <= DECODE_STEPS; ++step) {
    const totalLen = seqLen + step;
    started = performance.now();
    cache = await cache.update(buildFeeds(totalLen));
    ended = performance.now();
    updateMs += ended - started;

    started = performance.now();
    materialized = cache.materialize();
    ended = performance.now();
    materializeMs += ended - started;
    for (const tensor of Object.values(materialized)) tensor?.dispose?.();
  }

  const stats = cache.getStats();
  return {
    seq_len: seqLen,
    prefill_update_ms: prefillUpdateMs,
    prefill_materialize_ms: prefillMaterializeMs,
    decode_update_per_step_ms: (updateMs - prefillUpdateMs) / DECODE_STEPS,
    decode_materialize_per_step_ms: (materializeMs - prefillMaterializeMs) / DECODE_STEPS,
    total_ms: updateMs + materializeMs,
    packed_bytes: stats.packed_bytes,
    dense_bytes: stats.dense_bytes,
    compression_ratio: stats.packed_bytes ? stats.dense_bytes / stats.packed_bytes : 1,
    compressed_blocks: stats.compressed_blocks ?? null,
  };
}

const rows = [];
for (const { label, type, cfg } of CONFIGS) {
  const CacheClass = type === 'dynamic' ? DynamicCache : TurboQuantCache;
  for (const seqLen of SEQ_LENS) {
    rows.push({ config: label, ...(await simulate(CacheClass, cfg, seqLen)) });
  }
}

console.log(JSON.stringify({
  benchmark: 'cache_sweep',
  layers: LAYERS,
  num_kv_heads: NUM_KV_HEADS,
  head_dim: HEAD_DIM,
  decode_steps: DECODE_STEPS,
  rows,
}, null, 2));
