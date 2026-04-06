#!/usr/bin/env node
// Measures only TurboQuant/Dynamic cache materialization cost on an already
// built cache. This isolates dense rematerialization from update/pack cost.

import { TurboQuantCache, DynamicCache } from '../src/cache_utils.js';
import { Tensor } from '../src/utils/tensor.js';

const LAYERS = Number(process.env.LAYERS ?? 4);
const NUM_KV_HEADS = Number(process.env.NUM_KV_HEADS ?? 4);
const HEAD_DIM = Number(process.env.HEAD_DIM ?? 128);
const SEQ_LENS = (process.env.SEQ_LENS ?? '128,256,512,1024,2048')
  .split(',')
  .map(x => Number(x.trim()))
  .filter(Number.isFinite);
const RUNS = Number(process.env.RUNS ?? 20);
const WARMUP_RUNS = Number(process.env.WARMUP_RUNS ?? 3);

const TURBO_CONFIG = {
  b_key: Number(process.env.B_KEY ?? 3),
  b_value: Number(process.env.B_VALUE ?? 3),
  residual_length: Number(process.env.RESIDUAL_LENGTH ?? 64),
  eviction_batch: process.env.EVICTION_BATCH ? Number(process.env.EVICTION_BATCH) : 64,
  quantization: process.env.QUANTIZATION ?? 'sigma',
  sigma_k: Number(process.env.SIGMA_K ?? 2.5),
};

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? 0;
}

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

async function buildCache(CacheClass, config, seqLen) {
  const cache = new CacheClass(config);
  return await cache.update(buildFeeds(seqLen));
}

function safeDisposeMaterialized(materialized) {
  for (const tensor of Object.values(materialized)) {
    tensor?.dispose?.();
  }
}

async function benchmarkMaterialize(label, CacheClass, config, seqLen) {
  const cache = await buildCache(CacheClass, config, seqLen);
  const warmup = WARMUP_RUNS + RUNS;
  const timings = [];

  for (let i = 0; i < warmup; ++i) {
    const started = performance.now();
    const materialized = cache.materialize();
    const ended = performance.now();
    safeDisposeMaterialized(materialized);
    if (i >= WARMUP_RUNS) {
      timings.push(ended - started);
    }
  }

  const stats = cache.getStats();
  return {
    backend: label,
    seq_len: seqLen,
    runs: RUNS,
    avg_materialize_ms: average(timings),
    median_materialize_ms: median(timings),
    min_materialize_ms: Math.min(...timings),
    max_materialize_ms: Math.max(...timings),
    packed_bytes: stats.packed_bytes,
    dense_bytes: stats.dense_bytes,
    compression_ratio: stats.packed_bytes ? stats.dense_bytes / stats.packed_bytes : 1,
    compressed_blocks: stats.compressed_blocks ?? null,
  };
}

const rows = [];
for (const seqLen of SEQ_LENS) {
  rows.push(await benchmarkMaterialize('DynamicCache', DynamicCache, {}, seqLen));
  rows.push(await benchmarkMaterialize('TurboQuant', TurboQuantCache, TURBO_CONFIG, seqLen));
}

console.log(JSON.stringify({
  benchmark: 'materialize_only',
  layers: LAYERS,
  num_kv_heads: NUM_KV_HEADS,
  head_dim: HEAD_DIM,
  turbo_config: TURBO_CONFIG,
  rows,
}, null, 2));
