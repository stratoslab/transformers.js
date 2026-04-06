#!/usr/bin/env node
// End-to-end benchmark that separates prefill and decode timing by running the
// first model call manually, then continuing token-by-token with past_key_values.

import { AutoModelForCausalLM, AutoTokenizer } from '../src/transformers.js';
import { Tensor } from '../src/utils/tensor.js';

const MODEL_ID = process.env.MODEL_ID ?? 'onnx-community/Qwen2.5-0.5B-Instruct';
const DEVICE = process.env.DEVICE ?? 'webgpu';
const DTYPE = process.env.DTYPE ?? 'q4';
const MAX_NEW_TOKENS = Number(process.env.MAX_NEW_TOKENS ?? 64);
const RUNS = Number(process.env.RUNS ?? 3);
const B_KEY = Number(process.env.B_KEY ?? 3);
const B_VALUE = Number(process.env.B_VALUE ?? 3);
const RESIDUAL_LENGTH = Number(process.env.RESIDUAL_LENGTH ?? 64);
const EVICTION_BATCH = process.env.EVICTION_BATCH ? Number(process.env.EVICTION_BATCH) : 64;
const QUANTIZATION = process.env.QUANTIZATION ?? 'sigma';
const SIGMA_K = Number(process.env.SIGMA_K ?? 2.5);
const PROMPT =
  process.env.PROMPT ??
  'Summarize how a Canton workflow assistant should review a proposed token transfer and list the main risk checks.';

const messages = [
  { role: 'system', content: 'You are a concise assistant.' },
  { role: 'user', content: PROMPT },
];

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
}

function compressionRatio(stats) {
  if (!stats?.packed_bytes || !stats?.dense_bytes) return null;
  return stats.dense_bytes / stats.packed_bytes;
}

function appendToken(inputIds, token) {
  const oldData = inputIds.data;
  const batch = inputIds.dims[0];
  const seq = inputIds.dims[1];
  const next = new BigInt64Array(oldData.length + batch);
  next.set(oldData);
  for (let i = 0; i < batch; ++i) {
    next[oldData.length + i] = BigInt(token[i]);
  }
  return new Tensor('int64', next, [batch, seq + 1]);
}

function makePositionIds(batch, position) {
  return new Tensor('int64', new BigInt64Array(Array.from({ length: batch }, () => BigInt(position))), [batch, 1]);
}

async function disposeOutputs(outputs) {
  for (const value of Object.values(outputs ?? {})) {
    value?.dispose?.();
  }
}

async function runSplitBenchmark(model, inputs, cacheImplementation) {
  const cacheConfig = cacheImplementation === 'turboquant'
    ? {
        b_key: B_KEY,
        b_value: B_VALUE,
        residual_length: RESIDUAL_LENGTH,
        eviction_batch: EVICTION_BATCH,
        quantization: QUANTIZATION,
        sigma_k: SIGMA_K,
      }
    : undefined;

  const prefillTimings = [];
  const decodeTimings = [];
  const totalTimings = [];
  let lastResult = null;

  for (let run = 0; run < RUNS; ++run) {
    const startedTotal = performance.now();
    const startedPrefill = performance.now();
    let outputs = await model.forward({
      ...inputs,
      use_cache: true,
      cache_implementation: cacheImplementation,
      cache_config: cacheConfig,
    });
    const endedPrefill = performance.now();

    let pastKeyValues = outputs.past_key_values;
    const generated = [];
    let totalInputIds = inputs.input_ids;
    let nextToken = outputs.logits.slice(null, [-1, null], null).argmax(-1);
    generated.push(...nextToken.data);
    totalInputIds = appendToken(totalInputIds, nextToken.data);

    let decodeMs = 0;
    for (let step = 1; step < MAX_NEW_TOKENS; ++step) {
      const stepStarted = performance.now();
      const position = inputs.input_ids.dims[1] + step - 1;
      const stepOutputs = await model.forward({
        input_ids: nextToken,
        attention_mask: new Tensor('int64', new BigInt64Array(totalInputIds.size).fill(1n), totalInputIds.dims.slice()),
        past_key_values: pastKeyValues,
        position_ids: makePositionIds(nextToken.dims[0], position),
        use_cache: true,
        cache_implementation: cacheImplementation,
        cache_config: cacheConfig,
      });
      const stepEnded = performance.now();
      decodeMs += stepEnded - stepStarted;

      outputs = stepOutputs;
      pastKeyValues = stepOutputs.past_key_values;
      nextToken = stepOutputs.logits.slice(null, [-1, null], null).argmax(-1);
      generated.push(...nextToken.data);
      totalInputIds = appendToken(totalInputIds, nextToken.data);
    }

    const endedTotal = performance.now();
    prefillTimings.push(endedPrefill - startedPrefill);
    decodeTimings.push(decodeMs);
    totalTimings.push(endedTotal - startedTotal);

    lastResult = {
      sequences: totalInputIds,
      cache_stats: pastKeyValues?.getStats?.() ?? null,
      generated_tokens: generated.length,
    };

    await outputs?.past_key_values?.dispose?.();
    await disposeOutputs(outputs);
  }

  return {
    avg_prefill_ms: average(prefillTimings),
    avg_decode_total_ms: average(decodeTimings),
    avg_decode_per_token_ms: average(decodeTimings) / Math.max(MAX_NEW_TOKENS - 1, 1),
    avg_total_ms: average(totalTimings),
    result: lastResult,
  };
}

console.log(`Loading tokenizer: ${MODEL_ID}`);
const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
console.log(`Loading model: ${MODEL_ID} (dtype=${DTYPE}, device=${DEVICE})`);
const model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, { dtype: DTYPE, device: DEVICE });

const prompt = tokenizer.apply_chat_template(messages, {
  tokenize: true,
  add_generation_prompt: true,
  return_dict: true,
});

const dynamicBenchmark = await runSplitBenchmark(model, prompt, 'dynamic');
const turboBenchmark = await runSplitBenchmark(model, prompt, 'turboquant');

const rows = [
  {
    backend: 'dynamic',
    avg_prefill_ms: dynamicBenchmark.avg_prefill_ms,
    avg_decode_total_ms: dynamicBenchmark.avg_decode_total_ms,
    avg_decode_per_token_ms: dynamicBenchmark.avg_decode_per_token_ms,
    avg_total_ms: dynamicBenchmark.avg_total_ms,
    cache_stats: dynamicBenchmark.result.cache_stats,
    compression_ratio: compressionRatio(dynamicBenchmark.result.cache_stats),
  },
  {
    backend: 'turboquant',
    avg_prefill_ms: turboBenchmark.avg_prefill_ms,
    avg_decode_total_ms: turboBenchmark.avg_decode_total_ms,
    avg_decode_per_token_ms: turboBenchmark.avg_decode_per_token_ms,
    avg_total_ms: turboBenchmark.avg_total_ms,
    cache_stats: turboBenchmark.result.cache_stats,
    compression_ratio: compressionRatio(turboBenchmark.result.cache_stats),
  },
];

console.log(JSON.stringify({
  benchmark: 'turboquant_split',
  model_id: MODEL_ID,
  device: DEVICE,
  dtype: DTYPE,
  max_new_tokens: MAX_NEW_TOKENS,
  runs: RUNS,
  turbo_config: {
    b_key: B_KEY,
    b_value: B_VALUE,
    residual_length: RESIDUAL_LENGTH,
    eviction_batch: EVICTION_BATCH,
    quantization: QUANTIZATION,
    sigma_k: SIGMA_K,
  },
  rows,
}, null, 2));

await model.dispose();
