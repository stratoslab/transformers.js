import { AutoModelForCausalLM, AutoTokenizer } from '../src/transformers.js';

const MODEL_ID = process.env.MODEL_ID ?? 'onnx-community/Qwen2.5-0.5B-Instruct';
const DEVICE = process.env.DEVICE ?? 'webgpu';
const DTYPE = process.env.DTYPE ?? 'q4';
const MAX_NEW_TOKENS = Number(process.env.MAX_NEW_TOKENS ?? 128);
const RUNS = Number(process.env.RUNS ?? 3);
const B_KEY = Number(process.env.B_KEY ?? 4);
const B_VALUE = Number(process.env.B_VALUE ?? 8);
const RESIDUAL_LENGTH = Number(process.env.RESIDUAL_LENGTH ?? 64);
const EVICTION_BATCH = process.env.EVICTION_BATCH ? Number(process.env.EVICTION_BATCH) : null;
const QUANTIZATION = process.env.QUANTIZATION ?? 'minmax'; // 'minmax' | 'sigma'
const SIGMA_K = Number(process.env.SIGMA_K ?? 2.5);
const PROMPT =
  process.env.PROMPT ??
  'Summarize how a Canton workflow assistant should review a proposed token transfer and list the main risk checks.';

const messages = [
  { role: 'system', content: 'You are a concise assistant.' },
  { role: 'user', content: PROMPT },
];

function formatMs(value) {
  return `${value.toFixed(1)} ms`;
}

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
}

function lastAssistantText(decoded) {
  return decoded.split('\n').slice(-1)[0] ?? decoded;
}

function compressionRatio(stats) {
  if (!stats?.packed_bytes || !stats?.dense_bytes) return null;
  return stats.dense_bytes / stats.packed_bytes;
}

async function disposeGenerationResult(result) {
  await result?.past_key_values?.dispose?.();
  result?.sequences?.dispose?.();
}

async function runBenchmark(model, prompt, cacheImplementation) {
  const timings = [];
  let lastResult = null;

  for (let i = 0; i < RUNS; ++i) {
    const captureResult = i === RUNS - 1;

    const started = performance.now();
    const result = await model.generate({
      ...prompt,
      max_new_tokens: MAX_NEW_TOKENS,
      do_sample: false,
      return_dict_in_generate: captureResult,
      cache_implementation: cacheImplementation,
      cache_config:
        cacheImplementation === 'turboquant'
          ? {
              b_key: B_KEY,
              b_value: B_VALUE,
              residual_length: RESIDUAL_LENGTH,
              eviction_batch: EVICTION_BATCH,
              quantization: QUANTIZATION,
              sigma_k: SIGMA_K,
            }
          : undefined,
    });
    timings.push(performance.now() - started);

    if (captureResult) {
      lastResult = result;
    }
  }

  return {
    timings,
    averageMs: average(timings),
    result: lastResult,
  };
}

function printResult(label, benchmark, tokenizer) {
  const decoded = tokenizer.batch_decode(benchmark.result.sequences, {
    skip_special_tokens: true,
  })[0];
  console.log(`\n[${label}]`);
  console.log(`runs: ${benchmark.timings.map(formatMs).join(', ')}`);
  console.log(`avg: ${formatMs(benchmark.averageMs)}`);
  console.log('cache_stats:', benchmark.result.cache_stats);
  const ratio = compressionRatio(benchmark.result.cache_stats);
  if (ratio) {
    console.log(`compression ratio: ${ratio.toFixed(3)}x`);
  }
  console.log(`output: ${decoded}`);
  return decoded;
}

console.log(`Loading tokenizer: ${MODEL_ID}`);
const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);

console.log(`Loading model: ${MODEL_ID} (dtype=${DTYPE}, device=${DEVICE})`);
console.log(
  `TurboQuant config: b_key=${B_KEY}, b_value=${B_VALUE}, residual_length=${RESIDUAL_LENGTH}, ` +
    `eviction_batch=${EVICTION_BATCH ?? '(=residual_length)'}, quantization=${QUANTIZATION}` +
    (QUANTIZATION === 'sigma' ? `, sigma_k=${SIGMA_K}` : ''),
);
const model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
  dtype: DTYPE,
  device: DEVICE,
});

const prompt = tokenizer.apply_chat_template(messages, {
  tokenize: true,
  add_generation_prompt: true,
  return_dict: true,
});

console.log('Warming up...');
await model.generate({
  ...prompt,
  max_new_tokens: 8,
  do_sample: false,
  return_dict_in_generate: true,
  cache_implementation: 'dynamic',
});

const dynamicBenchmark = await runBenchmark(model, prompt, 'dynamic');
const turboBenchmark = await runBenchmark(model, prompt, 'turboquant');

const dynamicOutput = printResult('dynamic', dynamicBenchmark, tokenizer);
const turboOutput = printResult('turboquant', turboBenchmark, tokenizer);

console.log('\n[summary]');
console.log(`dynamic avg: ${formatMs(dynamicBenchmark.averageMs)}`);
console.log(`turboquant avg: ${formatMs(turboBenchmark.averageMs)}`);
console.log(
  `speed ratio (dynamic / turboquant): ${(dynamicBenchmark.averageMs / turboBenchmark.averageMs).toFixed(3)}`,
);
console.log(`outputs match exactly: ${dynamicOutput === turboOutput}`);
console.log(`dynamic tail: ${lastAssistantText(dynamicOutput)}`);
console.log(`turbo tail: ${lastAssistantText(turboOutput)}`);

await disposeGenerationResult(dynamicBenchmark.result);
await disposeGenerationResult(turboBenchmark.result);
await model.dispose();
