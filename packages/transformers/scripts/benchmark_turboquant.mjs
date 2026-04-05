import { AutoModelForCausalLM, AutoTokenizer } from '../src/transformers.js';

const MODEL_ID = process.env.MODEL_ID ?? 'onnx-community/Qwen2.5-0.5B-Instruct';
const DEVICE = process.env.DEVICE ?? 'webgpu';
const DTYPE = process.env.DTYPE ?? 'q4';
const MAX_NEW_TOKENS = Number(process.env.MAX_NEW_TOKENS ?? 128);
const RUNS = Number(process.env.RUNS ?? 3);
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

async function runBenchmark(model, prompt, cacheImplementation) {
  const timings = [];
  let lastResult = null;

  for (let i = 0; i < RUNS; ++i) {
    const started = performance.now();
    lastResult = await model.generate({
      ...prompt,
      max_new_tokens: MAX_NEW_TOKENS,
      do_sample: false,
      return_dict_in_generate: true,
      cache_implementation: cacheImplementation,
    });
    timings.push(performance.now() - started);
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
  console.log(`output: ${decoded}`);
}

console.log(`Loading tokenizer: ${MODEL_ID}`);
const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);

console.log(`Loading model: ${MODEL_ID} (dtype=${DTYPE}, device=${DEVICE})`);
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

printResult('dynamic', dynamicBenchmark, tokenizer);
printResult('turboquant', turboBenchmark, tokenizer);

console.log('\n[summary]');
console.log(`dynamic avg: ${formatMs(dynamicBenchmark.averageMs)}`);
console.log(`turboquant avg: ${formatMs(turboBenchmark.averageMs)}`);
console.log(
  `speed ratio (dynamic / turboquant): ${(dynamicBenchmark.averageMs / turboBenchmark.averageMs).toFixed(3)}`,
);

await model.dispose();
