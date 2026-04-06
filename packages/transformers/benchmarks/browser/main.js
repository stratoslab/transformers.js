import { AutoModelForCausalLM, AutoTokenizer } from '../../dist/transformers.web.js';

const elements = {
  modelId: document.getElementById('modelId'),
  device: document.getElementById('device'),
  dtype: document.getElementById('dtype'),
  runs: document.getElementById('runs'),
  maxNewTokens: document.getElementById('maxNewTokens'),
  warmupTokens: document.getElementById('warmupTokens'),
  bKey: document.getElementById('bKey'),
  bValue: document.getElementById('bValue'),
  residualLength: document.getElementById('residualLength'),
  evictionBatch: document.getElementById('evictionBatch'),
  quantization: document.getElementById('quantization'),
  sigmaK: document.getElementById('sigmaK'),
  prompt: document.getElementById('prompt'),
  runButton: document.getElementById('runButton'),
  resetButton: document.getElementById('resetButton'),
  status: document.getElementById('status'),
  cards: document.getElementById('cards'),
  rawOutput: document.getElementById('rawOutput'),
};

function setStatus(message) {
  elements.status.textContent = message;
}

function getConfig() {
  return {
    modelId: elements.modelId.value.trim(),
    device: elements.device.value,
    dtype: elements.dtype.value.trim(),
    runs: Number(elements.runs.value),
    maxNewTokens: Number(elements.maxNewTokens.value),
    warmupTokens: Number(elements.warmupTokens.value),
    prompt: elements.prompt.value.trim(),
    turbo: {
      b_key: Number(elements.bKey.value),
      b_value: Number(elements.bValue.value),
      residual_length: Number(elements.residualLength.value),
      eviction_batch: Number(elements.evictionBatch.value),
      quantization: elements.quantization.value,
      sigma_k: Number(elements.sigmaK.value),
    },
  };
}

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
}

function compressionRatio(stats) {
  if (!stats?.packed_bytes || !stats?.dense_bytes) return null;
  return stats.dense_bytes / stats.packed_bytes;
}

async function disposeGenerationResult(result) {
  await result?.past_key_values?.dispose?.();
  result?.sequences?.dispose?.();
}

async function loadModelAndTokenizer(config) {
  setStatus(`Loading tokenizer: ${config.modelId}`);
  const tokenizer = await AutoTokenizer.from_pretrained(config.modelId);
  setStatus(`Loading model: ${config.modelId} (${config.device}/${config.dtype})`);
  const model = await AutoModelForCausalLM.from_pretrained(config.modelId, {
    dtype: config.dtype,
    device: config.device,
  });
  return { tokenizer, model };
}

async function runGenerateBenchmark(model, promptInputs, benchmarkConfig) {
  const timings = [];
  let finalResult = null;

  for (let i = 0; i < benchmarkConfig.runs; ++i) {
    setStatus(`Running ${benchmarkConfig.label} (${i + 1}/${benchmarkConfig.runs})`);
    const started = performance.now();
    const result = await model.generate({
      ...promptInputs,
      max_new_tokens: benchmarkConfig.maxNewTokens,
      do_sample: false,
      return_dict_in_generate: true,
      cache_implementation: benchmarkConfig.cacheImplementation,
      cache_config: benchmarkConfig.cacheConfig,
    });
    const ended = performance.now();
    timings.push(ended - started);

    if (finalResult) {
      await disposeGenerationResult(finalResult);
    }
    finalResult = result;
  }

  return {
    label: benchmarkConfig.label,
    timings,
    averageMs: average(timings),
    result: finalResult,
  };
}

function renderCards(results) {
  elements.cards.innerHTML = '';
  for (const benchmark of results) {
    const ratio = compressionRatio(benchmark.result.cache_stats);
    const card = document.createElement('article');
    card.className = 'result-card';
    card.innerHTML = `
      <h3>${benchmark.label}</h3>
      <div class="metrics">
        <div class="metric">
          <span>Average latency</span>
          <strong>${benchmark.averageMs.toFixed(1)} ms</strong>
        </div>
        <div class="metric">
          <span>Compression ratio</span>
          <strong>${ratio ? `${ratio.toFixed(2)}x` : 'n/a'}</strong>
        </div>
      </div>
    `;
    elements.cards.appendChild(card);
  }
}

async function run() {
  const config = getConfig();
  if (!config.modelId || !config.prompt) {
    throw new Error('Model ID and prompt are required.');
  }

  elements.runButton.disabled = true;
  try {
    const { tokenizer, model } = await loadModelAndTokenizer(config);
    const messages = [
      { role: 'system', content: 'You are a concise assistant.' },
      { role: 'user', content: config.prompt },
    ];
    const promptInputs = tokenizer.apply_chat_template(messages, {
      tokenize: true,
      add_generation_prompt: true,
      return_dict: true,
    });

    setStatus('Running warmup');
    const warmup = await model.generate({
      ...promptInputs,
      max_new_tokens: config.warmupTokens,
      do_sample: false,
      return_dict_in_generate: true,
      cache_implementation: 'dynamic',
    });
    await disposeGenerationResult(warmup);

    const benchmarks = [
      {
        label: 'DynamicCache',
        cacheImplementation: 'dynamic',
        cacheConfig: undefined,
        runs: config.runs,
        maxNewTokens: config.maxNewTokens,
      },
      {
        label: 'TurboQuant',
        cacheImplementation: 'turboquant',
        cacheConfig: config.turbo,
        runs: config.runs,
        maxNewTokens: config.maxNewTokens,
      },
    ];

    const results = [];
    for (const benchmark of benchmarks) {
      results.push(await runGenerateBenchmark(model, promptInputs, benchmark));
    }

    renderCards(results);
    elements.rawOutput.textContent = JSON.stringify({
      benchmark: 'browser_generate_compare',
      config,
      rows: results.map(result => ({
        backend: result.label,
        average_ms: result.averageMs,
        timings_ms: result.timings,
        cache_stats: result.result.cache_stats,
        compression_ratio: compressionRatio(result.result.cache_stats),
      })),
    }, null, 2);

    for (const result of results) {
      await disposeGenerationResult(result.result);
    }
    await model.dispose();
    setStatus('Benchmark complete.');
  } catch (error) {
    console.error(error);
    setStatus(`Benchmark failed: ${error.message}`);
  } finally {
    elements.runButton.disabled = false;
  }
}

elements.runButton.addEventListener('click', () => {
  run();
});

elements.resetButton.addEventListener('click', () => {
  elements.cards.innerHTML = '';
  elements.rawOutput.textContent = 'No benchmark run yet.';
  setStatus('');
});
