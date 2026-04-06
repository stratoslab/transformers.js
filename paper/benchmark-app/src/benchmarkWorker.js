import {
  AutoModelForCausalLM,
  AutoProcessor,
  AutoTokenizer,
  Gemma4ForConditionalGeneration,
  TextStreamer,
} from "@huggingface/transformers";
import { DynamicCache, TurboQuantCache } from "@transformers-src/cache_utils.js";
import { Tensor } from "@transformers-src/utils/tensor.js";

const DEFAULT_MODEL_ID = "onnx-community/gemma-4-E2B-it-ONNX";

let tokenizer = null;
let processor = null;
let model = null;
let currentModelId = null;
let loadingPromise = null;

function isGemma4Model(modelId) {
  return modelId.toLowerCase().includes("gemma-4");
}

function post(message) {
  self.postMessage(message);
}

async function ensureLoaded(modelId) {
  if ((tokenizer || processor) && model && currentModelId === modelId) {
    post({ status: "ready", modelId });
    return;
  }

  if (loadingPromise) {
    return loadingPromise;
  }

  if (isGemma4Model(modelId)) {
    loadingPromise = Promise.all([
      AutoProcessor.from_pretrained(modelId, {
        progress_callback: (info) => post({ status: "progress", phase: "processor", info }),
      }),
      Gemma4ForConditionalGeneration.from_pretrained(modelId, {
        dtype: "q4f16",
        device: "webgpu",
        progress_callback: (info) => post({ status: "progress", phase: "model", info }),
      }),
    ])
      .then(([nextProcessor, nextModel]) => {
        processor = nextProcessor;
        tokenizer = nextProcessor.tokenizer;
        model = nextModel;
        currentModelId = modelId;
        post({ status: "ready", modelId });
      })
      .catch((error) => {
        post({
          status: "error",
          error: error instanceof Error ? error.message : String(error),
        });
        throw error;
      })
      .finally(() => {
        loadingPromise = null;
      });

    return loadingPromise;
  }

  loadingPromise = Promise.all([
    AutoTokenizer.from_pretrained(modelId, {
      progress_callback: (info) => post({ status: "progress", phase: "tokenizer", info }),
    }),
    AutoModelForCausalLM.from_pretrained(modelId, {
      dtype: "q4",
      device: "webgpu",
      progress_callback: (info) => post({ status: "progress", phase: "model", info }),
    }),
  ])
    .then(([nextTokenizer, nextModel]) => {
      processor = null;
      tokenizer = nextTokenizer;
      model = nextModel;
      currentModelId = modelId;
      post({ status: "ready", modelId });
    })
    .catch((error) => {
      post({
        status: "error",
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    })
    .finally(() => {
      loadingPromise = null;
    });

  return loadingPromise;
}

async function disposeGenerationResult(result) {
  await result?.past_key_values?.dispose?.();
  result?.sequences?.dispose?.();
}

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? 0;
}

function commonPrefixLength(a, b) {
  const limit = Math.min(a.length, b.length);
  let index = 0;
  while (index < limit && a[index] === b[index]) {
    index += 1;
  }
  return index;
}

function buildPromptInputs(modelId, prompt) {
  const messages = [
    { role: "system", content: "You are a concise assistant." },
    { role: "user", content: prompt },
  ];

  if (isGemma4Model(modelId)) {
    const chatPrompt = processor.apply_chat_template(messages, {
      add_generation_prompt: true,
      enable_thinking: false,
    });
    return processor(chatPrompt, null, null, {
      add_special_tokens: false,
    });
  }

  return tokenizer.apply_chat_template(messages, {
    tokenize: true,
    add_generation_prompt: true,
    return_dict: true,
  });
}

async function runCase(prompt, options) {
  const { runs, maxNewTokens, cacheImplementation, cacheConfig } = options;
  let lastResult = null;
  const timings = [];
  const ttfts = [];

  for (let i = 0; i < runs; i += 1) {
    const captureResult = i === runs - 1;
    const startedAt = performance.now();
    let firstTokenAt = null;
    const streamer = new TextStreamer(tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (text) => {
        if (text && firstTokenAt === null) {
          firstTokenAt = performance.now();
        }
      },
    });
    const result = await model.generate({
      ...prompt,
      max_new_tokens: maxNewTokens,
      do_sample: false,
      return_dict_in_generate: captureResult,
      cache_implementation: cacheImplementation,
      cache_config: cacheImplementation === "turboquant" ? cacheConfig : undefined,
      streamer,
    });
    const elapsed = performance.now() - startedAt;
    timings.push(elapsed);
    ttfts.push(firstTokenAt === null ? elapsed : firstTokenAt - startedAt);
    if (captureResult) {
      if (lastResult) {
        await disposeGenerationResult(lastResult);
      }
      lastResult = result;
    }
  }

  const promptLength = prompt.input_ids?.dims?.at(-1) ?? 0;
  const outputTokens = (lastResult.sequences?.dims?.at(-1) ?? 0) - promptLength;
  const output = tokenizer.batch_decode(lastResult.sequences, {
    skip_special_tokens: true,
  })[0];

  const averageMs = average(timings);
  const averageTtftMs = average(ttfts);
  const decodeMs = Math.max(averageMs - averageTtftMs, 1);

  const payload = {
    timings,
    ttfts,
    averageMs,
    averageTtftMs,
    outputTokens,
    decodeTokensPerSecond: outputTokens > 0 ? outputTokens / (decodeMs / 1000) : 0,
    cacheStats: lastResult.cache_stats,
    output,
  };

  await disposeGenerationResult(lastResult);
  return payload;
}

function makePresentTensor(seqLen, numKvHeads, headDim) {
  const data = new Float32Array(numKvHeads * seqLen * headDim);
  let flat = 0;
  for (let h = 0; h < numKvHeads; ++h) {
    for (let s = 0; s < seqLen; ++s) {
      for (let d = 0; d < headDim; ++d) {
        const k = h * 10000 + s * 131 + d;
        data[flat++] = Math.sin(k / 11) * 0.5 + Math.cos(k / 13) * 0.3;
      }
    }
  }
  return new Tensor("float32", data, [1, numKvHeads, seqLen, headDim]);
}

function buildSyntheticFeeds(seqLen, layers, numKvHeads, headDim) {
  const feeds = {};
  for (let l = 0; l < layers; ++l) {
    feeds[`present.${l}.key`] = makePresentTensor(seqLen, numKvHeads, headDim);
    feeds[`present.${l}.value`] = makePresentTensor(seqLen, numKvHeads, headDim);
  }
  return feeds;
}

function disposeTensorMap(entries) {
  for (const tensor of Object.values(entries ?? {})) {
    tensor?.dispose?.();
  }
}

async function buildSyntheticCache(CacheClass, config, params, seqLen) {
  const cache = new CacheClass(config);
  return await cache.update(buildSyntheticFeeds(seqLen, params.layers, params.numKvHeads, params.headDim));
}

async function runMaterializeOnly(params) {
  const rows = [];
  const seqLens = params.seqLens;
  const runs = params.runs;
  const warmupRuns = params.warmupRuns;

  for (const seqLen of seqLens) {
    for (const backend of [
      { label: "DynamicCache", CacheClass: DynamicCache, cfg: {} },
      ...params.turboConfigs.map((cfg) => ({
        label: cfg.label,
        CacheClass: TurboQuantCache,
        cfg,
      })),
    ]) {
      post({
        status: "phase",
        message: `Materialize-only: ${backend.label} @ seq=${seqLen}`,
      });
      const cache = await buildSyntheticCache(backend.CacheClass, backend.cfg, params, seqLen);
      const timings = [];
      for (let i = 0; i < warmupRuns + runs; ++i) {
        const started = performance.now();
        const materialized = cache.materialize();
        const ended = performance.now();
        disposeTensorMap(materialized);
        if (i >= warmupRuns) {
          timings.push(ended - started);
        }
      }
      const stats = cache.getStats();
      rows.push({
        benchmark: "materialize_only",
        backend: backend.label,
        seq_len: seqLen,
        avg_materialize_ms: average(timings),
        median_materialize_ms: median(timings),
        min_materialize_ms: Math.min(...timings),
        max_materialize_ms: Math.max(...timings),
        packed_bytes: stats.packed_bytes,
        dense_bytes: stats.dense_bytes,
        compression_ratio: stats.packed_bytes ? stats.dense_bytes / stats.packed_bytes : 1,
        compressed_blocks: stats.compressed_blocks ?? null,
      });
    }
  }
  return rows;
}

async function runCacheSweep(params) {
  const rows = [];
  const seqLens = params.seqLens;
  const decodeSteps = params.decodeSteps;

  for (const backend of [
    { label: "DynamicCache", CacheClass: DynamicCache, cfg: {} },
    ...params.turboConfigs.map((cfg) => ({
      label: cfg.label,
      CacheClass: TurboQuantCache,
      cfg,
    })),
  ]) {
    for (const seqLen of seqLens) {
      post({
        status: "phase",
        message: `Cache sweep: ${backend.label} @ seq=${seqLen}`,
      });
      let cache = new backend.CacheClass(backend.cfg);
      let updateMs = 0;
      let materializeMs = 0;
      let prefillUpdateMs = 0;
      let prefillMaterializeMs = 0;

      let started = performance.now();
      cache = await cache.update(buildSyntheticFeeds(seqLen, params.layers, params.numKvHeads, params.headDim));
      let ended = performance.now();
      updateMs += ended - started;
      prefillUpdateMs = ended - started;

      started = performance.now();
      let materialized = cache.materialize();
      ended = performance.now();
      materializeMs += ended - started;
      prefillMaterializeMs = ended - started;
      disposeTensorMap(materialized);

      for (let step = 1; step <= decodeSteps; ++step) {
        const totalLen = seqLen + step;
        started = performance.now();
        cache = await cache.update(buildSyntheticFeeds(totalLen, params.layers, params.numKvHeads, params.headDim));
        ended = performance.now();
        updateMs += ended - started;

        started = performance.now();
        materialized = cache.materialize();
        ended = performance.now();
        materializeMs += ended - started;
        disposeTensorMap(materialized);
      }

      const stats = cache.getStats();
      rows.push({
        benchmark: "cache_sweep",
        backend: backend.label,
        seq_len: seqLen,
        prefill_update_ms: prefillUpdateMs,
        prefill_materialize_ms: prefillMaterializeMs,
        decode_update_per_step_ms: (updateMs - prefillUpdateMs) / decodeSteps,
        decode_materialize_per_step_ms: (materializeMs - prefillMaterializeMs) / decodeSteps,
        total_ms: updateMs + materializeMs,
        packed_bytes: stats.packed_bytes,
        dense_bytes: stats.dense_bytes,
        compression_ratio: stats.packed_bytes ? stats.dense_bytes / stats.packed_bytes : 1,
        compressed_blocks: stats.compressed_blocks ?? null,
      });
    }
  }

  return rows;
}

async function runBrowserBenchmark({ modelId = DEFAULT_MODEL_ID, cases, runs = 1, sweepConfigs }) {
  await ensureLoaded(modelId);
  const normalizedCases = cases ?? [];
  const normalizedSweepConfigs = sweepConfigs ?? [];
  const results = [];

  for (let caseIndex = 0; caseIndex < normalizedCases.length; caseIndex += 1) {
    const benchmarkCase = normalizedCases[caseIndex];
    const encodedPrompt = await buildPromptInputs(modelId, benchmarkCase.prompt);

    post({
      status: "phase",
      message: `Warming up ${benchmarkCase.label} (${caseIndex + 1}/${normalizedCases.length})...`,
    });
    const warmup = await model.generate({
      ...encodedPrompt,
      max_new_tokens: Math.min(benchmarkCase.maxNewTokens ?? 32, 8),
      do_sample: false,
      return_dict_in_generate: true,
      cache_implementation: "dynamic",
    });
    await disposeGenerationResult(warmup);

    post({
      status: "phase",
      message: `Running dynamic baseline for ${benchmarkCase.label}...`,
    });
    const dynamic = await runCase(encodedPrompt, {
      runs,
      maxNewTokens: benchmarkCase.maxNewTokens ?? 32,
      cacheImplementation: "dynamic",
    });

    const sweepResults = [];
    for (let configIndex = 0; configIndex < normalizedSweepConfigs.length; configIndex += 1) {
      const cacheConfig = normalizedSweepConfigs[configIndex];
      post({
        status: "phase",
        message: `Running ${cacheConfig.label} on ${benchmarkCase.label} (${configIndex + 1}/${normalizedSweepConfigs.length})...`,
      });
      const turboquant = await runCase(encodedPrompt, {
        runs,
        maxNewTokens: benchmarkCase.maxNewTokens ?? 32,
        cacheImplementation: "turboquant",
        cacheConfig,
      });
      const prefixChars = commonPrefixLength(dynamic.output, turboquant.output);
      sweepResults.push({
        cacheConfig,
        dynamic,
        turboquant,
        comparison: {
          exactMatch: dynamic.output === turboquant.output,
          prefixAgreementChars: prefixChars,
          prefixAgreementRatio:
            dynamic.output.length > 0 ? prefixChars / dynamic.output.length : 1,
          speedRatio: dynamic.averageMs / turboquant.averageMs,
          compressionRatio:
            turboquant.cacheStats?.packed_bytes && turboquant.cacheStats?.dense_bytes
              ? turboquant.cacheStats.dense_bytes / turboquant.cacheStats.packed_bytes
              : null,
        },
      });
    }

    results.push({
      case: benchmarkCase,
      dynamic,
      sweepResults,
    });
  }

  return {
    kind: "browser_suite",
    modelId,
    runs,
    cases: normalizedCases,
    sweepConfigs: normalizedSweepConfigs,
    results,
    browser: {
      userAgent: navigator.userAgent,
      language: navigator.language,
    },
  };
}

async function runSyntheticBenchmark(data = {}) {
  const params = {
    seqLens: data.seqLens ?? [],
    decodeSteps: data.decodeSteps ?? 32,
    layers: data.layers ?? 4,
    numKvHeads: data.numKvHeads ?? 4,
    headDim: data.headDim ?? 128,
    runs: data.runs ?? 12,
    warmupRuns: data.warmupRuns ?? 3,
    turboConfigs: data.turboConfigs ?? [],
  };

  const materializeRows = await runMaterializeOnly(params);
  const sweepRows = await runCacheSweep(params);

  return {
    kind: "synthetic_cache",
    params,
    rows: [...materializeRows, ...sweepRows],
  };
}

self.addEventListener("message", async (event) => {
  const { type, data } = event.data ?? {};

  try {
    switch (type) {
      case "load":
        post({ status: "phase", message: "Loading benchmark model..." });
        await ensureLoaded(data?.modelId ?? DEFAULT_MODEL_ID);
        break;
      case "benchmark":
        post({ status: "phase", message: "Starting browser benchmark suite..." });
        post({ status: "complete", result: await runBrowserBenchmark(data ?? {}) });
        break;
      case "synthetic":
        post({ status: "phase", message: "Starting synthetic cache benchmarks..." });
        post({ status: "complete", result: await runSyntheticBenchmark(data ?? {}) });
        break;
      default:
        break;
    }
  } catch (error) {
    post({
      status: "error",
      error: error instanceof Error ? `${error.message}\n${error.stack ?? ""}` : String(error),
    });
  }
});
