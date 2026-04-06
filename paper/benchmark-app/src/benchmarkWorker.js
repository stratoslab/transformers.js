import {
  AutoModelForCausalLM,
  AutoProcessor,
  AutoTokenizer,
  Gemma4ForConditionalGeneration,
  TextStreamer,
} from "@huggingface/transformers";

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
  const {
    runs,
    maxNewTokens,
    cacheImplementation,
    cacheConfig,
  } = options;

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

async function runBenchmark({
  modelId = DEFAULT_MODEL_ID,
  cases,
  runs = 1,
  sweepConfigs,
}) {
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

  post({
    status: "complete",
    result: {
      modelId,
      runs,
      cases: normalizedCases,
      sweepConfigs: normalizedSweepConfigs,
      results,
      browser: {
        userAgent: navigator.userAgent,
        language: navigator.language,
      },
    },
  });
}

self.addEventListener("message", async (event) => {
  const { type, data } = event.data ?? {};

  try {
    switch (type) {
      case "ping":
        post({
          status: "phase",
          message: `Worker ready (${typeof navigator !== "undefined" && "gpu" in navigator ? "WebGPU API visible" : "WebGPU API missing"})`,
        });
        break;
      case "load":
        post({ status: "phase", message: "Loading benchmark model..." });
        await ensureLoaded(data?.modelId ?? DEFAULT_MODEL_ID);
        break;
      case "benchmark":
        await runBenchmark(data ?? {});
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
