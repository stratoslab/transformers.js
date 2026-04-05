# Implementation and Methods

Date: 2026-04-05

This note explains what was actually implemented for the browser TurboQuant experiment, how the code is structured, and how the benchmark was produced.

## Goal

The goal was not to bolt TurboQuant onto the app layer. The goal was to create a real experimental surface inside `transformers.js` so browser generation could choose between:

- the default dense cache path
- an experimental TurboQuant-style cache path

That required changes in two places:

- the forked `transformers.js` runtime
- the app-level Chrome benchmark harness

## Repositories involved

### 1. Forked runtime

Local path:

- `transformers.js/`

Remote:

- `https://github.com/stratoslab/transformers.js`

Key TurboQuant-related commits in the fork:

- `db06d73` Add pluggable generation cache and TurboQuant scaffold
- `4fab2e0` Pack TurboQuant cache tensors into low-bit storage
- `24c4c0b` Add rotation and residual correction to TurboQuant cache
- `dfbf815` Add TurboQuant cache stats and unit tests
- `0424f44` Expose cache stats from generate()
- `fa534d7` Add TurboQuant benchmark script and docs
- `3e268cc` Improve TurboQuant reconstruction and benchmark output
- `0b2b2d6` Fix DynamicCache base constructor
- `529939a` Fix GPU-safe cache stats and benchmark harness
- `337eb60` Tune TurboQuant defaults with dense residual window

### 2. Chrome benchmark app

Local path:

- root app in this repository

Key app-side commit:

- `7af609f` Add Chrome benchmark harness for TurboQuant

## What changed in `transformers.js`

### Cache abstraction

The most important architectural change is the new cache contract in:

- `packages/transformers/src/cache_utils.js`

The fork introduces:

- `PastKeyValues`
- `DynamicCache`
- `TurboQuantCache`

The cache contract supports three key operations:

- `update(decoderResults, options)`
- `materialize(decoderFeeds)`
- `getStats()`

This is the core design shift. Instead of assuming `past_key_values` is just a plain dense tensor map, generation can now hand control to a cache object with its own storage strategy.

### Generation integration

The generation path was patched in:

- `packages/transformers/src/models/modeling_utils.js`
- `packages/transformers/src/generation/configuration_utils.js`
- `packages/transformers/src/models/modeling_outputs.js`

The new behavior is:

- `cache_implementation` is now a generation option.
- `cache_implementation: "dynamic"` uses the dense baseline.
- `cache_implementation: "turboquant"` instantiates `TurboQuantCache`.
- `generate(..., { return_dict_in_generate: true })` returns `cache_stats`.

This made it possible to benchmark both paths through the same public generation API.

## What `TurboQuantCache` currently does

The current implementation is a TurboQuant-shaped approximation, not a faithful reproduction of the full paper algorithm.

### Implemented pieces

The current cache path includes:

- bit-packed low-bit storage for KV tensors
- optional Hadamard-style rotation when the head dimension is a power of two
- residual correction for keys
- a dense residual window controlled by `residual_length`
- dense tensor rematerialization before the next ONNX decoder call
- cache statistics reporting through `packed_bytes` and `dense_bytes`

The relevant implementation is primarily in:

- `packages/transformers/src/cache_utils.js`

### Not fully implemented

The current implementation does not yet provide:

- a fully faithful PolarQuant implementation
- a full QJL estimator as described in the paper
- compressed attention kernels
- GPU-native compressed cache consumption
- direct ORT/WebGPU support for compressed KV inputs

So the right description is:

- "TurboQuant-inspired browser cache implementation"

not:

- "full production TurboQuant port"

## Why dense rematerialization still exists

This is the central systems compromise in the current work.

`transformers.js` generation still ultimately has to feed the decoder model the tensors it expects. Today, that means the custom cache stores a compressed representation between steps, then reconstructs dense `past_key_values.*` tensors before the ONNX session runs.

That is why the current code uses a `materialize()` step. Conceptually:

1. Decoder produces `present.*` tensors.
2. `TurboQuantCache.update()` ingests and compresses them.
3. On the next decode step, `TurboQuantCache.materialize()` reconstructs dense tensors.
4. The ONNX session consumes those dense tensors as ordinary `past_key_values.*` inputs.

This is enough to test feasibility, but it is also the most likely reason performance still trails the baseline.

## Browser benchmark harness

The benchmark harness lives in the app repo:

- `src/BenchmarkApp.jsx`
- `src/benchmarkWorker.js`
- `src/benchmarkCases.js`
- `vite.config.js`

The important design decisions were:

- run the benchmark in a browser worker
- load the local forked `transformers.js` browser bundle, not the npm package
- test `dynamic` and `turboquant` on the same prompts and model
- use deterministic generation with `do_sample: false`
- collect latency, TTFT, throughput, cache stats, and output agreement

The benchmark worker uses:

- `Gemma4ForConditionalGeneration`
- `AutoProcessor`
- `AutoTokenizer`
- `TextStreamer`

For Gemma 4, the benchmark path loads:

- `onnx-community/gemma-4-E2B-it-ONNX`

and runs on:

- `device: "webgpu"`
- model dtype `q4f16`

The benchmark suite currently measures:

- end-to-end average latency
- average TTFT
- decode tokens per second
- packed vs dense cache bytes
- exact match
- prefix agreement ratio

## Current benchmark evidence

The current exported data is in:

- `paper/turboquant-benchmark.json`
- `paper/Chrome Benchmark.txt`

and the current analysis is summarized in:

- `paper/benchmark-analysis.md`

At the moment, the strongest supported findings are:

- the browser path works end-to-end on Chrome WebGPU
- the compressed path reduces packed KV size relative to the dense path
- the compressed path is slower than the dense baseline in the current sweep
- quality is directionally preserved on some long-context prompts but not stable enough for a "no quality loss" claim

## Why the current implementation underperforms

The likely reasons are structural:

- CPU-side packing and unpacking work adds overhead
- dense rematerialization before every decoder call adds overhead
- GPU download/upload costs can erase the memory savings
- the current quantizer is still an approximation of the paper algorithm
- conservative defaults keep more of the cache dense to protect output quality

The result is a browser implementation that is experimentally valid, but not yet competitive on latency.

## How this work helps anyway

Even with mixed benchmark results, the implementation is still useful research infrastructure.

It provides:

- a clear cache-extension seam inside `transformers.js`
- a benchmarkable browser path for future compression work
- concrete evidence that browser KV compression is bottlenecked by runtime data movement, not just quantizer design

That is valuable because it narrows future work from a broad question:

- "Can TurboQuant help in the browser?"

to a more precise one:

- "Can a browser runtime keep compressed KV state GPU-resident long enough for compression to pay off?"

## Immediate next steps

The most defensible next engineering steps are:

- integrate more of ONNX Runtime Web’s GPU-resident tensor path
- reduce full-cache dense rematerialization
- explore whether only older KV segments need aggressive compression
- improve the fidelity of the rotation and residual-correction stages
- rerun the benchmark across more prompts, more runs, and more hardware/browser combinations

## Code pointers

Fork runtime:

- `packages/transformers/src/cache_utils.js`
- `packages/transformers/src/models/modeling_utils.js`
- `packages/transformers/src/generation/configuration_utils.js`
- `packages/transformers/src/models/modeling_outputs.js`
- `packages/transformers/scripts/benchmark_turboquant.mjs`
- `packages/transformers/tests/utils/cache_utils.test.js`
- `packages/transformers/tests/utils/generation.test.js`

App-side benchmark harness:

- `src/BenchmarkApp.jsx`
- `src/benchmarkWorker.js`
- `src/benchmarkCases.js`
- `vite.config.js`
