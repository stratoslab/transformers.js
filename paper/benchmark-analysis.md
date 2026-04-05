# TurboQuant Benchmark Analysis

Source data: [turboquant-benchmark.json](./turboquant-benchmark.json)

## Scope

This analysis summarizes the Chrome WebGPU benchmark sweep for:

- Model: `onnx-community/gemma-4-E2B-it-ONNX`
- Runs per point: `2`
- Cases: `5`
- TurboQuant configs:
  - `Safe Default` = `b_key=4`, `b_value=8`, `residual_length=64`
  - `Mid Compression` = `b_key=4`, `b_value=8`, `residual_length=48`
  - `Key Heavy` = `b_key=3`, `b_value=8`, `residual_length=64`

Metrics reported:

- speed ratio = `dynamic average latency / turboquant average latency`
- compression ratio = `dense_bytes / packed_bytes`
- prefix agreement ratio = common output prefix length divided by dynamic output length
- exact match count
- TTFT
- decode tokens/sec

## Main findings

### 1. Current TurboQuant variants are slower than baseline in every tested condition

Average speed ratios by config:

- `Safe Default`: `0.538x`
- `Mid Compression`: `0.495x`
- `Key Heavy`: `0.573x`

Interpretation:

- A value below `1.0x` means TurboQuant is slower than the dynamic baseline.
- None of the tested operating points improved latency.
- `Key Heavy` is the least bad configuration on average, but it is still materially slower.

### 2. Compression gains are modest, not paper-strong

Average compression ratios by config:

- `Safe Default`: `1.290x`
- `Mid Compression`: `1.336x`
- `Key Heavy`: `1.328x`

Best observed compression point:

- `Long Context 2x` + `Key Heavy`: `1.568x`

Interpretation:

- The current implementation does not produce a dramatic KV-cache reduction.
- Gains are real but small relative to the overhead being introduced.

### 3. Output quality is directionally preserved on long contexts, but not exact

Average prefix agreement by config:

- `Safe Default`: `83.712%`
- `Mid Compression`: `69.892%`
- `Key Heavy`: `77.784%`

Exact matches:

- `0 / 5` for every config

Best observed prefix agreement:

- `Long Context 2x` + `Safe Default`: `96.644%`

Worst observed prefix agreement:

- `Policy Comparison` + `Mid Compression`: `35.786%`

Interpretation:

- Long-context benchmark cases retain high textual overlap with baseline.
- Shorter structured reasoning and comparison prompts are much more fragile.
- Current results are not stable enough to support a strong “no quality loss” claim.

## Per-case summary

### Risk Summary

- `Safe Default`: speed `0.801x`, compression `1.099x`, prefix `84.1%`
- `Mid Compression`: speed `0.726x`, compression `1.180x`, prefix `78.9%`
- `Key Heavy`: speed `0.807x`, compression `1.109x`, prefix `84.3%`

Takeaway:

- Quality is acceptable-ish for this short task, but the compression benefit is too small to justify the slowdown.

### Operations Checklist

- `Safe Default`: speed `0.691x`, compression `1.180x`, prefix `71.9%`
- `Mid Compression`: speed `0.637x`, compression `1.249x`, prefix `45.6%`
- `Key Heavy`: speed `0.695x`, compression `1.200x`, prefix `60.3%`

Takeaway:

- Formatting-heavy checklist output is sensitive to compression.
- `Mid Compression` is especially unstable here.

### Policy Comparison

- `Safe Default`: speed `0.664x`, compression `1.197x`, prefix `73.2%`
- `Mid Compression`: speed `0.614x`, compression `1.264x`, prefix `35.8%`
- `Key Heavy`: speed `0.669x`, compression `1.219x`, prefix `54.7%`

Takeaway:

- Comparative reasoning is currently one of the weakest areas.
- This case is the clearest evidence against claiming broad quality preservation.

### Long Context 1x

- `Safe Default`: speed `0.210x`, compression `1.479x`, prefix `92.7%`
- `Mid Compression`: speed `0.181x`, compression `1.488x`, prefix `92.7%`
- `Key Heavy`: speed `0.370x`, compression `1.545x`, prefix `93.0%`

Takeaway:

- Output overlap is high on long context.
- However, latency collapses badly, which likely reflects reconstruction overhead dominating any KV savings.

### Long Context 2x

- `Safe Default`: speed `0.323x`, compression `1.496x`, prefix `96.6%`
- `Mid Compression`: speed `0.315x`, compression `1.501x`, prefix `96.5%`
- `Key Heavy`: speed `0.323x`, compression `1.568x`, prefix `96.6%`

Takeaway:

- This is the strongest quality case in the current dataset.
- It is also the clearest proof that the implementation is not yet performance-competitive.

## Defensible claims right now

The current results support the following claims:

- An experimental TurboQuant-style KV-cache path can run inside browser-based Gemma 4 inference on Chrome WebGPU.
- The implementation can reduce packed KV size relative to dense caching.
- Larger prompt contexts show higher textual agreement with baseline than shorter structured reasoning tasks.
- The current implementation does not improve latency and is slower than the baseline across all tested cases.

## Claims that are not supported yet

The current data does **not** support the following claims:

- “TurboQuant improves end-to-end latency in the browser”
- “TurboQuant achieves substantial memory reduction”
- “TurboQuant preserves output quality”
- “TurboQuant is production-ready for Gemma 4 WebGPU inference”

## Recommendation for a paper

If this work is written up now, it should be framed as:

- a browser implementation study,
- a negative or mixed empirical result,
- or an engineering report on the tradeoff frontier rather than a breakthrough result.

If the goal is a stronger research-style paper, the next required steps are:

1. Improve the codec so compression materially exceeds `~1.5x` without major quality drift.
2. Reduce dense reconstruction overhead, which currently destroys speed.
3. Expand the sweep to more operating points and more runs per point.
4. Add additional quality metrics beyond prefix agreement, such as semantic similarity and task-specific rubric scoring.
5. Benchmark across multiple browsers, GPUs, and machines.

## Most promising current operating point

If one configuration must be used as the reference point for further work:

- `Safe Default` is the best quality-oriented setting overall.

Reason:

- It has the strongest average prefix agreement (`83.712%`) across the sweep.
- Its long-context behavior is especially strong (`92.7%` to `96.6%` prefix agreement).
- It is still too slow, but it is the cleanest quality-preserving baseline for future optimization work.
