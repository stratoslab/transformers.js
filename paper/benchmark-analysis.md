# TurboQuant Benchmark Analysis

Source data: [Chrome Benchmarkv2.txt](./Chrome%20Benchmarkv2.txt)

## Scope

This analysis summarizes the latest Chrome WebGPU benchmark sweep for:

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

### 1. The current result is no longer uniformly negative on latency

Average speed ratios by config:

- `Safe Default`: `1.054x`
- `Mid Compression`: `1.077x`
- `Key Heavy`: `1.039x`

Interpretation:

- A value above `1.0x` means TurboQuant is faster than the dynamic baseline on average across the five prompt cases.
- This does **not** mean TurboQuant is faster everywhere.
- The sweep shows a context-length crossover:
  - all three short cases are still slower than baseline (`0.731x` to `0.942x`)
  - `Long Context 2x` is faster for all three configs (`1.702x` to `2.143x`)
  - `Long Context 1x` is faster only for `Safe Default` (`1.093x`) and slower for the other two configs

The correct reading is:

- **TurboQuant is slower when cache pressure is low**
- **TurboQuant becomes competitive or faster once the context is large enough**

### 2. Compression is strongly context-dependent, not uniformly positive

Average compression ratios by config:

- `Safe Default`: `0.967x`
- `Mid Compression`: `1.044x`
- `Key Heavy`: `0.992x`

Best observed compression point:

- `Long Context 2x` + `Key Heavy`: `1.512x`

Worst observed compression point:

- `Risk Summary` + all configs: `0.667x`

Interpretation:

- On short prompts, the packed representation can be larger than dense.
- Compression only pays once the cache grows enough for the fixed packing overhead to amortize.
- The long-context cases are where the method becomes meaningfully compression-positive:
  - `Long Context 1x`: `1.389x` to `1.447x`
  - `Long Context 2x`: `1.446x` to `1.512x`

### 3. Quality is much stronger than in the earlier benchmark run

Average prefix agreement by config:

- `Safe Default`: `100.000%`
- `Mid Compression`: `95.277%`
- `Key Heavy`: `97.950%`

Exact matches:

- `Safe Default`: `5 / 5`
- `Mid Compression`: `3 / 5`
- `Key Heavy`: `3 / 5`

Worst observed prefix agreement:

- `Policy Comparison` + `Mid Compression`: `77.759%`

Interpretation:

- `Safe Default` now matches the dense baseline exactly on every case in this sweep.
- `Key Heavy` preserves exact output on all three short cases, but drifts on the two long-context cases.
- `Mid Compression` is still the least stable setting, but even its weakest case is materially stronger than in the earlier run.

## Per-case summary

### Risk Summary

- `Safe Default`: speed `0.942x`, compression `0.667x`, prefix `100.0%`
- `Mid Compression`: speed `0.889x`, compression `0.667x`, prefix `100.0%`
- `Key Heavy`: speed `0.886x`, compression `0.667x`, prefix `100.0%`

Takeaway:

- TurboQuant loses on both speed and compression for this small-cache case.
- There is no quality problem here; the issue is purely that the cache is too small for packing to pay off.

### Operations Checklist

- `Safe Default`: speed `0.779x`, compression `0.667x`, prefix `100.0%`
- `Mid Compression`: speed `0.756x`, compression `0.868x`, prefix `100.0%`
- `Key Heavy`: speed `0.731x`, compression `0.667x`, prefix `100.0%`

Takeaway:

- This is still a short-case loss for TurboQuant.
- `Mid Compression` is the only setting that avoids outright cache expansion here, but it remains slower than baseline.

### Policy Comparison

- `Safe Default`: speed `0.752x`, compression `0.667x`, prefix `100.0%`
- `Mid Compression`: speed `0.776x`, compression `0.851x`, prefix `77.8%`
- `Key Heavy`: speed `0.743x`, compression `0.667x`, prefix `100.0%`

Takeaway:

- The short comparative-reasoning case still favors the dense baseline on latency.
- `Mid Compression` again trades away quality without gaining enough speed or compression to justify it.

### Long Context 1x

- `Safe Default`: speed `1.093x`, compression `1.389x`, prefix `100.0%`
- `Mid Compression`: speed `0.820x`, compression `1.389x`, prefix `100.0%`
- `Key Heavy`: speed `0.762x`, compression `1.447x`, prefix `93.0%`

Takeaway:

- This is the first crossover point.
- `Safe Default` is slightly faster than baseline while also compressing the cache and preserving exact output.
- The more aggressive settings do not yet pay here.

### Long Context 2x

- `Safe Default`: speed `1.702x`, compression `1.446x`, prefix `100.0%`
- `Mid Compression`: speed `2.143x`, compression `1.446x`, prefix `98.6%`
- `Key Heavy`: speed `2.075x`, compression `1.512x`, prefix `96.7%`

Takeaway:

- This is the strongest positive result in the current dataset.
- All three TurboQuant settings beat the dense baseline on end-to-end latency.
- `Mid Compression` is the fastest here, while `Safe Default` is the cleanest quality-preserving point.

## Defensible claims right now

The current results support the following claims:

- A TurboQuant-inspired KV-cache path can run end-to-end inside browser-based Gemma 4 inference on Chrome WebGPU.
- The current implementation shows a **context-length crossover** rather than a uniform slowdown.
- TurboQuant is slower on short prompts with small caches, but can become faster on sufficiently long prompts.
- Cache compression is only beneficial once the cache is large enough; on short prompts the packed representation can exceed dense size.
- `Safe Default` is now a strong quality-preserving operating point in this sweep, with `5/5` exact matches.
- The benchmark still supports the systems claim that runtime behavior is dominated by the interaction between cache size and rematerialization overhead.

## Claims that are still not supported

The current data still does **not** support the following claims:

- “TurboQuant is faster than the dense baseline in general”  
  It is faster only once the prompt is long enough.
- “TurboQuant always reduces KV memory footprint”  
  It does not on short prompts.
- “The implementation is production-ready for browser Gemma 4 inference”  
  The run budget is still only `n=2`, and the evidence is from one Chrome/WebGPU environment.
- “The current implementation is a faithful reproduction of the published TurboQuant system”  
  It remains a TurboQuant-inspired browser approximation.

## Recommendation for the paper

The paper should now be framed as a **context-dependent tradeoff study**, not as a purely negative result.

The most defensible summary is:

- short-context browser inference still favors `DynamicCache`
- long-context browser inference can favor `TurboQuantCache`
- the crossover is driven by cache size and runtime overhead, not by a uniform algorithmic win

## Most promising current operating point

If one configuration must be used as the reference point for further work:

- `Safe Default` is the best general-purpose setting overall

Reason:

- It is the only configuration with `5/5` exact matches.
- It is the cleanest quality-preserving point on both long-context cases.
- It is already faster than baseline on `Long Context 1x` and strongly faster on `Long Context 2x`.
- It avoids the extra output drift of the more aggressive settings.
