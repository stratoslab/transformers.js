# TurboQuant in the Browser:
## An Experimental KV-Cache Compression Path for Gemma 4 on Chrome WebGPU

**Dhonam Pemba**  
Stratos Lab  
dhonam@stratoslab.xyz

**Kwang Wei Sim**  
Stratos Lab  
kwang@stratoslab.xyz

## Abstract

KV-cache growth is a primary bottleneck for long-context autoregressive inference, and the problem is especially acute in the browser, where GPU memory budgets, CPU-GPU transfer costs, and runtime abstraction overhead all interact. TurboQuant has recently been presented as a training-free, low-bit vector compression method capable of compressing LLM KV caches to roughly 3 bits while preserving quality and improving accelerator-side performance. In this work, we ask a narrower systems question: can a TurboQuant-style KV-cache strategy be integrated into a browser `transformers.js` stack and deliver practical benefit for real WebGPU inference?

We answer this by implementing an experimental TurboQuant-inspired cache path in a fork of `transformers.js`, exposing it through the generation API, and evaluating it on `onnx-community/gemma-4-E2B-it-ONNX` in Chrome WebGPU. Our implementation adds a pluggable cache abstraction, a `TurboQuantCache` with low-bit packed storage, rotation-based preprocessing, key residual correction, and a dense residual window, plus a browser benchmark harness that compares dense and compressed caching on the same prompt suite.

The main result is mixed. The browser integration is feasible and reduces packed KV size relative to dense caching, but the current implementation is slower than the dense baseline across all tested configurations. The best current quality-oriented configuration achieves 83.7% average prefix agreement with the dense baseline and 1.29x average compression, but only 0.538x speed ratio relative to baseline. These findings suggest that browser-side KV compression is technically viable, yet strongly constrained by dense rematerialization and CPU-GPU data movement. The work therefore contributes an implementation study, a benchmark harness, and a set of concrete systems lessons for future GPU-resident browser cache designs.

**Keywords:** browser inference, WebGPU, `transformers.js`, KV cache, TurboQuant, Gemma 4, ONNX Runtime Web, LLM systems

## 1. Introduction

Browser-hosted LLM inference has become increasingly attractive for privacy, latency locality, offline operation, and deployment simplicity. The recent maturation of `transformers.js`, ONNX Runtime Web, and WebGPU means that medium-scale generative models can now run directly in Chrome-class environments without a server-side inference stack. However, the system budget for browser inference remains tight. One of the most difficult pressure points is the autoregressive key-value (KV) cache, whose size grows with context length and decode length.

Recent work on TurboQuant argues that KV-cache compression can be achieved at very low bitrates without quality loss, and that accelerator-side implementations can benefit from both lower memory cost and faster attention-time operations. Those claims are promising, but they do not transfer automatically to the browser. A browser stack based on `transformers.js` and ONNX Runtime Web is constrained by API boundaries, tensor materialization rules, and data movement between CPU and GPU memory.

This paper investigates whether a TurboQuant-style cache path can be made to work inside browser inference and what happens when it is benchmarked in the actual target environment: Chrome WebGPU with Gemma 4. Our goal is not to claim a full reproduction of the original TurboQuant algorithm. Instead, our goal is to build and evaluate an experimental browser implementation that is close enough to surface the real systems bottlenecks.

Our contributions are:

1. A fork of `transformers.js` that adds a pluggable cache abstraction for text generation and an experimental `TurboQuantCache`.
2. A Chrome WebGPU benchmark harness for Gemma 4 that reports latency, time-to-first-token (TTFT), decode throughput, cache-size statistics, and output agreement.
3. An empirical analysis showing that browser KV compression is feasible but presently dominated by dense reconstruction and CPU-GPU transfer overhead.

## 2. Background and Motivation

### 2.1 KV-cache cost

For a decoder-only transformer, the dense KV-cache memory can be approximated as

$$
M_{\text{dense}} \approx 2 \cdot L \cdot H_{kv} \cdot T \cdot D \cdot b,
$$

where:

- \(L\) is the number of layers,
- \(H_{kv}\) is the number of key-value heads,
- \(T\) is the cached sequence length,
- \(D\) is the head dimension,
- \(b\) is bytes per element,
- and the factor of 2 accounts for keys and values.

In browser inference, the relevant cost is not just memory footprint. The system also pays for:

- tensor allocation,
- tensor ownership and disposal,
- CPU-to-GPU transfer,
- GPU-to-CPU transfer,
- and model-boundary rematerialization.

### 2.2 TurboQuant

TurboQuant has been publicly described as a training-free two-stage vector compression method that combines a PolarQuant stage with a 1-bit QJL residual correction stage. Google’s public discussion frames it as a way to compress KV caches to about 3 bits while preserving model quality and enabling large accelerator-side gains.

The central question for this paper is not whether TurboQuant is effective on H100-class kernels. It is whether a TurboQuant-style strategy remains beneficial when forced through a browser generation stack in which the decoder still expects dense past-key-value tensors.

### 2.3 Browser stack constraints

`transformers.js` runs ONNX models in-browser using ONNX Runtime. The official docs expose `device: 'webgpu'` for browser acceleration and explicitly note that WebGPU remains experimental in some browser contexts. ONNX Runtime Web’s WebGPU guide also highlights a critical detail: tensors default to CPU residency, and explicit IO binding is required to keep tensors on GPU buffers across model invocations. This is particularly important for transformer workloads that repeatedly feed prior outputs into subsequent decoder calls.

This implies a browser-specific challenge. If a compressed cache is stored off the model boundary but must be fully rehydrated into dense tensors before each decode step, then the algorithmic savings may be offset by runtime overhead.

## 3. System Design

### 3.1 Design objective

We aimed to introduce a cache implementation seam into `transformers.js` without rewriting the entire generation stack. The resulting design has two cache modes:

- `dynamic`: the original dense cache path
- `turboquant`: an experimental compressed cache path

The guiding principle was to keep the public generation API stable enough that the browser benchmark could switch only the cache implementation while keeping the model, prompt, and decode settings fixed.

### 3.2 Cache abstraction

The fork adds a `PastKeyValues` abstraction with three main methods:

- `update(decoderResults, options)`
- `materialize(decoderFeeds)`
- `getStats()`

This changes generation from a dense-map assumption to an object-owned cache strategy. In effect, the decoder can emit standard `present.*` tensors, while the cache implementation decides how to store them and how to materialize them for the next call.

### 3.3 TurboQuant-inspired cache

The current `TurboQuantCache` includes the following components:

- low-bit packed storage for keys and values,
- optional Hadamard-style rotation when the head dimension is a power of two,
- key residual correction,
- a dense recent-token residual window,
- cache-size reporting through `packed_bytes` and `dense_bytes`.

This implementation is intentionally described as **TurboQuant-inspired** rather than a full TurboQuant reproduction. It does not yet provide a faithful PolarQuant stage, a full QJL estimator, compressed attention kernels, or GPU-native consumption of compressed KV tensors.

### 3.4 Materialization model

The compressed cache is stored between decode steps, but before the next decoder call it is rematerialized into dense `past_key_values.*` tensors. The update-materialize loop is therefore:

1. Decoder produces `present.*`.
2. `TurboQuantCache.update()` ingests and compresses the tensors.
3. `TurboQuantCache.materialize()` reconstructs dense tensors.
4. The ONNX decoder consumes those dense tensors.

This is the central compromise in the current implementation. It makes the experiment possible without changing the ONNX graph interface, but it also introduces the most likely source of performance loss.

## 4. Metrics and Equations

We report three primary comparative metrics.

### 4.1 Speed ratio

$$
S = \frac{t_{\text{dynamic}}}{t_{\text{turbo}}},
$$

where \(t_{\text{dynamic}}\) is the average dense-cache latency and \(t_{\text{turbo}}\) is the average compressed-cache latency. A value greater than 1 would indicate that TurboQuant is faster than the dense baseline.

### 4.2 Compression ratio

$$
C = \frac{B_{\text{dense}}}{B_{\text{packed}}},
$$

where \(B_{\text{dense}}\) is the size of the equivalent dense cache and \(B_{\text{packed}}\) is the packed cache footprint reported by the compressed path.

### 4.3 Prefix agreement

Let \(y_d\) be the dense-cache output string, \(y_t\) the TurboQuant output string, and \(\mathrm{LCP}(y_d, y_t)\) the length of their longest common prefix. Then

$$
P = \frac{\mathrm{LCP}(y_d, y_t)}{|y_d|}.
$$

This is a lightweight textual agreement metric rather than a semantic correctness metric. It is useful for quick sweep analysis but should not be treated as a substitute for full task-level evaluation.

### 4.4 Approximate compressed-cache model

The current implementation can be viewed as a two-part cache:

$$
M_{\text{turbo}} \approx 2 \cdot L \cdot H_{kv} \cdot
\left[
T_r \cdot D \cdot b_{\text{fp}}
+
(T - T_r) \cdot D \cdot b_{\text{eff}}
\right],
$$

where:

- \(T_r\) is the dense residual-window length,
- \(b_{\text{fp}}\) is the dense residual precision cost,
- \(b_{\text{eff}}\) is the effective packed bitrate plus metadata overhead.

This formulation highlights a practical tension: increasing \(T_r\) protects quality but erodes compression, while decreasing \(T_r\) raises compression pressure and can increase output drift.

## 5. Experimental Setup

### 5.1 Model and runtime

We benchmarked:

- model: `onnx-community/gemma-4-E2B-it-ONNX`
- runtime: forked `transformers.js`
- execution environment: Chrome WebGPU
- model dtype: `q4f16`

### 5.2 Prompt suite

The benchmark suite consists of five prompt categories:

- Risk Summary
- Operations Checklist
- Policy Comparison
- Long Context 1x
- Long Context 2x

The first three cases are short structured enterprise tasks. The latter two are longer context-driven prompts intended to stress cache behavior under larger sequence lengths.

### 5.3 Configurations

We evaluated three TurboQuant operating points:

| Configuration | \(b_k\) | \(b_v\) | Residual length |
|---|---:|---:|---:|
| Safe Default | 4 | 8 | 64 |
| Mid Compression | 4 | 8 | 48 |
| Key Heavy | 3 | 8 | 64 |

Each point was run against the same dense baseline under deterministic generation.

## 6. Results

### 6.1 Aggregate configuration summary

Table 1 summarizes the average outcome across all five benchmark cases.

| Configuration | Avg. speed ratio | Avg. compression | Avg. prefix agreement | Exact matches |
|---|---:|---:|---:|---:|
| Safe Default | 0.538x | 1.290x | 83.7% | 0 / 5 |
| Key Heavy | 0.573x | 1.328x | 77.8% | 0 / 5 |
| Mid Compression | 0.495x | 1.336x | 69.9% | 0 / 5 |

The most important fact is that **all three configurations remain slower than the dense baseline**. The best speed ratio in aggregate is 0.573x, meaning the compressed path still takes substantially more time end-to-end than the dense path.

At the same time, compression gains are real but limited. The average compression ratios are clustered around 1.29x to 1.34x, which is far below the much stronger low-bit compression story reported for TurboQuant on accelerator-native implementations.

### 6.2 Per-case behavior

Table 2 highlights the per-case pattern.

| Case | Best speed ratio | Best compression | Best prefix agreement |
|---|---:|---:|---:|
| Risk Summary | 0.807x | 1.180x | 84.3% |
| Operations Checklist | 0.695x | 1.249x | 71.9% |
| Policy Comparison | 0.669x | 1.264x | 73.2% |
| Long Context 1x | 0.370x | 1.545x | 93.0% |
| Long Context 2x | 0.323x | 1.568x | 96.6% |

The short structured prompts are relatively tolerant from a latency standpoint, though still slower than baseline. The long-context prompts show the best textual overlap and the strongest compression, but they also experience the worst latency collapse. This pattern is exactly what one would expect if the system is paying large rematerialization and movement costs while processing larger caches.

### 6.3 Figures

The generated plots in the paper package provide the clearest visual summary:

- Speed-quality frontier: [speed_quality_frontier.png](/Users/dhonampemba/Development/canton/Stratos-Gemma-4/transformers.js/paper/figures/speed_quality_frontier.png)
- Prefix-agreement heatmap: [heatmap_prefix_agreement.png](/Users/dhonampemba/Development/canton/Stratos-Gemma-4/transformers.js/paper/figures/heatmap_prefix_agreement.png)
- Speed-ratio heatmap: [heatmap_speed_ratio.png](/Users/dhonampemba/Development/canton/Stratos-Gemma-4/transformers.js/paper/figures/heatmap_speed_ratio.png)
- Compression heatmap: [heatmap_compression_ratio.png](/Users/dhonampemba/Development/canton/Stratos-Gemma-4/transformers.js/paper/figures/heatmap_compression_ratio.png)
- Latency bars: [latency_bars.png](/Users/dhonampemba/Development/canton/Stratos-Gemma-4/transformers.js/paper/figures/latency_bars.png)
- TTFT and throughput bars: [ttft_tps_bars.png](/Users/dhonampemba/Development/canton/Stratos-Gemma-4/transformers.js/paper/figures/ttft_tps_bars.png)

Three observations stand out:

1. The quality-oriented point is `Safe Default`, which has the highest average prefix agreement.
2. The more aggressive `Mid Compression` setting raises compression only slightly but degrades both latency and textual agreement.
3. The long-context cases show high prefix agreement but severe slowdown, which is consistent with reconstruction overhead dominating the decode loop.

## 7. Discussion

### 7.1 Why the browser result differs from the TurboQuant headline

The public TurboQuant story is centered on:

- low-bit vector compression,
- efficient similarity estimation after quantization,
- and accelerator-friendly implementation.

Our browser result is mediated by a very different path:

- low-bit storage between decode steps,
- CPU-side packing and unpacking,
- dense reconstruction before every ONNX decoder call,
- and browser-managed WebGPU execution.

This makes the browser system more of a **runtime integration experiment** than a direct reproduction of the original accelerator setting. The result is not that TurboQuant is ineffective in principle; the result is that a cache-wrapper-only implementation is not enough to win in this environment.

### 7.2 What the implementation nevertheless proves

The work establishes three useful points.

First, `transformers.js` can support pluggable cache strategies for generation without requiring a total rewrite. Second, browser-based Gemma 4 inference can execute an experimental compressed KV-cache path end-to-end in Chrome WebGPU. Third, the dominant next bottleneck is now much clearer: the browser stack needs more GPU-resident cache handling and less full dense rematerialization.

### 7.3 Most plausible next optimization steps

The current results suggest a more promising next direction than simply tuning bit widths:

- keep more tensors GPU-resident using ONNX Runtime Web GPU-buffer outputs and IO binding,
- reduce or localize dense reconstruction,
- compress older cache segments more aggressively than the most recent window,
- and improve the fidelity of the TurboQuant-inspired stages.

If those system-level changes are not made, the browser implementation is unlikely to cross the threshold where compression benefits outweigh runtime overhead.

## 8. Limitations

This study has several limitations.

First, the current implementation is not a full PolarQuant + QJL reproduction. Second, the benchmark set is intentionally focused and small. Third, prefix agreement is only a lightweight proxy for output similarity; it does not substitute for semantic or task-specific correctness scoring. Fourth, all reported browser experiments are in Chrome WebGPU and do not yet establish cross-browser or cross-hardware generality. Finally, the current work reports mixed results rather than a competitive production configuration.

## 9. Conclusion

We presented an experimental browser implementation of a TurboQuant-style KV-cache path for `transformers.js` and evaluated it on Gemma 4 in Chrome WebGPU. The main result is mixed but informative: compressed caching is feasible in-browser and reduces packed KV size, but the current implementation remains slower than dense caching and does not yet justify strong quality-preservation claims.

The practical contribution of this work is therefore not a new state-of-the-art browser inference result. It is a clear experimental baseline, a forked runtime path, and a benchmark harness that expose the true systems bottlenecks of browser KV compression. The evidence suggests that future progress will depend less on simple quantizer tuning and more on GPU-resident cache handling, lower rematerialization cost, and tighter integration with the underlying runtime.

## Data Availability

The current paper package includes the benchmark export and derived assets:

- [turboquant-benchmark.json](/Users/dhonampemba/Development/canton/Stratos-Gemma-4/transformers.js/paper/turboquant-benchmark.json)
- [benchmark_rows.csv](/Users/dhonampemba/Development/canton/Stratos-Gemma-4/transformers.js/paper/tables/benchmark_rows.csv)
- [config_summary.csv](/Users/dhonampemba/Development/canton/Stratos-Gemma-4/transformers.js/paper/tables/config_summary.csv)
- [case_summary.csv](/Users/dhonampemba/Development/canton/Stratos-Gemma-4/transformers.js/paper/tables/case_summary.csv)

The software artifacts are described in:

- [implementation-methods.md](/Users/dhonampemba/Development/canton/Stratos-Gemma-4/transformers.js/paper/implementation-methods.md)

## Conflicts of Interest

The authors declare no conflicts of interest.

## AI Use Disclosure

AI-assisted tools were used during implementation and drafting support for this project. The human authors reviewed, edited, and take responsibility for the final manuscript text, code changes, benchmark design, and interpretation.

## References

1. Google Research. *TurboQuant: Redefining AI Efficiency with Extreme Compression*. Available at: <https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/>
2. TurboQuant OpenReview page. *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*. Available at: <https://openreview.net/forum?id=tO3ASKZlok>
3. TurboQuant arXiv PDF. Available at: <https://arxiv.org/pdf/2504.19874>
4. Hugging Face. *Transformers.js documentation*. Available at: <https://huggingface.co/docs/transformers.js/index>
5. Hugging Face. *Cache strategies*. Available at: <https://huggingface.co/docs/transformers/en/kv_cache>
6. ONNX Runtime. *Using the WebGPU Execution Provider*. Available at: <https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html>
7. ONNX Runtime. *Configuration reference*. Available at: <https://onnxruntime.ai/docs/genai/reference/config.html>
8. MDN Web Docs. *WebGPU API*. Available at: <https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API>
9. Preprints.org. *Instructions for Authors*. Available at: <https://www.preprints.org/instructions-for-authors>
10. Preprints.org. *Subjects*. Available at: <https://www.preprints.org/subject>
