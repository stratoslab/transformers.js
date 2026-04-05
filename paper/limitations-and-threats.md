# Limitations and Threats to Validity

This note expands the Limitations section of `draft-paper.md` with full prose and rationale for each limitation, suitable for use as a reference when responding to reviewers or revising the manuscript for a peer-reviewed venue.

---

## 1. Implementation fidelity: TurboQuant-inspired, not TurboQuant

The `TurboQuantCache` in the current fork is an *approximation* of the TurboQuant algorithm described in [1, 2]. It incorporates low-bit packed storage, an optional Hadamard-style rotation, and a residual correction for keys, but does not provide:

- a faithful PolarQuant rotation stage as specified in the paper;
- a full QJL residual estimator;
- compressed attention kernels;
- any mechanism to consume the compressed representation inside the ONNX attention computation.

The practical consequence is that the quality-compression tradeoff observed in this study is a property of a simplified implementation, not of TurboQuant proper. It is possible — though not demonstrated — that a closer reproduction would improve prefix agreement and compression simultaneously. Conversely, the performance penalty (rematerialization overhead) is structural and would remain even with a more faithful quantizer, unless the ONNX interface or execution model is also changed. Claims about TurboQuant specifically should therefore be framed as "a TurboQuant-inspired design" rather than a TurboQuant evaluation.

**Mitigation in the manuscript:** The paper consistently uses "TurboQuant-inspired" and "TurboQuant-style" terminology, and Section 3.4 explicitly lists what is not yet implemented.

---

## 2. Run budget: two runs per configuration

Each of the 15 configuration–case pairs (3 configurations × 5 cases) was benchmarked over **2 runs**. With n=2:

- no confidence interval can be meaningfully computed;
- variance cannot be estimated from the data;
- outlier runs cannot be identified or excluded;
- the reported mean could differ substantially from the true mean with only one additional run.

Some observed differences — in particular the TTFT variations on Long Context 1x (Safe Default: 8,778 ms, Key Heavy: 14,132 ms) and the absolute latency on long-context cases — may partly reflect run-to-run variance rather than stable systematic effects. The TPS and speed-ratio differences between configurations on short cases are also small enough that n=2 is insufficient to confirm them as statistically reliable.

**Implication for strong claims:** The per-case and per-configuration results in this paper should be treated as directional observations, not precise measurements. The aggregate patterns (all configurations slower than baseline, throughput collapse at large cache sizes) are robust because they are consistent across all 15 points. Individual cell values are not.

**Recommended fix:** Future benchmark runs should use a minimum of 5–10 runs per configuration. For a more complete study, a bootstrap resampling analysis over repeated runs would provide uncertainty estimates.

---

## 3. Single execution environment

All experiments were conducted in **Chrome WebGPU on a single machine**. The following sources of variation were not explored:

- **GPU model:** The hardware GPU adapter was not recorded in the benchmark export. Results are specific to the adapter under test; integrated graphics, discrete consumer GPUs, and server-class GPUs may show very different throughput, memory transfer speeds, and cache-operation timing.
- **Browser:** No cross-browser benchmarking was performed. Firefox Nightly and Safari (WebKit) have different WebGPU implementations that may produce different ONNX execution characteristics.
- **Operating system:** Not reported.
- **Browser version:** Not pinned in the benchmark export.

This single-environment design means that the absolute timing numbers (ms, tokens/s) are not portable to other systems, and the relative ratios (speed ratio, compression ratio) may change with different hardware or runtimes.

**Implication:** The paper's claims are scoped to Chrome WebGPU. Statements about "browser inference" in general should be understood as shorthand for this specific environment.

---

## 4. Quality evaluation: prefix agreement is not semantic correctness

The primary quality metric is **prefix agreement**, defined as the length of the longest common prefix between the TurboQuant output and the dense-baseline output, normalized by the length of the dense output. This metric has several well-known limitations:

- It gives full credit for character-level agreement up to the divergence point, regardless of what happens afterward.
- A high prefix agreement score is compatible with entirely wrong or hallucinated content after the agreement point.
- A low score may unfairly penalize an output that is semantically equivalent but expressed with different word choices or structure.
- It captures surface-form agreement with the dense baseline, not agreement with ground truth. The dense baseline is not validated as correct.

No semantic evaluation metrics (ROUGE, BERTScore, exact match on structured outputs, task-specific rubrics, or human preference ratings) were computed. The 0/5 exact-match rate confirms that no configuration exactly reproduces the baseline, but this is an extremely strict threshold.

**Implication:** Quality results in this paper should be understood as "degree of agreement with the dense baseline output," not as a measurement of task-level quality or semantic preservation.

---

## 5. Quality compared to dense baseline, not to ground truth

All quality comparisons are between the TurboQuant path and the dense-cache (`dynamic`) path. The dense path is the reference, but it is not itself validated as correct, high-quality, or consistent. If the dense path produces variable or incorrect outputs (which can happen with autoregressive models), then prefix agreement against the dense output is not the same as quality in any absolute sense.

For a more meaningful quality evaluation, a held-out reference dataset with human-annotated or rubric-scored expected outputs would be needed.

---

## 6. Limited prompt coverage

The benchmark suite contains **five prompt categories** covering enterprise text tasks. This set was not designed to be representative of any broader distribution of browser LLM use cases. It does not include:

- open-ended conversational generation;
- multi-turn dialogue with history;
- code generation;
- mathematical reasoning;
- multilingual inputs;
- adversarial or out-of-distribution inputs.

Generalization of quality and speed findings to these task types is not established. In particular, the strong quality results for long-context prompts (92–97% prefix agreement) may reflect properties of the specific long-context prompts chosen rather than a general benefit.

---

## 7. Single model

All experiments use **`onnx-community/gemma-4-E2B-it-ONNX` in q4f16 precision**. This is one model, one precision, and one model family. The KV-cache dimensionalities (head count, head dimension, layer count) specific to this model determine the cache sizes reported, which in turn determine the scale of the rematerialization overhead observed.

Other model families, sizes, architectures, or precision levels may exhibit different:

- cache-size scaling behavior;
- quantization sensitivity (quality degradation at low bit widths);
- TTFT and decode latency profiles.

---

## 8. Hardware transparency

The experimental hardware was not recorded in the benchmark output. This means readers cannot reproduce absolute latency numbers without equivalent hardware. The paper is transparent about this limitation in Section 5.4 and explicitly scopes timing claims to the tested environment.

---

## 9. No comparison with alternative cache strategies

The study benchmarks one TurboQuant-inspired approach against one dense baseline. It does not compare against:

- token eviction strategies (e.g., H2O, StreamingLLM);
- sliding-window attention;
- other quantization methods (e.g., KVQuant, INT4 uniform quantization);
- quantization applied only to keys, only to values, or with different per-layer granularity.

The relative merit of the TurboQuant-inspired approach versus these alternatives in the browser context is unknown.

---

## 10. Threat to internal validity: materialization model conflates compression and overhead

The current implementation conflates two effects that should ideally be separated:

1. The compression benefit (smaller packed representation reduces memory pressure and potentially GPU upload time).
2. The overhead cost (pack + unpack operations add CPU work; full dense reconstruction undoes the compression benefit for the ONNX call).

Because these effects are not measured independently in the current benchmark, it is not possible to determine precisely how much of the performance gap is due to CPU pack/unpack work versus GPU memory transfer costs versus ONNX graph execution overhead. This limits the precision of recommendations for future optimization.

---

## 11. Threat to external validity: preprint is not peer-reviewed

As of the current draft, this work is not peer-reviewed. The methods, analyses, and conclusions have not been independently evaluated. The benchmark results are derived from a single experimental run on a single machine by the implementation authors. Independent reproduction is encouraged.

---

## Summary table

| Limitation | Severity | Can be addressed in revision? |
|---|---|---|
| Partial TurboQuant reproduction | High (for algorithmic claims) | Partially — requires more implementation work |
| n=2 runs per configuration | High (for statistical claims) | Yes — rerun with more iterations |
| Single browser/GPU environment | Medium | Yes — add cross-env benchmarking |
| Prefix agreement only (no semantic eval) | High (for quality claims) | Partially — add task rubrics |
| Quality vs. baseline, not ground truth | Medium | Requires labelled test data |
| 5 prompts only | Medium | Add more task types |
| Single model | Medium | Add more models/sizes |
| No hardware spec | Low | Add to benchmark export |
| No comparison to alternatives | Medium | Add token-eviction baselines |
| Pack/unpack overhead not separated | Medium | Add instrumentation |
