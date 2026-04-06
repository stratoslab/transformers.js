# Figure Captions

Formal captions for the six PNG figures in `figures/`. These captions are suitable for inclusion in a Preprints.org manuscript submission. They are reproduced in-line in `draft-paper.md` and collected here for ease of reference.

---

## Figure 1 — `speed_quality_frontier.png`

**Speed-quality frontier for all 15 configuration–case benchmark points.** The horizontal axis is the end-to-end speed ratio relative to the dense baseline (a ratio of 1.0 matches the baseline; values above 1.0 are faster). The vertical axis is the prefix agreement with the dense-baseline output (1.0 = identical prefix; lower = earlier divergence). Each point represents one TurboQuant configuration (Safe Default, Key Heavy, or Mid Compression) applied to one of the five prompt cases. The dashed vertical line at x = 1.0 marks the dense-baseline speed level. In the latest Chrome sweep, the short structured cases remain left of this line, while the long-context cases move to or beyond it, showing a context-length crossover rather than a uniform slowdown.

---

## Figure 2 — `heatmap_compression_ratio.png`

**Compression ratio heatmap (case × configuration).** Rows are the five benchmark prompt cases; columns are the three TurboQuant configurations (Safe Default, Mid Compression, Key Heavy). Cell values are the compression ratio $C = B_{\text{dense}} / B_{\text{packed}}$, where values greater than 1.0 indicate that the packed cache is smaller than the dense equivalent. The latest Chrome sweep shows that compression is only beneficial at larger cache sizes: the three short cases range from `0.667x` to `0.868x`, while the two long-context cases range from `1.389x` to `1.512x`. The Key Heavy configuration achieves the strongest compression on the longest case.

---

## Figure 3 — `heatmap_prefix_agreement.png`

**Prefix agreement heatmap (case × configuration).** Rows are the five benchmark prompt cases; columns are the three TurboQuant configurations. Cell values are the prefix agreement ratio $P = \mathrm{LCP}(y_d, y_t) / |y_d|$, expressed as a percentage. The latest sweep is much stronger on quality than the earlier run: `Safe Default` is exact on all five cases, `Key Heavy` is exact on the three short cases, and even the weakest point (`Policy Comparison` under `Mid Compression`) remains at `77.8%` prefix agreement. The remaining quality loss is concentrated in the more aggressive configurations on longer or more structured prompts.

---

## Figure 4 — `heatmap_speed_ratio.png`

**Speed ratio heatmap (case × configuration).** Rows are the five benchmark prompt cases; columns are the three TurboQuant configurations. Cell values are the speed ratio $S = t_{\text{dynamic}} / t_{\text{turbo}}$, where values above 1.0 indicate that TurboQuant is faster than the dense baseline. The latest Chrome sweep shows a clear crossover: the three short cases remain below `1.0`, `Long Context 1x` is slightly above `1.0` for `Safe Default`, and `Long Context 2x` is above `1.7x` for all three settings. The pattern is consistent with cache size becoming large enough for the compressed path to amortize its extra machinery.

---

## Figure 5 — `latency_bars.png`

**Absolute generation latency (ms) by case and configuration.** Grouped bars show the dense baseline and the three TurboQuant configurations for each of the five prompt cases. The latest Chrome sweep reverses the earlier long-context pattern: for the three short cases, TurboQuant remains slower by hundreds to low thousands of milliseconds, but for `Long Context 2x` the compressed path falls to roughly `53.9–67.8 s` versus `115.5 s` for the dense baseline. Note that results shown are averages over `2` runs per configuration.

---

## Figure 6 — `ttft_tps_bars.png`

**Time-to-first-token (ms, top panel) and decode throughput (tokens/s, bottom panel) by case and configuration.** TTFT for short-context cases remains tightly clustered around `330–356 ms`, reflecting the absence of any cached KV state at the first decode step. For long-context cases, TTFT is still large but is lower for TurboQuant than for the dense baseline on `Long Context 2x`, and mixed on `Long Context 1x`. Decode throughput shows the same crossover as the latency metric: short cases remain in the `12.3–16.0 tok/s` range and trail the dense baseline, while the longest case improves to `4.46–6.39 tok/s` under TurboQuant as end-to-end latency crosses over in TurboQuant's favor.
