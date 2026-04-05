# Figure Captions

Formal captions for the six PNG figures in `figures/`. These captions are suitable for inclusion in a Preprints.org manuscript submission. They are reproduced in-line in `draft-paper.md` and collected here for ease of reference.

---

## Figure 1 — `speed_quality_frontier.png`

**Speed-quality frontier for all 15 configuration–case benchmark points.** The horizontal axis is the end-to-end speed ratio relative to the dense baseline (a ratio of 1.0 would match the baseline; values below 1.0 are slower). The vertical axis is the prefix agreement with the dense-baseline output (1.0 = identical prefix; lower = earlier divergence). Each point represents one TurboQuant configuration (Safe Default, Key Heavy, or Mid Compression) applied to one of the five prompt cases. The dashed vertical line at x = 1.0 marks the dense-baseline speed level; no point falls to the right of this line, confirming that no tested configuration outperforms the dense baseline on latency. Long-context cases cluster in the upper-left (high prefix agreement, severe slowdown); short structured cases cluster toward the lower-right (less slowdown, more output drift).

---

## Figure 2 — `heatmap_compression_ratio.png`

**Compression ratio heatmap (case × configuration).** Rows are the five benchmark prompt cases; columns are the three TurboQuant configurations (Safe Default, Mid Compression, Key Heavy). Cell values are the compression ratio $C = B_{\text{dense}} / B_{\text{packed}}$, where values greater than 1.0 indicate that the packed cache is smaller than the dense equivalent. Compression improves substantially for long-context cases (Long Context 1x: 1.48–1.55×; Long Context 2x: 1.50–1.57×) relative to short structured cases (Risk Summary: 1.10–1.18×; Operations Checklist: 1.18–1.25×; Policy Comparison: 1.20–1.26×). The Key Heavy configuration achieves the strongest compression on long-context cases.

---

## Figure 3 — `heatmap_prefix_agreement.png`

**Prefix agreement heatmap (case × configuration).** Rows are the five benchmark prompt cases; columns are the three TurboQuant configurations. Cell values are the prefix agreement ratio $P = \mathrm{LCP}(y_d, y_t) / |y_d|$, expressed as a percentage. Long-context cases retain high textual agreement with the baseline output (92–97%). Short structured cases show more variable results: Risk Summary achieves 79–84%; Operations Checklist 46–72%; Policy Comparison 36–73%. The Mid Compression configuration is consistently the worst on quality, reaching as low as 35.8% on Policy Comparison — the most severe quality divergence in the entire sweep.

---

## Figure 4 — `heatmap_speed_ratio.png`

**Speed ratio heatmap (case × configuration).** Rows are the five benchmark prompt cases; columns are the three TurboQuant configurations. Cell values are the speed ratio $S = t_{\text{dynamic}} / t_{\text{turbo}}$, where values below 1.0 indicate that TurboQuant is slower than the dense baseline. All 15 cells fall below 1.0. Short cases with small caches (Risk Summary, ~1.5 MB) have the best ratios (0.73–0.81×). Long-context cases with large caches (Long Context 2x, ~43.7 MB) have the worst (0.32–0.33×). The gradient from short to long cases is consistent with per-step reconstruction overhead scaling with cached sequence length.

---

## Figure 5 — `latency_bars.png`

**Absolute generation latency (ms) by case and configuration.** Grouped bars show the dense baseline and the three TurboQuant configurations for each of the five prompt cases. The logarithmic scale (if used) or the absolute axis highlights the order-of-magnitude difference between short-context and long-context cases. For Long Context 2x, TurboQuant generation takes approximately 171–176 seconds under all three configurations, versus approximately 55 seconds for the dense baseline. For Risk Summary, the difference is measured in hundreds of milliseconds. Note that results shown are averages over 2 runs per configuration.

---

## Figure 6 — `ttft_tps_bars.png`

**Time-to-first-token (ms, top panel) and decode throughput (tokens/s, bottom panel) by case and configuration.** TTFT for short-context cases (≤2.2 MB cache) is nearly identical across all configurations (≈260 ms), reflecting the absence of any cached KV state at the first decode step. For long-context cases, TTFT is more variable: Long Context 2x shows slightly lower TTFT for TurboQuant configurations than for the dense baseline under all three settings (−6% to −11%), while Long Context 1x TTFT results are mixed. Decode throughput (bottom) shows the clearest signal: on long-context cases, TurboQuant throughput collapses to 0.72–1.53 tokens/s versus 9.98–14.4 tokens/s for the dense baseline. On short cases the gap is smaller (14.5–18.5 tokens/s TurboQuant versus 22.3–23.6 tokens/s baseline). These patterns are consistent with per-step rematerialization overhead dominating for large caches.
