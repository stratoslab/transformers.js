# Cover Note: What Was Changed, What the Paper Can Claim, and What Still Blocks Submission

Date: 2026-04-05
Author of this note: revision pass on behalf of Dhonam Pemba and Kwang Wei Sim

---

## 1. Summary of changes made in this revision

### Main manuscript (`draft-paper.md`)

**Title:** Changed from a subtitle-style two-liner to a single clean title that signals a feasibility study rather than a breakthrough paper:
> *TurboQuant in the Browser: A Feasibility Study of KV-Cache Compression for Gemma 4 on Chrome WebGPU*

**Abstract:** Fully rewritten. The revised abstract is more precise (includes TPS numbers, the 13.5× throughput gap on Long Context 2x, the 0.50–0.57× speed ratios), correctly frames the paper as a cache-wrapper implementation study, and names the central finding (rematerialization overhead scales with cache size) up front.

**Introduction (§1):** Substantially expanded. Added explicit motivation for browser inference as a deployment target, the specific mechanism by which the browser stack makes KV compression difficult (CPU-resident tensor defaults, IO-binding requirements), a clearer statement of what the paper does and does not attempt, and a tighter three-item contributions list.

**Background (§2):** Restructured to three subsections: (1) KV-cache memory cost (retained from original), (2) TurboQuant claims and their accelerator-native context (expanded with more specificity), (3) the browser stack and its constraints (new — now explicitly covers IO binding, tensor residency defaults, and the rematerialization problem as a structural consequence of the ONNX interface).

**System Design (§3):** Added §3.4 "What is not yet implemented" as an explicit section (was buried in a paragraph). Added §3.5 "The materialization loop and its cost" to explain the four-step cycle that is responsible for the performance penalty. This makes the architecture clearer for readers who may not have browser ML implementation background.

**Metrics (§4):** Added TTFT (§4.4) and decode throughput (§4.5) as explicit formal metric definitions. These were reported in the results but not defined in the original draft.

**Experimental Setup (§5):** Added §5.4 "Run budget and reproducibility" — this acknowledges the n=2 limitation explicitly, lists the specific fork commit SHAs, names the app-side commit, and notes that hardware was not recorded. These details are important for an honest methods section.

**Results (§6):** Major revision.
- Added dense baseline row to Table 1 so readers can see the absolute TPS numbers, not just ratios.
- Added cache size column (MB) to Table 2 to make the relationship between cache size and performance visible directly in the table.
- Added Table 3: Decode Throughput and TTFT by case and configuration. This is new data that was previously buried in the figures.
- Expanded §6.2 prose to include the worst-case quality result (35.8% prefix agreement on Policy Comparison + Mid Compression) explicitly.
- Added §6.5 "Configuration-level observations" with an explicit note that Mid Compression is a dominated operating point (worse speed *and* worse quality than Safe Default on 3/5 cases).
- Fixed figure references: all absolute machine paths (`/Users/dhonampemba/...`) replaced with relative paths (`figures/filename.png`).
- Embedded figure captions as proper prose in the Results section.

**Discussion (§7):** Substantially strengthened.
- Added §7.2 "Quantitative evidence for the rematerialization bottleneck" which presents the key-numbers argument: Risk Summary cache = 1.5 MB → 1.28× throughput gap; Long Context 2x cache = 43.7 MB → 13.5× throughput gap. This is the most important new piece of analysis in the paper.
- Added §7.3 "The configuration landscape" with the Mid Compression dominance observation.
- Renamed §7.2 (original) → §7.4 "What the implementation establishes."

**Limitations (§8):** Completely rewritten. The original was a single paragraph. The revised section is a numbered list of 8 distinct limitations with clear explanations, covering: implementation fidelity, run budget, single environment, metric scope, quality reference (baseline vs. ground truth), prompt coverage, single model, and hardware transparency. A companion file `limitations-and-threats.md` provides extended rationale for each item.

**Future Work (§9):** Added as a new section with seven specific research directions: GPU-resident cache via IO binding, selective rematerialization, closer TurboQuant reproduction, compressed attention paths, statistical robustness, multi-environment evaluation, and richer quality evaluation.

**Conclusion:** Tightened. The conclusion now names the key quantitative finding (0.74 tokens/s vs 9.98 tokens/s on Long Context 2x) and frames the paper's contribution as a negative-but-informative baseline that narrows the engineering problem.

**Data Availability:** Paths fixed from absolute machine paths to relative filenames.

**References:** Consolidated. The original had three separate references to the TurboQuant paper (blog, OpenReview, arXiv). These are now two entries: the blog post (ref 1) and the technical paper with both URLs (ref 2). All other references retained. A note in the submission checklist flags that author names should be added to ref 2.

### New supporting files

- **`figure-captions.md`** — formal captions for all six figures, suitable for use in a journal submission.
- **`limitations-and-threats.md`** — extended prose on each limitation with reviewer-response notes and a severity-vs-fixability summary table.
- **`submission-checklist.md`** — full Preprints.org readiness checklist with a claims audit table that verifies every headline number against the CSV data.
- **`cover-note.md`** — this file.

---

## 2. The strongest defensible claims from the current evidence

These claims are directly supported by `tables/config_summary.csv` and `tables/benchmark_rows.csv`:

1. **The implementation works.** A TurboQuant-inspired compressed KV-cache path runs end-to-end inside browser-based Gemma 4 inference on Chrome WebGPU, implemented as a fork of `transformers.js` with a pluggable cache abstraction.

2. **Compression is real but modest.** Average KV-cache compression ratios of 1.29–1.34× are achieved. The best observed single-case compression is 1.57× (Long Context 2x, Key Heavy). This is substantially below the sub-4-bit story in the TurboQuant paper but is a genuine, measurable reduction.

3. **The compressed path is consistently slower than the dense baseline.** Average speed ratios of 0.495–0.573× are observed across all five cases and three configurations. No exception was found.

4. **The throughput gap scales with cache size.** Risk Summary (1.5 MB cache): TurboQuant achieves ~78% of baseline TPS. Long Context 2x (43.7 MB cache): TurboQuant achieves ~7.4% of baseline TPS. This ~10× difference in relative throughput as cache size grows ~29× is strong evidence for per-step reconstruction overhead scaling with cached sequence length.

5. **Quality is context-length-dependent.** Long-context prompts retain 92–97% prefix agreement with the baseline. Short structured prompts (Operations Checklist, Policy Comparison) are more sensitive, reaching as low as 35.8% prefix agreement under Mid Compression.

6. **Mid Compression is a dominated operating point.** On 3/5 prompt cases, Mid Compression is both slower than Safe Default and produces worse quality, while achieving only marginally better compression. This is a configuration guidance finding.

7. **The bottleneck is the runtime, not the quantizer.** The pattern of results — severe overhead scaling with cache size, near-identical TTFT for short contexts, large throughput collapse at long contexts — is consistent with per-step dense rematerialization as the dominant cost. Tuning the quantizer parameters (bit widths, residual window) is unlikely to fix this without also addressing the ONNX interface boundary.

---

## 3. What the evidence does NOT support

These claims should not appear in the manuscript:

- "TurboQuant improves browser LLM inference speed" — false in the current implementation.
- "TurboQuant preserves output quality in the browser" — quality is preserved only on long-context tasks, and even there exact match is 0/5.
- "The implementation achieves the compression levels described in the TurboQuant paper" — no: 1.29–1.57× versus the ~10× implied by 3-bit compression.
- "Results generalize to other browsers, GPUs, or models" — untested.

---

## 4. What still blocks true submission quality

The paper is significantly more polished than the original draft and is defensible as a preprint. However, the following gaps remain between the current state and a paper that would be competitive at a peer-reviewed venue:

### Blocking for Preprints.org submission

| Item | What's needed |
|---|---|
| File format | Convert `.md` to Word or LaTeX |
| Author confirmation | Both coauthors must review and approve |
| Repo accessibility | Fork and app repos must be publicly accessible |
| TurboQuant author names | Look up and add to References 1/2 |
| Hardware metadata | Record GPU, browser version, OS for the benchmark environment |
| Exact prompts | Deposit actual prompt strings, not just case labels |
| Reproduction instructions | Write `REPRODUCE.md` |

### Blocking for a competitive conference/journal paper

| Item | What's needed |
|---|---|
| Run count | Rerun with ≥5 runs per config to get variance estimates |
| Semantic quality metrics | Add task-specific correctness scoring or human eval |
| Multi-environment | Benchmark on ≥2 different GPU/browser combinations |
| Broader prompt suite | Add ≥10 diverse prompt types across domains |
| Comparison baselines | Compare against token eviction or sliding-window alternatives |
| Closer TurboQuant reproduction | Better PolarQuant + QJL to make algorithmic comparison fairer |

---

## 5. Recommended framing

The best short description of this paper for a submission cover letter or abstract:

> We present an experimental study of TurboQuant-inspired KV-cache compression for browser-based LLM inference using `transformers.js` and Chrome WebGPU. Our results are negative on latency — the compressed path is 1.75–2× slower than the dense baseline — but informative on mechanism: the overhead scales with KV-cache size in a way consistent with per-step full-tensor rematerialization, and we provide quantitative characterisation of this effect. The paper contributes an open implementation, a benchmark harness, and a concrete identification of the system-level bottleneck that future browser cache designs must address.

This framing is honest, defensible, and positions the paper as a systems contribution with clear negative results and a useful engineering lesson.
