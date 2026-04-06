# Cover Note: What Was Changed, What the Paper Can Claim, and What Still Blocks Submission

Date: 2026-04-06
Author of this note: revision pass on behalf of Dhonam Pemba and Kwang Wei Sim

---

## 1. Summary of changes made in this revision

### Main manuscript (`draft-paper.md`)

**Title:** Changed from a subtitle-style two-liner to a single clean title that signals a feasibility study rather than a breakthrough paper:
> *TurboQuant in the Browser: A Feasibility Study of KV-Cache Compression for Gemma 4 on Chrome WebGPU*

**Abstract:** Updated again for the latest Chrome benchmark run. The abstract now frames the result as a context-length crossover rather than a uniform slowdown, calls out the long-context speedups, and notes that compression is only beneficial once cache pressure is high enough.

**Introduction (§1):** Substantially expanded. Added explicit motivation for browser inference as a deployment target, the specific mechanism by which the browser stack makes KV compression difficult (CPU-resident tensor defaults, IO-binding requirements), a clearer statement of what the paper does and does not attempt, and a tighter three-item contributions list.

**Background (§2):** Restructured to three subsections: (1) KV-cache memory cost (retained from original), (2) TurboQuant claims and their accelerator-native context (expanded with more specificity), (3) the browser stack and its constraints (new — now explicitly covers IO binding, tensor residency defaults, and the rematerialization problem as a structural consequence of the ONNX interface).

**System Design (§3):** Added §3.4 "What is not yet implemented" as an explicit section (was buried in a paragraph). Added §3.5 "The materialization loop and its cost" to explain the four-step cycle that is responsible for the performance penalty. This makes the architecture clearer for readers who may not have browser ML implementation background.

**Metrics (§4):** Added TTFT (§4.4) and decode throughput (§4.5) as explicit formal metric definitions. These were reported in the results but not defined in the original draft.

**Experimental Setup (§5):** Added §5.4 "Run budget and reproducibility" — this acknowledges the n=2 limitation explicitly, lists the specific fork commit SHAs, names the app-side commit, and notes that hardware was not recorded. These details are important for an honest methods section.

**Results (§6):** Major revision.
- Updated Table 1 so the aggregate summary reflects the new crossover result rather than the earlier all-negative sweep.
- Added cache size column (MB) to Table 2 to make the relationship between cache size and performance visible directly in the table.
- Added Table 3: TurboQuant decode throughput and TTFT by case and configuration, using the fields reported in the latest benchmark capture.
- Expanded §6.2 prose to include the latest crossover pattern explicitly: the three short cases remain slower than baseline, while `Long Context 2x` is faster for all configurations and `Long Context 1x` is faster for `Safe Default`.
- Updated §6.5 "Configuration-level observations" so that `Mid Compression` is described as an aggressive high-performance point rather than a uniformly dominated one.
- Fixed figure references: all absolute machine paths (`/Users/dhonampemba/...`) replaced with relative paths (`figures/filename.png`).
- Embedded figure captions as proper prose in the Results section.

**Discussion (§7):** Substantially strengthened.
- Updated §7.2 "Quantitative evidence for the rematerialization bottleneck" so it now presents the result as a cache-size-driven crossover: short cases lose, large-cache cases win.
- Updated §7.3 "The configuration landscape" so it reflects the new behavior: `Mid Compression` is aggressive and fast on the longest case, but no longer simply dominated.
- Renamed §7.2 (original) → §7.4 "What the implementation establishes."

**Limitations (§8):** Completely rewritten. The original was a single paragraph. The revised section is a numbered list of 8 distinct limitations with clear explanations, covering: implementation fidelity, run budget, single environment, metric scope, quality reference (baseline vs. ground truth), prompt coverage, single model, and hardware transparency. A companion file `limitations-and-threats.md` provides extended rationale for each item.

**Future Work (§9):** Added as a new section with seven specific research directions: GPU-resident cache via IO binding, selective rematerialization, closer TurboQuant reproduction, compressed attention paths, statistical robustness, multi-environment evaluation, and richer quality evaluation.

**Conclusion:** Tightened again. The conclusion now frames the contribution as identifying the browser crossover regime in which compressed KV handling begins to pay off, while keeping the systems bottleneck story intact.

**Data Availability:** Paths fixed from absolute machine paths to relative filenames.

**References:** Consolidated. The original had three separate references to the TurboQuant paper (blog, OpenReview, arXiv). These are now two entries: the blog post (ref 1) and the technical paper with both URLs (ref 2). All other references retained. A note in the submission checklist flags that author names should be added to ref 2.

### New supporting files

- **`figure-captions.md`** — formal captions for all six figures, suitable for use in a journal submission.
- **`limitations-and-threats.md`** — extended prose on each limitation with reviewer-response notes and a severity-vs-fixability summary table.
- **`submission-checklist.md`** — full Preprints.org readiness checklist with a claims audit table that verifies every headline number against the CSV data.
- **`cover-note.md`** — this file.

---

## 2. The strongest defensible claims from the current evidence

These claims are directly supported by `Chrome Benchmarkv2.txt`:

1. **The implementation works.** A TurboQuant-inspired compressed KV-cache path runs end-to-end inside browser-based Gemma 4 inference on Chrome WebGPU, implemented as a fork of `transformers.js` with a pluggable cache abstraction.

2. **Compression is real only once context length is large enough.** Average compression is `0.967x` (Safe Default), `1.044x` (Mid Compression), and `0.992x` (Key Heavy), which means short prompts can actually expand the cache. The strongest single-case compression remains on `Long Context 2x` with `Key Heavy` at `1.512x`.

3. **The compressed path is no longer uniformly slower.** Average speed ratios are now `1.039–1.077x`, but this average hides a clear crossover. All three short cases remain slower than baseline, while `Long Context 2x` is faster for all three TurboQuant settings and `Long Context 1x` is faster for `Safe Default`.

4. **Latency behavior is strongly context-length-dependent.** Risk Summary still disfavors TurboQuant (`0.886–0.942x` speed ratios), but `Long Context 2x` shows the reverse (`1.702–2.143x`). This is the clearest evidence that the cache regime, not just the algorithm label, determines whether the compressed path pays off.

5. **Quality is materially stronger than in the earlier run.** `Safe Default` achieves `5/5` exact matches with `100%` average prefix agreement. `Mid Compression` and `Key Heavy` each achieve `3/5` exact matches. The weakest point in the sweep is now `Policy Comparison + Mid Compression` at `77.759%` prefix agreement.

6. **Safe Default is now the best reference operating point.** It is the only setting with exact matches on all five cases, and it is already faster than baseline on both long-context prompts.

7. **The bottleneck remains the runtime boundary.** The pattern of results is still consistent with a cache-size-driven crossover governed by rematerialization and runtime overhead. Short prompts do not amortize the extra cache machinery; long prompts increasingly do.

---

## 3. What the evidence does NOT support

These claims should not appear in the manuscript:

- "TurboQuant improves browser LLM inference speed in general" — not supported; the latest evidence shows a crossover, not a universal win.
- "TurboQuant always preserves output quality in the browser" — Safe Default is strong in the latest sweep, but the run budget is still too small for a blanket claim.
- "The implementation achieves the compression levels described in the TurboQuant paper" — still unsupported; long-context gains top out around `1.512x`, not the much larger accelerator-side story.
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

> We present an experimental study of TurboQuant-inspired KV-cache compression for browser-based LLM inference using `transformers.js` and Chrome WebGPU. The latest benchmark sweep shows a context-length crossover rather than a uniform slowdown: the compressed path is slower on short prompts, but faster on the longest tested context while preserving exact output under the Safe Default setting. The paper contributes an open implementation, a benchmark harness, and a concrete identification of cache-size-dependent runtime behavior that future browser cache designs must address.

This framing is honest, defensible, and positions the paper as a systems contribution with clear negative results and a useful engineering lesson.
