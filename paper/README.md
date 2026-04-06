# Paper Materials

This directory collects the current source material for the TurboQuant-in-browser writeup.

## Files

- [draft-paper.md](./draft-paper.md)
  - Full manuscript-style draft with title, authors, equations, methods, results, and figure/table references.
- [turboquant-research-review.md](./turboquant-research-review.md)
  - Literature and platform review covering TurboQuant, `transformers.js`, WebGPU, and ONNX Runtime Web.
- [implementation-methods.md](./implementation-methods.md)
  - What was implemented in the `transformers.js` fork and in the Chrome benchmark app, how it works, and what remains incomplete.
- [preprints-submission-notes.md](./preprints-submission-notes.md)
  - Recommended Preprints.org subject area, article type guidance, and author-instruction summary tailored to this project.
- [benchmark-analysis.md](./benchmark-analysis.md)
  - Analysis of the latest Chrome WebGPU benchmark sweep on `onnx-community/gemma-4-E2B-it-ONNX`.
- [turboquant-benchmark.json](./turboquant-benchmark.json)
  - Earlier exported benchmark data from the browser harness.
- [Chrome Benchmark.txt](./Chrome%20Benchmark.txt)
  - Earlier human-readable benchmark output capture.
- [Chrome Benchmarkv2.txt](./Chrome%20Benchmarkv2.txt)
  - Latest human-readable benchmark output capture used for the current manuscript updates.
- [figures](./figures)
  - Generated PNG figures for the current benchmark set.
- [tables](./tables)
  - Generated CSV summary tables and row-level benchmark exports.

## Recommended reading order

1. Start with [draft-paper.md](./draft-paper.md) for the current manuscript draft.
2. Use [turboquant-research-review.md](./turboquant-research-review.md) for the external context.
3. Read [implementation-methods.md](./implementation-methods.md) to understand what this project actually built.
4. Read [preprints-submission-notes.md](./preprints-submission-notes.md) before preparing a submission package.
5. Use [benchmark-analysis.md](./benchmark-analysis.md), [tables](./tables), and [figures](./figures) for the empirical section.

## Current status

The current evidence supports a context-dependent systems paper rather than a simple all-positive or all-negative performance claim.

- The TurboQuant-style cache path runs in Chrome WebGPU with Gemma 4.
- The latest Chrome sweep shows a context-length crossover: TurboQuant is slower on short prompts, but faster on the longest prompt and slightly faster on `Long Context 1x` under `Safe Default`.
- Compression is not uniformly positive: short prompts can expand the cache, while long-context cases reach roughly `1.39x` to `1.51x` compression.
- `Safe Default` is currently the strongest quality-preserving setting in the latest sweep, with `5/5` exact matches.
- The implementation is still TurboQuant-inspired rather than a faithful reproduction of the published accelerator-side system.
