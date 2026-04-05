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
  - Analysis of the current Chrome WebGPU benchmark sweep on `onnx-community/gemma-4-E2B-it-ONNX`.
- [turboquant-benchmark.json](./turboquant-benchmark.json)
  - Exported benchmark data from the browser harness.
- [Chrome Benchmark.txt](./Chrome%20Benchmark.txt)
  - Human-readable benchmark output capture.
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

The current evidence supports a browser-side engineering study, not a strong positive performance paper.

- The TurboQuant-style cache path runs in Chrome WebGPU with Gemma 4.
- The current implementation reduces packed KV size relative to dense caching.
- The current implementation is slower than the dense baseline in the tested Chrome/WebGPU sweep.
- The current implementation is not yet faithful enough to support a "no quality loss" claim.
