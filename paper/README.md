# TurboQuant Benchmark Artifacts

This directory now contains the benchmark assets for the TurboQuant fork work, not the manuscript-writing bundle.

The manuscript-style files were moved out of this repo to:

- `/Users/dhonampemba/Development/canton/turboquant-paper`

## What is here

- [benchmark-analysis.md](./benchmark-analysis.md)
  - Short benchmark summary and interpretation of the latest Chrome WebGPU run.
- [Chrome Benchmarkv2.txt](./Chrome%20Benchmarkv2.txt)
  - Primary human-readable benchmark capture for the latest run.
- [Chrome Benchmark.txt](./Chrome%20Benchmark.txt)
  - Earlier benchmark capture kept for comparison.
- [turboquant-benchmark.json](./turboquant-benchmark.json)
  - Earlier JSON export from the browser harness.
- [tables](./tables)
  - Generated CSV summaries and row-level benchmark exports.
- [figures](./figures)
  - Generated PNG figures from the benchmark data.
- [figures_and_tables.zip](./figures_and_tables.zip)
  - Packaged benchmark figures and tables.
- [benchmark-app](./benchmark-app)
  - Standalone browser benchmark app used to run the Chrome/WebGPU suite against the local fork.
- [generate_paper_assets.py](./generate_paper_assets.py)
  - Asset-generation script used to build the benchmark tables and figures.

## Current result summary

- The TurboQuant-style cache path runs end-to-end in Chrome WebGPU with Gemma 4.
- The latest run shows a context-length crossover:
  - short prompts still favor `DynamicCache`
  - long prompts favor `TurboQuantCache`
- Compression is only positive once cache pressure is high enough.
- `Safe Default` is the strongest overall setting in the current sweep, with `5/5` exact matches.
- The implementation remains TurboQuant-inspired rather than a faithful reproduction of the published accelerator-side system.
