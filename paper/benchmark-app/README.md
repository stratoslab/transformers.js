# Benchmark App

Standalone browser benchmark app for the TurboQuant Gemma 4 paper.

What it contains:
- The original Gemma 4 browser benchmark suite from the parent app repo.
- Synthetic cache benchmarks for:
  - materialize-only timing
  - cache update/materialize sweeps across sequence lengths

This app uses the local `transformers.js` build from:
- `../../packages/transformers/dist/transformers.web.js`

So benchmark runs reflect the current repo state, not a published npm release.

Run:

```bash
npm install
npm run dev
```

Then open the Vite URL in Chrome/Chromium with WebGPU enabled.
