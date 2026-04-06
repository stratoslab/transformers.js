# Preprints.org Submission Checklist

Recommended classification: **Computer Science and Mathematics → Artificial Intelligence and Machine Learning**
Article type: **Article**
Version: draft checklist as of 2026-04-06

Use this checklist before submitting or sharing `draft-paper.md` as a Preprints.org preprint. Items marked ✅ are completed in the current draft. Items marked ⚠️ need action before submission. Items marked ❌ are known gaps that should be resolved or explicitly acknowledged.

---

## Manuscript structure (Preprints.org requirements)

- [✅] Title on the first page
- [✅] Author names and affiliations on the first page
- [✅] Corresponding author email addresses on the first page
- [✅] Abstract present and clearly labeled
- [✅] Keywords present
- [✅] Introduction section
- [✅] Methods / system design section
- [✅] Results section with tables and figure references
- [✅] Discussion section
- [✅] Limitations section (Section 8 — significantly expanded)
- [✅] Future Work section (Section 9 — added)
- [✅] Conclusion section
- [✅] Data Availability section
- [✅] Conflicts of Interest statement
- [✅] AI Use Disclosure
- [✅] References section with numbered citations
- [⚠️] File format: current draft is `.md`; Preprints.org requires Word (.docx) or LaTeX. **The `.md` draft must be converted to Word or LaTeX before submission.**
- [⚠️] Graphical abstract (JPG or PNG): not yet created. Recommended before submission. The speed-quality frontier figure (`figures/speed_quality_frontier.png`) could serve as a graphical abstract with minor annotation.

---

## Content integrity

- [✅] No fabricated experimental results
- [✅] No unsupported performance claims (all headline numbers are in the benchmark data)
- [✅] "No quality loss" claim absent — not supported by benchmark data and not made
- [✅] "TurboQuant improves browser latency in general" claim absent — the latest results are crossover-dependent, not universal
- [✅] Mixed/tradeoff framing throughout — results are described as slower on short prompts and faster on the longest prompt
- [✅] Implementation correctly described as TurboQuant-*inspired*, not a full reproduction
- [✅] Figure paths updated to relative paths (`figures/filename.png`) — no absolute machine paths
- [✅] Data availability paths updated to relative/local filenames — no absolute machine paths
- [✅] All numbers in text traceable to the benchmark capture or derived directly from the reported case tables

---

## Authorship and AI disclosure

- [✅] Authors named on title page with affiliations and emails
- [✅] No AI tools listed as authors
- [✅] AI Use Disclosure section present and accurate
- [⚠️] Both coauthors (Dhonam Pemba, Kwang Wei Sim) must confirm they have reviewed the final manuscript and consent to the preprint posting before submission
- [⚠️] Both coauthors must confirm target journals permit preprint posting (check Sherpa Romeo if a specific journal is targeted)

---

## Ethics and intellectual property

- [✅] Conflicts of Interest declared (none)
- [✅] No copyrighted figures reproduced without permission (all figures generated from own benchmark data)
- [⚠️] Patent review: if the browser cache implementation has potential commercial or patent value, **do not post as a preprint** until patent applications are filed. Preprints.org explicitly warns that preprint posting can compromise a patent application.
- [✅] No human subjects research; no IRB/ethics board approval required
- [✅] Benchmark model (`onnx-community/gemma-4-E2B-it-ONNX`) is publicly available on Hugging Face; usage is consistent with its license

---

## Data and reproducibility

- [✅] Benchmark JSON export (`turboquant-benchmark.json`) present
- [✅] Derived summary CSVs present (`tables/`)
- [✅] Fork repo URL provided (`https://github.com/stratoslab/transformers.js`)
- [✅] Relevant commit SHAs listed in manuscript (Section 5.4)
- [⚠️] **Fork repo must be publicly accessible** before submission, or a persistent archive (Zenodo, OSF) must be created. If the repo is private, create a public snapshot or Zenodo deposit.
- [⚠️] **App repo URL and commit SHA** (`7af609f`) should be linked to a publicly accessible repository. The app-side benchmark harness (`BenchmarkApp.jsx`, `benchmarkWorker.js`, `benchmarkCases.js`) is not currently pointed to a public URL in the manuscript.
- [⚠️] Hardware/browser metadata: the benchmark export does not record GPU model, browser version, or OS. Consider adding a `metadata.json` or appending a reproducibility note to the benchmark JSON that captures: browser name and version, OS, GPU adapter description (obtainable from `navigator.gpu.requestAdapter()` → `requestAdapterInfo()`), date of run.
- [⚠️] Benchmark prompt definitions (the actual prompt strings used for each case) should be deposited alongside the JSON, not just the case labels. Reviewers and reproducers need the exact prompts.
- [⚠️] Instructions for rerunning the benchmark (build the fork, serve locally, load in Chrome, run the harness) should be in a `REPRODUCE.md` in the fork or deposited as a supplementary file.

---

## Reference quality

- [⚠️] References 1 and 2 currently cite the TurboQuant work without author names. Author names for the TurboQuant paper should be verified from the arXiv/OpenReview entry and added to the citation. Do not fabricate — look up the actual author list.
- [⚠️] Reference formatting is currently numbered list in Markdown; must be converted to the target format (APA, IEEE, etc.) when the document is converted to Word/LaTeX. Preprints.org does not mandate a specific citation style but asks for consistency.
- [✅] All referenced URLs were valid as of the drafting date (2026-04-05)
- [✅] No citation points to a non-existent or fabricated source

---

## Claims audit (pass/fail against benchmark data)

The following claims appear in the manuscript and are verified against `Chrome Benchmarkv2.txt`:

| Claim | Source | Status |
|---|---|---|
| "0.967–1.044× average compression" | Sweep Summary | ✅ Exact (0.967, 0.992, 1.044) |
| "1.039–1.077× average speed ratio" | Sweep Summary | ✅ Exact (1.039, 1.054, 1.077) |
| "95.277–100.000% average prefix agreement" | Sweep Summary | ✅ Exact |
| "`Safe Default` exact on all five cases" | Sweep Summary | ✅ Exact (`5`) |
| "`Mid Compression` and `Key Heavy` exact on 3/5 cases" | Sweep Summary | ✅ Exact (`3`, `3`) |
| "All three short cases remain slower than baseline" | Risk/Operations/Policy rows | ✅ Verified |
| "`Long Context 2x` is faster for all three TurboQuant configs" | Long Context 2x rows | ✅ Verified (`1.702x`, `2.143x`, `2.075x`) |
| "`Long Context 1x` is faster only for `Safe Default`" | Long Context 1x rows | ✅ Verified (`1.093x`, `0.820x`, `0.762x`) |
| "Short-case compression can be negative" | Risk/Operations/Policy rows | ✅ Verified (`0.667x` to `0.868x`) |
| "Best observed compression is `1.512x` on `Long Context 2x`, `Key Heavy`" | Long Context 2x row | ✅ Exact |
| "Worst observed prefix agreement is `77.759%` on Policy Comparison, Mid Compression" | Policy Comparison row | ✅ Exact |
| "Safe Default is the strongest quality-preserving setting" | Sweep Summary + case rows | ✅ Verified |

---

## Before final submission

1. **Convert to Word or LaTeX** (Preprints.org does not accept Markdown).
2. **Verify TurboQuant author names** and update References 1 and 2.
3. **Confirm coauthor approval** of final text.
4. **Make fork and app repos public** (or archive on Zenodo/OSF).
5. **Add hardware/browser metadata** to the benchmark export or a separate file.
6. **Include exact prompt strings** in the data package.
7. **Create a REPRODUCE.md** with benchmark replication instructions.
8. **Check patent exposure** if commercialisation is planned.
9. **Create graphical abstract** (optional but recommended by Preprints.org).
10. **Check Sherpa Romeo** if a specific target journal is in mind.
