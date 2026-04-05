# Preprints.org Submission Checklist

Recommended classification: **Computer Science and Mathematics → Artificial Intelligence and Machine Learning**
Article type: **Article**
Version: draft checklist as of 2026-04-05

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
- [✅] "TurboQuant improves browser latency" claim absent — not supported and not made
- [✅] Mixed/negative framing throughout — results accurately described as slower than baseline
- [✅] Implementation correctly described as TurboQuant-*inspired*, not a full reproduction
- [✅] Figure paths updated to relative paths (`figures/filename.png`) — no absolute machine paths
- [✅] Data availability paths updated to relative/local filenames — no absolute machine paths
- [✅] All numbers in text traceable to `tables/config_summary.csv` or `tables/benchmark_rows.csv`

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

The following claims appear in the manuscript and are verified against `config_summary.csv` and `benchmark_rows.csv`:

| Claim | Source | Status |
|---|---|---|
| "1.29–1.34× average compression" | config_summary.csv avg_compression_ratio | ✅ Exact (1.290, 1.328, 1.336) |
| "0.50–0.57× average speed ratio" | config_summary.csv avg_speed_ratio | ✅ Exact (0.495, 0.538, 0.573) |
| "69.9–83.7% average prefix agreement" | config_summary.csv avg_prefix_agreement_pct | ✅ Exact (69.9, 77.8, 83.7) |
| "18.5 tokens/s baseline TPS" | config_summary.csv avg_dynamic_tps | ✅ Exact (18.513) |
| "~9–10 tokens/s TurboQuant TPS" | config_summary.csv avg_turbo_tps | ✅ (10.0, 10.1, 9.1) |
| "0.74 tokens/s on Long Context 2x (Safe Default)" | benchmark_rows.csv turbo_tps | ✅ Exact (0.7397) |
| "9.98 tokens/s baseline on Long Context 2x" | benchmark_rows.csv dynamic_tps | ✅ Exact (9.978) |
| "13.5× throughput gap on Long Context 2x" | derived: 9.978/0.7397 | ✅ Exact (13.49×) |
| "1.28× throughput gap on Risk Summary" | derived: 23.56/18.39 | ✅ Exact (1.281×) |
| "43.7 MB dense cache on Long Context 2x" | benchmark_rows.csv dynamic_dense_bytes / (1024²) | ✅ (45,821,952 / 1,048,576 = 43.7 MB) |
| "1.5 MB dense cache on Risk Summary" | benchmark_rows.csv dynamic_dense_bytes / (1024²) | ✅ (1,603,584 / 1,048,576 = 1.53 MB) |
| "0/5 exact matches for all configs" | config_summary.csv exact_match_count | ✅ All 0 |
| "35.8% prefix agreement on Policy Comparison, Mid Compression" | benchmark_rows.csv prefix_agreement_pct | ✅ Exact (35.786%) |
| "96.6% prefix agreement on Long Context 2x, Safe Default" | benchmark_rows.csv prefix_agreement_pct | ✅ Exact (96.644%) |
| "Long Context 2x TurboQuant TTFT: 41–43 s vs 45.8 s baseline" | benchmark_rows.csv ttft fields | ✅ (41788, 43055, 40952 ms vs 45782 ms) |
| "Mid Compression dominated by Safe Default on 3/5 cases" | derived analysis | ✅ Verified |

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
