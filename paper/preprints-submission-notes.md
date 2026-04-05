# Preprints.org Submission Notes

Date: 2026-04-05

This note summarizes the relevant Preprints.org subject-area guidance and author instructions for the TurboQuant browser paper.

## Recommended subject area

### Best fit

- Top-level subject: `Computer Science and Mathematics`
- Recommended subarea: `Artificial Intelligence and Machine Learning`

Why this is the best fit:

- The work is centered on LLM inference, KV-cache compression, and generation quality/performance tradeoffs.
- Preprints.org explicitly presents `Computer Science and Mathematics` as covering algorithms, artificial intelligence, and computational methods.
- The subject browser for that area visibly includes the subcategory `Artificial Intelligence and Machine Learning`.

Relevant references:

- Subject-area announcement: <https://www.preprints.org/news/post/adjustment-of-subject-areas>
- Current subject listing: <https://www.preprints.org/subject>
- Computer Science and Mathematics browse page: <https://www.preprints.org/subject/browse/computer-science-and-mathematics>

### Secondary framing option

- Top-level subject: `Engineering`

Use this only if the paper is written primarily as a systems or implementation paper about runtime architecture, browser GPU execution, and deployment constraints. Even then, `Computer Science and Mathematics > Artificial Intelligence and Machine Learning` is still the cleaner primary classification.

### Not recommended as the primary category

- `Computer Science`
- `Software`

These appear as subareas within the broader computer science taxonomy, but the present work is more specifically about AI/ML inference than general software engineering.

## Subject-area language from Preprints.org

Preprints.org describes `Computer Science and Mathematics` as covering:

- algorithms
- artificial intelligence
- mathematical theories
- computational mathematics
- data analysis

and describes `Engineering` as covering broad engineering innovation and disciplinary expertise.

That makes the recommended positioning for this project:

- primary identity: AI inference / ML systems
- secondary identity: browser/WebGPU systems implementation

## Recommended article type

The author instructions state that the following manuscript types are generally suitable:

- `Article`
- `Review`
- `Conference Paper`
- `Data Descriptor`
- `Essay`
- `Brief Report`
- `Communication`
- `Short Note`
- `Technical Note`
- `Hypothesis`

For this project, the best options are:

- `Article` if you write this as a full experimental study
- `Technical Note` if you frame it as an engineering implementation report
- `Brief Report` only if the final manuscript is short and tightly scoped

Current recommendation:

- Use `Article`

Reason:

- You already have background, methods, implementation detail, and benchmark data.
- The paper benefits from a standard research structure with introduction, methods, results, and discussion.
- The instructions say original experimental research should have the structure of a research article.

## Why Preprints.org is a workable venue for the current result

The instructions explicitly state that:

- negative or non-significant results are welcome
- manuscripts can be posted after submission to a peer-reviewed venue if that venue permits preprints

That matters for this work because the current benchmark evidence is mixed rather than strongly positive.

This makes Preprints.org a reasonable fit if the manuscript is framed as:

- a browser implementation study
- an engineering report on KV-cache compression in `transformers.js`
- a mixed-result or negative-result study about why accelerator-side claims do not transfer directly to browser WebGPU inference

## Author-instruction summary

## Submission checklist

Before submission, Preprints.org says authors should confirm:

- all coauthors know about and agree to the posting
- target journals accept preprints
- all authors understand the withdrawal policy
- the work does not violate copyright, patents, trade secrets, privacy, or related rights
- research data are available

## Scope and manuscript suitability

Preprints.org accepts scientifically sound original research and comprehensive reviews across all fields.

For experimental work like this, the instructions say submissions should include:

- a comprehensive bibliography
- a research-article structure for original work

Recommended structure for this paper:

1. Introduction
2. Background
3. Browser Runtime Constraints
4. Implementation
5. Benchmark Design
6. Results
7. Limitations
8. Future Work

## Authorship and AI

The author instructions are explicit:

- AI tools and LLMs cannot be listed as authors
- use of AI tools must be disclosed in the Methods section or an equivalent section
- authors remain responsible for the final text

For this project, that means the manuscript should include a short disclosure such as:

- AI-assisted coding and drafting tools were used during implementation and manuscript preparation; all outputs were reviewed, edited, and validated by the human authors.

That disclosure should be refined to match the truth of your workflow, but it needs to exist.

## Formatting and file requirements

The instructions say:

- any manuscript style is allowed
- Preprints.org’s template is recommended
- journal/publisher logos must be removed if you use another template
- the first page must include title, authors, abstract, keywords, affiliations, and corresponding-author contact details
- submissions should be in Word or LaTeX
- LaTeX submissions must include all source files

They also recommend:

- a graphical abstract in JPG or PNG

## Data and supplementary material

Preprints.org requires authors to make associated data available where legally possible and encourages use of recognized repositories.

For this project, the following should be treated as submission-ready supplemental material:

- benchmark JSON exports
- benchmark prompts / benchmark-case definitions
- fork commit identifiers
- application commit identifiers
- configuration settings for all benchmark sweeps
- hardware/browser metadata used for runs

Strong recommendation:

- deposit the benchmark JSON, benchmark harness, and exact commit SHAs in a permanent repository or archive before submission

## Journal policy and patent caution

The instructions note:

- most journals accept preprints, but not all
- authors should check Sherpa Romeo or ask journals directly
- if you intend to file a patent, do not post a preprint because it can compromise the patent application

That patent warning is important for this project if you believe the browser-cache implementation has commercial or patent value.

## Ethics, permissions, and conflicts

Preprints.org requires:

- compliance with publishing ethics
- no plagiarism, fabrication, manipulation, fraud, or libel
- legal permission for reproduced copyrighted material
- a manuscript conflict-of-interest statement

For this paper, the likely simplest statement is:

- `Conflicts of Interest: The authors declare no conflicts of interest.`

Use that only if it is true.

## Withdrawal and permanence

The withdrawal policy is strict:

- once posted and DOI-registered, a preprint cannot be completely removed in the ordinary case
- revisions are preferred over removal
- only serious reasons justify withdrawal

The practical implication is:

- do not post until the coauthors are comfortable with the framing and public permanence of the result

## Screening criteria

Preprints.org screens submissions but does not peer review them.

The instructions say the screening checks include:

- English language
- not previously published
- ethics compliance
- genuine authorship
- disclosed conflicts
- all necessary figures and references
- no harmful, pseudoscientific, or unsupported strong claims
- AI use clearly disclosed

This is especially relevant here. Because your current results are mixed, the manuscript should avoid overclaiming. Strong unsupported claims about speedup or quality preservation are exactly the kind of thing that could create screening problems.

## Open-access terms

Preprints.org posts preprints under:

- `CC BY 4.0`

It also states that:

- authors grant a perpetual, non-exclusive distribution license
- authors certify they have the right to grant that license
- content may remain visible elsewhere even if withdrawn
- preprints are permanently archived at Portico

## Submission-specific recommendations for this project

Before submitting this paper, I recommend the following checklist.

### Recommended classification

- Subject: `Computer Science and Mathematics`
- Subarea: `Artificial Intelligence and Machine Learning`
- Type: `Article`

### Required manuscript-level items

- Add an AI-use disclosure in Methods
- Add a `Conflicts of Interest` section
- Add a `Data Availability` section
- Add a `Limitations` section
- Avoid positive claims not supported by the benchmark

### Recommended supplementary package

- `turboquant-benchmark.json`
- benchmark-case definitions
- Chrome/browser metadata
- exact model ID
- fork repo URL and commit SHA
- app repo URL and commit SHA
- instructions for rerunning the benchmark

### Recommended framing

Best current framing:

- `An Experimental Browser Implementation of TurboQuant-Style KV-Cache Compression for Gemma 4 on Chrome WebGPU`

Good framing themes:

- feasibility
- systems constraints
- browser runtime bottlenecks
- mixed empirical results
- lessons for future GPU-resident cache designs

Avoid framing like:

- "TurboQuant accelerates browser Gemma 4 inference"
- "TurboQuant preserves quality in the browser"

Those are not supported by the current benchmark set.

## Sources

- Preprints.org subject-area adjustment announcement  
  <https://www.preprints.org/news/post/adjustment-of-subject-areas>
- Preprints.org instructions for authors  
  <https://www.preprints.org/instructions-for-authors>
- Preprints.org subject listing  
  <https://www.preprints.org/subject>
- Preprints.org Computer Science and Mathematics browse page  
  <https://www.preprints.org/subject/browse/computer-science-and-mathematics>
