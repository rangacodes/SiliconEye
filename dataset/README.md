# SiliconEye corpus — column reference

Four files, no images. Run `python scripts/setup_images.py --dataset` from the
repository root to re-create the paper figures under `dataset/images/`
(README images are plain URLs; fetch them with any HTTP client).

## `figures.ndjson` — 9,776 rows, one per figure extracted from a source PDF

| column | meaning |
|---|---|
| `path` | stable id, `figures/<document>/fig_NNN.jpeg`; `setup_images.py --dataset` writes the image to `dataset/images/<document>/fig_NNN.jpeg`. Benchmark questions reference this (minus `figures/`) in their `source` field |
| `paper_key` | `arxiv:<id>` or `url:<document url>`; joins to `papers.ndjson` |
| `page` | 0-based page index in the PDF |
| `block_class` | Marker layout class: `Figure`, `Diagram` or `Picture` |
| `bbox_pts` | `[x0, y0, x1, y1]` crop rectangle in PDF points (1/72 in), recovered by matching the extracted crop back onto the rendered page |
| `width`, `height` | pixel size of the extracted crop (Marker renders at 192 dpi; embedded bitmaps keep their native size) |
| `bbox_match_score` | normalised cross-correlation of the crop against the page at the recovered box (1.0 = exact). 9,764 rows ≥ 0.5; treat lower rows as approximate |
| `caption` | nearest caption text found by the layout model (may be truncated or empty) |
| `is_useful` | Qwen3-VL-235B verdict: "would a hardware code-generation model need this figure?" (`true` / `false` / `null` for the 6 unclassified rows) |
| `figure_type` | VLM label: `block_diagram`, `schematic`, `fsm`, `waveform`, `spec_diagram`, `microarchitecture`, `dataflow`, `algorithm_diagram`, `chart`, `table`, `layout`, `photo`, `logo`, `screenshot`, `text`, `other` |
| `vlm_confidence` | VLM self-reported confidence in `[0, 1]` |
| `vlm_description` | one-sentence VLM description of the figure |
| `referring_repos` | GitHub repositories whose README links to this document |

## `papers.ndjson` — 419 rows, one per source document with ≥ 1 figure

| column | meaning |
|---|---|
| `paper_key` | as above |
| `source_url` | URL the PDF was downloaded from |
| `archive_url` | present only where the live URL no longer serves the pinned revision (Internet Archive copy) |
| `pdf_sha256`, `pdf_size_bytes` | the exact document revision the crop boxes refer to |
| `access_class` | `open` (arXiv/known OA host) or `unknown` (direct PDF link) |
| `source_types` | how the link was recognised in the README (`arxiv`, `pdf`, `doi`, `publisher:*`, `academic`) |
| `title` | first heading recovered by the layout model (best effort; 416/419) |
| `referring_repos`, `n_figures` | |

## `readme_images.ndjson` — 12,359 rows (11,638 unique by content)

| column | meaning |
|---|---|
| `repo` | GitHub `owner/name` |
| `src_url` | resolved URL of the image as referenced from the README |
| `readme_ref` | the reference as written in the README |
| `alt_text` | alt text, if any |
| `sha256`, `size_bytes`, `content_type` | of the image at crawl time (2026-08); GIFs and video were excluded |

These images have not been VLM-classified.

## `repos.csv` — 2,718 rows, one per repository contributing a paper figure or README image

| column | meaning |
|---|---|
| `full_name`, `stars`, `language`, `description`, `homepage`, `html_url` | GitHub metadata at crawl time (September 2025 snapshot) |
| `tier` | `tier_a` (paper-signal keywords / academic homepage) or `tier_b` (star-count tier) |
| `paper_figures`, `readme_images` | counts in this corpus |
| `trust_score` | README design-maturity score: weighted hits for `silicon` (1.4), `formal` (1.2), `production` (1.2), `compliance` (1.1), `fpga` (1.0), each category capped at 3 hits |
| `trust_categories` | categories with ≥ 1 hit |
| `silicon_marks` | membership in silicon registries: Tiny Tapeout, efabless caravel, flagship organisations (lowRISC, chipsalliance, riscv, pulp-platform, sifive, OpenROAD, YosysHQ, zeroasiccorp, …) |

## Provenance

Pipeline (September 2025 repository snapshot, figures extracted August 2026):
GitHub search for Verilog-primary repositories → fork removal and tiering →
README fetch → regex paper-link extraction → open-access PDF download → Marker
layout-model figure extraction → Qwen3-VL-235B classification (OpenRouter,
provider pinned to `parasail/fp8`) → trust-signal and registry enrichment.
Full details are in Appendix C of the paper.
