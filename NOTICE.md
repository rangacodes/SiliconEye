# Notice on third-party content

SiliconEye is built on figures that appear in research papers, component
datasheets and interface specifications published by third parties. Those
documents, and the figures in them, remain the copyright of their respective
publishers and authors. **This repository does not contain, and must never
contain, any of those figure rasters.** What it contains instead is:

- the question text, answer keys and option lists written by the SiliconEye
  authors (licensed under CC BY 4.0, see below);
- for every benchmark image, a *pointer* — the public URL of the source
  document, the SHA-256 of the exact revision, the page number and the crop
  rectangle — in [`benchmark/figure_sources.yaml`](benchmark/figure_sources.yaml);
- for the corpus, the same pointers plus our own metadata (captions as
  extracted by the layout model, VLM-generated type labels and one-sentence
  descriptions, trust and registry roll-ups) in [`dataset/`](dataset/);
- the code that re-creates the images on your machine from the source
  documents ([`scripts/setup_images.py`](scripts/setup_images.py)).

When you run `setup_images.py` you download the source documents yourself,
from their publishers (or from the Internet Archive where a URL has since
changed), and crop them locally. You are responsible for complying with each
publisher's terms for the resulting files. The licence status we found for the
documents behind the 19 benchmark images (checked 2026-10-10) is recorded in
the `license` field of each entry in `figure_sources.yaml`; in summary:

| Source | Terms |
|---|---|
| arXiv:1805.03648 (Kastner et al., *Parallel Programming for FPGAs*) | CC BY 4.0 — redistribution with attribution permitted |
| arXiv:2204.08742 (CoFHEE) | CC BY-NC-ND 4.0 |
| arXiv:1804.07858, arXiv:2209.09570 | arXiv non-exclusive licence; published versions © IEEE |
| arXiv:2002.03568 | arXiv non-exclusive licence; published version © IEICE |
| arXiv:1907.02217, arXiv:2011.00624, arXiv:2410.08476 | arXiv non-exclusive licence; © the authors |
| USENIX Security 2019 (VRASED) | open access; copyright retained by the authors |
| TI ADS7924 datasheet, Lattice iCE40 LP/HX datasheet, Winbond W25Q256FV datasheet | proprietary; see the publisher's terms of use |
| ULPI Specification 1.1 | all rights reserved by the ULPI promoters/adopters |
| GS1 EPC Gen2 Protocol Standard v2.0.1 | © GS1; reproduction requires permission |
| RISC-V Privileged Architecture v1.9 (Berkeley PDF) | © The Regents of the University of California (this revision predates the CC BY 4.0 releases) |

Please do not open pull requests that add figure images, thumbnails, or
derived rasters to this repository. If you believe a pointer in this
repository should be removed, open an issue and we will act on it promptly.

## Licences for what *is* in this repository

- Code (`scripts/`): MIT, see [`LICENSE`](LICENSE).
- Benchmark questions, answer keys and option text (`benchmark/questions.yaml`)
  and the corpus metadata we produced (`dataset/*.ndjson`, `dataset/repos.csv`,
  `benchmark/figure_sources.yaml`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Figure captions in `dataset/figures.ndjson` are short excerpts reproduced
  from the source documents for identification purposes only.
- Repository metadata in `dataset/repos.csv` (names, descriptions, star counts,
  URLs) is factual data obtained from the public GitHub API; the repositories
  themselves carry their own licences.
