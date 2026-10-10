# SiliconEye

**A multi-modal understanding benchmark for hardware design**, plus the
figure–paper–RTL corpus it was curated from.

SiliconEye asks whether vision-language models can actually *read* the
figures that hardware engineers work from — block diagrams, schematics,
finite-state machines and waveforms taken from real research papers,
datasheets and interface specifications — rather than answer from priors.
Every item is objectively gradable (exact-match multiple choice, multi-select,
integer or yes/no), the answer is visible only in the figure, and every source
document has an open-source Verilog implementation on GitHub.

> Paper: *SiliconEye: A Multi-modal Understanding Benchmark for Hardware
> Design* (preprint, 2026). See [Citation](#citation).

## What is in this repository

```
benchmark/
  questions.yaml         100 questions (id, topic, answer_type, question, options, correct_answer, …)
  figure_sources.yaml    where each of the 19 benchmark images comes from and how to regenerate it
  models.yaml            the 11 model variants evaluated in the paper (OpenRouter slugs + sampling knobs)
scripts/
  setup_images.py        downloads the source PDFs and re-creates benchmark/images/ (and the corpus)
  run_benchmark.py       runs questions × models through OpenRouter, grades, writes a report
  degrade_images.py      builds the degraded-image condition (resize / JPEG axes)
  score_reasoning.py     text-only LLM judge of reasoning traces (grounded vs. guessed)
  reasoning_report.py    renders the judge output as a Markdown report
dataset/
  figures.ndjson         9,776 paper figures: source, page, crop box, caption, VLM type label, repos
  papers.ndjson          the 419 source documents (URL, SHA-256, title, referring repos)
  readme_images.ndjson   12,359 images embedded in the READMEs of 2,530 Verilog repos (URLs only)
  repos.csv              2,718 repositories with trust-signal and silicon-registry roll-ups
  README.md              column reference
NOTICE.md                third-party content and licences — please read before redistributing anything
```

**No figure images are distributed.** The figures belong to their publishers
(see [`NOTICE.md`](NOTICE.md)). Instead, `figure_sources.yaml` and
`dataset/figures.ndjson` record a *pointer* for every image — public URL,
SHA-256 of the exact document revision, 0-based page number and crop rectangle
in PDF points — and `scripts/setup_images.py` rebuilds the images on your
machine, byte-for-byte at the geometry the paper's numbers were produced with.

## Quick start

```bash
git clone git@github.com:rangacodes/SiliconEye.git && cd SiliconEye
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Re-create the 19 benchmark images (≈45 MB of PDFs, ~30 s)
python scripts/setup_images.py

# 2. Put your OpenRouter key in the environment (or in a .env file at the repo root)
export OPENROUTER_API_KEY=sk-or-...

# 3. Run the benchmark (all 100 questions × every model in models.yaml)
python scripts/run_benchmark.py --run-id my_run --concurrency 4
#    or just a smoke test:
python scripts/run_benchmark.py --models claude_haiku_4_5_thinking_off --questions q2b_reset_out_edges --run-id smoke
```

Outputs land under `benchmark/runs/`: `{run_id}.ndjson` (one row per call —
raw response, reasoning trace, parsed answer, grade, full request config),
`{run_id}.md` (leaderboard + per-question table) and `{run_id}.config.json`.
Runs are resumable: re-running the same `--run-id` only issues the calls that
are missing.

### The degraded-image condition

The paper evaluates every item twice: with the original figure, and with the
figure downsampled to 5 % of its width and height (which removes essentially
all readable detail and gives a *blind baseline* for each question).

```bash
python scripts/degrade_images.py --axis resize --levels 5      # → benchmark/degraded/resize/5/
python scripts/run_benchmark.py --degraded resize/5 --run-id my_run_deg5
```

`--axis jpeg --levels 75 50 25 10 5` builds the JPEG-quality axis instead;
`--axis resize_roundtrip` downsamples and upsamples back to the original size.

### The reasoning-trace judge

For model variants that expose their reasoning, a **text-only** judge
(`z-ai/glm-5.3-flash` via OpenRouter) reads the question, the model's answer
and its reasoning trace — deliberately *not* the image — and scores how far
the trace is visually grounded (0.75–1.00), partially grounded (0.25–0.75) or
a blind guess (0.00–0.25):

```bash
python scripts/score_reasoning.py --input benchmark/runs/my_run.ndjson --out benchmark/runs/reasoning_scores/my_run_glm.ndjson
python scripts/reasoning_report.py --input benchmark/runs/reasoning_scores/my_run_glm.ndjson --out benchmark/runs/reasoning_scores/my_run_glm.md
```

The set of model ids treated as "reasoning-on" is `REASONING_MODELS` at the
top of `score_reasoning.py`; extend it if you add models.

## Question format

```yaml
- id: q2b_reset_out_edges
  author: Sriram
  category: visual            # visual | reasoning
  image: q2_source.jpeg       # basename under image_dir
  topic: fsm                  # block_diagram | schematic | fsm | waveform | fpga_arch
  source: url_https___www.usenix.org_system_files_sec19-nunes.pdf/fig_002.jpeg   # provenance in the corpus
  answer_type: numerical      # numerical | yes_no | multiple_choice | multiple_choice_multi
  question: |
    The image is a finite-state-machine diagram. ...
  correct_answer: 1
```

The runner appends an answer-format instruction per `answer_type` and grades
strictly: the final integer for `numerical`; `{Y}`/`{N}`, `{B}` or `{A, C}` in
the *last* curly-brace group of the response for the other types. See
`grade()` in `scripts/run_benchmark.py`.

Breakdown of the 100 items: 48 numerical · 30 multiple-choice · 18 yes/no ·
4 multi-select, over 19 figures from 15 documents (8 arXiv papers, 1 USENIX
paper, 3 component datasheets, 2 interface/protocol specifications, 1 ISA
manual).

## Reference results (from the paper)

Accuracy on the 100 items with the original figures vs. the 5 %-resized
figures (all models routed through OpenRouter, temperature 0):

| Model | Original | Degraded |
|---|---:|---:|
| Claude Opus 5 (thinking on) | 88.1 % | 59.5 % |
| Claude Opus 5 (thinking off) | 86.9 % | 45.2 % |
| Qwen3.8-27B (reasoning high) | 83.3 % | 29.8 % |
| Qwen3.8-27B (reasoning low) | 79.8 % | 36.9 % |
| Claude Sonnet 5 (thinking on) | 77.4 % | 40.5 % |
| Claude Sonnet 5 (thinking off) | 73.8 % | 38.1 % |
| Inkling-small (thinking on) | 72.6 % | 42.9 % |
| Claude Haiku 4.5 (thinking on) | 65.5 % | 31.0 % |
| Qwen3.8-27B (reasoning off) | 64.3 % | 39.3 % |
| Claude Haiku 4.5 (thinking off) | 63.1 % | 38.1 % |
| Inkling-small (thinking off) | 59.5 % | 38.1 % |

Format-weighted chance level is 18.1 %. Because the source documents are
publicly indexed and may have been seen in pre-training, treat SiliconEye as a
diagnostic comparison rather than a definitive ranking.

## The corpus

`dataset/` is the full extraction the benchmark was curated from: 81,901
Verilog-primary GitHub repositories → 10,897 READMEs → 1,886 paper links →
447 open-access PDFs → 9,776 figures (Marker layout model) classified by
Qwen3-VL-235B into hardware-design figure types (6,041 marked useful), plus
12,359 README-embedded images and design-maturity signals per repository.
Column reference in [`dataset/README.md`](dataset/README.md). To materialise
the corpus figures locally (447 PDFs, ≈1 GB):

```bash
python scripts/setup_images.py --dataset      # → dataset/images/<paper>/fig_NNN.jpeg
```

## Adding questions or models

- New question: add an entry to `questions.yaml`; if it uses a new figure, add
  the document and the image (URL, SHA-256, page, crop box, pixel size) to
  `figure_sources.yaml` so that `setup_images.py` can regenerate it. Do not
  commit the image itself.
- New model: add an entry to `models.yaml` (any OpenRouter slug; see the
  header comment for the knobs). Pin a provider for large open-weight models —
  unpinned routing produced a 33 % parse-error rate in our runs.

## Citation

```bibtex
@article{siliconeye2026,
  title   = {SiliconEye: A Multi-modal Understanding Benchmark for Hardware Design},
  author  = {Ranga, Sriram and Korkmaz, Mehmet An{\i}l and Bendale, Aneesh and
             Lakshminarayan, Pranav and Chattopadhyay, Anupam and Bhattacharjee, Debjyoti},
  year    = {2026},
  note    = {Preprint}
}
```

## Licence

Code is MIT-licensed; questions and the metadata we produced are CC BY 4.0.
Third-party documents referenced by this repository keep their own terms —
see [`NOTICE.md`](NOTICE.md).
