"""Regenerate the benchmark (and optionally the corpus) figure images.

SiliconEye distributes NO figure rasters: every benchmark image is a crop
of a page of a publicly downloadable PDF (research paper, datasheet or
specification), and the copyright of those documents stays with their
publishers. Instead, `benchmark/figure_sources.yaml` records, for every
image, the source document (URL + SHA-256 of the exact revision), the
0-based page number and the crop rectangle in PDF points. This script:

  1. downloads each source PDF into `cache/pdfs/` (verifying the SHA-256;
     if the live URL has changed or disappeared it falls back to the
     recorded `archive_url`, then to the Wayback Machine);
  2. renders each crop with PyMuPDF at the zoom that reproduces the exact
     pixel dimensions the benchmark was run with;
  3. writes `benchmark/images/q*_source.jpeg` and reports every mismatch.

Usage:
  python scripts/setup_images.py                 # the 19 benchmark images
  python scripts/setup_images.py --dataset       # + the 9,776-figure corpus
                                                 #   (447 PDFs, ~1 GB download)
  python scripts/setup_images.py --pdf-dir DIR   # use PDFs you downloaded
                                                 #   yourself (matched by SHA)
  python scripts/setup_images.py --force         # re-render existing images

The downloaded PDFs and the rendered images are for your own evaluation use;
check each document's terms before redistributing anything derived from it
(see NOTICE.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
import httpx
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "benchmark"
DATASET_DIR = ROOT / "dataset"
CACHE_DIR = ROOT / "cache" / "pdfs"

UA = {
    # Several datasheet hosts (GS1, Lattice) refuse requests without a
    # browser-like User-Agent.
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/pdf,*/*",
}
JPEG_QUALITY = 95


def eprint(*a, **k):
    print(*a, file=sys.stderr, flush=True, **k)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sanitize_key(paper_key: str) -> str:
    """`arxiv:1234.5678` -> `arxiv_1234.5678`; URLs -> filesystem-safe form.
    Mirrors the naming used in the released corpus `path` column."""
    out = []
    for ch in paper_key:
        out.append(ch if (ch.isalnum() or ch in "._-") else "_")
    return "".join(out)


# --- download ---------------------------------------------------------------

def _fetch(http: httpx.Client, url: str, retries: int = 4) -> bytes | None:
    r = None
    for attempt in range(retries):
        try:
            r = http.get(url)
        except Exception as e:
            eprint(f"      ! {type(e).__name__} fetching {url[:90]}")
            return None
        if r.status_code in (429, 503):
            # The Wayback Machine rate-limits aggressively; back off and retry.
            wait = 15 * (attempt + 1)
            eprint(f"      ~ HTTP {r.status_code}, retrying in {wait}s")
            time.sleep(wait)
            continue
        break
    if r is None or r.status_code != 200:
        eprint(f"      ! HTTP {r.status_code if r else '?'} for {url[:90]}")
        return None
    ct = r.headers.get("content-type", "")
    if "pdf" not in ct and not r.content[:5] == b"%PDF-":
        eprint(f"      ! not a PDF (content-type {ct[:40]!r}) at {url[:90]}")
        return None
    return r.content


def obtain_pdf(http: httpx.Client, doc: dict, pdf_dir: Path | None,
               allow_mismatch: bool) -> Path | None:
    """Return a local path to the PDF whose SHA-256 matches `doc['pdf_sha256']`.
    Order: cache -> --pdf-dir -> live url -> archive_url -> Wayback Machine."""
    want = doc["pdf_sha256"]
    cached = CACHE_DIR / f"{want}.pdf"
    if cached.exists():
        return cached
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if pdf_dir:
        for p in sorted(pdf_dir.glob("*.pdf")):
            if sha256_bytes(p.read_bytes()) == want:
                cached.write_bytes(p.read_bytes())
                return cached

    candidates = [("live", doc["url"])]
    if doc.get("archive_url"):
        candidates.append(("archive", doc["archive_url"]))
    # Wayback "id_" flag returns the archived bytes unmodified.
    candidates.append(("wayback", f"https://web.archive.org/web/2id_/{doc['url']}"))

    fallback: bytes | None = None
    for label, url in candidates:
        eprint(f"      fetching ({label}) {url[:100]}")
        body = _fetch(http, url)
        if body is None:
            continue
        if sha256_bytes(body) == want:
            cached.write_bytes(body)
            return cached
        eprint(f"      ! SHA-256 mismatch from {label} source — document revision differs")
        fallback = fallback or body
    if fallback is not None and allow_mismatch:
        p = CACHE_DIR / f"MISMATCH_{want[:16]}.pdf"
        p.write_bytes(fallback)
        eprint(f"      ! using mismatching revision (--allow-mismatch); crops may be wrong")
        return p
    return None


# --- render -----------------------------------------------------------------

def render_crop(pdf_path: Path, page_no: int, bbox: list[float],
                width: int, height: int, out_path: Path) -> tuple[int, int]:
    doc = fitz.open(pdf_path)
    page = doc[page_no]
    clip = fitz.Rect(*bbox)
    # Zoom chosen so that the crop comes out at the recorded pixel width.
    zoom = width / clip.width
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, colorspace=fitz.csRGB, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    # Rasterisation rounds the clip to whole pixels, which can leave the
    # output 1-2 px off the recorded size; snap to the exact dimensions so
    # every run sees byte-for-byte the same image geometry.
    if (img.width, img.height) != (width, height) and \
            abs(img.width - width) <= 3 and abs(img.height - height) <= 3:
        img = img.resize((width, height), Image.Resampling.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return img.width, img.height


def run(items: list[dict], docs: dict, out_root: Path, label: str,
        pdf_dir: Path | None, allow_mismatch: bool, force: bool) -> int:
    """items: list of {out: relative path, document, page, bbox_pts, width, height}"""
    by_doc: dict[str, list[dict]] = {}
    for it in items:
        by_doc.setdefault(it["document"], []).append(it)
    n_ok = n_skip = n_fail = n_dim = 0
    failures: list[str] = []
    t0 = time.time()
    with httpx.Client(follow_redirects=True, timeout=120, headers=UA) as http:
        for i, (key, its) in enumerate(by_doc.items(), 1):
            doc = docs[key]
            todo = [it for it in its if force or not (out_root / it["out"]).exists()]
            if not todo:
                n_skip += len(its)
                continue
            eprint(f"[{label}] ({i}/{len(by_doc)}) {key[:80]}  ({len(todo)} image(s))")
            pdf = obtain_pdf(http, doc, pdf_dir, allow_mismatch)
            if pdf is None:
                n_fail += len(todo)
                failures.append(f"{key}: could not obtain a matching PDF ({doc['url']})")
                continue
            for it in todo:
                out = out_root / it["out"]
                try:
                    w, h = render_crop(pdf, it["page"], it["bbox_pts"], it["width"], it["height"], out)
                except Exception as e:
                    n_fail += 1
                    failures.append(f"{it['out']}: {type(e).__name__}: {e}")
                    continue
                if abs(w - it["width"]) > 2 or abs(h - it["height"]) > 2:
                    n_dim += 1
                    eprint(f"      ! {it['out']}: rendered {w}x{h}, expected {it['width']}x{it['height']}")
                n_ok += 1
    eprint(f"[{label}] done in {time.time() - t0:.0f}s — rendered {n_ok}, skipped (existing) {n_skip}, "
           f"failed {n_fail}, dimension warnings {n_dim}")
    for f in failures:
        eprint(f"    FAILED  {f}")
    return n_fail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="store_true",
                    help="Also regenerate the released corpus figures into dataset/images/ "
                         "(reads dataset/figures.ndjson; 447 PDFs).")
    ap.add_argument("--pdf-dir", type=Path, default=None,
                    help="Directory of PDFs you downloaded yourself; used (matched by SHA-256) "
                         "before any network access.")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="If no byte-identical revision of a document can be found, crop the "
                         "revision that is available anyway (crops may be wrong).")
    ap.add_argument("--force", action="store_true", help="Re-render images that already exist.")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Benchmark image basenames to (re)build, e.g. q8_source.jpeg")
    args = ap.parse_args()

    spec = yaml.safe_load((BENCH_DIR / "figure_sources.yaml").read_text(encoding="utf-8"))
    docs = spec["documents"]
    items = [dict(out=im["image"], document=im["document"], page=im["page"], bbox_pts=im["bbox_pts"],
                  width=im["width"], height=im["height"]) for im in spec["images"]]
    if args.only:
        items = [it for it in items if it["out"] in set(args.only)]
    failed = run(items, docs, BENCH_DIR / "images", "benchmark", args.pdf_dir, args.allow_mismatch, args.force)

    if args.dataset:
        figs_path = DATASET_DIR / "figures.ndjson"
        papers_path = DATASET_DIR / "papers.ndjson"
        pdocs: dict[str, dict] = {}
        for line in papers_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                p = json.loads(line)
                pdocs[p["paper_key"]] = dict(url=p["source_url"], pdf_sha256=p["pdf_sha256"],
                                             archive_url=p.get("archive_url"))
        ditems = []
        for line in figs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            f = json.loads(line)
            if not f.get("bbox_pts"):
                continue
            ditems.append(dict(out=f["path"].removeprefix("figures/"), document=f["paper_key"], page=f["page"],
                               bbox_pts=f["bbox_pts"], width=f["width"], height=f["height"]))
        failed += run(ditems, pdocs, DATASET_DIR / "images", "dataset", args.pdf_dir, args.allow_mismatch, args.force)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
