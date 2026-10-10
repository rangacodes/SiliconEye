"""Generate degraded variants of the benchmark images.

Two independent axes are supported, one at a time via ``--axis``:

- ``jpeg``  — re-encode at successively lower JPEG quality levels.
              Levels are quality integers (100 = best). Naturally occurs
              when images are re-compressed by hosting/repackaging
              pipelines.

- ``resize`` — downsample to a fraction of the original resolution and
               re-save at high JPEG quality. Simulates low-DPI PDF exports
               and thumbnails. Levels are percent-of-original (100 = full).

For each source under ``benchmark/images/q*_source.jpeg``, the script
writes ``benchmark/degraded/{axis}/{level}/{basename}`` and emits a
Markdown stats table with resolution, disk size, and sharpness (variance
of the Laplacian) per (source × level).

Usage:
  python scripts/degrade_images.py                          # default: jpeg
  python scripts/degrade_images.py --axis jpeg --levels 75 50 25 10 5
  python scripts/degrade_images.py --axis resize --levels 75 50 25 10 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import laplace

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "benchmark"
CLEAN_DIR = BENCH_DIR / "images"
DEGRADED_ROOT = BENCH_DIR / "degraded"

# When downsampling, re-encode with a high quality so JPEG artefacts don't
# leak into the "resize" axis and confound the signal.
RESIZE_JPEG_QUALITY = 95


def laplacian_var(path: Path) -> float:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L")).astype(np.float64)
    return float(laplace(arr).var())


def image_stats(path: Path) -> dict:
    with Image.open(path) as im:
        w, h = im.size
    return {
        "path": path,
        "width": w,
        "height": h,
        "disk_kb": path.stat().st_size / 1024,
        "sharpness": laplacian_var(path),
    }


def degrade_jpeg(src: Path, im_rgb: Image.Image, level: int, out_path: Path) -> None:
    im_rgb.save(out_path, format="JPEG", quality=level, optimize=False)


def degrade_resize(src: Path, im_rgb: Image.Image, level: int, out_path: Path) -> None:
    """Downsample to `level` percent of each dimension; save as JPEG-q95 so
    JPEG blockiness isn't the dominant corruption."""
    w, h = im_rgb.size
    new_w = max(1, int(round(w * level / 100)))
    new_h = max(1, int(round(h * level / 100)))
    resized = im_rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized.save(out_path, format="JPEG", quality=RESIZE_JPEG_QUALITY, optimize=False)


def degrade_resize_roundtrip(src: Path, im_rgb: Image.Image, level: int, out_path: Path) -> None:
    """Downsample to `level` percent, then upsample back to the original
    dimensions. The output has the same pixel dimensions as the source
    but loses high-frequency detail — simulates readers zooming into a
    thumbnail-quality figure."""
    w, h = im_rgb.size
    new_w = max(1, int(round(w * level / 100)))
    new_h = max(1, int(round(h * level / 100)))
    small = im_rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
    restored = small.resize((w, h), Image.Resampling.LANCZOS)
    restored.save(out_path, format="JPEG", quality=RESIZE_JPEG_QUALITY, optimize=False)


AXIS_HANDLERS = {
    "jpeg": degrade_jpeg,
    "resize": degrade_resize,
    "resize_roundtrip": degrade_resize_roundtrip,
}


def label_for(axis: str, level: int) -> str:
    return {"jpeg": f"Q{level}", "resize": f"{level}%", "resize_roundtrip": f"{level}%↺"}[axis]


def degrade_one(src: Path, axis: str, levels: list[int]) -> list[dict]:
    """Run every degradation level for a single source. Includes a baseline
    row for the untouched source at the head."""
    rows: list[dict] = []
    src_stats = image_stats(src)
    src_stats["source_id"] = src.name
    src_stats["level"] = "source"
    rows.append(src_stats)

    axis_dir = DEGRADED_ROOT / axis
    with Image.open(src) as im:
        im_rgb = im.convert("RGB")
        handler = AXIS_HANDLERS[axis]
        for lvl in levels:
            out_dir = axis_dir / str(lvl)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / src.name
            handler(src, im_rgb, lvl, out_path)
            r = image_stats(out_path)
            r["source_id"] = src.name
            r["level"] = label_for(axis, lvl)
            rows.append(r)
    return rows


def write_stats(rows: list[dict], axis: str, levels: list[int]) -> Path:
    """Write a Markdown table grouped by source, source → highest level → lowest."""
    order = ["source"] + [label_for(axis, l) for l in sorted(set(levels), reverse=True)]

    metric_note = {
        "jpeg": "JPEG quality level (`Q{n}` where n = libjpeg quality). "
                "Resolution is preserved.",
        "resize": "Downsample percent of the original dimensions (`{n}%`). "
                  "Re-encoded at JPEG-q95 so JPEG artefacts don't dominate.",
        "resize_roundtrip": "Downsample to `{n}%` then upsample back to original "
                            "dimensions (`{n}%↺`). Same pixel dimensions as source, "
                            "but high-frequency detail is destroyed.",
    }[axis]

    lines = [
        f"# Image degradation stats — axis = `{axis}`\n",
        "Generated by `scripts/degrade_images.py`.\n",
        metric_note + "\n",
        "Sharpness = variance of the Laplacian of the greyscale image "
        "(higher = crisper edges). For the `resize` axis, sharpness is "
        "computed at the reduced resolution, so it drops mechanically with "
        "level.\n",
        "| source | level | resolution | disk (KB) | sharpness |",
        "|---|---|---|---:|---:|",
    ]
    by_source: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_source.setdefault(r["source_id"], {})[r["level"]] = r
    for sid in sorted(by_source):
        for lvl in order:
            r = by_source[sid].get(lvl)
            if not r:
                continue
            lines.append(
                f"| {sid} | {lvl} | {r['width']}×{r['height']} | "
                f"{r['disk_kb']:.1f} | {r['sharpness']:.0f} |"
            )
        lines.append("|  |  |  |  |  |")

    out = DEGRADED_ROOT / axis / "quality_stats.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=sorted(AXIS_HANDLERS), default="jpeg",
                    help="Which degradation axis to run. Each axis writes into "
                         "benchmark/degraded/{axis}/.")
    ap.add_argument("--levels", type=int, nargs="+", default=[75, 50, 25, 10, 5],
                    help="Level values for the chosen axis. For jpeg = libjpeg "
                         "quality; for resize = percent of original dimensions.")
    args = ap.parse_args()

    if not CLEAN_DIR.exists():
        raise SystemExit(f"missing clean sources dir: {CLEAN_DIR}")
    sources = sorted(CLEAN_DIR.glob("q*_source.*"))
    if not sources:
        raise SystemExit(f"no q*_source.* files in {CLEAN_DIR}")

    print(f"[stage9] axis={args.axis}  {len(sources)} sources × {len(args.levels)} levels "
          f"→ {DEGRADED_ROOT / args.axis}")
    all_rows: list[dict] = []
    for src in sources:
        all_rows.extend(degrade_one(src, args.axis, args.levels))

    md_path = write_stats(all_rows, args.axis, args.levels)
    total_variants = sum(1 for r in all_rows if r["level"] != "source")
    print(f"[stage9] wrote {total_variants} degraded variants")
    print(f"[stage9] wrote {md_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
