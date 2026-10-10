"""Render a Markdown report from GLM judge scores over
reasoning-on rows.

Reads:
  - benchmark/runs/reasoning_scores/full_v1_glm.ndjson  (per-row judge output)

Writes:
  - benchmark/runs/reasoning_scores/full_v1_glm.md
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def bucket(c: float) -> str:
    if c < 0.25:
        return "blind (0.00–0.25)"
    if c < 0.75:
        return "partial (0.25–0.75)"
    return "grounded (0.75–1.00)"


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def fmt(x: float) -> str:
    return f"{x:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="benchmark/runs/reasoning_scores/full_v1_glm.ndjson")
    ap.add_argument("--out", default="benchmark/runs/reasoning_scores/full_v1_glm.md")
    args = ap.parse_args()

    rows = load(ROOT / args.input)
    ok = [r for r in rows if r.get("status") == "ok" and r.get("confidence") is not None]
    err = [r for r in rows if r.get("status") != "ok"]

    lines: list[str] = []
    lines.append("# Reasoning-confidence judge report\n")
    lines.append(f"Judge: `z-ai/glm-5.3-flash` (text-only prompt, reasoning=low).\n")
    lines.append(f"Rows scored: **{len(ok)}/{len(rows)}** (errors: {len(err)}).\n")

    if ok:
        confs = [r["confidence"] for r in ok]
        lines.append(f"Overall confidence — mean={fmt(statistics.mean(confs))}, "
                     f"median={fmt(statistics.median(confs))}, "
                     f"min={fmt(min(confs))}, max={fmt(max(confs))}.\n")

    # Per-model breakdown
    per_model: dict[str, list[dict]] = defaultdict(list)
    for r in ok:
        per_model[r["model_id"]].append(r)

    lines.append("## Per-model summary\n")
    lines.append("| model | n | mean | median | blind | partial | grounded | acc | conf-when-correct | conf-when-wrong |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mid in sorted(per_model):
        rs = per_model[mid]
        confs = [r["confidence"] for r in rs]
        buckets = defaultdict(int)
        for c in confs:
            buckets[bucket(c)] += 1
        correct = [r for r in rs if r.get("is_correct")]
        wrong = [r for r in rs if not r.get("is_correct")]
        c_correct = [r["confidence"] for r in correct]
        c_wrong = [r["confidence"] for r in wrong]
        acc = len(correct) / len(rs) if rs else 0
        lines.append(
            f"| `{mid}` | {len(rs)} | {fmt(statistics.mean(confs))} | "
            f"{fmt(statistics.median(confs))} | "
            f"{buckets['blind (0.00–0.25)']} | "
            f"{buckets['partial (0.25–0.75)']} | "
            f"{buckets['grounded (0.75–1.00)']} | "
            f"{fmt(acc)} | "
            f"{fmt(statistics.mean(c_correct)) if c_correct else '—'} | "
            f"{fmt(statistics.mean(c_wrong)) if c_wrong else '—'} |"
        )

    # Confidence x correctness grid
    lines.append("\n## Confidence × correctness\n")
    lines.append("How often does confidence agree with correctness? Ideal: "
                 "grounded ⇒ correct, blind ⇒ wrong.\n")
    grid: dict[tuple[str, bool], int] = defaultdict(int)
    for r in ok:
        grid[(bucket(r["confidence"]), bool(r.get("is_correct")))] += 1
    lines.append("| bucket | correct | wrong | total | precision (correct/total) |")
    lines.append("|---|---:|---:|---:|---:|")
    for b in ["blind (0.00–0.25)", "partial (0.25–0.75)", "grounded (0.75–1.00)"]:
        c = grid[(b, True)]
        w = grid[(b, False)]
        t = c + w
        prec = c / t if t else 0
        lines.append(f"| {b} | {c} | {w} | {t} | {fmt(prec)} |")

    # Confident-but-wrong (grounded + wrong) — potentially the most misleading rows
    conf_wrong = sorted(
        [r for r in ok if r["confidence"] >= 0.75 and not r.get("is_correct")],
        key=lambda r: -r["confidence"],
    )
    lines.append(f"\n## Confident-but-wrong ({len(conf_wrong)})\n")
    lines.append("Model was scored ≥ 0.75 (grounded) but the final answer is wrong.\n")
    if conf_wrong:
        lines.append("| question | model | ans | correct | conf | judge summary |")
        lines.append("|---|---|---|---|---:|---|")
        for r in conf_wrong[:40]:
            s = r["summary"].replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| `{r['question_id']}` | `{r['model_id']}` | "
                f"{r.get('parsed_answer')} | {r['correct_answer']} | "
                f"{fmt(r['confidence'])} | {s} |"
            )
        if len(conf_wrong) > 40:
            lines.append(f"\n_…{len(conf_wrong)-40} more not shown._")

    # Blind guesses (any correctness) — the target of the exercise
    blind = sorted([r for r in ok if r["confidence"] < 0.25], key=lambda r: r["confidence"])
    lines.append(f"\n## Blind-guess traces ({len(blind)})\n")
    lines.append("Judge scored < 0.25 — model itself signaled it was guessing.\n")
    if blind:
        lines.append("| question | model | ans | correct | ✓/✗ | conf | judge summary |")
        lines.append("|---|---|---|---|---|---:|---|")
        for r in blind[:60]:
            s = r["summary"].replace("|", "\\|").replace("\n", " ")
            mark = "✓" if r.get("is_correct") else "✗"
            lines.append(
                f"| `{r['question_id']}` | `{r['model_id']}` | "
                f"{r.get('parsed_answer')} | {r['correct_answer']} | {mark} | "
                f"{fmt(r['confidence'])} | {s} |"
            )
        if len(blind) > 60:
            lines.append(f"\n_…{len(blind)-60} more not shown._")

    # Blind-but-correct — the "lucky guess" set
    lucky = [r for r in blind if r.get("is_correct")]
    lines.append(f"\n## Lucky guesses (blind AND correct): {len(lucky)}\n")

    # Errors
    if err:
        lines.append(f"\n## Judge errors ({len(err)})\n")
        for r in err[:20]:
            lines.append(f"- `{r['question_id']}` / `{r['model_id']}`: {r.get('status')} — "
                         f"{r.get('summary', '')[:200]}")

    out = ROOT / args.out
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
