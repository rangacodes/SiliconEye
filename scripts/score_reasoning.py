"""Judge each reasoning-on model's chain-of-thought for
self-expressed confidence.

The judge is DELIBERATELY TEXT-ONLY. We are not verifying the reasoning
against the image; we are asking: "did the model itself think it could
read the image well?" — flagging examples like:

  "I can't clearly read the image, but usually in such diagrams there is
   one edge per state into Reset, so the answer is 4."

For every row in a benchmark ndjson where the source model emitted a
non-empty `reasoning_response`, we send the JUDGE model:

  - the question text
  - the source model's final answer
  - the source model's reasoning trace

The judge returns:

  - a confidence score in [0, 1] (see prompt for the exact rubric)
  - a one-or-two sentence summary quoting the language cues

Streams results to `benchmark/runs/reasoning_scores/{basename}.ndjson`
as they land so a mid-run crash keeps everything already scored.

Usage:
  python scripts/score_reasoning.py                                        # default: full_v1.ndjson
  python scripts/score_reasoning.py --input benchmark/runs/full_v1.ndjson  # explicit
  python scripts/score_reasoning.py --judge z-ai/glm-5.3-flash              # (default) judge

Env vars:
  OPENROUTER_API_KEY (or in a .env file at the repo root)
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import re
import time
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parent.parent


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)


try:  # optional: load OPENROUTER_API_KEY from a .env at the repo root
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

BENCH_DIR = ROOT / "benchmark"
DEFAULT_INPUT = BENCH_DIR / "runs" / "full_v1.ndjson"
OUT_DIR = BENCH_DIR / "runs" / "reasoning_scores"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REASONING_MODELS = {
    "claude_haiku_4_5_thinking_on",
    "claude_sonnet_5_thinking_on",
    "claude_opus_5_thinking_on",
    "inkling_small_thinking_on",
    "qwen38_27b_reasoning_low",
    "qwen38_27b_reasoning_high",
}

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _guess_media_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"): return "image/jpeg"
    if ext == ".png":            return "image/png"
    if ext == ".gif":            return "image/gif"
    if ext == ".webp":           return "image/webp"
    if ext == ".bmp":            return "image/bmp"
    return "image/png"


def _normalize_image(field: str, image_dir: str) -> str:
    for pfx in ("images/", image_dir.rstrip("/") + "/"):
        if field.startswith(pfx):
            return field[len(pfx):]
    return field


PROMPT_TEMPLATE = """You are auditing an AI's chain-of-thought about a
hardware-design image that YOU CANNOT SEE. Your only inputs are the question
text and the AI's reasoning trace. You are NOT judging correctness. You are
only judging how confidently the AI itself claims to have read the image.

Look for language cues INSIDE the reasoning:

BLIND GUESS signals (confidence 0.00 – 0.25):
- Explicit admissions that the image is unclear or unreadable:
  "I can't clearly see", "the image is hard to read", "I'm unable to make out"
- Fallback to priors / typical patterns / textbook assumptions:
  "usually in such diagrams", "typically FSMs have", "by convention",
  "assuming symmetry", "it's standard for X to have Y"
- Estimating instead of counting: "there appear to be roughly", "I'll estimate"
- Committing to an answer despite the above hedges

PARTIAL signals (confidence 0.25 – 0.75):
- Some concrete observations mixed with hedging on other parts:
  "I can see A and B, but I'm not sure about C"
- Conditional language: "if I'm reading this correctly", "assuming X"
- Correcting itself mid-trace and settling on a guess for ambiguous parts
- Using labels visible in the image but inferring counts from partial cues

GROUNDED signals (confidence 0.75 – 1.00):
- Concrete visual observations: specific labels, positions, coordinates,
  counts stated as direct observations, arrow directions traced explicitly
- No fallback-to-priors language, no "usually" / "typically" / "should be"
- Confident enumeration: "row 1 has X, row 2 has Y, ..."

Example of a BLIND GUESS trace to calibrate on:
  "I can't read the image clearly, but I can see four states around the
   Reset state. Usually in such diagrams each surrounding state has one
   edge into Reset, so the answer should be 4."
→ confidence ~ 0.15. The model admits it can't read the image AND falls
   back to a textbook prior.

Return STRICT JSON only, no prose, no code fences:

{{
  "confidence": <float in [0.0, 1.0]>,
  "summary": "<one or two sentences describing WHY that score — quote or
              paraphrase the language cues that drove your decision>"
}}

===== QUESTION =====
{question}

===== FINAL ANSWER =====
{final_answer}

===== REASONING =====
{reasoning}
"""


def parse_json(text: str) -> dict | None:
    if not text: return None
    s = text.strip()
    # strip optional ```json fences
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.MULTILINE).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m: return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def _clamp_confidence(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= f <= 1.0):
        return max(0.0, min(1.0, f))
    return f


async def call_judge(
    http: httpx.AsyncClient,
    api_key: str,
    judge_slug: str,
    question_text: str,
    final_answer: str,
    reasoning_text: str,
    max_429_retries: int = 6,
) -> dict:
    """Call the judge. Text-only — image is intentionally NOT attached. The
    judge is looking for self-expressed uncertainty language in the reasoning
    trace, not verifying the reasoning against the actual image."""
    started = time.time()

    prompt = PROMPT_TEMPLATE.format(
        question=question_text.strip(),
        final_answer=str(final_answer),
        reasoning=reasoning_text.strip()[:8000],  # keep prompt bounded
    )

    payload = {
        "model": judge_slug,
        # Judge is a classification task, not a reasoning one. Reasoning-enabled
        # judges (like GLM-5.3-flash where reasoning is mandatory) still work if
        # we set effort=low: 0 reasoning tokens spent, content lands cleanly.
        # Non-reasoning judges (Gemini flash-lite, etc.) ignore this field.
        "max_tokens": 500,
        "temperature": 0.0,
        "top_p": 1.0,
        "response_format": {"type": "json_object"},
        "reasoning": {"effort": "low"},
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }],
    }
    r = None
    for attempt in range(max_429_retries):
        try:
            r = await http.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/rangacodes/SiliconEye",
                    "X-Title": "SiliconEye-reasoning-judge",
                },
                content=json.dumps(payload),
                timeout=45.0,
            )
        except Exception as e:
            return {"status": "error", "reason": f"net:{type(e).__name__}",
                    "elapsed": time.time() - started}
        if r.status_code == 429:
            # Backoff: 4, 8, 16, 32, 64, 90 (cap)
            wait = min(90, 4 * (2 ** attempt))
            await asyncio.sleep(wait)
            continue
        break
    if r is None:
        return {"status": "error", "reason": "no_response",
                "elapsed": time.time() - started}
    if r.status_code == 429:
        return {"status": "error", "reason": "http:429_after_retries",
                "raw": r.text[:400], "elapsed": time.time() - started}
    if r.status_code >= 400:
        return {"status": "error", "reason": f"http:{r.status_code}",
                "raw": r.text[:400], "elapsed": time.time() - started}
    try:
        body = r.json()
        text = body["choices"][0]["message"].get("content") or ""
        provider = body.get("provider")
    except Exception as e:
        return {"status": "error", "reason": f"schema:{type(e).__name__}",
                "raw": r.text[:400], "elapsed": time.time() - started}
    parsed = parse_json(text)
    if not parsed:
        return {"status": "parse_error", "raw": text[:400],
                "elapsed": time.time() - started, "provider": provider}
    conf = _clamp_confidence(parsed.get("confidence"))
    summary = (parsed.get("summary") or "").strip()
    return {
        "status": "ok",
        "confidence": conf,
        "summary": summary,
        "elapsed": time.time() - started,
        "raw_length": len(text),
        "provider": provider,
    }


async def run(input_path: Path, judge_slug: str, concurrency: int,
              limit: int | None = None, out_override: Path | None = None) -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPEN_ROUTER_KEY_V2")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set (export it or put it in .env)")

    # Load questions.yaml to map question_id → question text.
    qy = yaml.safe_load((BENCH_DIR / "questions.yaml").read_text())
    image_dir_default = qy.get("image_dir", "images")
    q_by_id = {q["id"]: q for q in qy["questions"]}

    rows = [json.loads(l) for l in input_path.open()]
    to_score: list[dict] = []
    for r in rows:
        if r.get("model_id") not in REASONING_MODELS: continue
        if not (r.get("reasoning_response") or "").strip(): continue
        if r["question_id"] not in q_by_id: continue  # e.g. deleted questions
        to_score.append(r)

    out_path = out_override or (OUT_DIR / input_path.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if limit is not None:
        to_score = to_score[:limit]
    done_keys: set[tuple[str, str]] = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                r = json.loads(line)
                if r.get("status") == "ok":
                    done_keys.add((r["question_id"], r["model_id"]))
            except json.JSONDecodeError:
                pass
    todo = [r for r in to_score if (r["question_id"], r["model_id"]) not in done_keys]
    eprint(f"[judge] input={input_path.name}  scoreable={len(to_score)}  already_done={len(done_keys)}  todo={len(todo)}")
    if not todo:
        return

    limits = httpx.Limits(max_keepalive_connections=concurrency * 2,
                          max_connections=concurrency * 4)
    write_lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)

    async def persist(row: dict) -> None:
        async with write_lock:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    async with httpx.AsyncClient(limits=limits) as http:
        async def worker(r: dict) -> dict:
            async with sem:
                q = q_by_id[r["question_id"]]
                # Judge is text-only — image is intentionally NOT sent so the
                # judge scores the reasoning's self-expressed confidence, not
                # its factual correctness against the image.
                res = await call_judge(
                    http, api_key, judge_slug,
                    q["question"], r.get("parsed_answer", ""),
                    r.get("reasoning_response", ""),
                )
                row = {
                    "question_id": r["question_id"],
                    "model_id": r["model_id"],
                    "repeat": r.get("repeat", 1),
                    "correct_answer": r.get("correct_answer"),
                    "parsed_answer": r.get("parsed_answer"),
                    "is_correct": r.get("is_correct"),
                    "reasoning_len": len(r.get("reasoning_response") or ""),
                    "judge_model": judge_slug,
                    **res,
                }
                await persist(row)
                return row

        tasks = [worker(r) for r in todo]
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            row = await coro
            st = row.get("status")
            mark = ("?" if st != "ok"
                    else "✓" if row.get("is_correct") is True
                    else "✗")
            conf = row.get("confidence")
            conf_s = f"{conf:.2f}" if conf is not None else "  — "
            eprint(f"  [{i}/{len(tasks)}] {mark} {row['question_id']:<32} "
                   f"{row['model_id']:<32} conf={conf_s} "
                   f"{(row.get('elapsed') or 0):.1f}s")
    try:
        display = out_path.resolve().relative_to(ROOT)
    except (TypeError, ValueError):
        display = out_path
    eprint(f"[judge] streamed → {display}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                    help="benchmark ndjson to score (default: full_v1.ndjson)")
    ap.add_argument("--judge", default="z-ai/glm-5.3-flash",
                    help="OpenRouter slug of the judge model")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap the number of rows scored (for sanity/A-B runs)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Override output ndjson path (default: reasoning_scores/{input.name})")
    args = ap.parse_args()
    asyncio.run(run(args.input, args.judge, args.concurrency,
                    limit=args.limit, out_override=args.out))


if __name__ == "__main__":
    main()
