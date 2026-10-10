"""Run the SiliconEye benchmark against one or more VLMs.

Usage:
  python scripts/run_benchmark.py                      # default models
  python scripts/run_benchmark.py --models qwen/qwen3-vl-235b-a22b-instruct anthropic/claude-haiku-4.5
  python scripts/run_benchmark.py --questions q1a_slices q2a_reset_in_edges
  python scripts/run_benchmark.py --repeats 3          # each question hit 3 times per model
  python scripts/run_benchmark.py --questions-file benchmark/questions.yaml
  python scripts/run_benchmark.py --run-id my-experiment

The runner:
  - loads a YAML question set (edit it any time to add more questions)
  - sends each (question × model × repeat) via OpenRouter
  - streams a per-call NDJSON row to disk so partial runs are recoverable
  - grades responses (extracts final integer / option letter)
  - writes a markdown summary at the end

Outputs (under benchmark/runs/):
  {run_id}.ndjson        one row per model call (raw + parsed + grade)
  {run_id}.md            per-run summary + per-question comparison + accuracy table

Env vars:
  OPENROUTER_API_KEY   required (or in a .env file at the repo root)
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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
RUNS_DIR = BENCH_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_QUESTIONS = BENCH_DIR / "questions.yaml"
DEFAULT_MODELS_YAML = BENCH_DIR / "models.yaml"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_models(path: Path, only_ids: list[str] | None = None) -> list[dict]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit(f"{path}: expected a top-level list of model configs")
    for m in raw:
        for req in ("id", "slug"):
            if req not in m:
                raise SystemExit(f"{path}: model missing '{req}'")
    if only_ids:
        keep = set(only_ids)
        raw = [m for m in raw if m["id"] in keep]
        missing = keep - {m["id"] for m in raw}
        if missing:
            raise SystemExit(f"unknown model ids in --models: {sorted(missing)}")
    return raw


def _guess_media_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"): return "image/jpeg"
    if ext == ".png":            return "image/png"
    if ext == ".gif":            return "image/gif"
    if ext == ".webp":           return "image/webp"
    if ext == ".bmp":            return "image/bmp"
    return "image/png"


def load_questions(path: Path) -> tuple[str, list[dict]]:
    """Load questions.yaml. Returns (image_dir, questions_list).

    Expected schema:
        image_dir: images        # subdirectory under benchmark/
        questions:
          - id: ...
            image: <basename>    # joined against image_dir at call-time
            ...
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "questions" not in raw:
        raise SystemExit(
            f"{path}: expected a mapping with top-level `image_dir` + `questions` keys"
        )
    image_dir = raw.get("image_dir") or "images"
    qs = raw["questions"]
    if not isinstance(qs, list):
        raise SystemExit(f"{path}: `questions` must be a list")
    for i, q in enumerate(qs):
        for req in ("id", "image", "question", "answer_type", "correct_answer"):
            if req not in q:
                raise SystemExit(f"{path}: question index {i} missing '{req}'")
        if q["answer_type"] not in ("numerical", "yes_no", "multiple_choice", "multiple_choice_multi"):
            raise SystemExit(f"{path}: question {q['id']}: unsupported answer_type={q['answer_type']}")
    return image_dir, qs


# --- answer parsing / grading -------------------------------------------------

INT_RE = re.compile(r"-?\d+")
MCQ_LETTER_RE = re.compile(r"\b([A-Za-z])\b")
YN_RE = re.compile(r"\b([YyNn])\b")
MCQ_MULTI_SET_RE = re.compile(r"\{([^{}]*)\}")


def extract_numerical(text: str) -> int | None:
    """Take the LAST integer in the response. Rationale: models often reason
    aloud with intermediate numbers before stating the final answer; the last
    integer is usually the concluded one."""
    if not text:
        return None
    ints = INT_RE.findall(text.strip())
    if not ints:
        return None
    try:
        return int(ints[-1])
    except ValueError:
        return None


def extract_yes_no(text: str) -> str | None:
    """STRICT parsing: the answer must appear inside a curly-brace set —
    e.g. {Y} or {N} — on the LAST such block in the response. This avoids
    spurious matches on words like "Nope", "Y-axis", or letters that
    incidentally appear as bare tokens inside a reasoning trace."""
    if not text:
        return None
    matches = MCQ_MULTI_SET_RE.findall(text)
    if not matches:
        return None
    inside = matches[-1]
    letters = [c.upper() for c in re.findall(r"[YyNn]", inside)]
    if len(letters) != 1:
        return None
    return letters[0]


def extract_mcq(text: str, valid_keys: list[str]) -> str | None:
    """STRICT parsing: the answer must appear inside a curly-brace set —
    e.g. {A} or {C} — on the LAST such block in the response, and the
    set must contain exactly one valid option letter. This avoids
    spurious matches on English articles and on distractor letters
    mentioned mid-reasoning."""
    if not text:
        return None
    valid = {k.upper() for k in valid_keys}
    matches = MCQ_MULTI_SET_RE.findall(text)
    if not matches:
        return None
    inside = matches[-1]
    letters = {c.upper() for c in re.findall(r"[A-Za-z]", inside) if c.upper() in valid}
    if len(letters) != 1:
        return None
    return next(iter(letters))


def extract_mcq_multi(text: str, valid_keys: list[str]) -> list[str] | None:
    """Extract the set of selected option letters.

    STRICT parsing: only letters that appear inside a curly-brace set
    literal — e.g. {A, C} or {"A", "B", "D"} — are counted. This avoids
    spurious matches on the English article "A" and other single-letter
    tokens that appear in a reasoning trace. If no `{...}` block appears
    in the response, or if the last such block contains no valid keys,
    the answer is treated as unparsed.

    We scan for the LAST `{...}` block in the response, because when
    thinking is on the model often narrates candidates in braces before
    committing to a final answer."""
    if not text:
        return None
    valid = {k.upper() for k in valid_keys}
    matches = MCQ_MULTI_SET_RE.findall(text)
    if not matches:
        return None
    # Use the last brace-set as the model's final answer.
    inside = matches[-1]
    letters = {m.upper() for m in re.findall(r"[A-Za-z]", inside)
               if m.upper() in valid}
    if not letters:
        # Explicit empty set {} means "none of the above" — return an
        # empty list so grading can compare against the correct set. The
        # grader will mark it wrong unless the ground truth is also
        # empty.
        if inside.strip() == "":
            return []
        return None
    return sorted(letters)


def _normalize_multi_answer(value: Any) -> list[str]:
    """Accept a YAML list ([A, D]), a comma/space-separated string ("A, D"),
    or a single letter, and return a sorted uppercase list."""
    if isinstance(value, (list, tuple, set)):
        items = [str(v).strip().upper() for v in value]
    else:
        items = [tok.strip().upper() for tok in re.split(r"[\s,]+", str(value)) if tok.strip()]
    return sorted({x for x in items if x})


def grade(q: dict, response_text: str) -> tuple[bool | None, Any]:
    """Return (is_correct, parsed_answer). is_correct is None if unparseable."""
    at = q["answer_type"]
    if at == "numerical":
        parsed = extract_numerical(response_text)
        if parsed is None:
            return (None, None)
        return (parsed == int(q["correct_answer"]), parsed)
    if at == "yes_no":
        parsed = extract_yes_no(response_text)
        if parsed is None:
            return (None, None)
        return (parsed == str(q["correct_answer"]).strip().upper(), parsed)
    if at == "multiple_choice":
        keys = [opt["key"].upper() for opt in (q.get("options") or [])]
        parsed = extract_mcq(response_text, keys)
        if parsed is None:
            return (None, None)
        return (parsed == str(q["correct_answer"]).upper(), parsed)
    if at == "multiple_choice_multi":
        keys = [opt["key"].upper() for opt in (q.get("options") or [])]
        parsed = extract_mcq_multi(response_text, keys)
        if parsed is None:
            return (None, None)
        expected = _normalize_multi_answer(q["correct_answer"])
        return (parsed == expected, parsed)
    return (None, None)


# --- API call -----------------------------------------------------------------

def build_payload(model_cfg: dict, question: dict, data_url: str) -> dict:
    """Assemble the OpenRouter chat-completions request body, threading every
    per-model knob (temperature, top_p, max_tokens, seed, provider pin) from
    the model config. Anything not set in the config is simply omitted so the
    provider's own default applies."""
    user_text = question["question"].strip()
    at = question["answer_type"]
    if at == "numerical":
        user_text = f"{user_text}\n\nRespond with a single integer, and NOTHING ELSE."
    elif at == "yes_no":
        user_text = (f"{user_text}\n\nGive your final answer as {{Y}} or {{N}} "
                     f"on the LAST line of your response. Only the letter inside "
                     f"the final {{...}} is graded.")
    elif at in ("multiple_choice", "multiple_choice_multi"):
        opts = question.get("options") or []
        opt_lines = "\n".join(f'  {o["key"]}) {o["text"]}' for o in opts)
        if at == "multiple_choice":
            tail = ("Give your final answer as a single option letter inside curly "
                    "braces on the LAST line of your response — e.g. {A} or {C}. "
                    "Only the letter inside the final {...} is graded.")
        else:
            tail = ("Give your final answer as a curly-brace set of option letters "
                    "on the LAST line of your response — for example {A, C} or {B} "
                    "or {A, B, D}. Only letters inside the final {...} are graded. "
                    "Use {} if you believe none of the options apply.")
        user_text = f"{user_text}\n\nOptions:\n{opt_lines}\n\n{tail}"

    payload: dict = {
        "model": model_cfg["slug"],
        "max_tokens": int(model_cfg.get("max_tokens", 2000)),
        "messages": [{
            "role": "user",
            "content": [
                # Text first, image second — safer default across providers;
                # a few multi-provider routes were previously silent-dropping
                # the image when it was the leading block.
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    }
    if "temperature" in model_cfg:
        payload["temperature"] = float(model_cfg["temperature"])
    if "top_p" in model_cfg:
        payload["top_p"] = float(model_cfg["top_p"])
    if "seed" in model_cfg:
        payload["seed"] = int(model_cfg["seed"])
    if model_cfg.get("provider"):
        payload["provider"] = {"order": [model_cfg["provider"]], "allow_fallbacks": False}
    # Reasoning controls — OpenRouter cross-provider surface. Use exactly one
    # of these:
    #   reasoning_effort: "low" | "medium" | "high"     (simple)
    #   reasoning: { enabled, effort, max_tokens, exclude }   (verbose)
    used_reasoning = False
    if "reasoning_effort" in model_cfg:
        payload["reasoning_effort"] = model_cfg["reasoning_effort"]
        used_reasoning = True
    if "reasoning" in model_cfg:
        payload["reasoning"] = model_cfg["reasoning"]
        used_reasoning = True
    # OpenRouter withholds the reasoning content field by default even when
    # the model spent reasoning tokens — you have to explicitly ask for it.
    # Always request inclusion when the config asks for any reasoning behaviour,
    # so we can inspect chain-of-thought in the ndjson.
    if used_reasoning:
        payload["include_reasoning"] = True
    return payload


async def call_model(
    http: httpx.AsyncClient,
    api_key: str,
    model_cfg: dict,
    question: dict,
    image_dir: str = "images",
) -> dict:
    # The `image` field is supposed to be a basename that's joined against
    # image_dir. Historically some questions embedded the image_dir prefix
    # inline ("images/q1_source.jpeg"). Strip a leading "images/" (or the
    # currently-active dir) so we don't double-nest.
    img_basename = question["image"]
    for legacy_prefix in ("images/", image_dir.rstrip("/") + "/"):
        if img_basename.startswith(legacy_prefix):
            img_basename = img_basename[len(legacy_prefix):]
            break
    img_path = (BENCH_DIR / image_dir / img_basename).resolve()
    if not img_path.exists():
        return {"status": "error", "reason": f"missing_image:{img_path}"}
    b64 = base64.standard_b64encode(img_path.read_bytes()).decode()
    data_url = f"data:{_guess_media_type(img_path)};base64,{b64}"

    payload = build_payload(model_cfg, question, data_url)
    timeout = float(model_cfg.get("timeout_s", 30))

    # Trimmed copy of the payload for on-disk provenance (drops the giant
    # base64 image but keeps every hyperparameter).
    payload_meta = {k: v for k, v in payload.items() if k != "messages"}

    started = time.time()
    try:
        r = await http.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/rangacodes/SiliconEye",
                "X-Title": "SiliconEye-eval",
            },
            content=json.dumps(payload),
            timeout=timeout,
        )
    except Exception as e:
        return {"status": "error", "reason": f"net:{type(e).__name__}",
                "elapsed": time.time() - started,
                "request_config": payload_meta}
    if r.status_code >= 400:
        return {"status": "error", "reason": f"http:{r.status_code}",
                "raw": r.text[:400],
                "elapsed": time.time() - started,
                "request_config": payload_meta}
    try:
        body = r.json()
        msg = body["choices"][0]["message"]
        text = msg.get("content")
        # Some providers put chain-of-thought in a separate `reasoning` field.
        # We keep it so the row is inspectable, but grading always uses content.
        reasoning_text = msg.get("reasoning") or msg.get("reasoning_content")
        usage = body.get("usage") or {}
        provider_id = body.get("provider") or (body.get("model_metadata") or {}).get("provider")
    except Exception as e:
        return {"status": "error", "reason": f"schema:{type(e).__name__}",
                "raw": r.text[:500],
                "elapsed": time.time() - started,
                "request_config": payload_meta}

    # If content is empty, note it distinctly — usually means max_tokens got
    # consumed by hidden reasoning tokens and the visible answer was truncated.
    finish_reason = (body["choices"][0].get("finish_reason") or "").lower()
    empty_content = not (text or "").strip()
    return {
        "status": "ok" if not empty_content else "empty_content",
        "raw_response": text,
        "reasoning_response": reasoning_text,
        "finish_reason": finish_reason,
        "elapsed": time.time() - started,
        "usage": usage,
        "provider_id_returned": provider_id,
        "request_config": payload_meta,
    }


# --- runner + summary ---------------------------------------------------------

async def run(
    model_cfgs: list[dict],
    questions: list[dict],
    repeats: int,
    run_id: str,
    concurrency: int,
    image_dir: str = "images",
) -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPEN_ROUTER_KEY_V2")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set (export it or put it in .env)")

    ndjson_path = RUNS_DIR / f"{run_id}.ndjson"
    md_path = RUNS_DIR / f"{run_id}.md"
    cfg_path = RUNS_DIR / f"{run_id}.config.json"
    # Allow slash-separated run_ids (e.g. "resize/deg_s25") so different
    # degradation axes can live in distinct subfolders under runs/.
    ndjson_path.parent.mkdir(parents=True, exist_ok=True)

    # Persist the model configs and question ids used, in full, alongside the
    # run outputs — makes reruns exactly reproducible.
    cfg_path.write_text(json.dumps({
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "image_dir": image_dir,
        "models": model_cfgs,
        "questions": [{"id": q["id"], "correct_answer": q["correct_answer"]} for q in questions],
        "repeats": repeats,
        "concurrency": concurrency,
    }, indent=2), encoding="utf-8")

    # Resume: skip (question × model_id × repeat) already recorded.
    done: set[tuple[str, str, int]] = set()
    if ndjson_path.exists():
        for r in _iter_ndjson(ndjson_path):
            k = (r.get("question_id"), r.get("model_id"), r.get("repeat"))
            if all(x is not None for x in k):
                done.add(k)  # type: ignore[arg-type]
        if done:
            eprint(f"[bench] {len(done)} calls already recorded — will skip on resume")

    calls: list[tuple[dict, dict, int]] = []
    for q in questions:
        for mc in model_cfgs:
            for i in range(1, repeats + 1):
                if (q["id"], mc["id"], i) in done:
                    continue
                calls.append((q, mc, i))
    eprint(f"[bench] run_id={run_id}  models={len(model_cfgs)}  questions={len(questions)}  "
           f"repeats={repeats}  todo={len(calls)}")

    if not calls:
        eprint("[bench] nothing to do — writing summary from existing rows")
        _write_summary(md_path, ndjson_path, model_cfgs, questions, repeats, run_id)
        return

    limits = httpx.Limits(max_keepalive_connections=concurrency * 2, max_connections=concurrency * 4)
    write_lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)

    async def persist(row: dict) -> None:
        async with write_lock:
            with ndjson_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    async with httpx.AsyncClient(limits=limits) as http:

        async def one_call(q: dict, mc: dict, repeat: int):
            async with sem:
                res = await call_model(http, api_key, mc, q, image_dir=image_dir)
                row: dict = {
                    "run_id": run_id,
                    "image_dir": image_dir,
                    "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "question_id": q["id"],
                    "model_id": mc["id"],
                    "model_slug": mc["slug"],
                    "repeat": repeat,
                    "correct_answer": q["correct_answer"],
                    "answer_type": q["answer_type"],
                    **res,
                }
                if res.get("status") == "ok":
                    ok, parsed = grade(q, res["raw_response"])
                    row["parsed_answer"] = parsed
                    row["is_correct"] = ok
                await persist(row)
                return row

        eprint(f"[bench] concurrency={concurrency}")
        tasks = [one_call(q, mc, i) for (q, mc, i) in calls]
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            row = await coro
            _log_row(i, len(tasks), row)

    _write_summary(md_path, ndjson_path, model_cfgs, questions, repeats, run_id)


def _iter_ndjson(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _log_row(i: int, total: int, row: dict) -> None:
    st = row.get("status")
    elapsed = row.get("elapsed", 0.0)
    if st == "ok":
        mark = "✓" if row.get("is_correct") is True else ("✗" if row.get("is_correct") is False else "?")
        eprint(f"  [{i}/{total}] {mark} {row['question_id']:<26} {row['model_id']:<32} "
               f"parsed={row.get('parsed_answer')!r:>8} vs {row['correct_answer']!r:<3} "
               f"{elapsed:.1f}s")
    elif st == "empty_content":
        eprint(f"  [{i}/{total}] ⌀ EMPTY {row['question_id']:<26} {row['model_id']:<32} "
               f"finish_reason={row.get('finish_reason')}  {elapsed:.1f}s")
    else:
        eprint(f"  [{i}/{total}] ✗ ERROR {row['question_id']:<26} {row['model_id']:<32} "
               f"{row.get('status')}:{row.get('reason','?')[:60]}")


def _write_summary(md_path: Path, ndjson_path: Path, model_cfgs: list[dict],
                   questions: list[dict], repeats: int, run_id: str) -> None:
    rows = list(_iter_ndjson(ndjson_path))
    # Index rows by (question_id, model_id) -> list of rows
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["question_id"], r["model_id"])].append(r)

    # Union current YAML configs with anything already in the ndjson from
    # prior calls under the same run_id — that way, adding models to the
    # yaml and re-running (or shrinking the yaml) doesn't wipe old results
    # from the summary. The current YAML wins when there's a conflict, so
    # if you edit a model's hyperparameters between invocations, the *latest*
    # config appears in the "Model configs" table.
    yaml_ids = {mc["id"] for mc in model_cfgs}
    ndjson_model_ids = {r["model_id"] for r in rows}
    for extra_id in sorted(ndjson_model_ids - yaml_ids):
        sample = next(r for r in rows if r["model_id"] == extra_id)
        rc = sample.get("request_config") or {}
        prov = rc.get("provider")
        prov_slug = prov["order"][0] if isinstance(prov, dict) and prov.get("order") else None
        model_cfgs.append({
            "id": extra_id,
            "slug": rc.get("model") or sample.get("model_slug") or "?",
            "provider": prov_slug,
            "temperature": rc.get("temperature"),
            "top_p": rc.get("top_p"),
            "max_tokens": rc.get("max_tokens"),
            "timeout_s": None,
            "reasoning": rc.get("reasoning"),
            "reasoning_effort": rc.get("reasoning_effort"),
            "_from_history": True,
        })

    yaml_qids = {q["id"] for q in questions}
    ndjson_qids = {r["question_id"] for r in rows}
    for extra_qid in sorted(ndjson_qids - yaml_qids):
        sample = next(r for r in rows if r["question_id"] == extra_qid)
        questions.append({
            "id": extra_qid,
            "topic": "?",
            "source": "(from history)",
            "image": None,
            "question": "(question text no longer in questions.yaml)",
            "correct_answer": sample.get("correct_answer"),
            "answer_type": sample.get("answer_type", "numerical"),
            "_from_history": True,
        })

    # Effective repeats = max repeat index seen in the ndjson (some models
    # may have been re-run more times than others as the yaml evolved).
    max_repeat = max((r.get("repeat") or 1) for r in rows) if rows else repeats
    repeats = max(repeats, max_repeat)

    # Per-model overall accuracy + latency
    per_model_stats: dict[str, dict] = {}
    for mc in model_cfgs:
        stats = {"correct": 0, "wrong": 0, "unparsed": 0, "empty": 0, "error": 0, "total": 0, "latencies": []}
        for q in questions:
            for r in grouped.get((q["id"], mc["id"]), []):
                stats["total"] += 1
                st = r.get("status")
                if st == "empty_content":
                    stats["empty"] += 1
                    if isinstance(r.get("elapsed"), (int, float)):
                        stats["latencies"].append(r["elapsed"])
                    continue
                if st != "ok":
                    stats["error"] += 1
                    continue
                ic = r.get("is_correct")
                if ic is True: stats["correct"] += 1
                elif ic is False: stats["wrong"] += 1
                else: stats["unparsed"] += 1
                if isinstance(r.get("elapsed"), (int, float)):
                    stats["latencies"].append(r["elapsed"])
        per_model_stats[mc["id"]] = stats

    lines = [
        f"# Benchmark run — `{run_id}`\n",
        f"- Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Question set: {len(questions)} unique questions × {repeats} repeat(s) × {len(model_cfgs)} model(s) "
        f"= {len(questions)*repeats*len(model_cfgs)} calls",
        f"- Raw log: [`{ndjson_path.name}`]({ndjson_path.name})",
        f"- Run config: [`{run_id}.config.json`]({run_id}.config.json)",
        "",
        "## Model configs (as sent)",
        "",
        "| id | slug | provider | temperature | top_p | max_tokens | timeout |",
        "|---|---|---|--:|--:|--:|--:|",
    ]
    for mc in model_cfgs:
        marker = " †" if mc.get("_from_history") else ""
        lines.append(
            f"| `{mc['id']}{marker}` | `{mc['slug']}` | "
            f"{'`'+mc['provider']+'`' if mc.get('provider') else '—'} | "
            f"{mc.get('temperature') if mc.get('temperature') is not None else '—'} | "
            f"{mc.get('top_p') if mc.get('top_p') is not None else '—'} | "
            f"{mc.get('max_tokens') if mc.get('max_tokens') is not None else '—'} | "
            f"{mc.get('timeout_s') if mc.get('timeout_s') is not None else '—'}s |"
        )
    if any(mc.get("_from_history") for mc in model_cfgs):
        lines.append("")
        lines.append("_† Config reconstructed from prior ndjson rows — not present in the current `models.yaml`._")

    lines += [
        "",
        "## Overall accuracy",
        "",
        "| Model | Correct | Wrong | Unparsed | Empty | Error | Accuracy | Avg latency |",
        "|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for mc in model_cfgs:
        s = per_model_stats[mc["id"]]
        acc = (100.0 * s["correct"] / max(1, s["total"]))
        lat = (sum(s["latencies"]) / max(1, len(s["latencies"])))
        lines.append(
            f"| `{mc['id']}` | {s['correct']} | {s['wrong']} | {s['unparsed']} | "
            f"{s['empty']} | {s['error']} | {acc:.1f}% | {lat:.2f}s |"
        )

    lines += ["", "## Per-question results", ""]
    for q in questions:
        # `image` is a basename under `image_dir`. Grab image_dir from the first
        # row we have for this question (rows persist the actual dir used at
        # call time — handles clean, JPEG-degraded, resize-degraded runs alike).
        sample_row = next(
            (r for rs in [grouped.get((q["id"], mc["id"]), []) for mc in model_cfgs]
             for r in rs),
            None,
        )
        row_image_dir = (sample_row or {}).get("image_dir") or "images"

        # Normalise the `image` field the same way the runner does — strip a
        # leading "images/" or the active image_dir prefix so basename+dir
        # composition doesn't double-nest.
        q_image = (q.get("image") or "")
        for legacy_prefix in ("images/", row_image_dir.rstrip("/") + "/"):
            if q_image.startswith(legacy_prefix):
                q_image = q_image[len(legacy_prefix):]
                break

        # Historical image_dir shims — the folder layout has been renamed a
        # couple of times as new degradation axes landed. When the recorded
        # dir is missing, transparently retarget to the current location so
        # old runs still render.
        def _resolve_image_dir(recorded: str) -> str:
            if (BENCH_DIR / recorded / q_image).exists():
                return recorded
            # images_degraded/{N}  →  degraded/jpeg/{N}   (post 2026-08 reorg)
            if recorded.startswith("images_degraded/"):
                shim = "degraded/jpeg/" + recorded.split("/", 1)[1]
                if (BENCH_DIR / shim / q_image).exists():
                    return shim
            return recorded  # fall through; will render as "not found"

        img_tag = "_(image reference unavailable)_"
        if q_image:
            resolved_dir = _resolve_image_dir(row_image_dir)
            img_abs = (BENCH_DIR / resolved_dir / q_image).resolve()
            try:
                img_rel = img_abs.relative_to(md_path.parent.resolve(), walk_up=True)
            except (TypeError, ValueError):
                img_rel = Path("..") / resolved_dir / q_image
            if img_abs.exists():
                img_tag = f'<img src="{img_rel}" width="480" />'
            else:
                img_tag = f"_image file `{resolved_dir}/{q_image}` not found on disk_"
            row_image_dir = resolved_dir  # for the label line below

        # Full question text as a Markdown blockquote (one `> ` per line).
        q_text = (q.get("question") or "").strip()
        quoted = "\n".join(f"> {line}" if line else ">" for line in q_text.splitlines())
        image_meta = f"_({row_image_dir}/{q_image})_" if q_image else ""

        lines += [
            f"### `{q['id']}` — {q.get('topic','?')} · correct = **{q['correct_answer']}**",
            "",
            f"_Source: {q.get('source','?')}_",
            "",
            img_tag,
            image_meta,
            "",
            quoted,
            "",
        ]
        lines.append("| Model | " + " | ".join(f"attempt {i}" for i in range(1, repeats + 1)) + " |")
        lines.append("|---|" + "|".join(["---"] * repeats) + "|")
        for mc in model_cfgs:
            rs = grouped.get((q["id"], mc["id"]), [])
            rs_sorted = sorted(rs, key=lambda r: r.get("repeat", 1))
            cells: list[str] = []
            for i in range(1, repeats + 1):
                r = next((x for x in rs_sorted if x.get("repeat") == i), None)
                if r is None:
                    cells.append("—"); continue
                st = r.get("status")
                if st == "empty_content":
                    cells.append(f"⌀ empty<br/><sub>finish={r.get('finish_reason','?')}</sub>")
                    continue
                if st != "ok":
                    cells.append(f"❌ err<br/><sub>{(r.get('reason') or '')[:30]}</sub>")
                    continue
                ic = r.get("is_correct")
                parsed = r.get("parsed_answer")
                mark = "✅" if ic is True else ("❌" if ic is False else "❓")
                cells.append(f"{mark} `{parsed}` <sub>{r.get('elapsed',0):.1f}s</sub>")
            lines.append(f"| `{mc['id']}` | " + " | ".join(cells) + " |")
        lines.append("")
        for mc in model_cfgs:
            rs = grouped.get((q["id"], mc["id"]), [])
            if not rs: continue
            example = next((r for r in rs if r.get("status") == "ok"), rs[0])
            raw = (example.get("raw_response") or example.get("reason") or "")[:600]
            if raw:
                lines.append(f"<details><summary>raw response — `{mc['id']}`</summary>\n\n```\n{raw}\n```\n</details>\n")
        lines.append("---\n")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    eprint(f"[bench] wrote {md_path.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-file", default=str(DEFAULT_MODELS_YAML),
                    help="YAML file describing the model roster")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Only run these model ids (default: all in models.yaml)")
    ap.add_argument("--questions-file", default=str(DEFAULT_QUESTIONS),
                    help="YAML file describing the question set")
    ap.add_argument("--questions", nargs="+", default=None,
                    help="Only run these question ids (default: all)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="Number of times each question is asked per model")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--run-id", default=None,
                    help="Output basename under benchmark/runs/. Default: timestamp.")
    ap.add_argument("--render-only", action="store_true",
                    help="Skip all API calls; just rebuild {run_id}.md from the existing ndjson.")
    ap.add_argument("--degraded", type=str, default=None, metavar="AXIS/LEVEL",
                    help="Serve degraded images instead of the clean sources. "
                         "Value is `AXIS/LEVEL` — e.g. `jpeg/25` or `resize/50` — "
                         "and maps to `benchmark/degraded/{AXIS}/{LEVEL}/`. "
                         "Backward-compat: a bare integer (e.g. `25`) is treated "
                         "as `jpeg/25`. Directory must be created first via "
                         "`scripts/degrade_images.py --axis {AXIS}`.")
    args = ap.parse_args()

    model_cfgs = load_models(Path(args.models_file), only_ids=args.models)
    image_dir, all_qs = load_questions(Path(args.questions_file))
    if args.degraded is not None:
        val = args.degraded.strip("/")
        # Backward-compat: a bare integer means the JPEG axis.
        if "/" not in val:
            val = f"jpeg/{val}"
        image_dir = f"degraded/{val}"
        candidate = (BENCH_DIR / image_dir).resolve()
        if not candidate.exists():
            raise SystemExit(
                f"--degraded {args.degraded}: {candidate} does not exist. "
                f"Run scripts/degrade_images.py --axis {val.split('/')[0]} first."
            )
    if args.questions:
        keep = set(args.questions)
        qs = [q for q in all_qs if q["id"] in keep]
        missing = keep - {q["id"] for q in qs}
        if missing:
            raise SystemExit(f"unknown question ids: {sorted(missing)}")
    else:
        qs = all_qs

    run_id = args.run_id or datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    (RUNS_DIR / run_id).parent.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        md_path = RUNS_DIR / f"{run_id}.md"
        ndjson_path = RUNS_DIR / f"{run_id}.ndjson"
        if not ndjson_path.exists():
            raise SystemExit(f"no ndjson to render: {ndjson_path}")
        _write_summary(md_path, ndjson_path, model_cfgs, qs, args.repeats, run_id)
        eprint(f"[bench] render-only: rebuilt {md_path.name}")
        return
    asyncio.run(run(model_cfgs, qs, args.repeats, run_id, args.concurrency, image_dir=image_dir))


if __name__ == "__main__":
    main()
