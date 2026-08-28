"""Batched acceptability judgments from Claude Opus 4.6 via the agy CLI.

The agy quota is account-wide with multi-day resets, so this instrument uses
a BATCHED protocol: 20 numbered sentences per call, strict-JSON ratings, 3
passes with per-pass shuffling (so batch context varies across repeats).
Batching is a protocol difference from the single-item Ollama cells and is
recorded in the cell id and metadata: the protocol is part of the instrument.

Writes into runs/multi/measurements.jsonl with (item, cell, repeat) dedup.
Unparseable chunks are retried once, then recorded as failures; no values are
ever fabricated.
"""

import json
import re
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from acceptometer.items import load_items
from acceptometer.elicit.base import CONTINUOUS, Measurement, load_measurements

MODEL = "Claude Opus 4.6 (Thinking)"
CELL = "claude-opus-4.6-agy/prompt_scalar_batch20"
OUT = Path("runs/multi/measurements.jsonl")
BATCH = 20
REPEATS = 3

PROMPT = """You are rating English sentences for acceptability to a native speaker, \
on a scale from 1 (completely unacceptable) to 7 (completely natural). Rate each \
sentence independently.

Reply with ONLY a JSON object mapping each sentence number to an integer rating, \
like {{"1": 5, "2": 2}}. No other text.

Sentences:
{sentences}"""


def call_agy(prompt: str) -> str:
    r = subprocess.run(
        ["agy", "--model", MODEL, "--dangerously-skip-permissions", "-p", prompt],
        capture_output=True, text=True, timeout=420)
    return r.stdout + "\n" + r.stderr


def parse_ratings(text: str, n: int) -> dict | None:
    for m in reversed(re.findall(r"\{[^{}]+\}", text, flags=re.S)):
        try:
            d = json.loads(m)
        except json.JSONDecodeError:
            continue
        out = {}
        for k, v in d.items():
            try:
                k, v = int(k), int(v)
            except (TypeError, ValueError):
                continue
            if 1 <= k <= n and 1 <= v <= 7:
                out[k] = float(v)
        if out:
            return out
    return None


def main() -> int:
    items = load_items("data/pilot_items.jsonl")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    have = set()
    if OUT.exists():
        for m in load_measurements(OUT):
            have.add((m.item_id, m.cell_id, m.repeat))

    now = datetime.now(timezone.utc).isoformat()
    prompt_sha = sha256(PROMPT.encode()).hexdigest()[:16]
    n_ok, failures = 0, []
    for rep in range(REPEATS):
        order = np.random.default_rng(rep).permutation(len(items))
        chunks = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
        for ci, chunk in enumerate(chunks):
            todo = [items[j] for j in chunk
                    if (items[j].item_id, CELL, rep) not in have]
            if not todo:
                continue
            listing = "\n".join(f"{k + 1}. {it.text}" for k, it in enumerate(todo))
            prompt = PROMPT.format(sentences=listing)
            ratings = None
            for attempt in range(2):
                try:
                    raw = call_agy(prompt)
                except subprocess.TimeoutExpired:
                    raw = ""
                ratings = parse_ratings(raw, len(todo))
                if ratings:
                    break
            if not ratings:
                failures.append({"repeat": rep, "chunk": ci, "n": len(todo)})
                print(f"rep {rep} chunk {ci}: PARSE FAILURE", flush=True)
                continue
            ms = []
            for k, it in enumerate(todo):
                if (k + 1) in ratings:
                    ms.append(Measurement(
                        item_id=it.item_id, cell_id=CELL, kind=CONTINUOUS,
                        value=ratings[k + 1], repeat=rep,
                        meta={"provider": "antigravity/agy", "model": MODEL,
                              "batch_size": BATCH, "protocol": "batched",
                              "prompt_sha": prompt_sha, "date": now}))
            with open(OUT, "a", encoding="utf-8") as fh:
                for m in ms:
                    fh.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")
            n_ok += len(ms)
            print(f"rep {rep} chunk {ci}: {len(ms)}/{len(todo)} ratings",
                  flush=True)
            time.sleep(2)
    print(f"TOTAL: {n_ok} measurements, {len(failures)} failed chunks")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
