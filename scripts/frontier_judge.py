"""Batched acceptability judgments from Codex and Claude Code frontier models.

The protocol matches ``scripts/agy_judge.py``: 20 numbered sentences per
call, strict-JSON ratings, and three passes with deterministic per-pass
shuffling.  Results append to ``runs/multi/measurements.jsonl`` and are
deduplicated by (item, cell, repeat), so interrupted runs are resumable.

No rating is fabricated or imputed.  A malformed response is retried once;
an inaccessible or mismatched model stops that judge without substituting a
different model.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from acceptometer.elicit.base import CONTINUOUS, Measurement, load_measurements
from acceptometer.items import load_items

OUT = Path("runs/multi/measurements.jsonl")
ITEMS = Path("data/pilot_items.jsonl")
BATCH = 20
REPEATS = 3
TIMEOUT = 420

PROMPT = """You are rating English sentences for acceptability to a native speaker, \
on a scale from 1 (completely unacceptable) to 7 (completely natural). Rate each \
sentence independently.

Reply with ONLY a JSON object mapping each sentence number to an integer rating, \
like {{"1": 5, "2": 2}}. No other text.

Sentences:
{sentences}"""


@dataclass(frozen=True)
class JudgeSpec:
    key: str
    model: str
    label: str
    provider: str
    harness: str
    cell_id: str
    reasoning: str


JUDGES = {
    spec.key: spec
    for spec in (
        JudgeSpec(
            key="opus-5",
            model="claude-opus-5",
            label="Claude Opus 5",
            provider="anthropic/claude-code",
            harness="claude-code",
            cell_id="claude-opus-5-claude-code/prompt_scalar_batch20",
            reasoning="model-default adaptive",
        ),
        JudgeSpec(
            key="gpt-5.6-sol",
            model="gpt-5.6-sol",
            label="GPT-5.6 Sol",
            provider="openai/codex",
            harness="codex",
            cell_id="gpt-5.6-sol-codex/prompt_scalar_batch20",
            reasoning="medium",
        ),
        JudgeSpec(
            key="gpt-5.6-terra",
            model="gpt-5.6-terra",
            label="GPT-5.6 Terra",
            provider="openai/codex",
            harness="codex",
            cell_id="gpt-5.6-terra-codex/prompt_scalar_batch20",
            reasoning="medium",
        ),
        JudgeSpec(
            key="gpt-5.6-luna",
            model="gpt-5.6-luna",
            label="GPT-5.6 Luna",
            provider="openai/codex",
            harness="codex",
            cell_id="gpt-5.6-luna-codex/prompt_scalar_batch20",
            reasoning="medium",
        ),
    )
}


class JudgeCallError(RuntimeError):
    """The requested judge could not produce a usable response."""


def parse_ratings(text: str, n: int) -> dict[int, float] | None:
    """Return a complete 1..n rating map from the last valid JSON object."""
    expected = set(range(1, n + 1))
    for match in reversed(re.findall(r"\{[^{}]+\}", text, flags=re.S)):
        try:
            data = json.loads(match)
        except json.JSONDecodeError:
            continue
        out: dict[int, float] = {}
        for key, value in data.items():
            if isinstance(key, bool) or isinstance(value, bool):
                continue
            try:
                key_int = int(key)
                value_int = int(value)
            except (TypeError, ValueError):
                continue
            if value_int == value and key_int in expected and 1 <= value_int <= 7:
                out[key_int] = float(value_int)
        if set(out) == expected:
            return out
    return None


def claude_result(stdout: str, model: str) -> str:
    """Extract the final text and verify that the requested Claude model ran."""
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise JudgeCallError("Claude Code returned invalid JSON") from exc
    if envelope.get("is_error"):
        raise JudgeCallError(str(envelope.get("result") or "Claude Code failed"))
    usage = envelope.get("modelUsage", {})
    if model not in usage:
        raise JudgeCallError(f"Claude Code did not report requested model {model}")
    result = envelope.get("result")
    if not isinstance(result, str):
        raise JudgeCallError("Claude Code response has no text result")
    return result


def call_claude(prompt: str, spec: JudgeSpec, timeout: int = TIMEOUT) -> str:
    command = [
        "claude",
        "--model",
        spec.model,
        "--safe-mode",
        "--restricted",
        "--tools",
        "",
        "--no-session-persistence",
        "--output-format",
        "json",
        "--prompt-suggestions",
        "false",
        "-p",
        prompt,
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout, check=False
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-500:]
        raise JudgeCallError(f"Claude Code exited {result.returncode}: {detail}")
    return claude_result(result.stdout, spec.model)


def call_codex(prompt: str, spec: JudgeSpec, timeout: int = TIMEOUT) -> str:
    with tempfile.TemporaryDirectory(prefix="acceptometer-codex-") as tmp:
        output = Path(tmp) / "last-message.txt"
        command = [
            "codex",
            "exec",
            "--model",
            spec.model,
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--cd",
            tmp,
            "--config",
            'model_reasoning_effort="medium"',
            "--output-last-message",
            str(output),
            prompt,
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[-500:]
            raise JudgeCallError(f"Codex exited {result.returncode}: {detail}")
        transcript = result.stdout + "\n" + result.stderr
        if f"model: {spec.model}" not in transcript:
            raise JudgeCallError(f"Codex did not report requested model {spec.model}")
        if not output.exists():
            raise JudgeCallError("Codex did not write its final response")
        return output.read_text(encoding="utf-8")


def call_judge(prompt: str, spec: JudgeSpec, timeout: int = TIMEOUT) -> str:
    if spec.harness == "claude-code":
        return call_claude(prompt, spec, timeout)
    if spec.harness == "codex":
        return call_codex(prompt, spec, timeout)
    raise JudgeCallError(f"unknown harness: {spec.harness}")


def cli_version(harness: str) -> str:
    command = (
        ["claude", "--version"]
        if harness == "claude-code"
        else ["codex", "--version"]
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return "unavailable"
    return (result.stdout or result.stderr).strip()[:200]


def append_measurements(path: Path, measurements: list[Measurement]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for measurement in measurements:
            handle.write(json.dumps(asdict(measurement), ensure_ascii=False) + "\n")


def run_judge(
    spec: JudgeSpec,
    items_path: Path,
    out_path: Path,
    batch_size: int = BATCH,
    repeats: int = REPEATS,
    pause: float = 2.0,
) -> tuple[int, list[dict]]:
    items = load_items(items_path)
    have: set[tuple[str, str, int]] = set()
    if out_path.exists():
        have = {
            (measurement.item_id, measurement.cell_id, measurement.repeat)
            for measurement in load_measurements(out_path)
        }

    now = datetime.now(timezone.utc).isoformat()
    prompt_sha = sha256(PROMPT.encode()).hexdigest()[:16]
    version = cli_version(spec.harness)
    n_ok = 0
    failures: list[dict] = []
    for repeat in range(repeats):
        order = np.random.default_rng(repeat).permutation(len(items))
        chunks = [
            order[index : index + batch_size]
            for index in range(0, len(order), batch_size)
        ]
        for chunk_index, chunk in enumerate(chunks):
            todo = [
                items[index]
                for index in chunk
                if (items[index].item_id, spec.cell_id, repeat) not in have
            ]
            if not todo:
                continue
            listing = "\n".join(
                f"{index + 1}. {item.text}" for index, item in enumerate(todo)
            )
            prompt = PROMPT.format(sentences=listing)
            ratings = None
            call_errors: list[str] = []
            for _attempt in range(2):
                try:
                    raw = call_judge(prompt, spec)
                except (JudgeCallError, subprocess.TimeoutExpired) as exc:
                    call_errors.append(str(exc))
                    continue
                ratings = parse_ratings(raw, len(todo))
                if ratings is not None:
                    break
            if ratings is None:
                failure = {
                    "judge": spec.key,
                    "repeat": repeat,
                    "chunk": chunk_index,
                    "n": len(todo),
                    "call_errors": call_errors,
                }
                failures.append(failure)
                print(
                    f"{spec.key} rep {repeat} chunk {chunk_index}: FAILURE",
                    flush=True,
                )
                if call_errors:
                    return n_ok, failures
                continue

            measurements = [
                Measurement(
                    item_id=item.item_id,
                    cell_id=spec.cell_id,
                    kind=CONTINUOUS,
                    value=ratings[index + 1],
                    repeat=repeat,
                    meta={
                        "provider": spec.provider,
                        "harness": spec.harness,
                        "harness_version": version,
                        "model": spec.model,
                        "model_label": spec.label,
                        "reasoning_effort": spec.reasoning,
                        "batch_size": batch_size,
                        "protocol": "batched",
                        "prompt_sha": prompt_sha,
                        "date": now,
                    },
                )
                for index, item in enumerate(todo)
            ]
            append_measurements(out_path, measurements)
            have.update((m.item_id, m.cell_id, m.repeat) for m in measurements)
            n_ok += len(measurements)
            print(
                f"{spec.key} rep {repeat} chunk {chunk_index}: "
                f"{len(measurements)}/{len(todo)} ratings",
                flush=True,
            )
            if pause:
                time.sleep(pause)
    return n_ok, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judge",
        action="append",
        choices=tuple(JUDGES),
        help="judge to run; repeat for several (default: all)",
    )
    parser.add_argument("--items", type=Path, default=ITEMS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--pause", type=float, default=2.0)
    args = parser.parse_args()

    selected = args.judge or list(JUDGES)
    total_ok = 0
    all_failures: list[dict] = []
    for key in selected:
        count, failures = run_judge(
            JUDGES[key],
            args.items,
            args.out,
            batch_size=args.batch_size,
            repeats=args.repeats,
            pause=args.pause,
        )
        total_ok += count
        all_failures.extend(failures)
        print(f"{key}: {count} new measurements, {len(failures)} failures")
    print(f"TOTAL: {total_ok} new measurements, {len(all_failures)} failures")
    return 0 if not all_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
