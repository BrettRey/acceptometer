"""Render the published eleven-judge table from tracked aggregate evidence.

The restricted pilot inputs and run directory are intentionally not in Git.
This script makes the redistributable aggregate table deterministic and can
refuse a release when ``PILOT-REPORT.md`` has drifted from that evidence.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

START_MARKER = "<!-- BEGIN GENERATED JUDGE TABLE -->"
END_MARKER = "<!-- END GENERATED JUDGE TABLE -->"
DEFAULT_EVIDENCE = Path("results/v0.1/judge-table.json")
DEFAULT_REPORT = Path("PILOT-REPORT.md")


def load_evidence(path: Path) -> dict:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    judges = evidence.get("judges")
    if evidence.get("schema_version") != 1 or not isinstance(judges, list):
        raise ValueError("unsupported or malformed judge-table evidence")
    ids = [judge.get("instrument_id") for judge in judges]
    if len(judges) != 11 or len(set(ids)) != 11:
        raise ValueError("v0.1 evidence must contain eleven unique judges")
    for judge in judges:
        for key in ("label", "r_all", "r_marginal_band", "r_outside_band"):
            if key not in judge:
                raise ValueError(f"judge row missing {key}: {judge!r}")
    source_hashes = evidence.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("v0.1 evidence must contain source SHA-256 hashes")
    for source, digest in source_hashes.items():
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"invalid SHA-256 for {source}: {digest!r}")
    return evidence


def format_correlation(value: float, *, digits: int = 2,
                       italic: bool = False, bold: bool = False) -> str:
    text = f"{value:.{digits}f}"
    if text.startswith("-0."):
        text = "-." + text[3:]
    elif text.startswith("0."):
        text = "." + text[2:]
    if bold:
        text = f"**{text}**"
    if italic:
        text = f"*{text}*"
    return text


def render_table(evidence: dict) -> str:
    judges = sorted(evidence["judges"], key=lambda row: -row["r_all"])
    best_band = max(round(row["r_marginal_band"], 2) for row in judges)
    bold_all_min = evidence["display"]["bold_r_all_min"]
    lines = [
        "| judge | r all | r band | r outside |",
        "|---|---|---|---|",
    ]
    for row in judges:
        all_value = format_correlation(
            row["r_all"], bold=round(row["r_all"], 2) >= bold_all_min
        )
        band_value = format_correlation(
            row["r_marginal_band"],
            bold=round(row["r_marginal_band"], 2) == best_band,
        )
        outside_value = format_correlation(row["r_outside_band"])
        lines.append(
            f"| {row['label']} | {all_value} | {band_value} | {outside_value} |"
        )
    ceiling = evidence["human"]["split_half_r_mean_200_splits"]
    ceiling_digits = evidence["display"]["human_ceiling_digits"]
    lines.append(
        f"| *human split-half ceiling* | "
        f"{format_correlation(ceiling, digits=ceiling_digits, italic=True)} | | |"
    )
    return "\n".join(lines)


def generated_block(evidence: dict) -> str:
    return f"{START_MARKER}\n{render_table(evidence)}\n{END_MARKER}"


def replace_generated_block(report: str, block: str) -> str:
    if report.count(START_MARKER) != 1 or report.count(END_MARKER) != 1:
        raise ValueError("report must contain exactly one generated-table marker pair")
    start = report.index(START_MARKER)
    end = report.index(END_MARKER, start) + len(END_MARKER)
    return report[:start] + block + report[end:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="update the report")
    mode.add_argument("--check", action="store_true", help="refuse report drift")
    args = parser.parse_args()

    evidence = load_evidence(args.evidence)
    block = generated_block(evidence)
    if not (args.write or args.check):
        print(render_table(evidence))
        return 0

    current = args.report.read_text(encoding="utf-8")
    expected = replace_generated_block(current, block)
    if args.check:
        if current != expected:
            print(f"REFUSED: {args.report} judge table has drifted from {args.evidence}")
            return 1
        print(f"OK: {args.report} judge table matches {args.evidence}")
        return 0

    args.report.write_text(expected, encoding="utf-8")
    print(f"updated {args.report} from {args.evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
