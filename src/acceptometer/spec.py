"""Loader for the frozen threshold specification (thresholds.yaml).

Modules read gate values from here so the spec file is the single versioned
authority; the warrant records the spec version and hash it was issued under.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

SPEC_FILE = Path(__file__).parent / "thresholds.yaml"


@lru_cache(maxsize=1)
def load_spec() -> dict:
    return yaml.safe_load(SPEC_FILE.read_text())


def spec_identity() -> dict:
    from .model.fit import sha256_file
    return {"spec_version": load_spec()["spec_version"],
            "spec_sha256": sha256_file(SPEC_FILE)}
