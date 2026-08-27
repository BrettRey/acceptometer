"""Prompted acceptability cells served by a local Ollama instance."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import yaml

from ..items import Item
from .base import BINARY, CONTINUOUS, CellSpec, Measurement

_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)


def _strip_thinking(text: str) -> str:
    return _THINK_BLOCK.sub("", text).strip()


def parse_binary(text: str) -> Optional[int]:
    """Parse a leading yes/no answer, rejecting answers containing both."""
    clean = _strip_thinking(text)
    match = re.match(r"^[\W_]*(yes|no)(?![\w'])", clean, re.IGNORECASE)
    if match is None:
        return None
    answers = {
        answer.lower()
        for answer in re.findall(r"\b(?:yes|no)\b", clean, re.IGNORECASE)
    }
    if len(answers) != 1:
        return None
    return 1 if match.group(1).lower() == "yes" else 0


def parse_scalar(text: str, lo: int = 1, hi: int = 7) -> Optional[float]:
    """Return the first standalone integer in the requested range."""
    clean = _strip_thinking(text)
    for match in re.finditer(r"(?<![\w.])[+-]?\d+(?!\w|\.\d)", clean):
        value = int(match.group())
        if lo <= value <= hi:
            return float(value)
    return None


class OllamaChatJudge:
    """Run registered prompts without contacting Ollama at import time."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        host: str = "http://localhost:11434",
        prompts_path: str = "prompts/prompts.yaml",
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.prompts_path = Path(prompts_path)
        with self.prompts_path.open(encoding="utf-8") as fh:
            registry = yaml.safe_load(fh)
        if not isinstance(registry, dict):
            raise ValueError(f"invalid prompt registry: {self.prompts_path}")
        self.prompts: dict[str, dict[str, str]] = {}
        for method in ("binary", "scalar"):
            variants = registry.get(method)
            if not isinstance(variants, dict) or not variants:
                raise ValueError(f"prompt registry is missing {method} variants")
            self.prompts[method] = {str(key): str(value) for key, value in variants.items()}
            if any("{sentence}" not in template for template in self.prompts[method].values()):
                raise ValueError(f"every {method} prompt must contain {{sentence}}")
        self.prompt_hashes = {
            method: {
                variant: hashlib.sha256(template.encode("utf-8")).hexdigest()
                for variant, template in variants.items()
            }
            for method, variants in self.prompts.items()
        }
        self.failures: list[dict[str, str]] = []
        self._client = httpx.Client(base_url=self.host, timeout=120.0)
        self._model_digest: dict | None = None

    def cells(self) -> list[CellSpec]:
        """Return one cell for every registered prompt variant."""
        cells = []
        for registry_method, cell_method, kind in (
            ("binary", "prompt_binary", BINARY),
            ("scalar", "prompt_scalar", CONTINUOUS),
        ):
            for variant in self.prompts[registry_method]:
                cells.append(
                    CellSpec(
                        cell_id=f"{self.model}/{cell_method}/{variant}",
                        model=f"ollama:{self.model}",
                        method=cell_method,
                        kind=kind,
                        prompt_variant=variant,
                        params={
                            "prompt_sha256": self.prompt_hashes[registry_method][variant]
                        },
                    )
                )
        return cells

    def score(
        self,
        items: list[Item],
        cell: CellSpec,
        repeats: int = 1,
    ) -> list[Measurement]:
        """Score items, retrying an unparseable response exactly once."""
        registry_method = {
            "prompt_binary": "binary",
            "prompt_scalar": "scalar",
        }.get(cell.method)
        if registry_method is None or cell.prompt_variant not in self.prompts[registry_method]:
            raise ValueError(f"unsupported Ollama prompt cell: {cell.cell_id}")
        template = self.prompts[registry_method][cell.prompt_variant]
        prompt_hash = self.prompt_hashes[registry_method][cell.prompt_variant]
        model_digest = self._get_model_digest()
        parser = parse_binary if cell.method == "prompt_binary" else parse_scalar
        out = []

        for item in items:
            prompt = template.format(sentence=item.text)
            for repeat in range(repeats):
                # all repeats share ONE sampling regime: mixing temp-0 and
                # stochastic draws in a single likelihood would attach a noise
                # estimate from one regime to a score from another
                temperature = 0.7
                raw = self._chat(prompt, temperature)
                value = parser(raw)
                if value is None:
                    raw = self._chat(prompt, temperature)
                    value = parser(raw)
                if value is None:
                    self.failures.append(
                        {
                            "item_id": item.item_id,
                            "cell_id": cell.cell_id,
                            "raw": raw,
                        }
                    )
                    continue
                out.append(
                    Measurement(
                        item_id=item.item_id,
                        cell_id=cell.cell_id,
                        kind=cell.kind,
                        value=float(value),
                        repeat=repeat,
                        meta={
                            "model_digest": model_digest,
                            "temperature": temperature,
                            "prompt_sha256": prompt_hash,
                            "date": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                )
        return out

    def _get_model_digest(self) -> dict:
        if self._model_digest is not None:
            return self._model_digest
        try:
            response = self._client.post("/api/show", json={"model": self.model})
            response.raise_for_status()
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"could not connect to Ollama at {self.host}; run `ollama serve`"
            ) from exc
        payload = response.json()
        digest = {"details": payload.get("details", {})}
        if payload.get("digest") is not None:
            digest["digest"] = payload["digest"]
        self._model_digest = digest
        return digest

    def _chat(self, prompt: str, temperature: float) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {"temperature": temperature},
        }
        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"could not connect to Ollama at {self.host}; run `ollama serve`"
            ) from exc
        return str(response.json()["message"]["content"])
