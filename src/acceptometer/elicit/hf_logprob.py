"""Exact sentence log-probability cells from local Hugging Face models."""

from __future__ import annotations

import math
import string
from datetime import datetime, timezone

from ..items import Item
from .base import CONTINUOUS, CellSpec, Measurement


class HFLogprobScorer:
    """Score causal-LM token probabilities in padded batches."""

    def __init__(
        self,
        model_id: str = "EleutherAI/pythia-160m",
        device: str | None = None,
    ) -> None:
        try:
            import torch
            import transformers
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "HFLogprobScorer requires the 'hf' optional dependencies"
            ) from exc

        self._torch = torch
        self._transformers = transformers
        self.model_id = model_id
        self.model_name = model_id.rsplit("/", 1)[-1]
        mps = getattr(torch.backends, "mps", None)
        self.device = str(
            device or ("mps" if mps is not None and mps.is_available() else "cpu")
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        model_revision = getattr(self.model.config, "_name_or_path", model_id)
        commit = getattr(self.model.config, "_commit_hash", None) or "unknown"
        self.revision = (f"{model_revision}; commit={commit}; "
                         f"transformers={transformers.__version__}")

    def cells(self) -> list[CellSpec]:
        """Return the three deterministic log-probability cells."""
        model = f"hf:{self.model_id}"
        return [
            CellSpec(
                cell_id=f"{self.model_name}/{method}",
                model=model,
                method=method,
                kind=CONTINUOUS,
                params={"deterministic": True},
            )
            for method in ("logprob_sum", "logprob_mean", "slor")
        ]

    def score(
        self,
        items: list[Item],
        cell: CellSpec,
        repeats: int = 1,
    ) -> list[Measurement]:
        """Score one cell; deterministic cells ignore requested repeats."""
        if cell.method not in {"logprob_sum", "logprob_mean", "slor"}:
            raise ValueError(f"unsupported HF log-probability cell: {cell.cell_id}")

        now = datetime.now(timezone.utc).isoformat()
        rows = self._sentence_scores(items)
        out = []
        for item, row in zip(items, rows, strict=True):
            meta = {
                "model_id": self.model_id,
                "revision": self.revision,
                "device": self.device,
                "date": now,
                "n_scored_tokens": row["n_tokens"],
                "first_token_skipped": row["first_token_skipped"],
            }
            if cell.method == "slor":
                meta["unigram_source"] = "wordfreq-zipf"
                meta["slor_definition"] = "(subword_logprob_sum - word_unigram_sum) / n_words"
            out.append(
                Measurement(
                    item_id=item.item_id,
                    cell_id=cell.cell_id,
                    kind=CONTINUOUS,
                    value=float(row[cell.method]),
                    meta=meta,
                )
            )
        return out

    def _sentence_scores(self, items: list[Item]) -> list[dict[str, float | int | bool]]:
        # prefer BOS; fall back to EOS as the conditioning token (standard
        # practice) so every sentence token is scored and the word-level
        # unigram correction stays aligned with the token-level model sum
        bos_id = self.tokenizer.bos_token_id
        if bos_id is None:
            bos_id = self.tokenizer.eos_token_id
        encoded = [
            self.tokenizer.encode(item.text, add_special_tokens=False) for item in items
        ]
        results: list[dict[str, float | int | bool]] = []

        for start in range(0, len(items), 16):
            batch_ids = encoded[start : start + 16]
            sequences = [([bos_id] + ids if bos_id is not None else ids) for ids in batch_ids]
            batch_sums = self._batch_logprob_sums(sequences)
            for item, token_ids, logprob_sum in zip(
                items[start : start + 16], batch_ids, batch_sums, strict=True
            ):
                n_tokens = len(token_ids) if bos_id is not None else max(len(token_ids) - 1, 0)
                # slor is a hybrid quantity here: subword model logprob minus a
                # word-level (wordfreq) unigram sum, normalized per WORD so the
                # two sums share a unit; when the first model token is unscored
                # (no BOS) the first word's unigram term is skipped to match
                unigram_sum, n_words = self._unigram_logprob_sum(
                    item, skip_first=bos_id is None)
                results.append(
                    {
                        "logprob_sum": logprob_sum,
                        "logprob_mean": logprob_sum / n_tokens if n_tokens else math.nan,
                        "slor": (
                            (logprob_sum - unigram_sum) / n_words
                            if n_tokens and n_words
                            else math.nan
                        ),
                        "n_tokens": n_tokens,
                        "n_words": n_words,
                        "first_token_skipped": bos_id is None,
                    }
                )
        return results

    def _batch_logprob_sums(self, sequences: list[list[int]]) -> list[float]:
        if not sequences:
            return []
        torch = self._torch
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = self.tokenizer.bos_token_id
        if pad_id is None:
            pad_id = 0
        width = max(max((len(sequence) for sequence in sequences), default=0), 1)
        input_ids = torch.full(
            (len(sequences), width), pad_id, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros(
            (len(sequences), width), dtype=torch.long, device=self.device
        )
        for row, sequence in enumerate(sequences):
            if sequence:
                input_ids[row, : len(sequence)] = torch.tensor(
                    sequence, dtype=torch.long, device=self.device
                )
                attention_mask[row, : len(sequence)] = 1

        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            log_probs = logits.log_softmax(dim=-1)

        sums = []
        for row, sequence in enumerate(sequences):
            total = 0.0
            for position in range(1, len(sequence)):
                total += log_probs[
                    row, position - 1, sequence[position]
                ].item()
            sums.append(total)
        return sums

    @staticmethod
    def _unigram_logprob_sum(item: Item, skip_first: bool = False) -> tuple[float, int]:
        from wordfreq import zipf_frequency

        total, n = 0.0, 0
        first = True
        for raw_word in item.text.split():
            word = raw_word.strip(string.punctuation).lower()
            if not word:
                continue
            if first and skip_first:
                first = False
                continue
            first = False
            zipf = max(float(zipf_frequency(word, item.language)), 1.0)
            total += (zipf - 9.0) * math.log(10.0)
            n += 1
        return total, n
