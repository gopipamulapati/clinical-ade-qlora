"""Generation-based evaluation: JSON validity, event-level P/R/F1, severity accuracy.

Run it on the base model and on base+adapter to get a before/after comparison.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .data import build_prompt
from .schema import Extraction, parse_output


@dataclass
class Scores:
    n: int
    json_valid_rate: float
    precision: float
    recall: float
    f1: float
    severity_accuracy: float
    exact_match_rate: float
    empty_note_accuracy: float


def score(preds: list[Extraction | None], golds: list[Extraction]) -> Scores:
    tp = fp = fn = sev_ok = sev_total = exact = 0
    empty_total = empty_ok = 0
    for pred, gold in zip(preds, golds, strict=True):
        gold_pairs = {(e.drug, e.reaction): e.severity for e in gold.events}
        pred_pairs = {(e.drug, e.reaction): e.severity for e in pred.events} if pred else {}
        hits = gold_pairs.keys() & pred_pairs.keys()
        tp += len(hits)
        fp += len(pred_pairs.keys() - gold_pairs.keys())
        fn += len(gold_pairs.keys() - pred_pairs.keys())
        sev_total += len(hits)
        sev_ok += sum(gold_pairs[k] == pred_pairs[k] for k in hits)
        exact += pred is not None and pred_pairs == gold_pairs
        if not gold.events:
            empty_total += 1
            empty_ok += pred is not None and not pred.events
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    n = len(golds)
    return Scores(
        n=n,
        json_valid_rate=round(sum(x is not None for x in preds) / max(n, 1), 4),
        precision=round(p, 4),
        recall=round(r, 4),
        f1=round(2 * p * r / (p + r), 4) if p + r else 0.0,
        severity_accuracy=round(sev_ok / sev_total, 4) if sev_total else 0.0,
        exact_match_rate=round(exact / max(n, 1), 4),
        empty_note_accuracy=round(empty_ok / empty_total, 4) if empty_total else 0.0,
    )


@torch.no_grad()
def generate(model, tokenizer, notes: list[str], max_new_tokens: int = 160, batch_size: int = 8) -> list[str]:
    outs: list[str] = []
    device = next(model.parameters()).device
    for i in range(0, len(notes), batch_size):
        prompts = [build_prompt(n, tokenizer) for n in notes[i : i + batch_size]]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        outs += tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    return outs


def evaluate(model, tokenizer, rows: list[dict], out_path: str | Path | None = None, **gen_kw) -> Scores:
    raw = generate(model, tokenizer, [r["note"] for r in rows], **gen_kw)
    preds = [parse_output(t) for t in raw]
    golds = [Extraction.model_validate_json(r["output"]) for r in rows]
    scores = score(preds, golds)
    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"scores": asdict(scores)}, indent=2))
        with out.with_suffix(".predictions.jsonl").open("w") as f:
            for r, t in zip(rows, raw, strict=True):
                f.write(json.dumps({"note": r["note"], "gold": r["output"], "raw": t}) + "\n")
    return scores


def comparison_table(before: Scores, after: Scores) -> str:
    rows = ["| metric | base | fine-tuned | delta |", "|---|---|---|---|"]
    for k, v in asdict(before).items():
        if k == "n":
            continue
        a = asdict(after)[k]
        rows.append(f"| {k} | {v:.3f} | {a:.3f} | {a - v:+.3f} |")
    return "\n".join(rows)
