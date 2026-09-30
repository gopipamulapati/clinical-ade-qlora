"""Synthetic clinical-note generator, prompt formatting and label-masked tokenization.

The notes are **fully synthetic** (templated, no patient data), so the dataset can be shared
and regenerated deterministically. They deliberately include the hard cases that make ADE
extraction non-trivial: negated symptoms, symptoms with a non-drug cause, symptoms that predate
the drug, several drugs per note, and severity cues that have to be mapped to a label.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .schema import ADE, INSTRUCTION, Extraction

DRUG_REACTIONS: dict[str, list[str]] = {
    "metformin": ["diarrhea", "nausea", "abdominal cramping"],
    "lisinopril": ["dry cough", "angioedema", "dizziness"],
    "atorvastatin": ["myalgia", "elevated liver enzymes"],
    "amoxicillin": ["rash", "diarrhea", "hives"],
    "warfarin": ["gum bleeding", "bruising", "hematuria"],
    "sertraline": ["nausea", "insomnia", "headache"],
    "amlodipine": ["ankle edema", "flushing", "headache"],
    "ciprofloxacin": ["tendon pain", "nausea", "photosensitivity"],
    "prednisone": ["hyperglycemia", "insomnia", "mood changes"],
    "gabapentin": ["drowsiness", "dizziness", "peripheral edema"],
    "ibuprofen": ["dyspepsia", "gi bleeding"],
    "hydrochlorothiazide": ["hypokalemia", "muscle cramps"],
    "levothyroxine": ["palpitations", "tremor"],
    "clopidogrel": ["bruising", "epistaxis"],
    "semaglutide": ["nausea", "vomiting", "constipation"],
}

SEVERITY_CUES: dict[str, list[str]] = {
    "mild": ["mild", "slight", "occasional", "minor"],
    "moderate": ["moderate", "bothersome", "persistent", "worsening"],
    "severe": ["severe", "debilitating", "significant"],
}
SEVERE_OUTCOMES = ["requiring an ED visit", "leading to hospital admission", "requiring the drug to be stopped"]

OTHER_CAUSES = ["a recent viral illness", "poor sleep", "dehydration", "a fall last week", "seasonal allergies"]
BACKGROUND_SX = ["fatigue", "headache", "nausea", "dizziness", "back pain", "cough", "rash"]
AGES = list(range(24, 89))
SEXES = ["male", "female"]
REASONS = ["follow-up", "medication review", "annual visit", "post-discharge check", "urgent care visit"]

POSITIVE_TEMPLATES = [
    "Patient reports {sev} {rx} since starting {drug}{outcome}.",
    "Since {drug} was initiated {when}, the patient has had {sev} {rx}{outcome}.",
    "{Rx_cap} ({sev}) is likely secondary to {drug}{outcome}.",
    "Developed {sev} {rx} after {drug} dose increase{outcome}.",
    "Suspected adverse reaction to {drug}: {sev} {rx}{outcome}.",
]
DISTRACTOR_TEMPLATES = [
    "Denies {sx}.",
    "No {sx} reported while on {drug}.",
    "{Sx_cap} attributed to {cause}, not medication related.",
    "History of {sx} predating {drug}, unchanged.",
    "Previous {sx} has fully resolved.",
]
WHEN = ["two weeks ago", "last month", "3 days ago", "in March", "at the last visit"]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def make_example(rng: random.Random) -> dict:
    drugs = rng.sample(sorted(DRUG_REACTIONS), k=rng.choice([1, 2, 2, 3]))
    n_events = rng.choices([0, 1, 2], weights=[0.2, 0.55, 0.25])[0]
    events: list[ADE] = []
    sentences: list[str] = []

    used: set[tuple[str, str]] = set()
    for _ in range(n_events):
        drug = rng.choice(drugs)
        rx = rng.choice(DRUG_REACTIONS[drug])
        if (drug, rx) in used:
            continue
        used.add((drug, rx))
        severity = rng.choice(list(SEVERITY_CUES))
        outcome = f", {rng.choice(SEVERE_OUTCOMES)}" if severity == "severe" and rng.random() < 0.6 else ""
        tpl = rng.choice(POSITIVE_TEMPLATES)
        sentences.append(
            tpl.format(
                sev=rng.choice(SEVERITY_CUES[severity]),
                rx=rx,
                Rx_cap=_cap(rx),
                drug=drug,
                when=rng.choice(WHEN),
                outcome=outcome,
            )
        )
        events.append(ADE(drug=drug, reaction=rx, severity=severity))

    # Hard negatives: symptoms that must NOT be extracted.
    for _ in range(rng.choice([0, 1, 1, 2])):
        sx = rng.choice(BACKGROUND_SX)
        if any(e.reaction == sx for e in events):
            continue
        sentences.append(
            rng.choice(DISTRACTOR_TEMPLATES).format(
                sx=sx, Sx_cap=_cap(sx), drug=rng.choice(drugs), cause=rng.choice(OTHER_CAUSES)
            )
        )

    rng.shuffle(sentences)
    header = (
        f"{rng.choice(AGES)}-year-old {rng.choice(SEXES)} seen for {rng.choice(REASONS)}. "
        f"Current medications: {', '.join(drugs)}."
    )
    footer = rng.choice(["Vitals stable.", "Plan discussed with patient.", "Follow up in 4 weeks.", ""])
    note = " ".join(s for s in [header, *sentences, footer] if s)
    return {"note": note, "output": Extraction(events=events).to_json()}


def generate(n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    return [make_example(rng) for _ in range(n)]


def write_splits(out_dir: str | Path, n_train: int = 2000, n_eval: int = 200, seed: int = 42) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Different seeds so no eval note is a verbatim duplicate of a training note.
    splits = {"train": generate(n_train, seed), "eval": generate(n_eval, seed + 1)}
    train_notes = {r["note"] for r in splits["train"]}
    splits["eval"] = [r for r in splits["eval"] if r["note"] not in train_notes]
    paths = {}
    for name, rows in splits.items():
        p = out / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        paths[name] = p
    return paths


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------------ prompting / tokenization


def build_prompt(note: str, tokenizer) -> str:
    """Use the model's chat template when it has one, else a plain instruction format."""
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": f"Clinical note:\n{note}"},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"### Instruction:\n{INSTRUCTION}\n\n### Clinical note:\n{note}\n\n### Response:\n"


def tokenize_example(example: dict, tokenizer, max_len: int) -> dict:
    """Tokenize prompt + target and mask the prompt with -100 so loss is only on the answer."""
    prompt_ids = tokenizer(build_prompt(example["note"], tokenizer), add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(example["output"], add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [tokenizer.eos_token_id]
    # Never truncate the answer; if too long, drop tokens from the start of the prompt.
    overflow = len(prompt_ids) + len(target_ids) - max_len
    if overflow > 0:
        prompt_ids = prompt_ids[overflow:]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


class SFTDataset:
    def __init__(self, rows: list[dict], tokenizer, max_len: int) -> None:
        self.items = [tokenize_example(r, tokenizer, max_len) for r in rows]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


def collate(batch: list[dict], pad_id: int) -> dict:
    import torch

    width = max(len(b["input_ids"]) for b in batch)

    def pad(seq, value):
        return seq + [value] * (width - len(seq))

    return {
        "input_ids": torch.tensor([pad(b["input_ids"], pad_id) for b in batch]),
        "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
        "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
    }
