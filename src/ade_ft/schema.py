"""Task definition: the instruction, the output schema, and robust parsing of model output."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError, field_validator

SEVERITIES = ("mild", "moderate", "severe")

INSTRUCTION = (
    "Extract every adverse drug event (ADE) from the clinical note. An ADE is a symptom the note "
    "attributes to a medication. Ignore symptoms that are denied, resolved before the drug was "
    "started, or caused by something else. Return ONLY JSON of the form "
    '{"events": [{"drug": str, "reaction": str, "severity": "mild"|"moderate"|"severe"}]}. '
    'Return {"events": []} if there are none.'
)


class ADE(BaseModel):
    drug: str
    reaction: str
    severity: str

    @field_validator("drug", "reaction")
    @classmethod
    def _norm(cls, v: str) -> str:
        return re.sub(r"\s+", " ", v.strip().lower())

    @field_validator("severity")
    @classmethod
    def _sev(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}")
        return v


class Extraction(BaseModel):
    events: list[ADE]

    def to_json(self) -> str:
        return json.dumps({"events": [e.model_dump() for e in self.events]}, separators=(", ", ": "))


def parse_output(text: str) -> Extraction | None:
    """Parse a model generation. Returns None when it is not valid schema-conformant JSON."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    candidate = match.group(0)
    # Models sometimes emit trailing text after a complete object; trim to the balanced brace.
    depth = 0
    for i, ch in enumerate(candidate):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            candidate = candidate[: i + 1]
            break
    try:
        return Extraction.model_validate(json.loads(candidate))
    except (json.JSONDecodeError, ValidationError):
        return None
