"""Typed training configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class LoraSettings(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])


class DataSettings(BaseModel):
    train: str = "data/train.jsonl"
    eval: str = "data/eval.jsonl"


class TrainConfig(BaseModel):
    base_model: str
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    lora: LoraSettings = Field(default_factory=LoraSettings)
    max_seq_len: int = 512
    learning_rate: float = 2e-4
    num_epochs: float = 2
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.05
    lr_scheduler: str = "cosine"
    gradient_checkpointing: bool = True
    seed: int = 42
    max_steps: int = -1
    data: DataSettings = Field(default_factory=DataSettings)
    output_dir: str = "outputs/run"

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> TrainConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        raw.update({k: v for k, v in overrides.items() if v is not None})
        return cls.model_validate(raw)
