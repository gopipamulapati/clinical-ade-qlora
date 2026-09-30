"""Supervised fine-tuning with LoRA/QLoRA using the plain HF Trainer (loss on answer tokens only)."""

from __future__ import annotations

import json
import logging
import math
from functools import partial
from pathlib import Path

from transformers import Trainer, TrainingArguments, set_seed

from .config import TrainConfig
from .data import SFTDataset, collate, read_jsonl
from .model import attach_lora, load_base, load_tokenizer, trainable_summary

log = logging.getLogger(__name__)

TINY = "tiny-random"


def resolve_base(cfg: TrainConfig, train_rows: list[dict]) -> str:
    """'tiny-random' -> build a throwaway tiny model (offline smoke runs)."""
    if cfg.base_model != TINY:
        return cfg.base_model
    from .tiny import build_tiny

    corpus = [r["note"] + " " + r["output"] for r in train_rows]
    return str(build_tiny(Path(cfg.output_dir) / "tiny-base", corpus))


def train(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    train_rows, eval_rows = read_jsonl(cfg.data.train), read_jsonl(cfg.data.eval)
    base = resolve_base(cfg, train_rows)

    tokenizer = load_tokenizer(base)
    model, quantized = load_base(cfg, base)
    model = attach_lora(model, cfg, quantized)
    params = trainable_summary(model)
    log.info("trainable params: %s", params)

    train_ds = SFTDataset(train_rows, tokenizer, cfg.max_seq_len)
    eval_ds = SFTDataset(eval_rows, tokenizer, cfg.max_seq_len)

    steps_per_epoch = math.ceil(len(train_ds) / (cfg.per_device_batch_size * cfg.gradient_accumulation_steps))
    total_steps = cfg.max_steps if cfg.max_steps > 0 else math.ceil(steps_per_epoch * cfg.num_epochs)
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=warmup_steps,  # computed: warmup_ratio was removed in transformers 5
        lr_scheduler_type=cfg.lr_scheduler,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],
        seed=cfg.seed,
        bf16=quantized and cfg.bnb_4bit_compute_dtype == "bfloat16",
        fp16=quantized and cfg.bnb_4bit_compute_dtype == "float16",
        optim="paged_adamw_8bit" if quantized else "adamw_torch",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=partial(collate, pad_id=tokenizer.pad_token_id),
    )
    result = trainer.train()
    metrics = {**result.metrics, **trainer.evaluate(), "params": params, "quantized_4bit": quantized, "base": base}

    adapter_dir = Path(cfg.output_dir) / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    (Path(cfg.output_dir) / "train_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    return metrics
