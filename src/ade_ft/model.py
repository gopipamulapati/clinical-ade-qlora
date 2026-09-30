"""Load base model (4-bit when a CUDA GPU + bitsandbytes are available) and attach LoRA adapters."""

from __future__ import annotations

import logging

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import TrainConfig

log = logging.getLogger(__name__)


def _can_quantize() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return False
    return True


def load_tokenizer(name_or_path: str):
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # correct for batched generation; training pads manually
    return tok


def load_base(cfg: TrainConfig, name_or_path: str | None = None):
    path = name_or_path or cfg.base_model
    kwargs: dict = {}
    quantized = cfg.load_in_4bit and _can_quantize()
    if cfg.load_in_4bit and not quantized:
        log.warning("4-bit requested but CUDA/bitsandbytes unavailable - loading full precision (LoRA, not QLoRA).")
    if quantized:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        kwargs["torch_dtype"] = getattr(torch, cfg.bnb_4bit_compute_dtype)
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    return model, quantized


def attach_lora(model, cfg: TrainConfig, quantized: bool):
    if quantized:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)
    elif cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    return model


def trainable_summary(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "pct": round(100 * trainable / total, 4)}


def load_for_inference(base: str, adapter: str | None = None, cfg: TrainConfig | None = None):
    cfg = cfg or TrainConfig(base_model=base, load_in_4bit=False)
    model, _ = load_base(cfg, base)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model
