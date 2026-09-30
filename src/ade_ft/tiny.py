"""A tiny random Llama + locally trained BPE tokenizer.

Lets the full train -> eval -> merge -> serve path run offline on CPU in seconds (unit tests
and CI), with no model download. It learns the output *format*, not medicine.
"""

from __future__ import annotations

from pathlib import Path

from .schema import INSTRUCTION


def build_tiny(out_dir: str | Path, corpus: list[str], vocab_size: int = 512) -> Path:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(corpus + [INSTRUCTION], trainer=trainer)
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<pad>"
    )
    hf_tok.save_pretrained(out)

    cfg = LlamaConfig(
        vocab_size=hf_tok.vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        bos_token_id=hf_tok.bos_token_id,
        eos_token_id=hf_tok.eos_token_id,
        pad_token_id=hf_tok.pad_token_id,
    )
    LlamaForCausalLM(cfg).save_pretrained(out)
    return out
