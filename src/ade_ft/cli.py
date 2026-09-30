"""ade-ft CLI: gen-data -> train -> compare (base vs fine-tuned) -> merge -> serve."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import typer

from .config import TrainConfig
from .data import read_jsonl, write_splits

app = typer.Typer(add_completion=False, help="QLoRA fine-tuning for clinical ADE extraction.")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@app.command("gen-data")
def gen_data(out: str = "data", n_train: int = 2000, n_eval: int = 200, seed: int = 42) -> None:
    """Generate the synthetic train/eval splits."""
    paths = write_splits(out, n_train, n_eval, seed)
    for name, p in paths.items():
        typer.echo(f"{name}: {p} ({sum(1 for _ in p.open())} rows)")


@app.command()
def train(config: str, max_steps: int = typer.Option(None, help="Override for quick runs.")) -> None:
    """Fine-tune with LoRA (QLoRA on GPU)."""
    from .train import train as run

    cfg = TrainConfig.from_yaml(config, max_steps=max_steps)
    metrics = run(cfg)
    typer.echo(json.dumps({k: metrics[k] for k in ("train_loss", "eval_loss", "params") if k in metrics}, indent=2))


def _eval_one(base: str, adapter: str | None, rows: list[dict], out: Path, label: str):
    from .evaluate import evaluate
    from .model import load_for_inference, load_tokenizer

    model = load_for_inference(base, adapter)
    tok = load_tokenizer(adapter or base)
    scores = evaluate(model, tok, rows, out / f"{label}.json")
    typer.echo(f"{label}: {json.dumps(asdict(scores))}")
    return scores


@app.command()
def compare(
    config: str,
    limit: int = typer.Option(200, help="Eval rows to use."),
    out: str = typer.Option(None, help="Report dir (default: <output_dir>/eval)."),
) -> None:
    """Evaluate the base model and base+adapter on the eval split; write a markdown table."""
    from .evaluate import comparison_table

    cfg = TrainConfig.from_yaml(config)
    rows = read_jsonl(cfg.data.eval)[:limit]
    out_dir = Path(out or Path(cfg.output_dir) / "eval")
    base = str(Path(cfg.output_dir) / "tiny-base") if cfg.base_model == "tiny-random" else cfg.base_model
    before = _eval_one(base, None, rows, out_dir, "base")
    after = _eval_one(base, str(Path(cfg.output_dir) / "adapter"), rows, out_dir, "finetuned")
    table = comparison_table(before, after)
    (out_dir / "comparison.md").write_text(f"# {cfg.base_model} - base vs QLoRA (n={len(rows)})\n\n{table}\n")
    typer.echo(table)


@app.command()
def merge(config: str, out: str = typer.Option(None)) -> None:
    """Merge the adapter into full-precision base weights for adapter-free serving (vLLM, TGI)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = TrainConfig.from_yaml(config)
    base = str(Path(cfg.output_dir) / "tiny-base") if cfg.base_model == "tiny-random" else cfg.base_model
    adapter = Path(cfg.output_dir) / "adapter"
    target = Path(out or Path(cfg.output_dir) / "merged")
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base), adapter).merge_and_unload()
    model.save_pretrained(target)
    AutoTokenizer.from_pretrained(adapter).save_pretrained(target)
    typer.echo(f"merged model written to {target}")


if __name__ == "__main__":
    app()
