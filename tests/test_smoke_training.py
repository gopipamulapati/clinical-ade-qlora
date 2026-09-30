"""End-to-end on a tiny random model: train -> adapter saved -> evaluate -> merge -> serve."""

import json

import pytest

from ade_ft.config import DataSettings, LoraSettings, TrainConfig
from ade_ft.data import write_splits

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from ade_ft.train import train

    root = tmp_path_factory.mktemp("run")
    paths = write_splits(root / "data", n_train=64, n_eval=8)
    cfg = TrainConfig(
        base_model="tiny-random",
        load_in_4bit=True,  # silently falls back to full precision on CPU
        lora=LoraSettings(r=4, alpha=8, dropout=0.0, target_modules=["q_proj", "v_proj"]),
        max_seq_len=256,
        learning_rate=1e-3,
        per_device_batch_size=8,
        max_steps=4,
        gradient_checkpointing=False,
        data=DataSettings(train=str(paths["train"]), eval=str(paths["eval"])),
        output_dir=str(root / "out"),
    )
    metrics = train(cfg)
    return cfg, metrics, root


def test_training_produces_adapter_and_finite_loss(trained):
    cfg, metrics, root = trained
    adapter = root / "out" / "adapter"
    assert (adapter / "adapter_config.json").exists()
    assert json.loads((adapter / "adapter_config.json").read_text())["r"] == 4
    assert metrics["train_loss"] == pytest.approx(metrics["train_loss"])  # not NaN
    assert 0 < metrics["params"]["pct"] < 10  # only LoRA weights train
    assert metrics["quantized_4bit"] is False


def test_evaluate_runs_on_adapter(trained):
    from ade_ft.data import read_jsonl
    from ade_ft.evaluate import evaluate
    from ade_ft.model import load_for_inference, load_tokenizer

    cfg, _, root = trained
    base, adapter = str(root / "out" / "tiny-base"), str(root / "out" / "adapter")
    model, tok = load_for_inference(base, adapter), load_tokenizer(adapter)
    scores = evaluate(model, tok, read_jsonl(cfg.data.eval)[:4], root / "eval.json", max_new_tokens=16)
    assert scores.n == 4 and 0.0 <= scores.f1 <= 1.0
    assert (root / "eval.predictions.jsonl").exists()


def test_serve_endpoint(trained, monkeypatch):
    from fastapi.testclient import TestClient

    from ade_ft import serve

    _, _, root = trained
    monkeypatch.setenv("ADE_BASE", str(root / "out" / "tiny-base"))
    monkeypatch.setenv("ADE_ADAPTER", str(root / "out" / "adapter"))
    serve._load.cache_clear()
    client = TestClient(serve.app)
    r = client.post("/extract", json={"note": "Patient reports mild rash since starting amoxicillin."})
    assert r.status_code == 200
    assert set(r.json()) == {"events", "valid_json", "raw"}
