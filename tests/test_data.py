import re

from ade_ft.config import TrainConfig
from ade_ft.data import DRUG_REACTIONS, build_prompt, collate, generate, tokenize_example, write_splits
from ade_ft.schema import Extraction


class CharTok:
    """Minimal tokenizer stub: one id per character, no chat template."""

    eos_token_id = 1
    chat_template = None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 250 + 2 for c in text]}


def test_generation_is_deterministic_and_valid():
    a, b = generate(50, seed=7), generate(50, seed=7)
    assert a == b
    for row in a:
        ex = Extraction.model_validate_json(row["output"])
        for e in ex.events:
            assert e.reaction in DRUG_REACTIONS[e.drug]
            assert e.drug in row["note"] and e.reaction in row["note"].lower()


def test_negated_symptoms_are_not_labelled():
    for row in generate(300, seed=1):
        labelled = {e.reaction for e in Extraction.model_validate_json(row["output"]).events}
        for sx in re.findall(r"Denies ([a-z ]+)\.", row["note"]):
            assert sx not in labelled


def test_dataset_has_empty_and_multi_event_notes():
    counts = [len(Extraction.model_validate_json(r["output"]).events) for r in generate(300)]
    assert 0 in counts and max(counts) >= 2


def test_splits_do_not_leak(tmp_path):
    paths = write_splits(tmp_path, 300, 60)
    train = {line for line in paths["train"].read_text().splitlines()}
    assert not any(line in train for line in paths["eval"].read_text().splitlines())


def test_prompt_is_masked_and_answer_kept():
    tok = CharTok()
    row = generate(1)[0]
    item = tokenize_example(row, tok, max_len=4096)
    n_prompt = len(tok(build_prompt(row["note"], tok))["input_ids"])
    assert item["labels"][:n_prompt] == [-100] * n_prompt
    assert item["labels"][n_prompt:] == item["input_ids"][n_prompt:]
    assert item["labels"][-1] == tok.eos_token_id


def test_truncation_never_cuts_answer():
    tok = CharTok()
    row = generate(1)[0]
    item = tokenize_example(row, tok, max_len=len(row["output"]) + 20)
    assert len(item["input_ids"]) == len(row["output"]) + 20
    assert sum(label != -100 for label in item["labels"]) == len(row["output"]) + 1


def test_collate_pads_labels_with_ignore_index():
    batch = collate(
        [
            {"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "labels": [-100, 6, 7]},
            {"input_ids": [5], "attention_mask": [1], "labels": [5]},
        ],
        pad_id=0,
    )
    assert batch["input_ids"].tolist()[1] == [5, 0, 0]
    assert batch["labels"].tolist()[1] == [5, -100, -100]
    assert batch["attention_mask"].tolist()[1] == [1, 0, 0]


def test_config_yaml_and_overrides():
    cfg = TrainConfig.from_yaml("configs/qwen2.5-1.5b-qlora.yaml", max_steps=10)
    assert cfg.load_in_4bit and cfg.lora.r == 16 and "down_proj" in cfg.lora.target_modules
    assert cfg.max_steps == 10
