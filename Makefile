CONFIG ?= configs/qwen2.5-1.5b-qlora.yaml

.PHONY: install test data train compare merge serve smoke

install:
	pip install -e ".[dev,serve]"

test:
	pytest -q

data:
	ade-ft gen-data --n-train 2000 --n-eval 200

train: data
	ade-ft train $(CONFIG)

compare:
	ade-ft compare $(CONFIG)

merge:
	ade-ft merge $(CONFIG)

serve:
	ADE_BASE=$$(python -c "from ade_ft.config import TrainConfig as C;print(C.from_yaml('$(CONFIG)').base_model)") \
	ADE_ADAPTER=$$(python -c "from ade_ft.config import TrainConfig as C;print(C.from_yaml('$(CONFIG)').output_dir)")/adapter \
	uvicorn ade_ft.serve:app --port 8000

smoke:           ## full pipeline on CPU in ~2 minutes, no downloads
	$(MAKE) data train compare CONFIG=configs/tiny-cpu.yaml
