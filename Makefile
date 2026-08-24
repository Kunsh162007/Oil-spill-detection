# Oil Spill Detection & Vessel Attribution - SIH 2026 / SIH26143
PY ?= .venv/Scripts/python.exe
CONFIG ?= configs/demo_synthetic.yaml
PORT ?= 8000

.PHONY: help setup demo serve test test-fast coverage clean \
        fetch-scene fetch-wind fetch-ais train train-lookalike eval ablation \
        latency bakeoff docker-train

help:
	@echo "Oil Spill Detection & Vessel Attribution"
	@echo ""
	@echo "  make setup           create the venv and install dependencies"
	@echo "  make demo            generate the synthetic demo scenes"
	@echo "  make serve           run the API + map UI on http://127.0.0.1:$(PORT)"
	@echo "  make test            full test suite"
	@echo "  make test-fast       skip the slow end-to-end tests"
	@echo "  make coverage        test suite with a coverage report"
	@echo ""
	@echo "  make fetch-scene CONFIG=configs/fetch_elsa3.yaml   Sentinel-1 imagery"
	@echo "  make fetch-wind  CONFIG=...                        ERA5 wind"
	@echo "  make fetch-ais   CONFIG=...                        AIS tracks"
	@echo ""
	@echo "  make train CONFIG=configs/train_baseline.yaml      segmentation"
	@echo "  make train-lookalike CONFIG=...                    physics model"
	@echo "  make eval RUN=runs/<name>                          results table"
	@echo "  make ablation                                      wind ablation"
	@echo "  make latency                                       per-stage timing"
	@echo "  make bakeoff                                       model selection sweep"

setup:
	python -m venv .venv || py -3.11 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -r requirements-ml.txt
	@echo "Done. Copy .env.example to .env and fill in credentials for real data."

demo:
	$(PY) scripts/make_demo_scene.py
	$(PY) scripts/make_demo_scene.py --calm-wind --name CALM_WIND_DEMO \
		--bbox "74.20,9.05,74.80,9.65" --seed 21

serve:
	OILSPILL_CONFIG=$(CONFIG) $(PY) -m uvicorn api.main:app --reload --port $(PORT)

test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -m "not slow"

coverage:
	$(PY) -m pytest --cov=core --cov=ingest --cov=detect --cov=drift \
		--cov=attribute --cov=decision --cov=api --cov-report=term-missing

fetch-scene:
	$(PY) scripts/fetch_sentinel.py --config $(CONFIG)

fetch-wind:
	$(PY) scripts/fetch_wind.py --config $(CONFIG)

fetch-ais:
	$(PY) scripts/fetch_ais.py --config $(CONFIG)

train:
	$(PY) scripts/train.py --config $(CONFIG)

train-lookalike:
	$(PY) scripts/train.py --config $(CONFIG) --stage lookalike

eval:
	$(PY) scripts/eval.py --run $(RUN)

ablation:
	$(PY) scripts/eval.py --wind-ablation --config $(CONFIG)

latency:
	$(PY) scripts/eval.py --latency --config $(CONFIG)

bakeoff:
	$(PY) scripts/bakeoff.py --config configs/bakeoff.yaml

docker-train:
	docker build -t oilspill-train -f docker/Dockerfile.train .

clean:
	$(PY) -c "import shutil,glob,os; [os.remove(p) for p in glob.glob('data/cache/*.pkl')]; [shutil.rmtree(d, ignore_errors=True) for d in ('.pytest-tmp','.pytest_cache','htmlcov')]"
