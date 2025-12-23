SHELL := /bin/bash

.PHONY: build build-nc sh run run-bonds run-crypto run-commodities run-indices run-forex baseline-all list-strategies lock lock-update discover-crypto manifest-status dashboard

build:
	docker-compose build

build-nc:
	docker-compose build --no-cache

sh:
	docker-compose run --rm app bash

run:
	docker-compose run --rm app bash -lc "poetry install && poetry run python -m src.main run --config config/example.yaml"

run-bonds:
	docker-compose run --rm app bash -lc "poetry install && RUN_ID=bonds-$(shell date +%Y%m%d%H%M) poetry run python -m src.main run --config config/collections/bonds_majors.yaml"

run-crypto:
	docker-compose run --rm app bash -lc "poetry install && RUN_ID=crypto-$(shell date +%Y%m%d%H%M) poetry run python -m src.main run --config config/collections/crypto_majors.yaml"

run-commodities:
	docker-compose run --rm app bash -lc "poetry install && RUN_ID=commodities-$(shell date +%Y%m%d%H%M) poetry run python -m src.main run --config config/collections/commodities_majors.yaml"

run-indices:
	docker-compose run --rm app bash -lc "poetry install && RUN_ID=indices-$(shell date +%Y%m%d%H%M) poetry run python -m src.main run --config config/collections/indices_majors.yaml"

run-forex:
	docker-compose run --rm app bash -lc "poetry install && RUN_ID=forex-$(shell date +%Y%m%d%H%M) poetry run python -m src.main run --config config/collections/forex_majors.yaml"

baseline-all:
	docker-compose run --rm app bash -lc "poetry install && ./scripts/run_baselines.sh"

list-strategies:
	docker-compose run --rm app bash -lc "poetry install && poetry run python -m src.main list-strategies --strategies-path /ext/strategies"

discover-crypto:
	# Usage: make discover-crypto EXCHANGE=binance QUOTE=USDT TOP=100 OUT=config/collections/crypto_top100.yaml NAME=crypto_top100
	docker-compose run --rm app bash -lc "poetry install && poetry run python -m src.main discover-symbols --exchange $(EXCHANGE) --quote $(QUOTE) --top-n $(TOP) --name $(NAME) --output $(OUT)"

manifest-status:
	docker-compose run --rm app bash -lc "poetry install && poetry run python -m src.main manifest-status --reports-dir reports --latest"

dashboard:
	docker-compose run --rm --service-ports app bash -lc "poetry install && poetry run python -m src.main dashboard --reports-dir reports --host 0.0.0.0 --port 8000"

lock:
	docker-compose run --rm app bash -lc "poetry lock --no-update && git add poetry.lock"

lock-update:
	docker-compose run --rm app bash -lc "poetry lock && git add poetry.lock"
