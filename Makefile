SHELL := /bin/bash

.PHONY: build build-nc sh run run-stocks-dividend run-stocks-large-cap-value run-stocks-large-cap-growth run-stocks-mid-cap run-stocks-small-cap run-stocks-international run-stocks-emerging run-bonds-global run-bonds-high-yield run-bonds-corporate run-bonds-municipal run-bonds-tips run-bonds-us-treasuries run-crypto run-commodities baseline-all list-strategies lock lock-update discover-crypto manifest-status dashboard

build:
	docker-compose build

build-nc:
	docker-compose build --no-cache

sh:
	docker-compose run --rm app bash

run:
	docker-compose run --rm app bash -lc "poetry install && poetry run python -m src.main run --config config/example.yaml"

run-stocks-dividend:
	docker-compose run --rm app bash -lc "RUN_ID=stocks-dividend-$(shell date +%Y%m%d%H%M) poetry run python -m src.main run --config config/collections/stocks_dividend.yaml"

run-stocks-large-cap-value:
	docker-compose run --rm app bash -lc "poetry install && RUN_ID=stocks-large-cap-value-$(shell date +%Y%m%d%H%M) poetry run python -m src.main run --config config/collections/stocks_large_cap_value.yaml"

run-stocks-large-cap-growth:
	docker-compose run --rm app bash -lc "poetry install && RUN_ID=stocks-large-cap-growth-$(shell date +%Y%m%d%H%M) poetry run python -m src.main run --config config/collections/stocks_large_cap_growth.yaml"

run-stocks-mid-cap:
	docker-compose run --rm app bash -lc "poetry install && RUN_ID=stocks-mid-cap-$(shell date +%Y%m%d%H%M) poetry run python -m src.main run --config config/collections/stocks_mid_cap.yaml"

run-stocks-small-cap:
	docker-compose run --rm app bash -lc "poetry install && RUN_ID=stocks-small-cap-$(shell date +%Y%m%d%H%M) poetry run python -m src.main run --config config/collections/stocks_small_cap.yaml"

run-stocks-international:
	docker-compose run --rm app bash -lc "poetry install && RUN_ID=stocks-international-$(shell date +%Y%m%d%H%M) poetry run python -m src.main run --config config/collections/stocks_international.yaml"

run-stocks-emerging:
	docker-compose run --rm app bash -lc "poetry install && RUN_ID=stocks-emerging-$(shell date +%Y%m%d%H%M) poetry run python -m src.main run --config config/collections/stocks_emerging.yaml"

run-bonds-global:
	docker-compose run --rm app bash -lc "poetry install && RUN_ID=bonds-global-$(shell date +%Y%m%d%H%M) poetry run python -m src.main run --config config/collections/bonds_global.yaml"

run-bonds-high-yield:
	docker-compose run --rm app bash -lc "poetry install && RUN_ID=bonds-high-yield-$(shell date +%Y%m%d%H%M) poetry run python -m src.main run --config config/collections/bonds_high_yield.yaml"

run-bonds-corporate:
	docker-compose run --rm app bash -lc "poetry install && RUN_ID=bonds-corporate-$(shell date +%Y%m%d%H%M) poetry run python -m src.main run --config config/collections/bonds_corporate.yaml"

run-bonds-municipal:
	docker-compose run --rm app bash -lc "poetry install && RUN_ID=bonds-municipal-$(shell date +%Y%m%d%H%M) poetry run python -m src.main run --config config/collections/bonds_municipal.yaml"

run-bonds-tips:
	docker-compose run --rm app bash -lc "poetry install && RUN_ID=bonds-tips-$(shell date +%Y%m%d%H%M) poetry run python -m src.main run --config config/collections/bonds_tips.yaml"

run-bonds-us-treasuries:
	docker-compose run --rm app bash -lc "poetry install && RUN_ID=bonds-us-treasuries-$(shell date +%Y%m%d%H%M) poetry run python -m src.main run --config config/collections/bonds_us_treasuries.yaml"

run-crypto:
	docker-compose run --rm app bash -lc "poetry install && RUN_ID=crypto-$(shell date +%Y%m%d%H%M) poetry run python -m src.main run --config config/collections/crypto.yaml"

run-commodities:
	docker-compose run --rm app bash -lc "poetry install && RUN_ID=commodities-$(shell date +%Y%m%d%H%M) poetry run python -m src.main run --config config/collections/commodities.yaml"

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
