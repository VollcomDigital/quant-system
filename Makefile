SHELL := /bin/bash

.PHONY: build build-nc refresh-image refresh-image-nc sh run run-integration restore-integration-cache run-stocks-dividend run-stocks-large-cap-value run-stocks-large-cap-growth run-stocks-mid-cap run-stocks-small-cap run-stocks-international run-stocks-emerging run-bonds-global run-bonds-high-yield run-bonds-corporate run-bonds-municipal run-bonds-tips run-bonds-us-treasuries run-crypto run-commodities list-strategies lock lock-update discover-crypto manifest-status dashboard tests coverage precommit-coverage

INTEGRATION_CACHE_ROOT := .cache/integration
INTEGRATION_DATA_CACHE_DIR := $(INTEGRATION_CACHE_ROOT)/data
INTEGRATION_FIXTURE_DIR := tests/fixtures/data_cache

build:
	docker-compose build

build-nc:
	docker-compose build --no-cache

refresh-image:
	docker-compose build --pull app

refresh-image-nc:
	docker-compose build --pull --no-cache app

sh:
	docker-compose run --rm app bash

run:
	docker-compose run --rm app bash -lc "python -m src.main run --config config/example.yaml"

restore-integration-cache:
	mkdir -p "$(INTEGRATION_DATA_CACHE_DIR)/yfinance"
	cp -f "$(INTEGRATION_FIXTURE_DIR)/yfinance/BTC-USD_1d.parquet" "$(INTEGRATION_DATA_CACHE_DIR)/yfinance/BTC-USD_1d.parquet"

run-integration: restore-integration-cache
	docker-compose run --rm -e DATA_CACHE_DIR=/app/$(INTEGRATION_DATA_CACHE_DIR) app bash -lc "python -m src.main run --config config/example.integration.yaml --only-cached"

run-stocks-dividend:
	docker-compose run --rm app bash -lc "RUN_ID=stocks-dividend-$(shell date +%Y%m%d%H%M) python -m src.main run --config config/collections/stocks_dividend.yaml"

run-stocks-large-cap-value:
	docker-compose run --rm app bash -lc "RUN_ID=stocks-large-cap-value-$(shell date +%Y%m%d%H%M) python -m src.main run --config config/collections/stocks_large_cap_value.yaml"

run-stocks-large-cap-growth:
	docker-compose run --rm app bash -lc "RUN_ID=stocks-large-cap-growth-$(shell date +%Y%m%d%H%M) python -m src.main run --config config/collections/stocks_large_cap_growth.yaml"

run-stocks-mid-cap:
	docker-compose run --rm app bash -lc "RUN_ID=stocks-mid-cap-$(shell date +%Y%m%d%H%M) python -m src.main run --config config/collections/stocks_mid_cap.yaml"

run-stocks-small-cap:
	docker-compose run --rm app bash -lc "RUN_ID=stocks-small-cap-$(shell date +%Y%m%d%H%M) python -m src.main run --config config/collections/stocks_small_cap.yaml"

run-stocks-international:
	docker-compose run --rm app bash -lc "RUN_ID=stocks-international-$(shell date +%Y%m%d%H%M) python -m src.main run --config config/collections/stocks_international.yaml"

run-stocks-emerging:
	docker-compose run --rm app bash -lc "RUN_ID=stocks-emerging-$(shell date +%Y%m%d%H%M) python -m src.main run --config config/collections/stocks_emerging.yaml"

run-bonds-global:
	docker-compose run --rm app bash -lc "RUN_ID=bonds-global-$(shell date +%Y%m%d%H%M) python -m src.main run --config config/collections/bonds_global.yaml"

run-bonds-high-yield:
	docker-compose run --rm app bash -lc "RUN_ID=bonds-high-yield-$(shell date +%Y%m%d%H%M) python -m src.main run --config config/collections/bonds_high_yield.yaml"

run-bonds-corporate:
	docker-compose run --rm app bash -lc "RUN_ID=bonds-corporate-$(shell date +%Y%m%d%H%M) python -m src.main run --config config/collections/bonds_corporate.yaml"

run-bonds-municipal:
	docker-compose run --rm app bash -lc "RUN_ID=bonds-municipal-$(shell date +%Y%m%d%H%M) python -m src.main run --config config/collections/bonds_municipal.yaml"

run-bonds-tips:
	docker-compose run --rm app bash -lc "RUN_ID=bonds-tips-$(shell date +%Y%m%d%H%M) python -m src.main run --config config/collections/bonds_tips.yaml"

run-bonds-us-treasuries:
	docker-compose run --rm app bash -lc "RUN_ID=bonds-us-treasuries-$(shell date +%Y%m%d%H%M) python -m src.main run --config config/collections/bonds_us_treasuries.yaml"

run-crypto:
	docker-compose run --rm app bash -lc "RUN_ID=crypto-$(shell date +%Y%m%d%H%M) python -m src.main run --config config/collections/crypto.yaml"

run-commodities:
	docker-compose run --rm app bash -lc "RUN_ID=commodities-$(shell date +%Y%m%d%H%M) python -m src.main run --config config/collections/commodities.yaml"

list-strategies:
	docker-compose run --rm app bash -lc "python -m src.main list-strategies --strategies-path /ext/strategies"

discover-crypto:
	# Usage: make discover-crypto EXCHANGE=binance QUOTE=USDT TOP=100 OUT=config/collections/crypto_top100.yaml NAME=crypto_top100
	docker-compose run --rm app bash -lc "python -m src.main discover-symbols --exchange $(EXCHANGE) --quote $(QUOTE) --top-n $(TOP) --name $(NAME) --output $(OUT)"

manifest-status:
	docker-compose run --rm app bash -lc "python -m src.main manifest-status --reports-dir reports --latest"

dashboard:
	docker-compose run --rm -p 8000:8000 app bash -lc "python -m src.main dashboard --reports-dir reports --host 0.0.0.0 --port 8000"

tests:
	docker-compose run --rm app bash -lc "pytest -q -vv"

coverage: 
	docker-compose run --rm app bash -lc "pytest -q --cov=src --cov-report=term-missing --cov-report=html --cov-fail-under=80"

precommit-coverage:
	docker-compose run --rm app bash -lc "pre-commit run pytest-coverage --all-files"

lock:
	docker-compose run --rm app bash -lc "poetry lock --no-update"

lock-update:
	docker-compose run --rm app bash -lc "poetry lock"
