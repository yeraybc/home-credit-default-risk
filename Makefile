PYTHON := python3
PIP    := .venv/bin/pip
PY     := .venv/bin/python

.PHONY: help setup install lint format test train serve docker-build docker-up clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

setup: ## crea el entorno e instala dependencias
	python3 -m venv .venv
	$(PIP) install --upgrade pip -q
	$(PIP) install -r requirements.txt -q
	$(PIP) install -e . -q

install: ## instala dependencias en el venv existente
	$(PIP) install -r requirements.txt -q
	$(PIP) install -e . -q

lint: ## linter
	.venv/bin/ruff check src/ api/ tests/

format: ## formatea con black
	.venv/bin/black src/ api/ tests/

test: ## tests
	.venv/bin/pytest tests/ -v --tb=short

train: ## entrena el modelo
	$(PY) -m src.models.train

audit: ## muestra el data audit
	$(PY) -c "from src.data.loader import load_all_tables, data_audit; print(data_audit(load_all_tables()).to_string())"

serve: ## api en http://localhost:8000
	.venv/bin/uvicorn api.main:app --reload --port 8000

docker-build: ## construye imagen docker
	docker build -t home-credit-scoring:latest -f docker/Dockerfile .

docker-up: ## levanta api + evidently
	docker compose -f docker/docker-compose.yml up --build

docker-down: ## para contenedores
	docker compose -f docker/docker-compose.yml down

clean: ## elimina caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
