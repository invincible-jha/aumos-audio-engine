.PHONY: install test test-quick lint format typecheck clean all docker-build docker-run

all: lint typecheck test

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=aumos_audio_engine --cov-report=term-missing

test-quick:
	pytest tests/ -x -q --no-header

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/aumos_audio_engine/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	rm -rf dist/ build/ *.egg-info

docker-build:
	docker build -t aumos/audio-engine:dev .

docker-run:
	docker compose -f docker-compose.dev.yml up -d

migrate:
	alembic -c src/aumos_audio_engine/migrations/alembic.ini upgrade head

migrate-create:
	alembic -c src/aumos_audio_engine/migrations/alembic.ini revision --autogenerate -m "$(MSG)"
