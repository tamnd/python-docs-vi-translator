.DEFAULT_GOAL := help
.PHONY: help sync fmt lint type test cover check build clean

help: ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-8s %s\n", $$1, $$2}'

sync: ## Install the locked environment
	uv sync --locked --all-groups

fmt: ## Format
	uv run ruff format .
	uv run ruff check --fix-only .

lint: ## Lint
	uv run ruff format --check .
	uv run ruff check .

type: ## Type check
	uv run mypy --strict src/

test: ## Run the tests
	uv run pytest

cover: ## Run the tests with the coverage floor
	uv run pytest --cov=pydocvi --cov-report=term-missing --cov-fail-under=85

check: lint type cover ## Everything CI runs

build: ## Build the wheel and sdist
	uv build

clean: ## Remove build and cache artefacts
	rm -rf dist .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
