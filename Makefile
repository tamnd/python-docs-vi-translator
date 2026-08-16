.DEFAULT_GOAL := help
.PHONY: help sync fmt lint type test cover secrets check build clean

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

secrets: ## Refuse to ship anything key-shaped
	@if git grep -nIE 'sk-[A-Za-z0-9._-]{8,}' -- .; then \
		echo "a key-shaped string is in a tracked file"; exit 1; \
	fi
	@if git ls-files | grep -E 'routes\.json$$'; then \
		echo "a route file is tracked, keys belong in the environment"; exit 1; \
	fi
	@echo "no key-shaped string and no route file is tracked"

check: lint type cover secrets ## Everything CI runs

build: ## Build the wheel and sdist
	uv build

clean: ## Remove build and cache artefacts
	rm -rf dist .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
