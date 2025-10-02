.DEFAULT_GOAL := help

SRC_DIR := src
TEST_DIR := tests

.PHONY: help
help:
	@echo "Please use 'make <target>' where target is one of:"
	@echo ""
	@echo "  format      - Format code using ruff"
	@echo "  lint        - Check code style using ruff"
	@echo "  lint-fix    - Check and fix code style using ruff"
	@echo "  type-check  - Check types using mypy"
	@echo "  static-fix  - Apply all fixes (format, lint, and type-check)"
	@echo "  test        - Run unit tests with pytest and generate coverage report"
	@echo "  clean       - Remove cached files and temporary files"
	@echo ""

.PHONY: format
format:
	uv run ruff format $(SRC_DIR) $(TEST_DIR)

.PHONY: lint
lint:
	uv run ruff check $(SRC_DIR) $(TEST_DIR)

.PHONY: lint-fix
lint-fix:
	uv run ruff check --fix $(SRC_DIR) $(TEST_DIR)

.PHONY: type-check
type-check:
	uv run mypy $(SRC_DIR) $(TEST_DIR)

.PHONY: static-fix
static-fix: format lint-fix type-check

.PHONY: test
test:
	uv run pytest -n auto --cov=$(SRC_DIR) --cov-report=term-missing $(TEST_DIR)

.PHONY: clean
clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache .coverage
	find $(SRC_DIR) $(TEST_DIR) -type d -name "__pycache__" -exec rm -rf {} +
