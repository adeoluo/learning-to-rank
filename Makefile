.PHONY: help setup lint format type test coverage rust-check rust-develop parity validation data evaluation report demos diagram docker-build docker-up clean all changelog cut-release

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies (builds the Rust core) and pre-commit hooks
	uv sync --all-extras --dev
	uv run pre-commit install

format: ## Format automatically
	uv run ruff format .
	uv run ruff check --fix .

lint: ## Lint (no fixes)
	uv run ruff format --check .
	uv run ruff check .

type: ## Type-check
	uv run mypy src

test: ## Run tests
	uv run pytest

coverage: ## Tests with coverage gate
	uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=80

rust-check: ## Rust core: fmt, clippy (with and without Python), unit tests
	cd rust && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo clippy --all-targets --features python -- -D warnings && cargo test

rust-develop: ## Rebuild the Rust core into .venv after editing rust/ (uv sync also builds it)
	uv sync --all-extras --dev --group rust
	uv run --no-sync maturin develop --uv --release

parity: ## Rust vs numpy reference parity tests
	LTR_REQUIRE_RUST=1 uv run --no-sync pytest tests/test_rust_parity.py -v

validation: ## Tests against LightGBM, XGBoost, scikit-learn and ranx
	uv sync --all-extras --dev --group validation
	uv run --no-sync pytest -m validation -v

evaluation: ## Every full-scale benchmark (hours on a laptop; needs make data), then the tables
	uv sync --all-extras --dev --group validation
	cd benchmarks && uv run --no-sync python gbdt_mslr.py all
	cd benchmarks && uv run --no-sync python neural_mslr.py all
	cd benchmarks && uv run --no-sync python ultr_mslr.py
	cd benchmarks && uv run --no-sync python segments_mslr.py
	cd benchmarks && uv run --no-sync python throughput.py
	cd benchmarks && uv run --no-sync python istella.py prepare && uv run --no-sync python istella.py all
	$(MAKE) report

report: ## Print the README result tables from benchmarks/results
	uv run --no-sync python benchmarks/report.py

demos: ## Re-record the demo GIFs with vhs (needs make data)
	for t in train compare clicks serve; do vhs docs/demo/$$t.tape; done

data: ## Download and prepare MSLR-WEB10K (and Istella-S) under ../.bench-data/learning-to-rank
	scripts/download_data.sh

diagram: ## Render every diagram in docs/architecture to SVG (house theme, white background)
	@for f in docs/architecture/*.mmd; do \
		npx -y @mermaid-js/mermaid-cli -q -i "$$f" -o "$${f%.mmd}.svg" -c mermaid.config.json -b white || exit 1; \
	done

docker-build: ## Build the image
	docker build -t learning-to-rank .

docker-up: ## Run via docker compose
	docker compose up --build

clean: ## Remove caches and artifacts
	rm -rf .pytest_cache .pytest-tmp .mypy_cache .ruff_cache dist build *.egg-info coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +

all: lint type coverage ## Full local gate (mirror of continuous integration)

changelog: ## Regenerate CHANGELOG.md from Conventional Commits (git-cliff)
	git cliff --output CHANGELOG.md && perl -0pi -e 's/\n+\z/\n/' CHANGELOG.md

cut-release: ## Cut a release locally: version bump, changelog, tag, GitHub release (RELEASE_VERSION=vX.Y.Z optional)
	scripts/release.sh $(RELEASE_VERSION)
