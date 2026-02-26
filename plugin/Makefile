# Makefile – common developer tasks for yProv4ML MLflow Plugin
.DEFAULT_GOAL := help
PYTHON        ?= python
PIP           ?= pip

# ─── Installation ──────────────────────────────────────────────────────────────

.PHONY: install
install: ## Install the plugin (core only)
	$(PIP) install git+https://github.com/zdeniztas/yProvML.git@experiment_unification
	$(PIP) install -e .

.PHONY: install-examples
install-examples: ## Install plugin + training-example deps (torch, codecarbon, …)
	$(MAKE) install
	$(PIP) install -e ".[examples]"

.PHONY: install-dashboard
install-dashboard: ## Install plugin + Streamlit dashboard deps
	$(MAKE) install
	$(PIP) install -e ".[dashboard]"

.PHONY: install-dev
install-dev: ## Install everything needed for development
	$(PIP) install git+https://github.com/zdeniztas/yProvML.git@experiment_unification
	$(PIP) install -e ".[dev]"

# ─── Testing ───────────────────────────────────────────────────────────────────

.PHONY: test
test: ## Run the full test suite
	pytest test/ -v --tb=short

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	pytest test/ -v --tb=short --cov=yprov_mlflow_plugin --cov-report=term-missing

.PHONY: test-fast
test-fast: ## Run tests, stop on first failure
	pytest test/ -x --tb=short

# ─── Linting ───────────────────────────────────────────────────────────────────

.PHONY: lint
lint: ## Run ruff linter
	ruff check yprov_mlflow_plugin/ examples/ src/ test/

.PHONY: fmt
fmt: ## Auto-format with ruff
	ruff format yprov_mlflow_plugin/ examples/ src/ test/

.PHONY: typecheck
typecheck: ## Run mypy type checker
	mypy yprov_mlflow_plugin/ --ignore-missing-imports

# ─── Running ───────────────────────────────────────────────────────────────────

.PHONY: demo
demo: ## Run the minimal CIFAR-10 / toy-metrics demo
	$(PYTHON) examples/demo.py

.PHONY: demo-dry
demo-dry: ## Run demo with 1 epoch (quick smoke test)
	$(PYTHON) examples/demo.py --epochs 1

.PHONY: dashboard
dashboard: ## Launch the Streamlit decision-support dashboard
	streamlit run src/decision_app.py

.PHONY: prov-to-csv
prov-to-csv: ## Convert PROV JSON files to CSV (usage: make prov-to-csv ROOT=data/prov OUT=data/unified)
	$(PYTHON) src/prov_to_csv.py --root $(or $(ROOT),data/prov) --out-dir $(or $(OUT),data/unified)

.PHONY: mlflow-ui
mlflow-ui: ## Launch the MLflow UI (pass TRACKING_URI=yprov+file:///path/to/mlruns)
	mlflow ui --backend-store-uri $(or $(TRACKING_URI),yprov+file:///$(PWD)/mlruns)

# ─── Verification ──────────────────────────────────────────────────────────────

.PHONY: verify
verify: ## Verify the plugin is correctly installed
	@$(PYTHON) -c "import mlflow; print('mlflow', mlflow.__version__)"
	@$(PYTHON) -c "import prov4ml; print('prov4ml OK')" 2>/dev/null || \
		echo "⚠  prov4ml not installed – run: make install"
	@$(PYTHON) -c "\
from importlib.metadata import entry_points; \
eps = [ep for ep in entry_points().get('mlflow.tracking_store', []) if 'yprov' in ep.name]; \
print(f'✅ {len(eps)} yprov tracking_store entry points registered') if eps \
else print('❌ No yprov entry points found – run: make install')"

# ─── Cleanup ───────────────────────────────────────────────────────────────────

.PHONY: clean
clean: ## Remove build artefacts and caches
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .ruff_cache/ .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

.PHONY: clean-runs
clean-runs: ## Remove local MLflow run data (mlruns/, mlartifacts/, data/, outputs/)
	@echo "⚠  This will delete mlruns/, mlartifacts/, data/, outputs/"
	@read -p "Are you sure? [y/N] " ans; \
		if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
			rm -rf mlruns/ mlartifacts/ data/ outputs/; \
			echo "Cleaned."; \
		else echo "Aborted."; fi

# ─── Help ──────────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
