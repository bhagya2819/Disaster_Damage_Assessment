.PHONY: help env install hooks lint format test smoke app clean

help:
	@echo "Targets:"
	@echo "  env       - create conda env from environment.yml"
	@echo "  install   - pip install -e . (editable)"
	@echo "  hooks     - install pre-commit hooks"
	@echo "  lint      - run ruff"
	@echo "  format    - run black + ruff --fix"
	@echo "  test      - run pytest"
	@echo "  smoke     - run tests/test_smoke.py only"
	@echo "  app       - launch Streamlit app"
	@echo "  clean     - remove caches and build artefacts"

env:
	conda env create -f environment.yml

install:
	pip install -e .

hooks:
	pre-commit install

lint:
	ruff check src tests

format:
	black src tests
	ruff check --fix src tests

test:
	pytest

smoke:
	pytest tests/test_smoke.py -v

app:
	streamlit run app/streamlit_app.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
