.PHONY: setup test all demo clean-eval
setup: ; bash scripts/setup.sh
test: ; . .venv/bin/activate && python -m pytest -q tests
all: ; bash scripts/run_all.sh
demo: ; . .venv/bin/activate && python app/app.py
