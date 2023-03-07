# Makefile
SHELL = /bin/bash

# Styling
.PHONY: style
style:
	python3 -m black .
	python3 -m flake8
	python3 -m isort .

# Environment
.PHONY: venv
venv:
	python3 -m venv env && \
	source env/bin/activate && \
	python3 -m pip install --upgrade pip && \
	python3 -m pip install pip setuptools wheel && \
	python3 -m pip install -e .

# Cleaning
.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".trash" | xargs rm -rf
	rm -f .coverage

.PHONY: help
help:
	@echo "Commands:"
	@echo "venv    : creates a virtual environment."
	@echo "style   : executes style formatting."
	@echo "clean   : cleans all unnecessary files."
