.PHONY: all clean install test lint run train evaluate save-model

# Variables
PYTHON = python3
PIP = pip3
PYTEST = pytest
FLAKE8 = flake8
VENV = venv
VENV_BIN = $(VENV)/bin

all: install lint test

$(VENV)/bin/activate:
    $(PYTHON) -m venv $(VENV)
    $(VENV_BIN)/pip install --upgrade pip
    $(VENV_BIN)/pip install -r requirements.txt

install: $(VENV)/bin/activate

clean:
    rm -rf __pycache__
    rm -rf $(VENV)
    rm -rf .pytest_cache
    rm -rf .coverage
    rm -rf saved_models/*
    rm -rf logs/*
    rm -rf checkpoints/*
    find . -type f -name "*.pyc" -delete
    find . -type d -name "__pycache__" -delete

lint:
    $(VENV_BIN)/flake8 src/ tests/
    $(VENV_BIN)/black src/ tests/

test:
    PYTHONPATH=src $(VENV_BIN)/pytest tests/ -v

run:
    PYTHONPATH=src $(VENV_BIN)/python main.py

train:
    PYTHONPATH=src $(VENV_BIN)/python -c "from src.models.training import train_model; train_model()"

evaluate: