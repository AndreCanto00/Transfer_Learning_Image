.PHONY: all clean install test lint run train evaluate save-model setup-dev

# Variables
PYTHON = python3
PIP = pip3
PYTEST = pytest
FLAKE8 = flake8
VENV = venv
VENV_BIN = $(VENV)/bin
SRC_DIR = src
TESTS_DIR = tests

all: install lint test

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(VENV_BIN)/pip install --upgrade pip
	$(VENV_BIN)/pip install -r requirements.txt

install: $(VENV)/bin/activate

clean:
	rm -rf **/__pycache__
	rm -rf $(VENV)
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf saved_models/*
	rm -rf logs/*
	rm -rf checkpoints/*
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

lint:
	PYTHONPATH=$(PWD)/$(SRC_DIR) $(VENV_BIN)/flake8 $(SRC_DIR)/ $(TESTS_DIR)/
	PYTHONPATH=$(PWD)/$(SRC_DIR) $(VENV_BIN)/black $(SRC_DIR)/ $(TESTS_DIR)/

test:
	PYTHONPATH=$(PWD)/$(SRC_DIR) $(VENV_BIN)/pytest $(TESTS_DIR)/ -v

run:
	PYTHONPATH=$(PWD)/$(SRC_DIR) $(VENV_BIN)/python main.py

train:
	PYTHONPATH=$(PWD)/$(SRC_DIR) $(VENV_BIN)/python -c "from src.models.training import train_model; train_model()"

evaluate:
	PYTHONPATH=$(PWD)/$(SRC_DIR) $(VENV_BIN)/python -c "from src.models.evaluation import evaluate_model; evaluate_model()"

save-model:
	PYTHONPATH=$(PWD)/$(SRC_DIR) $(VENV_BIN)/python -c "from src.models.model import save_pretrained_model; save_pretrained_model()"

setup-dev: install
	$(VENV_BIN)/pip install -r requirements-dev.txt