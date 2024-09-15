RED := \033[0;31m
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RESET := \033[0m

ENV_NAME := "py39"
PY_VERSION := "3.9"

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@echo "-------------------"
	@echo "$(YELLOW)list$(RESET) : List available targets with descriptions"
	@echo ""
	@echo "== Env =="
	@echo "$(GREEN)install$(RESET) : Installs a new py39 env and sets-up poetry packages"
	@echo ""
	@echo ""
	@echo ""

list: help

install:
	@echo "$(GREEN) [CONDA] Creating [$(ENV_NAME)] python env $(RESET)"
	conda create --name $(ENV_NAME) python=3.9 -y
	@echo "Activating the environment..."
	@bash -c "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(ENV_NAME) \
	&& pip install poetry \
	poetry env use $(which python)"
	@echo "Installing Packages"
	@echo "Changing to pyproject.toml location..."
	@bash -c " PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring poetry install"
	@echo "Installing NPM packages"
	@bash -c "cd client && npm install"
