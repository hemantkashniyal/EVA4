ENV_FILE := "./environment.yaml"
ENV_NAME := $(shell cat $(ENV_FILE) | grep "name:" | cut -d ":" -f 2 | sed -e 's/^[ \t]*//')
BASH_SHELL := $(shell which bash)
SHELL := $(BASH_SHELL)

default:

.PHONY: check_conda_venv_inactive
check_conda_venv_inactive:
	[ ${CONDA_DEFAULT_ENV} != $(ENV_NAME) ] && (echo "venv $(ENV_NAME) not active") || (echo "Error: venv $(ENV_NAME) active" && exit 1)

.PHONY: check_conda_venv_active
check_conda_venv_active:
	[ ${CONDA_DEFAULT_ENV} = $(ENV_NAME) ] && (echo "venv $(ENV_NAME) active") || (echo "Error: venv $(ENV_NAME) not active" && exit 1)

.PHONY: environment
environment: check_conda_venv_inactive
	conda env update --prune -f $(ENV_FILE)
	conda env list | grep $(ENV_NAME)

.PHONY: environment_clean
environment_clean: check_conda_venv_inactive
	conda env remove --name $(ENV_NAME)

.PHONY: install
install: #check_conda_venv_active
	pip install -r ./requirements.txt

.PHONY: stop_jupyterlab
stop_jupyterlab:
	docker-compose -f ./docker/jupyterlab/docker-compose.yaml stop || true	

.PHONY: start_jupyterlab
start_jupyterlab: stop_jupyterlab
	docker-compose -f ./docker/jupyterlab/docker-compose.yaml up -d
	echo Access JupyterLab: http://localhost:28888