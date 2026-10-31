#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mi_bici
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format


## Run unit tests
.PHONY: test
test:
	pytest tests/


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

ARGS ?=

# mlflow run le pasa a cada corrida la URI de la terminal, así que se toma la del .env
-include .env
export MLFLOW_TRACKING_URI

# Mismo experimento que EXPERIMENT_NAME en config.py; --env-manager local usa el .venv activo
MLFLOW_RUN = mlflow run . --env-manager local --experiment-name mibici-equipo03

## Train baselines and candidates and log them to MLflow
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m mi_bici.modeling.train $(ARGS)


## Corre con mlflow run el baseline, los candidatos y la búsqueda de P1, P2 y P3
# Al final de cada pregunta corre como candidato la mejor combinación de su búsqueda
# (params=mejor lee la etiqueta mejor_params del run padre), porque make registry no toma los
# hijos de la búsqueda
.PHONY: experimentos
experimentos:
	@test -n "$(MLFLOW_TRACKING_URI)" || { echo "Falta MLFLOW_TRACKING_URI: cp .env.example .env"; exit 1; }
	$(MLFLOW_RUN) -e train -P question=p1 -P model=baseline
	$(MLFLOW_RUN) -e train -P question=p1 -P model=rf
	$(MLFLOW_RUN) -e train -P question=p1 -P model=ridge
	$(MLFLOW_RUN) -e train -P question=p1 -P model=hgb
	$(MLFLOW_RUN) -e search -P question=p1 -P model=hgb
	$(MLFLOW_RUN) -e train -P question=p1 -P model=hgb -P params=mejor
	$(MLFLOW_RUN) -e train -P question=p2 -P model=baseline
	$(MLFLOW_RUN) -e train -P question=p2 -P model=hgb
	$(MLFLOW_RUN) -e train -P question=p2 -P model=logreg
	$(MLFLOW_RUN) -e train -P question=p2 -P model=rf
	$(MLFLOW_RUN) -e search -P question=p2 -P model=hgb
	$(MLFLOW_RUN) -e train -P question=p2 -P model=hgb -P params=mejor
	$(MLFLOW_RUN) -e train -P question=p3 -P model=baseline
	$(MLFLOW_RUN) -e train -P question=p3 -P model=kmeans
	$(MLFLOW_RUN) -e train -P question=p3 -P model=birch
	$(MLFLOW_RUN) -e train -P question=p3 -P model=gmm
	$(MLFLOW_RUN) -e search -P question=p3 -P model=kmeans
	$(MLFLOW_RUN) -e train -P question=p3 -P model=kmeans -P params=mejor


## Registra el mejor modelo de cada pregunta con el alias champion
.PHONY: registry
registry:
	$(PYTHON_INTERPRETER) -m mi_bici.modeling.registry


## Valida los datos procesados con DeepChecks: integridad y drift entre entrenamiento y prueba
.PHONY: validar-datos
validar-datos:
	$(PYTHON_INTERPRETER) -m mi_bici.validacion integridad
	$(PYTHON_INTERPRETER) -m mi_bici.validacion drift


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) mi_bici/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
