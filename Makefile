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


## Corre las pruebas sin las marcadas como lentas, que levantan MLflow
.PHONY: test-fast
test-fast:
	pytest -m "not lento"


## Revisa el código con bandit y las dependencias con pip-audit
.PHONY: security
security:
	bandit -r mi_bici -c pyproject.toml -q
	# DeepChecks 0.19.1 necesita setuptools<81 (usa pkg_resources). Los dos avisos de setuptools son al
	# armar un sdist en macOS y al instalar eggs en Windows, y este proyecto no hace ninguno de los dos
	pip-audit -r requirements.txt --ignore-vuln PYSEC-2026-3447 --ignore-vuln GHSA-xmj3-9gf7-h52w


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

ARGS ?=
MOTOR ?= docker
IMAGE_NAME ?= mibici-api
PORT ?= 8000

# mlflow run le pasa a cada corrida la URI de la terminal, así que se toma la del .env
-include .env
export MLFLOW_TRACKING_URI

# Desde dentro del contenedor, 127.0.0.1 es el propio contenedor; se cambia por el host según el motor
HOST_CONTENEDOR = $(if $(filter podman,$(MOTOR)),host.containers.internal,host.docker.internal)
TRACKING_CONTENEDOR = $(subst localhost,$(HOST_CONTENEDOR),$(subst 127.0.0.1,$(HOST_CONTENEDOR),$(or $(MLFLOW_TRACKING_URI),http://127.0.0.1:5001)))
EXTRA_CONTENEDOR = $(if $(filter podman,$(MOTOR)),,--add-host=host.docker.internal:host-gateway)

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


## Levanta la API localmente con uvicorn
.PHONY: api
api:
	uvicorn mi_bici.api.app:app --host 127.0.0.1 --port $(PORT)


## Construye la imagen de la API (usa MOTOR=podman para Podman)
.PHONY: image
image:
	$(MOTOR) build -t $(IMAGE_NAME) .


## Corre el contenedor de la API con el puerto y las variables de entorno
# Publica el puerto solo en 127.0.0.1 para que la API no quede abierta a la red local
.PHONY: container
container:
	$(MOTOR) run --rm -p 127.0.0.1:$(PORT):8000 $(EXTRA_CONTENEDOR) \
		-e MLFLOW_TRACKING_URI=$(TRACKING_CONTENEDOR) \
		$(if $(MODELO_URI),-e MODELO_URI=$(MODELO_URI)) \
		$(IMAGE_NAME)


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
