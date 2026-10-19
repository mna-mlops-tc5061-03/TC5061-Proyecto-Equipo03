# mi_bici

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Propuesta y base reproducible de ML para el sistema MiBici. Con los viajes que publica el
[portal de datos abiertos](https://mibici.net/es/datos-abiertos/) se responden tres preguntas de
operación: demanda por estación (regresión), riesgo de quedarse sin bicis o sin anclajes
(clasificación) y grupos de estaciones con patrones de uso parecidos (clustering).

## Preguntas

| | Pregunta | Tarea | Métrica principal |
|---|---|---|---|
| **P1** | ¿Cuántos viajes saldrán de cada estación por día? | Regresión | MAE (en viajes) |
| **P2** | ¿Qué estaciones están en riesgo de quedarse sin bicis o sin anclajes en la siguiente hora? | Clasificación | Recall |
| **P3** | ¿Qué grupos de estaciones tienen patrones de uso parecidos? | Clustering | Silhouette |

Las tres se formularon con el [Machine Learning Canvas](https://www.ownml.co/machine-learning-canvas), disponible en `references/`.

## Datos

Se usan los datos abiertos de [MiBici](https://mibici.net/es/datos-abiertos/), el sistema de bicicleta pública del Área Metropolitana de Guadalajara:

- **Viajes** (`datos_abiertos_AAAA_MM.csv`): un archivo por mes, con identificador del viaje y del usuario, género, año de nacimiento, fecha y hora de inicio y fin, y estaciones de origen y destino.
- **Estaciones** (`nomenclatura_AAAA_MM.csv`): catálogo con identificador, nombre y ubicación de cada estación.

El periodo de análisis y el corte de prueba se definen en `mi_bici/config.py`. Los archivos descargados se guardan sin modificar en `data/raw/`, y las versiones limpias en Parquet en `data/processed/`. Ninguno se versiona: se reconstruyen con la ingesta, y `data/manifest.json` registra el número de filas y el hash de cada archivo como versión de los datos.

## Instalación

Requiere Python 3.13. Los comandos se corren desde la raíz del repositorio.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` fija la versión exacta de cada dependencia, así que el entorno es el mismo en
cualquier computadora.

## Ejecución

Con el entorno activado (`source .venv/bin/activate`):

```bash
make data     # descarga, limpia y versiona los datos de MiBici
make train    # registra en MLflow un baseline y un candidato de P1, P2 y P3 (6 runs)
mlflow ui     # ver los runs en http://127.0.0.1:5000
make test     # corre las pruebas de tests/
make lint     # revisa el formato del código
```

Para correr solo una parte: `python -m mi_bici.modeling.train --pregunta P2 --modelo candidato`.

## Configuración

Todo lo que define una ejecución vive en `mi_bici/config.py`: rutas, URLs de los datos, periodo y
corte de prueba, semilla, umbrales y el experimento de MLflow (`mibici-equipo03`). Ningún otro
módulo escribe estos valores a mano.

MLflow guarda los runs en `mlflow.db` local. Para usar un servidor compartido basta con definir
`MLFLOW_TRACKING_URI` en un archivo `.env`, que no se versiona.

Cada run se registra igual, desde `mi_bici/modeling/base.py`:

| Qué | Contenido |
|---|---|
| Parámetros | `fecha_inicio`, `fecha_fin`, `corte_prueba`, `data_version`, `semilla`, `filas_train`, modelo e hiperparámetros |
| Métricas | Con el conjunto en el nombre: `test_mae`, `test_recall`, `train_silhouette`… |
| Etiquetas | `equipo`, `pregunta`, `tipo` (baseline o candidato), `git_commit` y `git_dirty` |
| Artefactos | Gráficas o tablas de cada pregunta, y el modelo en los candidatos |

`data_version` son los primeros 12 caracteres del hash de `data/manifest.json` (sin su fecha de
creación), así que dos ingestas con los mismos datos dan la misma versión.

## Buenas prácticas aplicadas

| Práctica | Cómo se aplica |
|---|---|
| Estructura estándar | Plantilla de Cookiecutter Data Science |
| Datos inmutables y fuera de Git | `data/raw/` no se modifica ni se versiona; solo `data/manifest.json` |
| Reproducibilidad | Dependencias con versión fija, semilla y configuración centralizada en `config.py`, versión de datos por hash |
| Código en módulos | Los notebooks solo exploran; la lógica vive en `mi_bici/` |
| Registro de experimentos | Todos los runs se registran en MLflow con los mismos parámetros, métricas y etiquetas |
| Variables sin fuga de información | Las variables usan solo información disponible antes del momento a predecir |
| Pruebas y estilo | Pruebas unitarias en `tests/` y formato revisado con ruff |
| Secretos fuera del código | Credenciales y URIs en `.env`, que no se versiona |
| Flujo de trabajo | GitHub Flow, rama `main` protegida, Pull Requests con revisión y Conventional Commits |

## Estructura

Generada con [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) 2.3.0.

```
├── mi_bici/                 # código del proyecto
│   ├── config.py            # rutas, URLs, periodo, semilla, umbrales y experimento de MLflow
│   ├── dataset.py           # ingesta: descarga, limpieza, Parquet y manifiesto
│   ├── features.py          # variables de P1, P2 y P3, sin información del futuro
│   ├── plots.py             # gráficas que se registran como artefactos
│   └── modeling/
│       ├── base.py          # ExperimentoBase: registro común de cada run en MLflow
│       ├── evaluate.py      # métricas por tipo de tarea (test_*, train_*)
│       └── train.py         # experimentos de P1, P2 y P3 (baseline y candidato)
├── data/                    # raw, interim, processed y external. No se versiona
├── models/                  # modelos serializados. No se versionan
├── notebooks/               # exploración, con la convención 1.0-iniciales-descripcion.ipynb
├── references/              # ML Canvas, diagramas y referencias
├── reports/                 # reporte de la propuesta y figures/
├── tests/                   # pruebas unitarias
├── docs/                    # documentación con mkdocs
├── Makefile                 # make data, make train, make test, make lint, make format
├── MLproject                # punto de entrada de MLflow
├── requirements.txt         # dependencias con versiones fijas
└── pyproject.toml           # metadatos del paquete y configuración de ruff
```

`.gitignore` excluye los datos de MiBici, `mlflow.db`, `mlruns/`, `mlartifacts/`, `.env` y los
modelos serializados. De `data/` solo se versiona `data/manifest.json`, la versión de los datos.

## Flujo de trabajo

Seguimos [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow):

1. Nadie trabaja directo en `main`.
2. Cada cambio sale de `main` actualizado, en su propia rama.
3. Entra a `main` solo por Pull Request, con al menos una revisión de otro integrante.

**Ramas:** `tipo/descripcion-corta`, en minúsculas y con guiones. Por ejemplo `feat/ingesta` o
`docs/readme`.

**Commits:** [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/), con el formato
`tipo(alcance opcional): descripción corta`. Por ejemplo `feat(ingesta): manifiesto de datos` o
`docs: instalación en el README`. Si un cambio rompe algo que ya funcionaba, se marca con `!`
(`feat!: ...`) y se explica en el cuerpo del commit con `BREAKING CHANGE:`. El título del Pull
Request sigue el mismo formato.

| Tipo | Cuándo |
|---|---|
| `feat` | Funcionalidad nueva |
| `fix` | Corrección de un error |
| `docs` | Solo documentación |
| `chore` | Configuración o dependencias |
| `style` | Formato, sin cambiar la lógica |
| `refactor` | Reorganizar código sin cambiar lo que hace |
| `test` | Pruebas |
| `perf` | Mejora de rendimiento |
| `build` / `ci` | Empaquetado o integración continua |

**Docstrings:** estilo Google en español, para que todo el código se lea igual. El resumen va en
una línea, con verbo en tercera persona (Calcula, Registra, Descarga). Si hace falta, sigue un
párrafo con el porqué y las secciones `Args` y `Returns`.

```python
def metricas_regresion(y, pred) -> dict:
    """Calcula las métricas de regresión de P1.

    El MAE se lee en viajes, que es lo que el operador entiende.

    Args:
        y: Valores reales del conjunto de prueba.
        pred: Valores predichos.

    Returns:
        `test_mae`, `test_rmse` y `test_r2`.
    """
```

## Integrantes

| Integrante | Matrícula | Rol |
|---|---|---|
| Anaí Barrera | A01797380 | Exploración de datos y documentación |
| Gerardo González Martínez | A01840096 | Configuración del proyecto y registro de experimentos |
| Miguel Castillo | A01840045 | Ingesta y versión de los datos |
| Sebastián Estrada | A00805402 | ML Canvas, diagramas y referencias |