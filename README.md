# mi_bici

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Propuesta y base reproducible de ML para el sistema MiBici. Con los viajes que publica el
[portal de datos abiertos](https://mibici.net/es/datos-abiertos/) se responden tres preguntas de
operación: demanda por estación (regresión), riesgo de quedarse sin bicis o sin anclajes
(clasificación) y grupos de estaciones con patrones de uso parecidos (clustering).

## Instalación

Requiere Python 3.13. Los comandos se corren desde la raíz del repositorio.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` fija la versión exacta de cada dependencia, así que el entorno es el mismo en
cualquier computadora.

## Configuración

Todo lo que define una ejecución vive en `mi_bici/config.py`: rutas, URLs de los datos, periodo y
corte de prueba, semilla, umbrales y el experimento de MLflow (`mibici-equipo03`). Ningún otro
módulo escribe estos valores a mano.

MLflow guarda los runs en `mlflow.db` local. Para usar un servidor compartido basta con definir
`MLFLOW_TRACKING_URI` en un archivo `.env`, que no se versiona.

## Estructura

Generada con [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) 2.3.0.

```
├── mi_bici/                 # código del proyecto
│   ├── config.py            # rutas, URLs, periodo, semilla, umbrales y experimento de MLflow
│   ├── dataset.py           # ingesta de los datos de MiBici (plantilla, pendiente)
│   ├── features.py          # variables de cada pregunta (plantilla, pendiente)
│   ├── plots.py             # gráficas (plantilla, pendiente)
│   └── modeling/            # entrenamiento y evaluación (plantilla, pendiente)
├── data/                    # raw, interim, processed y external. No se versiona
├── models/                  # modelos serializados. No se versionan
├── notebooks/               # exploración, con la convención 1.0-iniciales-descripcion.ipynb
├── references/              # ML Canvas, diagramas y referencias
├── reports/                 # reporte de la propuesta y figures/
├── docs/                    # documentación con mkdocs
├── Makefile                 # make requirements, make lint, make format, make clean
├── requirements.txt         # dependencias con versiones fijas
└── pyproject.toml           # metadatos del paquete y configuración de ruff
```

`.gitignore` excluye los datos de MiBici, `mlflow.db`, `mlruns/`, `mlartifacts/`, `.env` y los
modelos serializados. De `data/` solo se versionará `data/manifest.json`, la versión de los datos.

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
