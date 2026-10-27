"""Corre los experimentos de P1, P2 y P3 y los registra en MLflow.

Cada pregunta vive en su archivo (`p1_demanda.py`, `p2_riesgo.py` y `p3_grupos.py`) con su
catálogo de modelos. Un run es una pregunta y un modelo del catálogo.

python -m mi_bici.modeling.train
python -m mi_bici.modeling.train --question p2 --model hgb --params "learning_rate=0.05"
python -m mi_bici.modeling.train --question p2 --model hgb --params mejor
"""

import ast

from loguru import logger
import mlflow
import typer

from mi_bici.config import EXPERIMENT_NAME, TRACKING_URI
from mi_bici.modeling.base import version_datos
from mi_bici.modeling.p1_demanda import ExperimentoDemanda
from mi_bici.modeling.p2_riesgo import ExperimentoRiesgo
from mi_bici.modeling.p3_grupos import ExperimentoGrupos

app = typer.Typer()

EXPERIMENTOS = {"P1": ExperimentoDemanda, "P2": ExperimentoRiesgo, "P3": ExperimentoGrupos}
# Valor de --params que toma la mejor combinación de la búsqueda más reciente
MEJOR = "mejor"


def leer_valor(texto: str):
    """Lee un valor con su tipo: números, booleanos y `None`. El resto queda como texto."""
    try:
        return ast.literal_eval(texto)
    except (ValueError, SyntaxError):
        return texto


def leer_params(texto: str) -> dict:
    """Convierte `"k=v,..."` en un diccionario de hiperparámetros.

    Args:
        texto: Hiperparámetros separados por comas. Vacío si no se indica ninguno.

    Returns:
        Diccionario con los hiperparámetros, cada valor con su tipo.
    """
    params = {}
    for par in filter(None, texto.split(",")):
        clave, valor = (parte.strip() for parte in par.split("=", 1))
        params[clave] = leer_valor(valor)
    return params


def mejores_params(pregunta: str, modelo: str) -> str:
    """Lee la mejor combinación de la búsqueda más reciente de un modelo con los datos actuales.

    Args:
        pregunta: `"P1"`, `"P2"` o `"P3"`.
        modelo: Clave del catálogo de la pregunta.

    Returns:
        La etiqueta `mejor_params` del run padre, con el formato de `--params`.

    Raises:
        ValueError: Si no hay una búsqueda terminada de ese modelo con la misma `data_version`.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    filtro = (
        f"tags.question = '{pregunta.lower()}' and tags.stage_of_work = 'search' "
        f"and params.modelo = '{modelo}' and params.data_version = '{version_datos()}' "
        "and attributes.status = 'FINISHED'"
    )
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=filtro,
        order_by=["attributes.start_time DESC"],
        output_format="list",
    )
    # Los hijos de la búsqueda tienen las mismas etiquetas, pero solo el padre guarda mejor_params
    for run in runs:
        if "mejor_params" in run.data.tags:
            return run.data.tags["mejor_params"]
    raise ValueError(
        f"No hay una búsqueda de {pregunta}-{modelo} con los datos actuales. Córrela primero con "
        f"mlflow run . -e search -P question={pregunta.lower()} -P model={modelo}"
    )


@app.command()
def main(
    question: str = typer.Option("todas", help="p1, p2, p3 o todas"),
    model: str = typer.Option("todos", help="Clave del catálogo de la pregunta o todos"),
    params: str = typer.Option(
        "", help='Hiperparámetros "k=v,..." que reemplazan al catálogo, o "mejor" (de la búsqueda)'
    ),
) -> None:
    """Corre los modelos pedidos y registra cada uno como un run en MLflow."""
    pregunta = question.upper()
    if pregunta not in (*EXPERIMENTOS, "TODAS"):
        raise typer.BadParameter(f"question debe ser una de {[*EXPERIMENTOS, 'todas']}")
    if pregunta == "TODAS" and (model != "todos" or params):
        raise typer.BadParameter("model y params requieren una sola pregunta")
    if params == MEJOR and model == "todos":
        raise typer.BadParameter(f"params={MEJOR} requiere un solo modelo")
    for p in EXPERIMENTOS if pregunta == "TODAS" else [pregunta]:
        clase = EXPERIMENTOS[p]
        claves = list(clase.catalogo) if model == "todos" else [model]
        for clave in claves:
            texto = params
            if params == MEJOR:
                try:
                    texto = mejores_params(p, clave)
                except ValueError as error:
                    raise typer.BadParameter(str(error)) from error
                logger.info(f"{p}-{clave}: mejor combinación de la búsqueda {texto}")
            experimento = clase(clave, leer_params(texto))
            run_id, metricas = experimento.ejecutar()
            resumen = "  ".join(f"{k}={v:.3f}" for k, v in metricas.items())
            logger.success(f"{p}-{clave}: {resumen} (run {run_id})")


if __name__ == "__main__":
    app()
