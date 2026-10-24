"""Corre los experimentos de P1, P2 y P3 y los registra en MLflow.

Cada pregunta vive en su archivo (`p1_demanda.py`, `p2_riesgo.py` y `p3_grupos.py`) con su
catálogo de modelos. Un run es una pregunta y un modelo del catálogo.

python -m mi_bici.modeling.train
python -m mi_bici.modeling.train --question p2 --model hgb --params "learning_rate=0.05"
"""

import ast

from loguru import logger
import typer

from mi_bici.modeling.p1_demanda import ExperimentoDemanda
from mi_bici.modeling.p2_riesgo import ExperimentoRiesgo
from mi_bici.modeling.p3_grupos import ExperimentoGrupos

app = typer.Typer()

EXPERIMENTOS = {"P1": ExperimentoDemanda, "P2": ExperimentoRiesgo, "P3": ExperimentoGrupos}


def leer_params(texto: str) -> dict:
    """Convierte `"k=v,..."` en un diccionario de hiperparámetros.

    Los números, booleanos y `None` se leen con su tipo, y el resto queda como texto.

    Args:
        texto: Hiperparámetros separados por comas. Vacío si no se indica ninguno.

    Returns:
        Diccionario con los hiperparámetros.
    """
    params = {}
    for par in filter(None, texto.split(",")):
        clave, valor = (parte.strip() for parte in par.split("=", 1))
        try:
            params[clave] = ast.literal_eval(valor)
        except (ValueError, SyntaxError):
            params[clave] = valor
    return params


@app.command()
def main(
    question: str = typer.Option("todas", help="p1, p2, p3 o todas"),
    model: str = typer.Option("todos", help="Clave del catálogo de la pregunta o todos"),
    params: str = typer.Option("", help='Hiperparámetros "k=v,..." que reemplazan al catálogo'),
) -> None:
    """Corre los modelos pedidos y registra cada uno como un run en MLflow."""
    pregunta = question.upper()
    if pregunta not in (*EXPERIMENTOS, "TODAS"):
        raise typer.BadParameter(f"question debe ser una de {[*EXPERIMENTOS, 'todas']}")
    if pregunta == "TODAS" and (model != "todos" or params):
        raise typer.BadParameter("model y params requieren una sola pregunta")

    for p in EXPERIMENTOS if pregunta == "TODAS" else [pregunta]:
        clase = EXPERIMENTOS[p]
        claves = list(clase.catalogo) if model == "todos" else [model]
        for clave in claves:
            experimento = clase(clave, leer_params(params))
            run_id, metricas = experimento.ejecutar()
            resumen = "  ".join(f"{k}={v:.3f}" for k, v in metricas.items())
            logger.success(f"{p}-{clave}: {resumen} (run {run_id})")


if __name__ == "__main__":
    app()
