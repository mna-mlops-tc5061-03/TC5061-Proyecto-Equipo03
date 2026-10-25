"""Corre los experimentos de P1, P2 y P3 y los registra en MLflow.

Cada pregunta vive en su archivo (`p1_demanda.py`, `p2_riesgo.py` y `p3_grupos.py`) con su
catálogo de modelos. Un run es una pregunta y un modelo del catálogo.

python -m mi_bici.modeling.train
python -m mi_bici.modeling.train --question p2 --model hgb --params "learning_rate=0.05"
"""

import ast

from loguru import logger
import typer

from mi_bici.config import HIPERPARAMETROS_GRIDS
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
    grid: bool = typer.Option(
        False, "--grid", help="Corre la cuadrícula de hiperparámetros para el modelo o pregunta"
    ),
) -> None:
    """Corre los modelos pedidos y registra cada uno como un run en MLflow."""
    pregunta = question.upper()
    if pregunta not in (*EXPERIMENTOS, "TODAS"):
        raise typer.BadParameter(f"question debe ser una de {[*EXPERIMENTOS, 'todas']}")
    if pregunta == "TODAS" and (model != "todos" or params or grid):
        raise typer.BadParameter("model, params y grid requieren una sola pregunta")
    if params and grid:
        raise typer.BadParameter("params y grid son mutuamente excluyentes")
    for p in EXPERIMENTOS if pregunta == "TODAS" else [pregunta]:
        clase = EXPERIMENTOS[p]
        claves = list(clase.catalogo) if model == "todos" else [model]
        for clave in claves:
            # Determinamos la lista de combinaciones de hiperparámetros
            if grid:
                grid_pregunta = HIPERPARAMETROS_GRIDS.get(p, {})
                if clave not in grid_pregunta:
                    logger.warning(
                        f"No hay cuadrícula definida para {p}-{clave}, se usará por defecto."
                    )
                    lista_params = [{}]
                else:
                    lista_params = grid_pregunta[clave]
            else:
                lista_params = [leer_params(params)]
            for config_params in lista_params:
                experimento = clase(clave, config_params)
                run_id, metricas = experimento.ejecutar()
                resumen = "  ".join(f"{k}={v:.3f}" for k, v in metricas.items())
                logger.success(
                    f"{p}-{clave} ({config_params or 'default'}): {resumen} (run {run_id})"
                )


if __name__ == "__main__":
    app()
