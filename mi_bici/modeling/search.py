"""Corre una búsqueda de hiperparámetros con nested runs en MLflow.

Un run padre por búsqueda y un run hijo por combinación. Los hijos se miden en validación y el
padre guarda la mejor combinación en `mejor_params`, lista para pasarla a `--params` de train.

python -m mi_bici.modeling.search --question p1 --model hgb
python -m mi_bici.modeling.search --question p2 --model hgb --grid "learning_rate=0.05|0.1"
"""

from loguru import logger
import typer

from mi_bici.modeling.train import EXPERIMENTOS, leer_valor

app = typer.Typer()


def leer_cuadricula(texto: str) -> dict:
    """Convierte `"k=v1|v2,..."` en una cuadrícula de hiperparámetros.

    Args:
        texto: Hiperparámetros separados por comas y sus valores por `|`. Vacío para usar la
            cuadrícula del catálogo.

    Returns:
        Diccionario con la lista de valores de cada hiperparámetro, cada uno con su tipo.
    """
    cuadricula = {}
    for par in filter(None, texto.split(",")):
        clave, valores = (parte.strip() for parte in par.split("=", 1))
        cuadricula[clave] = [leer_valor(valor.strip()) for valor in valores.split("|")]
    return cuadricula


@app.command()
def main(
    question: str = typer.Option("p1", help="p1, p2 o p3"),
    model: str = typer.Option("hgb", help="Clave del catálogo de la pregunta"),
    grid: str = typer.Option("", help='Cuadrícula "k=v1|v2,..."; vacía usa la del catálogo'),
) -> None:
    """Prueba cada combinación de la cuadrícula y registra la búsqueda en MLflow."""
    pregunta = question.upper()
    if pregunta not in EXPERIMENTOS:
        raise typer.BadParameter(f"question debe ser una de {list(EXPERIMENTOS)}")
    experimento = EXPERIMENTOS[pregunta](model)
    run_id, mejor = experimento.buscar(leer_cuadricula(grid))
    logger.success(f"{pregunta}-{model}: mejor combinación {mejor} (run padre {run_id})")


if __name__ == "__main__":
    app()
