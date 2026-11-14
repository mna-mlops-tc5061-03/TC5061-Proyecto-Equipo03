"""Reentrena el champion de P1 con datos más recientes y lo registra como challenger.

Usa el mismo estimador con los mismos hiperparámetros que el champion. Lo único que cambia es la
ventana de entrenamiento, que llega hasta el día antes de `CORTE_AB` e incluye los meses que el
champion no vio. Así el A/B (#136) mide el efecto de los datos nuevos. El champion no se mueve.

python -m mi_bici.modeling.reentrenar
"""

from pathlib import Path

from loguru import logger
import mlflow
from mlflow import MlflowClient
import pandas as pd
from sklearn.base import clone
import typer

from mi_bici.config import (
    ALIAS_PRODUCCION,
    ALIAS_RESPALDO,
    COLUMNA_FECHA_P1,
    CORTE_AB,
    FECHA_INICIO,
    MODELOS_REGISTRO,
    VARIABLES_P1,
)
from mi_bici.dataset import load_processed_stations, load_processed_travels
from mi_bici.features import demanda_diaria
from mi_bici.modeling.p1_demanda import ExperimentoDemanda
from mi_bici.modeling.registry import buscar_runs, comprobar_carga, registrar_version
from mi_bici.seguimiento import conectar_mlflow

app = typer.Typer()

# Último día de la ventana de reentrenamiento: octubre queda fuera para evaluar el A/B
ULTIMO_DIA = (pd.Timestamp(CORTE_AB) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


class ReentrenamientoDemanda(ExperimentoDemanda):
    """Ajusta una copia sin entrenar del estimador del champion con la ventana nueva.

    No tiene validación ni prueba: la comparación con el champion la hace el A/B en un periodo
    que ninguna de las dos versiones vio.

    Args:
        clave: Clave del catálogo con que se entrenó el champion, por ejemplo `"hgb"`.
        estimador: Estimador del champion. Se copia sin sus parámetros aprendidos.
        version_origen: Versión del champion en el Registry.
    """

    columna_fecha = None

    def __init__(self, clave: str, estimador, version_origen: str):
        super().__init__(clave)
        self.estimador = estimador
        self.version_origen = version_origen
        self.etapa = "retraining"

    def crear_modelo(self, **extra):
        """Devuelve el estimador del champion sin entrenar, con sus mismos hiperparámetros."""
        return clone(self.estimador)

    def preparar(self) -> None:
        """Construye la demanda diaria desde `FECHA_INICIO` hasta el día antes de `CORTE_AB`."""
        demanda = demanda_diaria(
            load_processed_travels(hasta=ULTIMO_DIA), load_processed_stations()
        )
        self.train = demanda[demanda[COLUMNA_FECHA_P1] < CORTE_AB]
        self.test = None
        self.X_train = self.train[VARIABLES_P1]

    def artefactos(self, carpeta: Path) -> list[Path]:
        """No guarda gráficas: la de real contra predicho necesita un periodo de prueba."""
        return []

    def parametros(self) -> dict:
        """Reemplaza los cortes de validación y prueba por la ventana del reentrenamiento."""
        parametros = super().parametros()
        for clave in ("corte_validacion", "corte_prueba", "metrica_seleccion"):
            parametros.pop(clave)
        return {**parametros, "fecha_fin": ULTIMO_DIA, "corte_ab": CORTE_AB}

    def etiquetas(self) -> dict:
        """Agrega la versión del champion de la que sale el reentrenamiento."""
        return {**super().etiquetas(), "source_version": self.version_origen}


def describir(version_origen: str, run: pd.Series) -> str:
    """Explica de dónde sale la versión reentrenada.

    Args:
        version_origen: Versión del champion que se reentrenó.
        run: Run del reentrenamiento, como fila de `buscar_runs()`.

    Returns:
        La descripción en Markdown.
    """
    return (
        f"Reentrenamiento de la v{version_origen} con los mismos hiperparámetros y datos de "
        f"{FECHA_INICIO} a {ULTIMO_DIA}. Se compara con `@{ALIAS_PRODUCCION}` en el A/B, con "
        f"datos desde {CORTE_AB}. Run `{run.run_id}`, commit `{run['tags.git_commit'][:7]}`, "
        f"datos `{run['params.data_version']}`."
    )


def reentrenar(cliente: MlflowClient, nombre: str = MODELOS_REGISTRO["P1"]) -> dict:
    """Reentrena el champion, lo registra como versión nueva y le asigna `@challenger`.

    Args:
        cliente: Cliente de MLflow.
        nombre: Nombre del modelo registrado.

    Returns:
        Las versiones del champion y del challenger y el Run ID del reentrenamiento.

    Raises:
        RuntimeError: Si al terminar el champion ya no es la misma versión.
    """
    champion = cliente.get_model_version_by_alias(nombre, ALIAS_PRODUCCION)
    # El número de versión llega como texto o como entero según el backend de MLflow
    origen = str(champion.version)
    clave = cliente.get_run(champion.run_id).data.params["modelo"]
    estimador = mlflow.sklearn.load_model(f"models:/{nombre}@{ALIAS_PRODUCCION}")
    logger.info(f"Reentrenando {nombre} v{origen} ({clave}) hasta {ULTIMO_DIA}")

    run_id, metricas = ReentrenamientoDemanda(clave, estimador, origen).ejecutar()
    run = buscar_runs().set_index("run_id", drop=False).loc[run_id]
    version = str(registrar_version(cliente, nombre, run))
    cliente.set_model_version_tag(nombre, version, "source_version", origen)
    cliente.set_model_version_tag(nombre, version, "fecha_fin", ULTIMO_DIA)
    cliente.update_model_version(nombre, version, description=describir(origen, run))
    cliente.set_registered_model_alias(nombre, ALIAS_RESPALDO, version)

    if str(cliente.get_model_version_by_alias(nombre, ALIAS_PRODUCCION).version) != origen:
        raise RuntimeError(f"{nombre}@{ALIAS_PRODUCCION} cambió durante el reentrenamiento")
    return {
        "champion": origen,
        "challenger": version,
        "run_id": run_id,
        "train_mae": metricas["train_mae"],
    }


@app.command()
def main() -> None:
    """Reentrena el champion de P1 y comprueba que el challenger se puede cargar."""
    conectar_mlflow()
    cliente = MlflowClient()
    resumen = reentrenar(cliente)
    filas = comprobar_carga(MODELOS_REGISTRO["P1"], ALIAS_RESPALDO)
    logger.success(
        f"{MODELOS_REGISTRO['P1']} v{resumen['challenger']} @{ALIAS_RESPALDO} desde la "
        f"v{resumen['champion']} @{ALIAS_PRODUCCION} (train_mae = {resumen['train_mae']:.3f}); "
        f"predice {filas} filas de su ejemplo"
    )


if __name__ == "__main__":
    app()
