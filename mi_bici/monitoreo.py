"""Monitoreo de P1 con Evidently: drift por variable y calidad de los datos.

Compara el periodo de referencia (entrenamiento del champion de P1) con el periodo actual
(producción), incluyendo la predicción del champion y el objetivo `viajes`.

python -m mi_bici.monitoreo
"""

from evidently import DataDefinition
import mlflow
import pandas as pd

from mi_bici.config import (
    ALIAS_PRODUCCION,
    CATEGORICAS_P1,
    COLUMNA_FECHA_P1,
    COLUMNA_PREDICCION,
    MODELOS_REGISTRO,
    NUMERICAS_P1,
    PERIODO_ACTUAL,
    PERIODO_REFERENCIA,
    TRACKING_URI,
    VARIABLES_P1,
)
from mi_bici.dataset import load_processed_stations, load_processed_travels
from mi_bici.features import demanda_diaria
from mi_bici.seguimiento import conectar_mlflow


def definicion_datos() -> DataDefinition:
    """Define las columnas numéricas y categóricas de P1 para Evidently."""
    return DataDefinition(
        numerical_columns=list(NUMERICAS_P1),
        categorical_columns=list(CATEGORICAS_P1),
    )


def conjuntos(modelo=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Arma la demanda diaria de P1 con la predicción del champion y separa referencia y actual.

    Calcula `demanda_diaria` una sola vez hasta `PERIODO_ACTUAL[1]` antes de separar, así los
    rezagos de los primeros días de agosto salen de finales de julio.

    Args:
        modelo: Modelo con método `predict`. Si es `None`, carga el champion de P1 desde MLflow.

    Returns:
        `(referencia, actual)` filtrados por `PERIODO_REFERENCIA` y `PERIODO_ACTUAL`.
    """
    viajes = load_processed_travels(hasta=PERIODO_ACTUAL[1])
    estaciones = load_processed_stations()
    demanda = demanda_diaria(viajes, estaciones)

    if modelo is None:
        conectar_mlflow(TRACKING_URI)
        uri = f"models:/{MODELOS_REGISTRO['P1']}@{ALIAS_PRODUCCION}"
        modelo = mlflow.pyfunc.load_model(uri)

    demanda[COLUMNA_PREDICCION] = modelo.predict(demanda[VARIABLES_P1])
    fechas = pd.to_datetime(demanda[COLUMNA_FECHA_P1])
    referencia = demanda[fechas.between(*PERIODO_REFERENCIA)].reset_index(drop=True)
    actual = demanda[fechas.between(*PERIODO_ACTUAL)].reset_index(drop=True)
    return referencia, actual
