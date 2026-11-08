"""Fixtures para probar la API sin servidor de MLflow ni datos de MiBici.

`cargar_modelo` se sustituye por una función que devuelve un modelo falso con la misma forma que el
real: un pyfunc con `predict`, un clasificador con `predict_proba` y `threshold`, y los tipos de la
firma. Así el `lifespan` de la app corre igual que en producción.
"""

from fastapi.testclient import TestClient
from mlflow.exceptions import MlflowException
import numpy as np
import pandas as pd
import pytest

import mi_bici.api.app as modulo_app

# Tipos de la firma del champion de P2: las columnas enteras llegan como int32
TIPOS_FIRMA = {
    "flujo_neto_lag_1": np.dtype("float64"),
    "flujo_neto_lag_2": np.dtype("float64"),
    "salidas_lag_1": np.dtype("float64"),
    "llegadas_lag_1": np.dtype("float64"),
    "riesgo_semana_anterior": np.dtype("float64"),
    "hora": np.dtype("int32"),
    "dia_semana": np.dtype("int32"),
    "festivo": np.dtype("bool"),
    "capacidad": np.dtype("int32"),
}
UMBRAL = 0.5


class ClasificadorFalso:
    """Imita al clasificador de scikit-learn: la probabilidad crece con las salidas del bloque."""

    threshold = UMBRAL

    def __init__(self):
        self.ultimos_datos: pd.DataFrame | None = None

    def predict_proba(self, datos: pd.DataFrame) -> np.ndarray:
        self.ultimos_datos = datos
        positiva = np.clip(datos["salidas_lag_1"].to_numpy() / 20, 0, 1)
        return np.column_stack([1 - positiva, positiva])


class ModeloFalso:
    """Imita al pyfunc: `predict` aplica el umbral a la probabilidad del clasificador."""

    def __init__(self, clasificador: ClasificadorFalso):
        self.clasificador = clasificador
        self.ultimos_datos: pd.DataFrame | None = None

    def predict(self, datos: pd.DataFrame) -> np.ndarray:
        self.ultimos_datos = datos
        return (self.clasificador.predict_proba(datos)[:, 1] >= UMBRAL).astype(int)


@pytest.fixture
def servicio() -> dict:
    """Lo mismo que devuelve `cargar_modelo`, pero con el modelo falso."""
    clasificador = ClasificadorFalso()
    return {
        "modelo": ModeloFalso(clasificador),
        "clasificador": clasificador,
        "tipos": TIPOS_FIRMA,
        "nombre": "mibici-equipo03-p2-riesgo",
        "version": "7",
        "alias": "champion",
    }


@pytest.fixture
def cliente(servicio, monkeypatch):
    """Cliente con el modelo falso cargado al arrancar."""
    monkeypatch.setattr(modulo_app, "cargar_modelo", lambda *args, **kwargs: servicio)
    with TestClient(modulo_app.app) as c:
        yield c


@pytest.fixture
def cliente_sin_modelo(monkeypatch):
    """Cliente cuyo modelo no se pudo cargar al arrancar, como cuando no hay servidor de MLflow."""

    def falla(*args, **kwargs):
        raise MlflowException("sin servidor de MLflow")

    monkeypatch.setattr(modulo_app, "cargar_modelo", falla)
    with TestClient(modulo_app.app) as c:
        yield c
