"""Fixtures para probar la API sin servidor de MLflow ni datos de MiBici.

`cargar_modelo` se sustituye por una función que devuelve, para la URI de cada pregunta, un modelo
falso con la misma forma que el real: un pyfunc con `predict`, el estimador de scikit-learn y los
tipos de la firma. Así el `lifespan` de la app corre igual que en producción.
"""

from fastapi.testclient import TestClient
from mlflow.exceptions import MlflowException
import numpy as np
import pandas as pd
import pytest

import mi_bici.api.app as modulo_app

# Tipos de las firmas de los champions: las columnas enteras llegan como int32
TIPOS_FIRMA_P1 = {
    "viajes_lag_1": np.dtype("float64"),
    "viajes_lag_7": np.dtype("float64"),
    "viajes_media_7": np.dtype("float64"),
    "dia_semana": np.dtype("int32"),
    "festivo": np.dtype("bool"),
    "capacidad": np.dtype("int32"),
}
TIPOS_FIRMA_P2 = {
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
    """Imita al clasificador de P2: la probabilidad crece con las salidas del bloque."""

    threshold = UMBRAL

    def __init__(self):
        self.ultimos_datos: pd.DataFrame | None = None

    def predict_proba(self, datos: pd.DataFrame) -> np.ndarray:
        self.ultimos_datos = datos
        positiva = np.clip(datos["salidas_lag_1"].to_numpy() / 20, 0, 1)
        return np.column_stack([1 - positiva, positiva])


class ModeloRiesgoFalso:
    """Imita al pyfunc de P2: `predict` aplica el umbral a la probabilidad del clasificador."""

    def __init__(self, clasificador: ClasificadorFalso):
        self.clasificador = clasificador
        self.ultimos_datos: pd.DataFrame | None = None

    def predict(self, datos: pd.DataFrame) -> np.ndarray:
        self.ultimos_datos = datos
        return (self.clasificador.predict_proba(datos)[:, 1] >= UMBRAL).astype(int)


class ModeloDemandaFalso:
    """Imita al pyfunc de P1: predice la media de los últimos 7 días más 1."""

    def __init__(self):
        self.ultimos_datos: pd.DataFrame | None = None

    def predict(self, datos: pd.DataFrame) -> np.ndarray:
        self.ultimos_datos = datos
        return datos["viajes_media_7"].to_numpy() + 1


@pytest.fixture
def servicio_riesgo() -> dict:
    """Lo mismo que devuelve `cargar_modelo` para P2, pero con el modelo falso."""
    clasificador = ClasificadorFalso()
    return {
        "modelo": ModeloRiesgoFalso(clasificador),
        "estimador": clasificador,
        "tipos": TIPOS_FIRMA_P2,
        "nombre": "mibici-equipo03-p2-riesgo",
        "version": "7",
        "alias": "champion",
    }


@pytest.fixture
def servicio_demanda() -> dict:
    """Lo mismo que devuelve `cargar_modelo` para P1, pero con el modelo falso."""
    modelo = ModeloDemandaFalso()
    return {
        "modelo": modelo,
        "estimador": modelo,
        "tipos": TIPOS_FIRMA_P1,
        "nombre": "mibici-equipo03-p1-demanda",
        "version": "3",
        "alias": "champion",
    }


def _iniciar(monkeypatch, servicios: dict):
    """Arranca la app con un `cargar_modelo` que devuelve el servicio de cada URI o falla."""

    def cargar(uri, *args, **kwargs):
        if servicios.get(uri) is None:
            raise MlflowException(f"sin servidor de MLflow para {uri}")
        return servicios[uri]

    monkeypatch.setattr(modulo_app, "cargar_modelo", cargar)
    return TestClient(modulo_app.app)


URI_P1 = modulo_app.PREGUNTAS["demanda"].uri
URI_P2 = modulo_app.PREGUNTAS["riesgo"].uri


@pytest.fixture
def cliente(servicio_demanda, servicio_riesgo, monkeypatch):
    """Cliente con los dos modelos falsos cargados al arrancar."""
    with _iniciar(monkeypatch, {URI_P1: servicio_demanda, URI_P2: servicio_riesgo}) as c:
        yield c


@pytest.fixture
def cliente_sin_modelo(monkeypatch):
    """Cliente sin ningún modelo cargado, como cuando no hay servidor de MLflow."""
    with _iniciar(monkeypatch, {}) as c:
        yield c


@pytest.fixture
def cliente_solo_riesgo(servicio_riesgo, monkeypatch):
    """Cliente donde P2 cargó y P1 no, para ver que una pregunta no tumba a la otra."""
    with _iniciar(monkeypatch, {URI_P2: servicio_riesgo}) as c:
        yield c
