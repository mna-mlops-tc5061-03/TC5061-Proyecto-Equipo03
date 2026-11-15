"""Fixtures sintéticos para probar el monitoreo de drift de P1 sin MLflow ni datos de MiBici."""

import mlflow
from mlflow import MlflowClient
import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest

from mi_bici.config import (
    CATEGORICAS_P1,
    COLUMNA_FECHA_P1,
    COLUMNA_PREDICCION,
    EXPERIMENT_NAME,
    NUMERICAS_P1,
    PERIODO_ACTUAL,
    PERIODO_REFERENCIA,
)
import mi_bici.monitoreo as modulo_monitoreo

FILAS_SINTETICAS = 1200
COLUMNAS_PERSONALES = ("genero", "nacimiento", "usuario_id")


class ModeloDemandaFalso:
    """Imita al pyfunc champion de P1: predice `viajes_media_7 + 1` y guarda la entrada."""

    def __init__(self) -> None:
        self.ultimos_datos: pd.DataFrame | None = None

    def predict(self, datos: pd.DataFrame) -> np.ndarray:
        self.ultimos_datos = datos.copy()
        return datos["viajes_media_7"].to_numpy() + 1.0


def _tabla_p1(rng: np.random.Generator, n: int, inicio: str, fin: str) -> pd.DataFrame:
    """Construye una tabla sintética de demanda diaria con las columnas de P1."""
    fechas = pd.date_range(inicio, fin, freq="D")
    viajes_media = rng.normal(30.0, 2.0, n).clip(1.0)
    return pd.DataFrame(
        {
            COLUMNA_FECHA_P1: rng.choice(fechas, size=n),
            "estacion_id": rng.integers(1, 20, size=n),
            "viajes_lag_1": rng.normal(30.0, 1.0, n),
            "viajes_lag_7": rng.normal(30.0, 1.0, n),
            "viajes_media_7": viajes_media,
            "capacidad": rng.integers(15, 35, size=n),
            "viajes": rng.normal(30.0, 2.0, n).clip(0.0),
            COLUMNA_PREDICCION: viajes_media + 1.0,
            "dia_semana": rng.integers(0, 7, size=n),
            "festivo": rng.binomial(1, 0.05, size=n),
        }
    )


@pytest.fixture
def modelo_falso(monkeypatch) -> ModeloDemandaFalso:
    """Sustituye la carga del champion en MLflow por `ModeloDemandaFalso` sin servidor."""
    modelo = ModeloDemandaFalso()
    monkeypatch.setattr(modulo_monitoreo, "conectar_mlflow", lambda *args, **kwargs: None)
    monkeypatch.setattr("mlflow.pyfunc.load_model", lambda *args, **kwargs: modelo)
    return modelo


@pytest.fixture
def mlflow_temporal(tmp_path, monkeypatch) -> MlflowClient:
    """Crea un servidor local temporal de MLflow con SQLite para verificar runs y artefactos."""
    uri_previa = mlflow.get_tracking_uri()
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(modulo_monitoreo, "TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=(tmp_path / "artefactos").as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    yield MlflowClient(tracking_uri=uri)
    mlflow.set_tracking_uri(uri_previa)


@pytest.fixture
def viajes_y_estaciones_monitoreo(estaciones, monkeypatch) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inyecta viajes sintéticos (con columnas personales) desde julio hasta agosto de 2026."""
    dias = pd.date_range("2026-07-15", "2026-08-10", freq="D")
    filas = [
        (dia + pd.Timedelta(hours=hora), est_id, (est_id % 3) + 1)
        for dia in dias
        for est_id in (1, 2, 3)
        for hora in (8, 14, 19)
    ]
    inicio = pd.Series([f[0] for f in filas])
    viajes = pd.DataFrame(
        {
            "usuario_id": [f"usr_{i}" for i in range(len(filas))],
            "genero": ["M", "F"] * (len(filas) // 2) + ["M"] * (len(filas) % 2),
            "nacimiento": 1992,
            "inicio": inicio.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "fin": (inicio + pd.Timedelta(minutes=12)).dt.strftime("%Y-%m-%d %H:%M:%S"),
            "origen_id": [f[1] for f in filas],
            "destino_id": [f[2] for f in filas],
            "duracion_min": 12.0,
        }
    )
    monkeypatch.setattr(modulo_monitoreo, "load_processed_travels", lambda **kwargs: viajes.copy())
    monkeypatch.setattr(modulo_monitoreo, "load_processed_stations", lambda: estaciones.copy())
    return viajes, estaciones


@pytest.fixture
def referencia_y_actual() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dos conjuntos sintéticos de P1 (>1000 filas) con las mismas distribuciones base."""
    rng = np.random.default_rng(42)
    referencia = _tabla_p1(rng, FILAS_SINTETICAS, *PERIODO_REFERENCIA)
    # Partimos de una permutación de las columnas de P1 para que las distribuciones sean idénticas
    actual = referencia.copy()
    fechas_actual = pd.date_range(*PERIODO_ACTUAL, freq="D")
    actual[COLUMNA_FECHA_P1] = rng.choice(fechas_actual, size=FILAS_SINTETICAS)
    for col in (*NUMERICAS_P1, *CATEGORICAS_P1):
        actual[col] = rng.permutation(referencia[col].to_numpy())
    return referencia, actual
