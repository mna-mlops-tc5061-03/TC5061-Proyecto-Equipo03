import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from mi_bici.config import (
    COLUMNA_FECHA_P2,
    OBJETIVO_P1,
    OBJETIVO_P2,
    SEMILLA,
    VARIABLES_P1,
    VARIABLES_P2,
)
from mi_bici.features import VARIABLES_PERFIL_USO
from mi_bici.modeling.train import EXPERIMENTOS

FILAS = 200
# Con más de 10,000 filas, HistGradientBoosting hace early stopping y separa la validación al azar.
# Con menos no usa la semilla y la prueba de P1 pasaría aunque la semilla no llegara al modelo
FILAS_P1 = 12_000
# Un candidato de cada pregunta. En P3 el k va fijo para que la prueba sea rápida
CANDIDATOS = [
    ("P1", "hgb", {}),
    ("P2", "rf", {}),
    ("P3", "kmeans", {"n_clusters": 3}),
    ("P3", "gmm", {"n_components": 3}),
]


def datos_sinteticos(pregunta: str) -> pd.DataFrame:
    """Genera siempre los mismos datos para una pregunta, con sus variables y su objetivo."""
    rng = np.random.default_rng(0)
    if pregunta == "P3":
        return pd.DataFrame(
            rng.random((FILAS, len(VARIABLES_PERFIL_USO))), columns=VARIABLES_PERFIL_USO
        )
    if pregunta == "P1":
        datos = pd.DataFrame(rng.normal(size=(FILAS_P1, len(VARIABLES_P1))), columns=VARIABLES_P1)
        datos[OBJETIVO_P1] = rng.poisson(20, FILAS_P1)
    else:
        datos = pd.DataFrame(rng.normal(size=(FILAS, len(VARIABLES_P2))), columns=VARIABLES_P2)
        datos[OBJETIVO_P2] = (rng.random(FILAS) > 0.7).astype(int)
        bloques = pd.date_range("2026-06-01", periods=FILAS // 10, freq="3h")
        datos[COLUMNA_FECHA_P2] = np.repeat(bloques, 10)
    return datos


def entrenar_y_evaluar(pregunta: str, modelo: str, params: dict, semilla: int):
    """Entrena y evalúa un candidato sin MLflow.

    Returns:
        Las predicciones (la probabilidad en P2 y el grupo en P3) y las métricas.
    """
    datos = datos_sinteticos(pregunta)
    experimento = EXPERIMENTOS[pregunta](modelo, params, semilla=semilla)
    experimento.train = datos
    if pregunta == "P3":
        experimento.perfil = datos.copy()
        experimento.X_escalado = StandardScaler().fit_transform(datos)
        experimento.entrenar(datos)
        metricas = experimento.evaluar(datos, "train")
        return experimento.perfil["grupo"].to_numpy(), metricas
    experimento.entrenar(datos)
    metricas = experimento.evaluar(datos, "val")
    pred = experimento.puntaje if pregunta == "P2" else experimento.pred
    return pred, metricas


class TestReproducibilidad:
    @pytest.mark.parametrize(("pregunta", "modelo", "params"), CANDIDATOS)
    def test_misma_semilla_mismas_predicciones_y_metricas(self, pregunta, modelo, params):
        """Verifica que entrenar dos veces con la misma semilla da el mismo resultado."""
        pred_1, metricas_1 = entrenar_y_evaluar(pregunta, modelo, params, SEMILLA)
        pred_2, metricas_2 = entrenar_y_evaluar(pregunta, modelo, params, SEMILLA)

        assert pred_1 == pytest.approx(pred_2)
        assert metricas_1 == pytest.approx(metricas_2)

    @pytest.mark.parametrize(("pregunta", "modelo", "params"), CANDIDATOS[:3])
    def test_otra_semilla_cambia_las_predicciones(self, pregunta, modelo, params):
        """Verifica que la semilla sí llega al modelo: con otra, hgb, rf y kmeans predicen distinto."""
        pred_1, _ = entrenar_y_evaluar(pregunta, modelo, params, SEMILLA)
        pred_2, _ = entrenar_y_evaluar(pregunta, modelo, params, SEMILLA + 1)

        assert pred_1 != pytest.approx(pred_2)
