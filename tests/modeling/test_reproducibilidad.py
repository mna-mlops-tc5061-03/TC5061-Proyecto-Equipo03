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
    variables = VARIABLES_P1 if pregunta == "P1" else VARIABLES_P2
    datos = pd.DataFrame(rng.normal(size=(FILAS, len(variables))), columns=variables)
    if pregunta == "P1":
        datos[OBJETIVO_P1] = rng.poisson(20, FILAS)
    else:
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

    @pytest.mark.parametrize(("pregunta", "modelo", "params"), [CANDIDATOS[1], CANDIDATOS[2]])
    def test_otra_semilla_cambia_las_predicciones(self, pregunta, modelo, params):
        """Verifica que la semilla sí llega al modelo: con otra, rf y kmeans predicen distinto."""
        pred_1, _ = entrenar_y_evaluar(pregunta, modelo, params, SEMILLA)
        pred_2, _ = entrenar_y_evaluar(pregunta, modelo, params, SEMILLA + 1)

        assert pred_1 != pytest.approx(pred_2)
