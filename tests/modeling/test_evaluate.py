import numpy as np
import pytest

from mi_bici.config import METRICA_SELECCION
from mi_bici.modeling.evaluate import (
    metricas_clasificacion,
    metricas_clustering,
    metricas_regresion,
)
from mi_bici.modeling.train import EXPERIMENTOS


class TestEvaluate:
    def test_prefijo_en_el_nombre_de_las_metricas(self):
        """Verifica que el prefijo indica el conjunto evaluado y que prueba es el valor por defecto."""
        y, pred = [0, 1, 1, 0], [0, 1, 0, 0]

        assert set(metricas_regresion(y, pred, "val")) == {"val_mae", "val_rmse", "val_r2"}
        assert "test_average_precision" in metricas_clasificacion(y, pred, pred)

    def test_metrica_de_seleccion_no_usa_prueba(self):
        """Verifica que cada pregunta elige con una métrica de validación o de entrenamiento."""
        assert set(METRICA_SELECCION) == set(EXPERIMENTOS)
        assert not any(metrica.startswith("test_") for metrica, _ in METRICA_SELECCION.values())

    def test_clasificacion_sin_bloques_en_riesgo_da_cero(self):
        """Verifica que sin positivos las métricas valen 0 y se avisa, en lugar de fallar."""
        with pytest.warns(UserWarning, match="No positive class"):
            metricas = metricas_clasificacion([0, 0, 0, 0], [0, 1, 0, 0], [0.1, 0.9, 0.2, 0.3])

        assert metricas == {
            "test_recall": 0.0,
            "test_precision": 0.0,
            "test_f1": 0.0,
            "test_average_precision": 0.0,
        }

    def test_clustering_con_un_solo_grupo_falla(self):
        """Verifica que con un solo grupo la silueta no se puede calcular y se lanza ValueError."""
        X = np.random.default_rng(0).normal(size=(10, 2))

        with pytest.raises(ValueError, match="Number of labels is 1"):
            metricas_clustering(X, [0] * 10)
