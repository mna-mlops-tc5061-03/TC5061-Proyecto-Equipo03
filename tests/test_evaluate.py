from mi_bici.config import METRICA_SELECCION
from mi_bici.modeling.evaluate import metricas_clasificacion, metricas_regresion
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
