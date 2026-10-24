import pytest

from mi_bici.modeling.train import EXPERIMENTOS, ExperimentoRiesgo, leer_params


class TestTrain:
    def test_leer_params_convierte_tipos(self):
        """Verifica que los números, booleanos y None conservan su tipo y el resto queda como texto."""
        params = leer_params("learning_rate=0.05, max_iter=200,early_stopping=False,class_weight=balanced,max_depth=None")

        assert params == {
            "learning_rate": 0.05,
            "max_iter": 200,
            "early_stopping": False,
            "class_weight": "balanced",
            "max_depth": None,
        }
        assert leer_params("") == {}

    def test_catalogo_tiene_baseline_y_candidato(self):
        """Verifica que cada pregunta tiene un baseline sin estimador y al menos un candidato."""
        for clase in EXPERIMENTOS.values():
            assert clase.catalogo["baseline"].clase is None
            assert any(m.clase is not None for m in clase.catalogo.values())

    def test_params_del_run_reemplazan_al_catalogo(self):
        """Verifica que los hiperparámetros del run se suman a los del catálogo y llegan al estimador."""
        experimento = ExperimentoRiesgo("hgb", {"learning_rate": 0.05}, semilla=7)
        estimador = experimento.crear_modelo()

        assert experimento.tipo == "candidato"
        assert estimador.class_weight == "balanced"
        assert estimador.learning_rate == 0.05
        assert estimador.random_state == 7

    def test_modelo_desconocido_falla(self):
        """Verifica que una clave fuera del catálogo se rechaza con un mensaje claro."""
        with pytest.raises(ValueError, match="modelo debe ser uno de"):
            ExperimentoRiesgo("xgboost")
