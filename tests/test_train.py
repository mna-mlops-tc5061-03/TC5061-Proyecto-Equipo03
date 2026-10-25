import pytest

from mi_bici.modeling.p1_demanda import ExperimentoDemanda
from mi_bici.modeling.p2_riesgo import ExperimentoRiesgo
from mi_bici.modeling.train import EXPERIMENTOS, leer_params

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


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

    def test_catalogo_p1_contiene_modelos_requeridos(self):
        """Verifica que P1 contenga baseline, hgb, rf y ridge en su catálogo con cuadrículas."""
        catalogo = ExperimentoDemanda.catalogo
        for clave in ("baseline", "hgb", "rf", "ridge"):
            assert clave in catalogo
        assert set(catalogo["rf"].cuadricula) == {"n_estimators", "max_depth"}
        assert set(catalogo["ridge"].cuadricula) == {"alpha"}

    def test_ridge_utiliza_standard_scaler(self):
        """Verifica que el modelo ridge construya un Pipeline con StandardScaler y Ridge."""
        experimento = ExperimentoDemanda("ridge")
        modelo = experimento.crear_modelo()
        assert isinstance(modelo, Pipeline)
        assert isinstance(modelo.named_steps["escalado"], StandardScaler)
        assert isinstance(modelo.named_steps["modelo"], Ridge)

    def test_random_forest_recibe_parametros_personalizados(self):
        """Verifica que RandomForestRegressor acepte hiperparámetros y mantenga semilla."""
        experimento = ExperimentoDemanda("rf", {"n_estimators": 40, "max_depth": 8}, semilla=123)
        estimador = experimento.crear_modelo()
        assert isinstance(estimador, RandomForestRegressor)
        assert estimador.n_estimators == 40
        assert estimador.max_depth == 8
        assert estimador.random_state == 123

