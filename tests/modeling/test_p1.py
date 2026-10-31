from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mi_bici.modeling.p1_demanda import ExperimentoDemanda


class TestTrainP1:
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
