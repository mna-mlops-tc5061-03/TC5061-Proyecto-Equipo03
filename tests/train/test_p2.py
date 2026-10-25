from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mi_bici.config import UMBRAL_DECISION_P2
from mi_bici.modeling.p2_riesgo import ExperimentoRiesgo


class TestTrainP2:
    def test_catalogo_p2_contiene_modelos_requeridos(self):
        """Verifica que P2 contenga baseline, hgb, logreg y rf en su catálogo con cuadrículas."""
        catalogo = ExperimentoRiesgo.catalogo
        for clave in ("baseline", "hgb", "logreg", "rf"):
            assert clave in catalogo
        assert set(catalogo["rf"].cuadricula) == {"n_estimators", "max_depth"}
        assert set(catalogo["logreg"].cuadricula) == {"C"}
        assert set(catalogo["hgb"].cuadricula) == {"learning_rate", "max_leaf_nodes"}

    def test_logreg_p2_utiliza_standard_scaler_y_balanceo(self):
        """Verifica que logreg en P2 construya un Pipeline con StandardScaler y LogisticRegression balanceado."""
        experimento = ExperimentoRiesgo("logreg")
        modelo = experimento.crear_modelo()
        assert isinstance(modelo, Pipeline)
        assert isinstance(modelo.named_steps["escalado"], StandardScaler)
        assert isinstance(modelo.named_steps["modelo"], LogisticRegression)
        assert modelo.named_steps["modelo"].class_weight == "balanced"

    def test_random_forest_p2_es_clasificador_balanceado(self):
        """Verifica que rf en P2 sea RandomForestClassifier balanceado."""
        experimento = ExperimentoRiesgo("rf", {"n_estimators": 30}, semilla=99)
        estimador = experimento.crear_modelo()
        assert isinstance(estimador, RandomForestClassifier)
        assert estimador.class_weight == "balanced"
        assert estimador.n_estimators == 30
        assert estimador.random_state == 99

    def test_umbral_decision_inicial_viene_de_config(self):
        """Verifica que ExperimentoRiesgo inicialice su umbral_decision con el valor de config."""
        experimento = ExperimentoRiesgo("hgb")
        assert experimento.umbral_decision == UMBRAL_DECISION_P2
