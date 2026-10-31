import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mi_bici.config import (
    CAPACIDAD_REBALANCEO,
    COLUMNA_FECHA_P2,
    OBJETIVO_P2,
    UMBRAL_DECISION_P2,
    VARIABLES_P2,
)
from mi_bici.modeling.p2_riesgo import ExperimentoRiesgo, umbral_por_capacidad


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

    def test_umbral_por_capacidad_con_puntajes_conocidos(self):
        """Verifica que el umbral sea el más bajo que no rebasa la capacidad por bloque."""
        puntaje = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        bloques = ["b1", "b1", "b1", "b2", "b2", "b2"]  # 2 bloques
        # Con capacidad=2 por bloque caben hasta 4 alertas en total -> umbral 0.6
        assert umbral_por_capacidad(puntaje, bloques, capacidad=2) == 0.6
        # Si hay empate en 0.6 que haría pasar de 2 alertas en 1 bloque, sube al siguiente (0.9)
        assert umbral_por_capacidad(np.array([0.9, 0.6, 0.6, 0.4]), ["b1"] * 4, capacidad=2) == 0.9

    def test_modelo_guardado_predice_con_umbral_de_validacion(self, tmp_path):
        """Verifica que el modelo guardado con MLflow predice igual que predict_proba >= val_umbral_decision."""
        rng = np.random.default_rng(42)
        datos = pd.DataFrame(
            {col: rng.normal(size=40) for col in VARIABLES_P2}
            | {
                COLUMNA_FECHA_P2: np.repeat(["2026-06-01", "2026-06-02"], 20),
                OBJETIVO_P2: (rng.random(40) > 0.7).astype(int),
            }
        )
        experimento = ExperimentoRiesgo("hgb")
        experimento.train = datos
        experimento.entrenar(datos.iloc[:25])
        metricas = experimento.evaluar(datos.iloc[25:], "val")
        assert metricas["val_alertas_por_bloque"] <= CAPACIDAD_REBALANCEO

        ruta = tmp_path / "modelo"
        mlflow.sklearn.save_model(
            experimento.modelo_a_guardar(),
            ruta,
            skops_trusted_types=list(experimento.tipos_confiables),
        )
        X = datos[VARIABLES_P2]
        esperado = (
            experimento.modelo.predict_proba(X)[:, 1] >= metricas["val_umbral_decision"]
        ).astype(int)
        np.testing.assert_array_equal(mlflow.pyfunc.load_model(ruta).predict(X), esperado)
