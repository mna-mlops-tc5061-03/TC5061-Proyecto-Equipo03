from loguru import logger
import numpy as np
import pandas as pd
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mi_bici.config import K_CANDIDATOS
from mi_bici.features import VARIABLES_PERFIL_USO
from mi_bici.modeling.p3_grupos import ExperimentoGrupos


class TestTrainP3:
    def test_catalogo_p3_contiene_modelos_requeridos(self):
        """Verifica que P3 contenga baseline, kmeans, birch y gmm en su catálogo con cuadrículas."""
        catalogo = ExperimentoGrupos.catalogo
        for clave in ("baseline", "kmeans", "birch", "gmm"):
            assert clave in catalogo
        assert set(catalogo["kmeans"].cuadricula) == {"n_clusters"}
        assert set(catalogo["birch"].cuadricula) == {"n_clusters"}
        assert set(catalogo["gmm"].cuadricula) == {"n_components"}

    def test_birch_utiliza_standard_scaler(self):
        """Verifica que el modelo birch construya un Pipeline con StandardScaler y Birch."""
        experimento = ExperimentoGrupos("birch")
        modelo = experimento.crear_modelo(n_clusters=4)
        assert isinstance(modelo, Pipeline)
        assert isinstance(modelo.named_steps["escalado"], StandardScaler)
        assert isinstance(modelo.named_steps["modelo"], Birch)
        assert modelo.named_steps["modelo"].n_clusters == 4

    def test_gmm_utiliza_standard_scaler_y_semilla(self):
        """Verifica que el modelo gmm construya un Pipeline con StandardScaler y GaussianMixture."""
        experimento = ExperimentoGrupos("gmm", semilla=123)
        modelo = experimento.crear_modelo(n_components=5)
        assert isinstance(modelo, Pipeline)
        assert isinstance(modelo.named_steps["escalado"], StandardScaler)
        assert isinstance(modelo.named_steps["modelo"], GaussianMixture)
        assert modelo.named_steps["modelo"].n_components == 5
        assert modelo.named_steps["modelo"].random_state == 123

    def test_gmm_acepta_n_clusters_como_alias(self):
        """Verifica que gmm traduzca n_clusters a n_components si se pasa en los parámetros."""
        experimento = ExperimentoGrupos("gmm", {"n_clusters": 6})
        modelo = experimento.crear_modelo()
        assert isinstance(modelo, Pipeline)
        assert modelo.named_steps["modelo"].n_components == 6

    def test_k_en_borde_detecta_extremos(self, caplog):
        """Verifica la etiqueta k_en_borde y el log_warning cuando k cae en el borde de K_CANDIDATOS."""
        experimento = ExperimentoGrupos("kmeans")
        experimento.modelo = experimento.crear_modelo()
        k_min = min(K_CANDIDATOS)
        experimento.k = k_min
        experimento.k_en_borde = experimento.k in (min(K_CANDIDATOS), max(K_CANDIDATOS))
        assert experimento.etiquetas()["k_en_borde"] == "True"
        experimento.k = 5
        experimento.k_en_borde = experimento.k in (min(K_CANDIDATOS), max(K_CANDIDATOS))
        assert experimento.etiquetas()["k_en_borde"] == "False"

    def test_entrenar_marca_k_en_borde(self):
        """Verifica que entrenar() marca k_en_borde y avisa en el log cuando k es el menor del rango."""
        rng = np.random.default_rng(0)
        datos = pd.DataFrame(
            rng.normal(size=(30, len(VARIABLES_PERFIL_USO))), columns=VARIABLES_PERFIL_USO
        )
        experimento = ExperimentoGrupos("kmeans", {"n_clusters": min(K_CANDIDATOS)})
        experimento.X_escalado = StandardScaler().fit_transform(datos)
        avisos = []
        sink = logger.add(avisos.append, level="WARNING")
        try:
            experimento.entrenar(datos)
        finally:
            logger.remove(sink)

        assert experimento.etiquetas()["k_en_borde"] == "True"
        assert any("borde" in aviso for aviso in avisos)

    def test_candidatos_registran_estabilidad_y_baseline_no(self):
        """Verifica que kmeans, birch y gmm registran las dos métricas de estabilidad y baseline no."""
        rng = np.random.default_rng(42)
        datos = pd.DataFrame(
            rng.normal(size=(40, len(VARIABLES_PERFIL_USO))), columns=VARIABLES_PERFIL_USO
        )
        datos["municipio"] = ["GDL", "ZAP", "TLA", "TLQ"] * 10
        X_escalado = StandardScaler().fit_transform(datos[VARIABLES_PERFIL_USO])

        base = ExperimentoGrupos("baseline")
        base.X_escalado = X_escalado
        metricas_base = base.evaluar(datos, "train")
        assert "train_silhouette_std" not in metricas_base
        assert "train_grupo_menor_min" not in metricas_base

        for clave in ("kmeans", "birch", "gmm"):
            exp = ExperimentoGrupos(clave, {"n_clusters": 3})
            exp.X_escalado = X_escalado
            exp.entrenar(datos)
            metricas = exp.evaluar(datos, "train")
            assert "train_silhouette_std" in metricas
            assert "train_grupo_menor_min" in metricas
            assert 0.0 < metricas["train_grupo_menor_min"] <= 1.0

    def test_modelo_determinista_da_desviacion_cero(self):
        """Verifica que un modelo determinista como Birch da desviación estándar 0."""
        rng = np.random.default_rng(42)
        datos = pd.DataFrame(
            rng.normal(size=(40, len(VARIABLES_PERFIL_USO))), columns=VARIABLES_PERFIL_USO
        )
        exp = ExperimentoGrupos("birch", {"n_clusters": 3})
        exp.X_escalado = StandardScaler().fit_transform(datos[VARIABLES_PERFIL_USO])
        exp.entrenar(datos)
        metricas = exp.evaluar(datos, "train")

        assert metricas["train_silhouette_std"] == 0.0
        assert 0.0 < metricas["train_grupo_menor_min"] <= 1.0

    def test_hijos_busqueda_no_miden_estabilidad(self):
        """Verifica que los hijos de la búsqueda no calculan las métricas de estabilidad."""
        rng = np.random.default_rng(42)
        datos = pd.DataFrame(
            rng.normal(size=(40, len(VARIABLES_PERFIL_USO))), columns=VARIABLES_PERFIL_USO
        )
        exp = ExperimentoGrupos("kmeans", {"n_clusters": 3})
        exp.etapa = "search"
        exp.X_escalado = StandardScaler().fit_transform(datos[VARIABLES_PERFIL_USO])
        exp.entrenar(datos)
        metricas = exp.evaluar(datos, "train")

        assert "train_silhouette_std" not in metricas
        assert "train_grupo_menor_min" not in metricas
