import numpy as np
import pandas as pd

from loguru import logger
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
        