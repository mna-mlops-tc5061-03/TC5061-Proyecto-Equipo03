from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
