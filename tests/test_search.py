import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from mi_bici.modeling import base
from mi_bici.modeling.base import ExperimentoBase, Modelo
from mi_bici.modeling.evaluate import metricas_regresion
from mi_bici.modeling.search import leer_cuadricula


class ExperimentoFalso(ExperimentoBase):
    """Experimento chico con datos sintéticos para probar la búsqueda sin leer MiBici."""

    pregunta = "P1"
    catalogo = {  # noqa: RUF012
        "baseline": Modelo("regla"),
        "ridge": Modelo("lineal", Ridge, cuadricula={"alpha": [0.1, 1000.0]}),
    }
    columna_fecha = "fecha"

    def preparar(self):
        fechas = pd.date_range("2026-04-01", "2026-07-31", freq="D")
        x = np.arange(len(fechas), dtype=float)
        self.train = pd.DataFrame({"fecha": fechas, "x": x, "y": 2 * x + 1})
        self.X_train = self.train[["x"]]

    def entrenar(self, datos):
        self.modelo = self.crear_modelo()
        self.modelo.fit(datos[["x"]], datos["y"])

    def evaluar(self, datos, prefijo):
        return metricas_regresion(datos["y"], self.modelo.predict(datos[["x"]]), prefijo)

    def artefactos(self, carpeta):
        return []


class TestSearch:
    def test_leer_cuadricula_convierte_tipos(self):
        """Verifica que cada hiperparámetro queda con su lista de valores y sus tipos."""
        cuadricula = leer_cuadricula("learning_rate=0.05|0.1, max_leaf_nodes=15|31,loss=squared_error")

        assert cuadricula == {
            "learning_rate": [0.05, 0.1],
            "max_leaf_nodes": [15, 31],
            "loss": ["squared_error"],
        }
        assert leer_cuadricula("") == {}

    def test_busqueda_con_un_hijo_por_combinacion(self, tmp_path, monkeypatch):
        """Verifica el padre, un hijo por combinación medido solo en validación y la mejor combinación."""
        monkeypatch.setattr(base, "TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")

        run_id, mejor = ExperimentoFalso("ridge").buscar()

        runs = mlflow.search_runs(experiment_names=[base.EXPERIMENT_NAME])
        hijos = runs[runs["tags.mlflow.parentRunId"] == run_id]
        assert len(hijos) == 2
        assert set(runs["tags.stage_of_work"]) == {"search"}
        assert not any(c.startswith("metrics.test_") for c in hijos.dropna(axis=1, how="all"))
        assert mejor["alpha"] == 0.1
        padre = mlflow.get_run(run_id)
        assert padre.data.tags["mejor_params"] == "alpha=0.1"
