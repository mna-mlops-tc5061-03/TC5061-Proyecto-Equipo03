import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from mi_bici.config import (
    ALIAS_PRODUCCION,
    ALIAS_RESPALDO,
    CORTE_AB,
    EXPERIMENT_NAME,
    MODELOS_REGISTRO,
    VARIABLES_P1,
)
from mi_bici.modeling import base, reentrenar
from mi_bici.modeling.base import ARBOLES_HGB
from mi_bici.modeling.registry import comprobar_carga

NOMBRE = MODELOS_REGISTRO["P1"]


def demanda_sintetica() -> pd.DataFrame:
    """Arma una demanda diaria de dos estaciones que cruza `CORTE_AB`, sin datos de MiBici."""
    corte = pd.Timestamp(CORTE_AB)
    fechas = pd.date_range(corte - pd.Timedelta(days=40), corte + pd.Timedelta(days=10))
    filas = len(fechas) * 2
    generador = np.random.default_rng(0)
    demanda = pd.DataFrame(
        {
            "fecha": np.repeat(fechas, 2),
            "estacion_id": np.tile([1, 2], len(fechas)),
            **{v: generador.integers(0, 20, filas) for v in VARIABLES_P1},
        }
    )
    demanda["viajes"] = demanda["viajes_lag_7"] + generador.integers(0, 3, filas)
    return demanda


@pytest.fixture
def registro(tmp_path, monkeypatch):
    """Registra un champion de P1 en un MLflow temporal y sustituye la lectura de datos."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    uri_previa = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    monkeypatch.setattr(base, "TRACKING_URI", uri)
    mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=(tmp_path / "artefactos").as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    lecturas = []
    monkeypatch.setattr(reentrenar, "load_processed_travels", lambda hasta: lecturas.append(hasta))
    monkeypatch.setattr(reentrenar, "load_processed_stations", lambda: None)
    monkeypatch.setattr(
        reentrenar, "demanda_diaria", lambda viajes, estaciones: demanda_sintetica()
    )

    datos = demanda_sintetica()
    champion = HistGradientBoostingRegressor(max_iter=5, max_leaf_nodes=7, random_state=42)
    champion.fit(datos[VARIABLES_P1], datos["viajes"])
    with mlflow.start_run() as run:
        mlflow.set_tags({"question": "p1", "git_commit": "0123456789abcdef"})
        mlflow.log_params({"modelo": "hgb", "data_version": "abc123"})
        mlflow.sklearn.log_model(
            champion,
            name="model",
            input_example=datos[VARIABLES_P1].head(3),
            skops_trusted_types=list(ARBOLES_HGB),
        )
    cliente = MlflowClient()
    version = mlflow.register_model(f"runs:/{run.info.run_id}/model", NOMBRE).version
    cliente.set_registered_model_alias(NOMBRE, ALIAS_PRODUCCION, version)
    yield cliente, lecturas
    mlflow.set_tracking_uri(uri_previa)


class TestReentrenar:
    @pytest.mark.lento
    def test_challenger_nuevo_sin_mover_al_champion(self, registro):
        """Verifica la versión nueva con @challenger, su linaje y que el champion sigue igual."""
        cliente, lecturas = registro

        resumen = reentrenar.reentrenar(cliente)

        assert (resumen["champion"], resumen["challenger"]) == ("1", "2")
        assert str(cliente.get_model_version_by_alias(NOMBRE, ALIAS_PRODUCCION).version) == "1"
        challenger = cliente.get_model_version_by_alias(NOMBRE, ALIAS_RESPALDO)
        assert str(challenger.version) == "2"
        assert challenger.run_id == resumen["run_id"]
        assert {"source_run", "git_commit", "data_version"} <= set(challenger.tags)
        assert challenger.tags["source_version"] == "1"
        assert challenger.description.startswith("Reentrenamiento de la v1")
        assert comprobar_carga(NOMBRE, ALIAS_RESPALDO) == 3
        # Los datos se leen hasta el día antes de CORTE_AB
        assert lecturas == [reentrenar.ULTIMO_DIA]

    @pytest.mark.lento
    def test_mismos_hiperparametros_y_sin_filas_de_octubre(self, registro):
        """Verifica que el run tiene los hiperparámetros del champion y ninguna fila desde CORTE_AB."""
        _, _ = registro
        estimador = mlflow.sklearn.load_model(f"models:/{NOMBRE}@{ALIAS_PRODUCCION}")
        experimento = reentrenar.ReentrenamientoDemanda("hgb", estimador, "1")

        run_id, _ = experimento.ejecutar()

        assert experimento.train["fecha"].max() < pd.Timestamp(CORTE_AB)
        assert experimento.modelo.get_params() == estimador.get_params()
        assert experimento.modelo is not estimador
        run = mlflow.get_run(run_id)
        assert run.data.tags["stage_of_work"] == "retraining"
        assert run.data.params["corte_ab"] == CORTE_AB
        assert run.data.params["filas_train"] == str(
            (demanda_sintetica()["fecha"] < CORTE_AB).sum()
        )
        assert "corte_prueba" not in run.data.params
