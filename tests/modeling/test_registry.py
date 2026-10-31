import mlflow
from mlflow import MlflowClient
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from mi_bici.config import ALIAS_PRODUCCION, ALIAS_RESPALDO, EXPERIMENT_NAME, MODELOS_REGISTRO
from mi_bici.modeling.registry import (
    buscar_runs,
    comprobar_carga,
    elegir,
    mejora,
    metrica_prueba,
    registrar,
)

PARTICION = {"data_version": "abc123", "corte_validacion": "2026-06-01", "corte_prueba": "2026-08-01"}
COMMIT = "0123456789abcdef"


def fila(run_id, etapa, pregunta="P1", git_dirty="False", data_version="abc123", **metricas):
    """Arma un run como los devuelve mlflow.search_runs."""
    return {
        "run_id": run_id,
        "tags.question": pregunta,
        "tags.stage_of_work": etapa,
        "tags.git_dirty": git_dirty,
        "tags.git_commit": COMMIT,
        **{f"params.{k}": v for k, v in PARTICION.items()},
        "params.data_version": data_version,
        **{f"metrics.{k}": v for k, v in metricas.items()},
    }


@pytest.fixture
def experimento(tmp_path):
    """Crea el experimento del equipo en un MLflow temporal con SQLite, que admite el Registry."""
    uri_previa = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=(tmp_path / "artefactos").as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    yield
    mlflow.set_tracking_uri(uri_previa)


def correr(etapa, val_mae, test_mae):
    """Registra un run de P1. Los candidatos guardan un modelo con ejemplo de entrada."""
    X = pd.DataFrame({"viajes_lag_7": [1.0, 2.0, 3.0, 4.0], "capacidad": [10.0, 12.0, 10.0, 12.0]})
    with mlflow.start_run() as run:
        mlflow.set_tags({"question": "P1", "stage_of_work": etapa, "git_dirty": "False", "git_commit": COMMIT})
        mlflow.log_params(PARTICION)
        mlflow.log_metrics({"val_mae": val_mae, "test_mae": test_mae})
        if etapa == "candidate":
            modelo = LinearRegression().fit(X, [1.0, 2.0, 3.0, 4.0])
            mlflow.sklearn.log_model(modelo, name="model", input_example=X.head(3))
    return run.info.run_id


class TestRegistry:
    def test_metrica_de_prueba_y_mejora(self):
        """Verifica que la métrica de prueba sale de la de validación y que la mejora positiva es mejor."""
        assert metrica_prueba("val_mae") == "test_mae"
        assert metrica_prueba("train_silhouette") is None
        assert mejora(5.0, 10.0, mayor_es_mejor=False) == 0.5
        assert mejora(0.3, 0.2, mayor_es_mejor=True) == pytest.approx(0.5)

    def test_elige_al_que_gana_en_validacion_y_en_prueba(self):
        """Verifica que un candidato que solo gana en validación se descarta y que el resto se ordena."""
        runs = pd.DataFrame(
            [
                fila("base", "baseline", val_mae=6.9, test_mae=6.9),
                fila("segundo", "candidate", val_mae=5.5, test_mae=5.8),
                fila("mejor", "candidate", val_mae=5.0, test_mae=5.5),
                fila("sobreajustado", "candidate", val_mae=4.0, test_mae=7.5),
            ]
        )

        base, aceptados = elegir(runs, "P1")

        assert base.run_id == "base"
        assert list(aceptados.run_id) == ["mejor", "segundo"]

    def test_mayor_es_mejor_y_p3_sin_prueba(self):
        """Verifica que en P2 y P3 gana el valor más alto y que P3 se compara sin prueba."""
        runs = pd.DataFrame(
            [
                fila("base", "baseline", pregunta="P3", train_silhouette=0.02),
                fila("peor", "candidate", pregunta="P3", train_silhouette=0.01),
                fila("kmeans", "candidate", pregunta="P3", train_silhouette=0.10),
            ]
        )

        _, aceptados = elegir(runs, "P3")

        assert list(aceptados.run_id) == ["kmeans"]

    def test_solo_compara_runs_limpios_con_los_mismos_datos(self):
        """Verifica que se ignoran los runs con cambios sin guardar, de otros datos o de otra pregunta."""
        runs = pd.DataFrame(
            [
                fila("base_sucio", "baseline", git_dirty="True", val_mae=9.0, test_mae=9.0),
                fila("base", "baseline", val_mae=6.9, test_mae=6.9),
                fila("otros_datos", "candidate", data_version="zzz999", val_mae=5.0, test_mae=5.0),
                fila("sucio", "candidate", git_dirty="True", val_mae=5.0, test_mae=5.0),
                fila("otra_pregunta", "candidate", pregunta="P2", val_mae=5.0, test_mae=5.0),
                fila("busqueda", "search", val_mae=5.0, test_mae=5.0),
                fila("valido", "candidate", val_mae=6.0, test_mae=6.0),
            ]
        )

        base, aceptados = elegir(runs, "P1")

        assert base.run_id == "base"
        assert list(aceptados.run_id) == ["valido"]

    def test_sin_baseline_falla(self):
        """Verifica que sin un baseline limpio no se elige nada y el error dice por qué."""
        runs = pd.DataFrame([fila("candidato", "candidate", val_mae=5.0, test_mae=5.0)])
        sin_runs = pd.DataFrame(columns=["run_id", "status", "start_time"])

        with pytest.raises(ValueError, match="no hay un baseline"):
            elegir(runs, "P1")
        with pytest.raises(ValueError, match="no hay un baseline"):
            elegir(sin_runs, "P1")

    def test_registra_champion_y_challenger_con_linaje(self, experimento):
        """Verifica los aliases, el linaje en las etiquetas y que el champion carga y predice."""
        correr("baseline", 6.9, 6.9)
        mejor = correr("candidate", 5.0, 5.5)
        segundo = correr("candidate", 5.5, 5.8)
        cliente = MlflowClient()
        nombre = MODELOS_REGISTRO["P1"]

        resumen = registrar("P1", buscar_runs(), cliente)

        champion = cliente.get_model_version_by_alias(nombre, ALIAS_PRODUCCION)
        assert champion.run_id == mejor
        assert champion.tags == {
            "experiment": EXPERIMENT_NAME,
            "source_run": mejor,
            "git_commit": COMMIT,
            "data_version": "abc123",
        }
        assert champion.description.startswith("Elegida por `val_mae` = 5.000")
        assert cliente.get_model_version_by_alias(nombre, ALIAS_RESPALDO).run_id == segundo
        assert f"@{ALIAS_PRODUCCION}" in cliente.get_registered_model(nombre).description
        assert resumen["mejora"] == pytest.approx((6.9 - 5.0) / 6.9)
        assert comprobar_carga(nombre) == 3

    def test_volver_a_registrar_no_repite_versiones(self, experimento):
        """Verifica que correr el registro dos veces reutiliza la versión del mismo run."""
        correr("baseline", 6.9, 6.9)
        correr("candidate", 5.0, 5.5)
        cliente = MlflowClient()

        registrar("P1", buscar_runs(), cliente)
        registrar("P1", buscar_runs(), cliente)

        versiones = cliente.search_model_versions(f"name = '{MODELOS_REGISTRO['P1']}'")
        assert len(versiones) == 1
        assert ALIAS_RESPALDO not in cliente.get_registered_model(MODELOS_REGISTRO["P1"]).aliases
