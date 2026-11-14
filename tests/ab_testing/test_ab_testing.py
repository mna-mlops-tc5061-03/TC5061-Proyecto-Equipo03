from types import SimpleNamespace

import mlflow

# Se importa el módulo real: mlflow.pyfunc es un LazyLoader hasta que alguien lo importa, y el
# reemplazo de load_model se perdería al cargarse
import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest

from mi_bici import ab_testing
from mi_bici.config import (
    ALIAS_PRODUCCION,
    ALIAS_RESPALDO,
    CORTE_AB,
    EXPERIMENT_NAME,
    PERIODO_ACTUAL,
    VARIABLES_P1,
)

ESTACIONES = 30
DIAS = 31


def demanda(desde: str = CORTE_AB, dias: int = DIAS) -> pd.DataFrame:
    """Arma una demanda diaria sintética de varias estaciones, sin datos de MiBici."""
    fechas = pd.date_range(desde, periods=dias)
    generador = np.random.default_rng(0)
    datos = pd.DataFrame(
        {
            "fecha": np.repeat(fechas, ESTACIONES),
            "estacion_id": np.tile(np.arange(ESTACIONES), dias),
            **{v: generador.integers(0, 20, dias * ESTACIONES) for v in VARIABLES_P1},
        }
    )
    datos["dia_semana"] = datos["fecha"].dt.dayofweek
    datos["viajes"] = generador.integers(0, 40, len(datos)).astype(float)
    return datos


def predicciones(datos: pd.DataFrame, error: float, semilla: int) -> np.ndarray:
    """Predice el objetivo con un error aleatorio de tamaño `error`."""
    ruido = np.random.default_rng(semilla).normal(0, error, len(datos))
    return datos["viajes"].to_numpy() + ruido


class ModeloFalso:
    """Modelo del Registry que devuelve predicciones fijas."""

    def __init__(self, pred):
        self.pred = pred

    def predict(self, X):
        return self.pred[: len(X)]


class ClienteFalso:
    """Cliente de MLflow con champion y challenger que anota cada cambio de alias."""

    def __init__(self):
        self.aliases = {ALIAS_PRODUCCION: "1", ALIAS_RESPALDO: "3"}
        self.cambios = []

    def get_model_version_by_alias(self, nombre, alias):
        return SimpleNamespace(version=self.aliases[alias])

    def set_registered_model_alias(self, nombre, alias, version):
        self.cambios.append((alias, version))
        self.aliases[alias] = version


@pytest.fixture(scope="module")
def mlflow_temporal(tmp_path_factory):
    """Crea un MLflow temporal con SQLite, uno para todo el módulo porque crearlo tarda."""
    carpeta = tmp_path_factory.mktemp("mlflow")
    uri_previa = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{carpeta / 'mlflow.db'}")
    mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=(carpeta / "artefactos").as_uri())
    yield
    mlflow.set_tracking_uri(uri_previa)


@pytest.fixture
def ab(mlflow_temporal, tmp_path, monkeypatch):
    """Prepara el A/B con datos sintéticos y modelos falsos.

    Devuelve una función que fija los errores de A y de B y corre `ab_test`.
    """
    monkeypatch.setattr(ab_testing, "AB_DIR", tmp_path / "ab_testing")
    datos = demanda()
    monkeypatch.setattr(ab_testing, "datos_ab", lambda: datos)

    def correr(error_a: float, error_b: float, mover: bool):
        modelos = {
            ALIAS_PRODUCCION: ModeloFalso(predicciones(datos, error_a, 1)),
            ALIAS_RESPALDO: ModeloFalso(predicciones(datos, error_b, 2)),
        }
        monkeypatch.setattr(
            mlflow.pyfunc, "load_model", lambda uri: modelos[uri.rsplit("@", 1)[1]]
        )
        cliente = ClienteFalso()
        tabla, resultado = ab_testing.ab_test(cliente, mover=mover)
        return cliente, tabla, resultado

    return correr


class TestComparar:
    def test_b_claramente_mejor_se_promueve(self):
        """Si B se equivoca mucho menos, p < 0.05, el IC excluye 0 y la regla lo promueve."""
        datos = demanda()

        resultado = ab_testing.comparar(
            datos, predicciones(datos, 5.0, 1), predicciones(datos, 1.0, 2)
        )

        assert resultado["delta_mae"] > 0
        assert resultado["p_valor"] < 0.05
        assert resultado["ic_bajo"] > 0
        assert resultado["promover"]

    def test_modelos_identicos_no_se_promueven(self):
        """Caso límite: con las mismas predicciones no hay diferencia ni promoción."""
        datos = demanda()
        pred = predicciones(datos, 3.0, 1)

        resultado = ab_testing.comparar(datos, pred, pred.copy())

        assert resultado["delta_mae"] == 0
        assert resultado["p_valor"] == 1.0
        assert not resultado["promover"]

    def test_b_peor_no_se_promueve(self):
        """Aunque la diferencia sea significativa, si favorece a A no se promueve B."""
        datos = demanda()

        resultado = ab_testing.comparar(
            datos, predicciones(datos, 1.0, 1), predicciones(datos, 5.0, 2)
        )

        assert resultado["p_valor"] < 0.05
        assert resultado["ic_alto"] < 0
        assert not resultado["promover"]

    def test_desglose_habiles_y_fin_de_semana(self):
        """La diferencia se separa entre días hábiles y fines de semana."""
        datos = demanda()
        pred_a = datos["viajes"].to_numpy() + np.where(datos["dia_semana"] >= 5, 4.0, 1.0)

        resultado = ab_testing.comparar(datos, pred_a, datos["viajes"].to_numpy())

        assert resultado["delta_mae_habiles"] == pytest.approx(1.0)
        assert resultado["delta_mae_fin_de_semana"] == pytest.approx(4.0)


class TestBootstrap:
    def test_misma_semilla_mismo_intervalo(self):
        """El intervalo es reproducible con la semilla de config.py."""
        datos = demanda()
        diferencias = np.random.default_rng(3).normal(0.5, 1, len(datos))
        grupos = datos["estacion_id"].to_numpy()

        primero = ab_testing.intervalo_bootstrap(diferencias, grupos)
        segundo = ab_testing.intervalo_bootstrap(diferencias, grupos)

        assert primero == segundo
        assert primero[0] < diferencias.mean() < primero[1]

    def test_remuestrear_estaciones_da_un_intervalo_mas_ancho(self):
        """Si cada estación tiene su propia diferencia, remuestrear filas subestima la incertidumbre."""
        datos = demanda()
        grupos = datos["estacion_id"].to_numpy()
        por_estacion = np.random.default_rng(4).normal(0.5, 2, ESTACIONES)
        diferencias = por_estacion[grupos]

        bajo, alto = ab_testing.intervalo_bootstrap(diferencias, grupos)
        bajo_filas, alto_filas = ab_testing.intervalo_bootstrap(
            diferencias, np.arange(len(diferencias))
        )

        assert (alto - bajo) > 3 * (alto_filas - bajo_filas)


class TestDatos:
    def test_ninguna_fila_antes_de_corte_ab(self, monkeypatch):
        """El A/B solo evalúa desde CORTE_AB, y los rezagos pueden venir de antes."""
        lecturas = []
        monkeypatch.setattr(
            ab_testing, "load_processed_travels", lambda hasta: lecturas.append(hasta)
        )
        monkeypatch.setattr(ab_testing, "load_processed_stations", lambda: None)
        anterior = (pd.Timestamp(CORTE_AB) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        monkeypatch.setattr(
            ab_testing, "demanda_diaria", lambda viajes, estaciones: demanda(anterior, 20)
        )

        datos = ab_testing.datos_ab()

        assert datos["fecha"].min() == pd.Timestamp(CORTE_AB)
        assert len(datos) == 10 * ESTACIONES
        assert lecturas == [PERIODO_ACTUAL[1]]


class TestPromocion:
    @pytest.mark.lento
    def test_sin_promover_no_se_mueve_ningun_alias(self, ab):
        """Aunque la regla favorezca a B, sin --promover los aliases quedan igual."""
        cliente, _, resultado = ab(5.0, 1.0, mover=False)

        assert resultado["promover"]
        assert cliente.cambios == []

    @pytest.mark.lento
    def test_promover_intercambia_champion_y_challenger(self, ab):
        """Con --promover y la regla a favor, B pasa a champion y A a challenger."""
        cliente, tabla, resultado = ab(5.0, 1.0, mover=True)

        assert cliente.aliases == {ALIAS_PRODUCCION: "3", ALIAS_RESPALDO: "1"}
        assert list(tabla["version"]) == [
            "champion (v1)",
            "challenger (v3)",
            "baseline (viajes_lag_7)",
        ]
        run = mlflow.get_run(resultado["run_id"])
        assert run.data.tags["stage_of_work"] == "ab_test"
        assert run.data.tags["decision"] == "promover"
        assert run.data.params["promover_solicitado"] == "True"

    @pytest.mark.lento
    def test_promover_sin_evidencia_no_mueve_nada(self, ab):
        """Con --promover pero modelos iguales en error, la regla no promueve y nada cambia."""
        cliente, _, resultado = ab(3.0, 3.0, mover=True)

        assert not resultado["promover"]
        assert cliente.cambios == []
        assert mlflow.get_run(resultado["run_id"]).data.tags["decision"] == "mantener"
