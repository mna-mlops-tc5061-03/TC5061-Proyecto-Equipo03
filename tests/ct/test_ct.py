"""Pruebas del ciclo CT (monitoreo -> reentrenar -> A/B -> promoción) con modelos y datos chicos."""

import json
from pathlib import Path
import subprocess

import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from typer.testing import CliRunner

from mi_bici import ab_testing, ct, monitoreo
from mi_bici.config import (
    AB_DIR,
    ALIAS_PRODUCCION,
    ALIAS_RESPALDO,
    CORTE_AB,
    CT_DIR,
    EVIDENTLY_DIR,
    EXPERIMENT_NAME,
    MODELOS_REGISTRO,
    PERIODO_ACTUAL,
    PERIODO_REFERENCIA,
    PROJ_ROOT,
    VARIABLES_P1,
)
from mi_bici.modeling import base, reentrenar
from mi_bici.modeling.base import ARBOLES_HGB

NOMBRE = MODELOS_REGISTRO["P1"]
ESTACIONES = 15


def _demanda_ct(desplazamiento_actual: float) -> pd.DataFrame:
    """Arma una demanda diaria sintética de 15 estaciones entre junio y octubre de 2026.

    En `PERIODO_REFERENCIA` (jun–jul) los viajes rondan 15–25. Si `desplazamiento_actual > 0`, a
    partir de agosto la demanda sube a `15..25 + desplazamiento_actual`: un modelo de árboles
    entrenado solo en la referencia no puede extrapolar, dispara el umbral de MAE (> 1.15×) y el
    reentrenamiento con agosto–septiembre sí aprende el nivel nuevo y gana el A/B en octubre.
    """
    fechas = pd.date_range("2026-06-01", PERIODO_ACTUAL[1], freq="D")
    n_dias = len(fechas)
    filas = n_dias * ESTACIONES
    rng = np.random.default_rng(42)

    fechas_rep = np.repeat(fechas, ESTACIONES)
    estaciones_rep = np.tile(np.arange(1, ESTACIONES + 1), n_dias)
    en_actual = fechas_rep >= pd.Timestamp(PERIODO_ACTUAL[0])

    nivel = rng.uniform(15.0, 25.0, filas) + np.where(en_actual, desplazamiento_actual, 0.0)
    viajes = np.round(nivel + rng.normal(0.0, 0.5, filas), 2).clip(1.0)
    return pd.DataFrame(
        {
            "fecha": fechas_rep,
            "estacion_id": estaciones_rep,
            "viajes_lag_1": np.round(nivel + rng.normal(0.0, 0.4, filas), 2).clip(1.0),
            "viajes_lag_7": np.round(nivel + rng.normal(0.0, 0.6, filas), 2).clip(1.0),
            "viajes_media_7": np.round(nivel, 2),
            "dia_semana": pd.DatetimeIndex(fechas_rep).dayofweek.astype("int32"),
            "festivo": rng.binomial(1, 0.05, filas).astype("int64"),
            "capacidad": rng.integers(15, 30, filas, dtype="int64"),
            "viajes": viajes,
        }
    )


@pytest.fixture
def entorno_ct(tmp_path, monkeypatch):
    """Prepara un MLflow temporal con SQLite, un champion v1 y datos sintéticos configurables."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    uri_previa = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    for modulo in (base, monitoreo, ct):
        monkeypatch.setattr(modulo, "TRACKING_URI", uri)
    mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=(tmp_path / "artefactos").as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    def preparar(desplazamiento_actual: float) -> tuple[MlflowClient, pd.DataFrame]:
        datos = _demanda_ct(desplazamiento_actual)
        ref = datos[datos["fecha"].between(*PERIODO_REFERENCIA)]

        for modulo in (monitoreo, reentrenar, ab_testing):
            monkeypatch.setattr(modulo, "load_processed_travels", lambda hasta=None: None)
            monkeypatch.setattr(modulo, "load_processed_stations", lambda: None)
            monkeypatch.setattr(
                modulo, "demanda_diaria", lambda viajes, estaciones, df=datos: df.copy()
            )

        champion = HistGradientBoostingRegressor(max_iter=15, max_leaf_nodes=7, random_state=42)
        champion.fit(ref[VARIABLES_P1], ref["viajes"])
        with mlflow.start_run(run_name="P1-champion-inicial") as run:
            mlflow.set_tags({"question": "p1", "git_commit": "0123456789abcdef"})
            mlflow.log_params({"modelo": "hgb", "data_version": "sintetico"})
            mlflow.sklearn.log_model(
                champion,
                name="model",
                input_example=ref[VARIABLES_P1].head(3),
                skops_trusted_types=list(ARBOLES_HGB),
            )
        cliente = MlflowClient()
        version = str(mlflow.register_model(f"runs:/{run.info.run_id}/model", NOMBRE).version)
        cliente.set_registered_model_alias(NOMBRE, ALIAS_PRODUCCION, version)
        return cliente, datos

    yield preparar, tmp_path / "ct"
    mlflow.set_tracking_uri(uri_previa)


class TestReglaDisparador:
    def test_disparador_usa_columna_reentrenar_como_unica_fuente_de_verdad(self):
        """Verifica que `evaluar_disparador` obedece `por_mes['reentrenar']` sin duplicar la regla."""
        mae_ref = 5.0

        # Aunque el MAE esté arriba de 1.15×, si monitoreo marcó reentrenar=False no dispara
        sin_disparo = pd.DataFrame(
            {
                "mes": ["2026-08", "2026-09"],
                "mae_champion": [4.5, 9.0],
                "reentrenar": [False, False],
            }
        )
        res_no = ct.evaluar_disparador(sin_disparo, mae_ref)
        assert not res_no["disparado"]
        assert res_no["meses_disparados"] == []

        # Y si monitoreo marcó reentrenar=True, dispara aunque el MAE esté debajo de 1.15×
        con_disparo = pd.DataFrame(
            {
                "mes": ["2026-08", "2026-09", "2026-10"],
                "mae_champion": [4.5, 5.1, 7.0],
                "reentrenar": [False, True, True],
            }
        )
        res_si = ct.evaluar_disparador(con_disparo, mae_ref)
        assert res_si["disparado"]
        assert res_si["meses_disparados"] == ["2026-09", "2026-10"]
        assert res_si["mes_peor"] == "2026-10"

    def test_ct_escribe_en_carpeta_propia_sin_tocar_evidently_ni_ab_oficiales(self):
        """Verifica que la carpeta por defecto de CT (`CT_DIR`) es distinta de las versionadas."""
        assert CT_DIR != EVIDENTLY_DIR
        assert CT_DIR != AB_DIR
        assert CT_DIR.name == "ct"

    def test_cli_no_promueve_por_defecto_y_activa_con_bandera(self, monkeypatch):
        """Verifica que `python -m mi_bici.ct` pasa `promover=False` salvo con `--promover`."""
        llamadas = []
        monkeypatch.setattr(ct, "conectar_mlflow", lambda *a, **kw: None)
        monkeypatch.setattr(ct, "MlflowClient", lambda: object())

        def ct_falso(cliente, promover=False):
            llamadas.append(promover)
            return {
                "decision": "aprobado_sin_promover" if not promover else "promovido",
                "motivo": "prueba CLI",
                "run_id": "run_cli",
            }

        monkeypatch.setattr(ct, "ciclo_ct", ct_falso)
        runner = CliRunner()

        res_default = runner.invoke(ct.app, [])
        res_promover = runner.invoke(ct.app, ["--promover"])

        assert res_default.exit_code == 0, res_default.output
        assert res_promover.exit_code == 0, res_promover.output
        assert llamadas == [False, True]

    def test_make_ct_solo_pasa_promover_con_variable_explicita(self):
        """Verifica que `make ct` no pasa `--promover` por defecto y sí con `PROMOVER=1`."""
        sin_var = subprocess.run(
            ["make", "-n", "ct"], cwd=PROJ_ROOT, capture_output=True, text=True, check=True
        )
        con_var = subprocess.run(
            ["make", "-n", "ct", "PROMOVER=1"],
            cwd=PROJ_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        assert "-m mi_bici.ct" in sin_var.stdout
        assert "--promover" not in sin_var.stdout
        assert "--promover" in con_var.stdout


class TestCicloCT:
    @pytest.mark.lento
    def test_ct_sin_promover_corre_punta_a_punta_sin_mover_champion(self, entorno_ct):
        """Con disparador activo y A/B favorable, sin `PROMOVER=1` `@champion` queda en v1."""
        preparar, salida_dir = entorno_ct
        cliente, _ = preparar(desplazamiento_actual=30.0)

        resumen = ct.ciclo_ct(cliente, promover=False, salida_dir=salida_dir)

        assert (salida_dir / "evidently" / "p1_por_mes.csv").exists()
        assert (salida_dir / "ab_testing" / "p1_ab.csv").exists()
        assert resumen["disparado"] is True
        assert resumen["decision"] == "aprobado_sin_promover"
        assert resumen["promovido"] is False
        assert str(cliente.get_model_version_by_alias(NOMBRE, ALIAS_PRODUCCION).version) == "1"
        assert str(cliente.get_model_version_by_alias(NOMBRE, ALIAS_RESPALDO).version) == "2"

        # Cada etapa dejó su run en MLflow y el último registra la decisión y el motivo
        for etapa, key in (
            ("monitoring", "monitoreo"),
            ("retraining", "reentrenamiento"),
            ("ab_test", "ab_test"),
        ):
            run_etapa = cliente.get_run(resumen["runs"][key])
            assert run_etapa.data.tags["stage_of_work"] == etapa

        run_ct = cliente.get_run(resumen["run_id"])
        assert run_ct.data.tags["stage_of_work"] == "ct"
        assert run_ct.data.tags["decision"] == "aprobado_sin_promover"
        assert "PROMOVER" in run_ct.data.tags["motivo"]
        assert run_ct.data.params["promover_solicitado"] == "False"

    @pytest.mark.lento
    def test_ct_con_promover_mueve_champion_a_nueva_y_challenger_a_anterior(self, entorno_ct):
        """Con `PROMOVER=1` y A/B favorable, `@champion` pasa a v2 y `@challenger` a v1."""
        preparar, salida_dir = entorno_ct
        cliente, _ = preparar(desplazamiento_actual=30.0)

        resumen = ct.ciclo_ct(cliente, promover=True, salida_dir=salida_dir)

        assert resumen["disparado"] is True
        assert resumen["decision"] == "promovido"
        assert resumen["promovido"] is True
        assert str(cliente.get_model_version_by_alias(NOMBRE, ALIAS_PRODUCCION).version) == "2"
        assert str(cliente.get_model_version_by_alias(NOMBRE, ALIAS_RESPALDO).version) == "1"

        run_ct = cliente.get_run(resumen["run_id"])
        assert run_ct.data.tags["decision"] == "promovido"
        assert "v1" in run_ct.data.tags["motivo"] and "v2" in run_ct.data.tags["motivo"]
        artefactos = {a.path for a in cliente.list_artifacts(resumen["run_id"], path="data")}
        assert "data/decision_ct.json" in artefactos

    @pytest.mark.lento
    def test_ct_sin_disparador_no_reentrena_ni_corre_ab(self, entorno_ct):
        """Si ningún mes supera `1.15 × referencia`, no se reentrena ni se crea v2."""
        preparar, salida_dir = entorno_ct
        cliente, _ = preparar(desplazamiento_actual=0.0)

        resumen = ct.ciclo_ct(cliente, promover=True, salida_dir=salida_dir)

        assert resumen["disparado"] is False
        assert resumen["decision"] == "sin_reentrenamiento"
        assert resumen["promovido"] is False
        assert resumen["runs"]["reentrenamiento"] is None
        assert resumen["runs"]["ab_test"] is None
        assert str(cliente.get_model_version_by_alias(NOMBRE, ALIAS_PRODUCCION).version) == "1"
        assert ct._alias_opcional(cliente, NOMBRE, ALIAS_RESPALDO) is None

        run_ct = cliente.get_run(resumen["run_id"])
        assert run_ct.data.tags["stage_of_work"] == "ct"
        assert run_ct.data.tags["decision"] == "sin_reentrenamiento"

    @pytest.mark.lento
    def test_ct_con_promover_pero_ab_desfavorable_mantiene_champion(self, entorno_ct, monkeypatch):
        """Aunque `PROMOVER=1` y el disparador se active, si el A/B rechaza no mueve `@champion`."""
        preparar, salida_dir = entorno_ct
        cliente, _ = preparar(desplazamiento_actual=30.0)

        ab_original = ab_testing.comparar

        def comparar_desfavorable(datos, pred_a, pred_b, *args, **kwargs):
            # Intercambiamos las predicciones para simular un challenger peor que el champion
            return ab_original(datos, pred_b, pred_a, *args, **kwargs)

        monkeypatch.setattr(ab_testing, "comparar", comparar_desfavorable)

        resumen = ct.ciclo_ct(cliente, promover=True, salida_dir=salida_dir)

        assert resumen["disparado"] is True
        assert resumen["decision"] == "mantener"
        assert resumen["promovido"] is False
        assert str(cliente.get_model_version_by_alias(NOMBRE, ALIAS_PRODUCCION).version) == "1"
        assert str(cliente.get_model_version_by_alias(NOMBRE, ALIAS_RESPALDO).version) == "2"

        run_ct = cliente.get_run(resumen["run_id"])
        assert run_ct.data.tags["decision"] == "mantener"
        ruta_json = mlflow.artifacts.download_artifacts(
            run_id=resumen["run_id"], artifact_path="data/decision_ct.json"
        )
        assert json.loads(Path(ruta_json).read_text(encoding="utf-8"))["decision"] == "mantener"
        assert pd.Timestamp(CORTE_AB) == pd.Timestamp("2026-10-01")
