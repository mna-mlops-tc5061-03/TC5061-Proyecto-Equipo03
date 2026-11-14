"""Pruebas del reporte de drift de P1 con datos sintéticos y sin servidor de MLflow."""

from evidently import Report
import numpy as np
import pandas as pd
import pytest

from mi_bici.config import (
    COLUMNA_FECHA_P1,
    COLUMNA_PREDICCION,
    NUMERICAS_P1,
    PERIODO_ACTUAL,
    PERIODO_REFERENCIA,
    VARIABLES_P1,
)
from mi_bici.monitoreo import (
    COLUMNAS_P1,
    conjuntos,
    correr_reporte,
    evaluar_por_mes,
    registrar_monitoreo,
)

COLUMNAS_PERSONALES = ("genero", "nacimiento", "usuario_id")


def _fila_variable(tabla: pd.DataFrame, variable: str) -> pd.Series:
    return tabla.loc[tabla["variable"] == variable].iloc[0]


class TestDeteccionDrift:
    def test_columna_desplazada_marca_drift_e_identica_no(self, referencia_y_actual):
        """Verifica que desplazar una columna numérica marca drift solo en esa variable."""
        referencia, actual = referencia_y_actual
        actual = actual.assign(viajes_lag_1=actual["viajes_lag_1"] + 25.0)

        tabla, drift_share = correr_reporte(actual, referencia, html=None, csv=None)

        desplazada = _fila_variable(tabla, "viajes_lag_1")
        identica = _fila_variable(tabla, "capacidad")
        assert bool(desplazada["drift"]) is True
        assert desplazada["valor"] >= desplazada["umbral"]
        assert bool(identica["drift"]) is False
        assert identica["valor"] == pytest.approx(0.0)
        assert int(tabla["drift"].sum()) == 1
        assert drift_share == pytest.approx(1 / len(COLUMNAS_P1))

    def test_categorica_desplazada_marca_drift_e_identica_no(self, referencia_y_actual):
        """Verifica que cambiar la proporción de una categórica marca drift y otra idéntica no."""
        referencia, actual = referencia_y_actual
        rng = np.random.default_rng(7)
        actual = actual.assign(festivo=rng.binomial(1, 0.85, size=len(actual)))

        tabla, _ = correr_reporte(actual, referencia, html=None, csv=None)

        assert bool(_fila_variable(tabla, "festivo")["drift"]) is True
        assert bool(_fila_variable(tabla, "dia_semana")["drift"]) is False

    def test_sin_desplazamiento_no_marca_drift(self, referencia_y_actual):
        """Verifica que si todas las columnas mantienen su distribución no se reporta drift."""
        referencia, actual = referencia_y_actual

        tabla, drift_share = correr_reporte(actual, referencia, html=None, csv=None)

        assert not tabla["drift"].any()
        assert drift_share == pytest.approx(0.0)


class TestOrdenArgumentos:
    def test_orden_actual_primero_y_referencia_segundo(self, referencia_y_actual, monkeypatch):
        """Verifica que `correr_reporte` pasa `(actual, referencia)` a Evidently y no al revés.

        Evidently normaliza la distancia de Wasserstein dividiendo entre la desviación estándar de
        `referencia` (`std(referencia)`). Si 1 % de las filas de `actual` tienen un pico (+100), la
        distancia sin normalizar es ~1.0:
        - Con el orden correcto `(actual, referencia)`, se divide entre `std(referencia) ≈ 1.0`,
          dando `valor ≈ 1.0 >= 0.1` (`drift = True`).
        - Si se invierten `(referencia, actual)`, se divide entre `std(actual) ≈ 10.0`, cayendo a
          `valor < 0.1` (`drift = False`).
        """
        referencia, actual = referencia_y_actual
        actual = actual.copy()
        # 1 % de filas con pico eleva std(actual) a ~10 mientras std(referencia) sigue en ~1
        indices_pico = actual.index[: len(actual) // 100]
        actual.loc[indices_pico, "viajes_lag_1"] = actual.loc[indices_pico, "viajes_lag_1"] + 100.0

        orden_recibido = []
        run_original = Report.run

        def espiar_run(self, current_data, reference_data, *args, **kwargs):
            orden_recibido.append((current_data.as_dataframe(), reference_data.as_dataframe()))
            return run_original(self, current_data, reference_data, *args, **kwargs)

        monkeypatch.setattr(Report, "run", espiar_run)

        tabla_correcta, _ = correr_reporte(actual, referencia, html=None, csv=None)
        tabla_invertida, _ = correr_reporte(referencia, actual, html=None, csv=None)

        primero, segundo = orden_recibido[0]
        assert primero["viajes_lag_1"].std() > segundo["viajes_lag_1"].std()
        assert bool(_fila_variable(tabla_correcta, "viajes_lag_1")["drift"]) is True
        assert bool(_fila_variable(tabla_invertida, "viajes_lag_1")["drift"]) is False

    def test_registrar_monitoreo_pasa_actual_primero_a_correr_reporte(
        self, referencia_y_actual, tmp_path, monkeypatch
    ):
        """Verifica que `registrar_monitoreo(referencia, actual)` invierte a `(actual, referencia)`."""
        referencia, actual = referencia_y_actual
        actual = actual.copy()
        indices_pico = actual.index[: len(actual) // 100]
        actual.loc[indices_pico, "viajes_lag_1"] = actual.loc[indices_pico, "viajes_lag_1"] + 100.0

        monkeypatch.setattr("mi_bici.monitoreo.conectar_mlflow", lambda *a, **kw: None)
        monkeypatch.setattr("mi_bici.monitoreo.mlflow.set_experiment", lambda *a, **kw: None)
        monkeypatch.setattr("mi_bici.monitoreo.mlflow.set_tags", lambda *a, **kw: None)
        monkeypatch.setattr("mi_bici.monitoreo.mlflow.log_params", lambda *a, **kw: None)
        monkeypatch.setattr("mi_bici.monitoreo.mlflow.log_metrics", lambda *a, **kw: None)
        monkeypatch.setattr("mi_bici.monitoreo.mlflow.log_artifact", lambda *a, **kw: None)

        class _RunFalso:
            info = type("Info", (), {"run_id": "run_sintetico"})()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        monkeypatch.setattr("mi_bici.monitoreo.mlflow.start_run", lambda *a, **kw: _RunFalso())

        _, tabla, metricas = registrar_monitoreo(
            referencia,
            actual,
            html=None,
            csv=tmp_path / "drift.csv",
            por_mes_csv=tmp_path / "mes.csv",
            por_mes_png=tmp_path / "mes.png",
        )

        assert bool(_fila_variable(tabla, "viajes_lag_1")["drift"]) is True
        assert metricas["drift_share"] == pytest.approx(1 / len(COLUMNAS_P1))


class TestPeriodosSinTraslape:
    def test_periodos_en_config_no_se_traslapan(self):
        """Verifica que `PERIODO_REFERENCIA` termina estrictamente antes de `PERIODO_ACTUAL`."""
        ref_inicio, ref_fin = map(pd.Timestamp, PERIODO_REFERENCIA)
        act_inicio, act_fin = map(pd.Timestamp, PERIODO_ACTUAL)

        assert ref_inicio <= ref_fin
        assert act_inicio <= act_fin
        assert ref_fin < act_inicio

    def test_conjuntos_no_traslapa_fechas_y_calcula_rezagos_con_julio(
        self, viajes_y_estaciones_monitoreo, modelo_falso
    ):
        """Verifica que `conjuntos()` separa sin traslape y que agosto conserva rezagos de julio."""
        referencia, actual = conjuntos()

        fechas_ref = pd.to_datetime(referencia[COLUMNA_FECHA_P1])
        fechas_act = pd.to_datetime(actual[COLUMNA_FECHA_P1])

        assert not referencia.empty and not actual.empty
        assert fechas_ref.max() < fechas_act.min()
        assert set(fechas_ref).isdisjoint(set(fechas_act))
        assert fechas_ref.between(*PERIODO_REFERENCIA).all()
        assert fechas_act.between(*PERIODO_ACTUAL).all()

        primero_agosto = actual.loc[fechas_act == pd.Timestamp("2026-08-01")]
        assert not primero_agosto.empty
        assert (
            primero_agosto[["viajes_lag_1", "viajes_lag_7", "viajes_media_7"]].notna().all().all()
        )


class TestSinColumnasPersonales:
    def test_conjuntos_no_incluye_columnas_personales(
        self, viajes_y_estaciones_monitoreo, modelo_falso
    ):
        """Verifica que `COLUMNAS_P1` y las tablas de `conjuntos()` excluyen datos personales."""
        referencia, actual = conjuntos()

        assert set(COLUMNAS_PERSONALES).isdisjoint(COLUMNAS_P1)
        assert set(COLUMNAS_PERSONALES).isdisjoint(referencia.columns)
        assert set(COLUMNAS_PERSONALES).isdisjoint(actual.columns)

    def test_correr_reporte_filtra_columnas_personales_y_filas_en_csv_y_html(
        self, referencia_y_actual, tmp_path
    ):
        """Verifica que si entran columnas personales a `correr_reporte`, no salen en CSV ni HTML."""
        referencia, actual = referencia_y_actual
        canario = "usuario_privado_canario_999"
        for df in (referencia, actual):
            df["genero"] = "F"
            df["nacimiento"] = 1988
            df["usuario_id"] = canario

        ruta_html = tmp_path / "p1_drift.html"
        ruta_csv = tmp_path / "p1_drift.csv"
        tabla, _ = correr_reporte(actual, referencia, html=ruta_html, csv=ruta_csv)

        assert set(tabla["variable"]) == set(COLUMNAS_P1)
        assert set(COLUMNAS_PERSONALES).isdisjoint(tabla["variable"])
        assert list(pd.read_csv(ruta_csv).columns) == [
            "variable",
            "metodo",
            "valor",
            "umbral",
            "drift",
        ]

        contenido_html = ruta_html.read_text(encoding="utf-8")
        assert canario not in contenido_html
        for col_personal in COLUMNAS_PERSONALES:
            assert f'"{col_personal}"' not in contenido_html


class TestModeloFalsoSinMLflow:
    def test_conjuntos_sustituye_champion_por_modelo_falso(
        self, viajes_y_estaciones_monitoreo, modelo_falso
    ):
        """Verifica que `conjuntos()` agrega `prediccion` usando el modelo falso sin MLflow."""
        referencia, actual = conjuntos()

        assert modelo_falso.ultimos_datos is not None
        assert list(modelo_falso.ultimos_datos.columns) == VARIABLES_P1
        for df in (referencia, actual):
            assert COLUMNA_PREDICCION in df.columns
            np.testing.assert_allclose(df[COLUMNA_PREDICCION], df["viajes_media_7"] + 1.0)

    def test_evaluar_por_mes_detecta_mes_con_drift_y_mayor_error(
        self, referencia_y_actual, tmp_path
    ):
        """Verifica la tabla y gráfica mensual cuando octubre presenta drift y mayor MAE."""
        referencia, base_actual = referencia_y_actual
        meses_fechas = ("2026-08-15", "2026-09-15", "2026-10-15")
        partes = []
        for fecha_txt in meses_fechas:
            copia = base_actual.copy()
            copia[COLUMNA_FECHA_P1] = pd.Timestamp(fecha_txt)
            if fecha_txt.startswith("2026-10"):
                for col in NUMERICAS_P1:
                    if col != "capacidad":
                        copia[col] = copia[col] + 25.0
                # El objetivo crece más que la predicción para elevar el MAE del champion en octubre
                copia["viajes"] = copia["viajes"] + 10.0
            partes.append(copia)
        actual = pd.concat(partes, ignore_index=True)

        ruta_csv = tmp_path / "p1_por_mes.csv"
        ruta_png = tmp_path / "p1_por_mes.png"
        por_mes = evaluar_por_mes(actual, referencia, csv=ruta_csv, png=ruta_png)

        assert list(por_mes["mes"]) == ["2026-08", "2026-09", "2026-10"]
        assert list(por_mes["reentrenar"]) == [False, False, True]
        assert por_mes.loc[2, "drift_share"] >= 0.5
        assert por_mes.loc[2, "mae_champion"] > por_mes.loc[0, "mae_champion"]
        assert ruta_csv.exists() and ruta_png.exists()
