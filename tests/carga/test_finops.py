"""Pruebas del análisis FinOps con CSV sintéticos, sin depender de corridas reales."""

import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from mi_bici.carga.finops import (
    analizar_finops,
    app,
    costo_por_inferencia,
    escalabilidad_mibici,
    escalon_sostenible,
    leer_limite_cpu_api,
    resumen_por_escalon,
)
from mi_bici.config import (
    BLOQUE_HORAS,
    DIAS_MES,
    HORAS_MES,
    LATENCIA_OBJETIVO_MS,
    LOTE_CARGA,
    PRECIO_HORA_USD,
    TASA_ERROR_MAX,
)

ESCALONES_PRUEBA = [(10, 60), (50, 60), (100, 60), (200, 60)]


def _construir_historia_y_recursos(
    perfiles: list[dict],
    incluir_rampa: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genera `locust_stats_history` y `recursos` sintéticos para 4 escalones.

    Cada perfil define `usuarios`, `rps_1`, `rps_lote`, `rps_health`, `p50_ms`, `p95_1_ms`,
    `p95_lote_ms`, `tasa_error`, `cpu_api` y `mem_api`.
    """
    t0 = 1_700_000_000
    filas_hist = []
    filas_rec = []
    acum = {
        "Aggregated": [0.0, 0.0],
        "/predict/demanda [1]": [0.0, 0.0],
        "/predict/riesgo [1]": [0.0, 0.0],
        f"/predict/demanda [{LOTE_CARGA}]": [0.0, 0.0],
        f"/predict/riesgo [{LOTE_CARGA}]": [0.0, 0.0],
        "/health": [0.0, 0.0],
    }

    for indice, perfil in enumerate(perfiles):
        u_obj = perfil["usuarios"]
        u_prev = perfiles[indice - 1]["usuarios"] if indice > 0 else 0
        for segundo_local in range(0, 60, 10):
            ts = t0 + indice * 60 + segundo_local
            en_rampa = incluir_rampa and segundo_local == 0 and u_prev < u_obj
            user_count = (u_prev + u_obj) // 2 if en_rampa else u_obj
            factor = 0.5 if en_rampa else 1.0

            rps_1 = perfil["rps_1"] * factor
            rps_lote = perfil["rps_lote"] * factor
            rps_health = perfil["rps_health"] * factor
            rps_tot = rps_1 + rps_lote + rps_health
            err = perfil["tasa_error"]

            rutas = [
                (
                    "Aggregated",
                    "",
                    rps_tot,
                    perfil["p50_ms"],
                    max(perfil["p95_1_ms"], perfil["p95_lote_ms"]),
                ),
                ("/predict/demanda [1]", "POST", rps_1 / 2, perfil["p50_ms"], perfil["p95_1_ms"]),
                ("/predict/riesgo [1]", "POST", rps_1 / 2, perfil["p50_ms"], perfil["p95_1_ms"]),
                (
                    f"/predict/demanda [{LOTE_CARGA}]",
                    "POST",
                    rps_lote / 2,
                    perfil["p50_ms"] * 2,
                    perfil["p95_lote_ms"],
                ),
                (
                    f"/predict/riesgo [{LOTE_CARGA}]",
                    "POST",
                    rps_lote / 2,
                    perfil["p50_ms"] * 2,
                    perfil["p95_lote_ms"],
                ),
                ("/health", "GET", rps_health, 1.0, 5.0),
            ]
            for nombre, tipo, rps, p50, p95 in rutas:
                fallos_s = rps * err
                acum[nombre][0] += rps * 10
                acum[nombre][1] += fallos_s * 10
                filas_hist.append(
                    {
                        "Timestamp": ts,
                        "User Count": user_count,
                        "Type": tipo,
                        "Name": nombre,
                        "Requests/s": rps,
                        "Failures/s": fallos_s,
                        "50%": p50,
                        "95%": p95,
                        "Total Request Count": acum[nombre][0],
                        "Total Failure Count": acum[nombre][1],
                    }
                )
            filas_rec.append(
                {
                    "epoch": ts,
                    "contenedor": "api",
                    "cpu_pct": perfil["cpu_api"] * factor,
                    "mem_mib": perfil["mem_api"],
                }
            )
            filas_rec.append(
                {
                    "epoch": ts,
                    "contenedor": "mlflow",
                    "cpu_pct": 1.0,
                    "mem_mib": 2200.0,
                }
            )

    return pd.DataFrame(filas_hist), pd.DataFrame(filas_rec)


@pytest.fixture
def datos_carga_sinteticos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cuatro escalones: 10 y 50 cumplen SLO; 100 falla por latencia p95; 200 falla por errores."""
    perfiles = [
        {
            "usuarios": 10,
            "rps_1": 6.0,
            "rps_lote": 2.0,
            "rps_health": 1.0,
            "p50_ms": 5.0,
            "p95_1_ms": 15.0,
            "p95_lote_ms": 40.0,
            "tasa_error": 0.0,
            "cpu_api": 10.0,
            "mem_api": 190.0,
        },
        {
            "usuarios": 50,
            "rps_1": 30.0,
            "rps_lote": 10.0,
            "rps_health": 5.0,
            "p50_ms": 12.0,
            "p95_1_ms": 45.0,
            "p95_lote_ms": 180.0,
            "tasa_error": 0.002,
            "cpu_api": 40.0,
            "mem_api": 210.0,
        },
        {
            "usuarios": 100,
            "rps_1": 60.0,
            "rps_lote": 20.0,
            "rps_health": 10.0,
            "p50_ms": 200.0,
            "p95_1_ms": 620.0,
            "p95_lote_ms": 850.0,
            "tasa_error": 0.005,
            "cpu_api": 85.0,
            "mem_api": 230.0,
        },
        {
            "usuarios": 200,
            "rps_1": 80.0,
            "rps_lote": 30.0,
            "rps_health": 10.0,
            "p50_ms": 90.0,
            "p95_1_ms": 250.0,
            "p95_lote_ms": 300.0,
            "tasa_error": 0.05,
            "cpu_api": 98.0,
            "mem_api": 250.0,
        },
    ]
    return _construir_historia_y_recursos(perfiles, incluir_rampa=True)


class TestRpsSostenibleYMeseta:
    def test_elige_escalon_correcto_y_excluye_rampa(self, datos_carga_sinteticos):
        """Verifica que se elige 50 u (100 falla p95, 200 falla error) y se excluye la rampa."""
        historia, recursos = datos_carga_sinteticos

        escalones = resumen_por_escalon(historia, recursos, escalones=ESCALONES_PRUEBA)
        mejor = escalon_sostenible(escalones)

        assert list(escalones["usuarios"]) == [10, 50, 100, 200]
        assert list(escalones["cumple_slo"]) == [True, True, False, False]
        assert int(mejor["usuarios"]) == 50
        # Si entrara la fila de rampa (factor 0.5), el RPS bajaría de 45.0 y la CPU bajaría de 40.0
        assert mejor["rps"] == pytest.approx(45.0)
        assert mejor["cpu_media_pct"] == pytest.approx(40.0)
        assert mejor["mem_max_mib"] == pytest.approx(210.0)


class TestCostoYEscalabilidad:
    def test_formula_costo_por_inferencia_y_lotes(self):
        """Verifica la fórmula `PRECIO_HORA_USD / (rps * pred_por_req * 3600)` para 1 y 470."""
        rps = 50.0
        unit_1, mil_1 = costo_por_inferencia(rps, PRECIO_HORA_USD, predicciones_por_peticion=1)
        unit_lote, mil_lote = costo_por_inferencia(
            rps, PRECIO_HORA_USD, predicciones_por_peticion=LOTE_CARGA
        )

        esperado_1 = PRECIO_HORA_USD / (rps * 3600)
        esperado_lote = PRECIO_HORA_USD / (rps * LOTE_CARGA * 3600)
        assert unit_1 == pytest.approx(esperado_1)
        assert mil_1 == pytest.approx(esperado_1 * 1000)
        assert unit_lote == pytest.approx(esperado_lote)
        assert mil_lote == pytest.approx(esperado_lote * 1000)
        assert unit_1 / unit_lote == pytest.approx(LOTE_CARGA)

    def test_analizar_finops_separa_modos_y_mezcla_real(self, datos_carga_sinteticos):
        """Verifica `por_peticion`, `por_prediccion` (sin `/health`), `una_estacion` y `lote`."""
        historia, recursos = datos_carga_sinteticos

        _, finops = analizar_finops(historia, recursos, escalones=ESCALONES_PRUEBA)
        por_modo = finops.set_index("modo")

        assert list(finops["modo"]) == ["por_peticion", "por_prediccion", "una_estacion", "lote"]
        # En el escalón de 50 u: rps[1]=30, rps[470]=10, /health=5 → mezcla = 30 + 470*10 = 4,730
        assert por_modo.loc["por_peticion", "rps_sostenible"] == pytest.approx(45.0)
        assert por_modo.loc["por_prediccion", "rps_sostenible"] == pytest.approx(40.0)
        assert por_modo.loc["por_prediccion", "predicciones_por_segundo"] == pytest.approx(
            30.0 + LOTE_CARGA * 10.0
        )
        assert por_modo.loc["por_prediccion", "costo_inferencia_usd"] == pytest.approx(
            PRECIO_HORA_USD / ((30.0 + LOTE_CARGA * 10.0) * 3600)
        )
        assert por_modo.loc["una_estacion", "costo_inferencia_usd"] == pytest.approx(
            PRECIO_HORA_USD / (30.0 * 3600)
        )
        assert por_modo.loc["lote", "costo_inferencia_usd"] == pytest.approx(
            PRECIO_HORA_USD / (10.0 * LOTE_CARGA * 3600)
        )
        # Costo efectivo mensual = (PRECIO_HORA_USD * 720 / demanda_mensual) * 1000
        demanda_esperada = (LOTE_CARGA + LOTE_CARGA * (24 // BLOQUE_HORAS)) * DIAS_MES
        assert por_modo.loc["por_prediccion", "demanda_mensual_pred"] == demanda_esperada
        assert por_modo.loc["por_prediccion", "costo_efectivo_1000_usd"] == pytest.approx(
            round((PRECIO_HORA_USD * HORAS_MES / demanda_esperada) * 1000, 4)
        )

    def test_escalabilidad_calcula_instancias_pico(self):
        """Verifica que con capacidad holgada basta 1 instancia y con baja capacidad escala."""
        holgada = escalabilidad_mibici(rps_sostenible=40.0, predicciones_por_peticion=LOTE_CARGA)
        saturada = escalabilidad_mibici(rps_sostenible=5.0, predicciones_por_peticion=1)

        assert holgada["instancias_pico"] == 1
        # Pico = 2 * 470 = 940 pred en 60 s = 15.67 pred/s → ceil(15.67 / 5.0) = 4 instancias
        assert saturada["instancias_pico"] == 4


class TestFalloSinEscalonValidoYCli:
    def test_ningun_escalon_cumple_lanza_mensaje_claro_sin_division_por_cero(
        self, datos_carga_sinteticos
    ):
        """Verifica que si p95 supera el objetivo en todos los escalones no divide entre cero."""
        historia, recursos = datos_carga_sinteticos
        historia_lenta = historia.assign(**{"95%": LATENCIA_OBJETIVO_MS + 100.0})

        with pytest.raises(ValueError, match="Ningún escalón .* cumplió el objetivo de servicio"):
            analizar_finops(historia_lenta, recursos, escalones=ESCALONES_PRUEBA)

    def test_cli_genera_csvs_y_figuras_y_falla_claro_si_no_cumple_slo(
        self, datos_carga_sinteticos, tmp_path
    ):
        """Verifica el flujo completo de `make finops` y el código de salida 1 sin escalón válido."""
        historia, recursos = datos_carga_sinteticos
        hist_csv = tmp_path / "locust_stats_history.csv"
        rec_csv = tmp_path / "recursos.csv"
        maq_json = tmp_path / "maquina.json"
        historia.to_csv(hist_csv, index=False)
        recursos.to_csv(rec_csv, index=False)
        maq_json.write_text(
            json.dumps({"contenedores": {"api": {"cpus": 2.0, "mem_gib": 4.0}}}),
            encoding="utf-8",
        )
        assert leer_limite_cpu_api(maq_json) == pytest.approx(200.0)

        runner = CliRunner()
        res_ok = runner.invoke(
            app,
            [
                "--historia-csv",
                str(hist_csv),
                "--recursos-csv",
                str(rec_csv),
                "--maquina-json",
                str(maq_json),
                "--salida-finops",
                str(tmp_path / "finops.csv"),
                "--salida-escalones",
                str(tmp_path / "escalones.csv"),
                "--figuras-dir",
                str(tmp_path / "figs"),
            ],
        )
        assert res_ok.exit_code == 0, res_ok.output
        assert (tmp_path / "finops.csv").exists()
        assert (tmp_path / "escalones.csv").exists()
        for png in ("latencia_carga.png", "rps_y_error_carga.png", "recursos_carga.png"):
            assert (tmp_path / "figs" / png).exists()

        # Caso sin escalón válido: falla por tasa de error > TASA_ERROR_MAX
        hist_err = tmp_path / "hist_err.csv"
        historia.assign(
            **{
                "Failures/s": historia["Requests/s"] * (TASA_ERROR_MAX + 0.1),
                "Total Failure Count": historia["Total Request Count"] * (TASA_ERROR_MAX + 0.1),
            }
        ).to_csv(hist_err, index=False)
        res_err = runner.invoke(
            app,
            [
                "--historia-csv",
                str(hist_err),
                "--recursos-csv",
                str(rec_csv),
                "--maquina-json",
                str(maq_json),
                "--salida-finops",
                str(tmp_path / "finops_err.csv"),
                "--salida-escalones",
                str(tmp_path / "escalones_err.csv"),
                "--figuras-dir",
                str(tmp_path / "figs_err"),
            ],
        )
        assert res_err.exit_code == 1
        assert not (tmp_path / "finops_err.csv").exists()
