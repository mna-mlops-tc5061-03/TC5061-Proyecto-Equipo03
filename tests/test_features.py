import numpy as np
import pandas as pd
import pytest

from mi_bici.config import CORTE_PRUEBA, FESTIVOS, MIN_DIAS_PERFIL, UMBRAL_RIESGO
from mi_bici.features import (
    VARIABLES_PERFIL_USO,
    agregar_calendario,
    agregar_rezagos,
    convertir_fechas,
    demanda_diaria,
    etiqueta_riesgo,
    flujo_por_bloque,
    perfil_estacion,
)


@pytest.fixture
def serie_dos_estaciones() -> pd.DataFrame:
    """Diez días de viajes de dos estaciones, ordenados por estación y fecha."""
    fechas = pd.date_range("2026-06-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "estacion_id": [1] * 10 + [2] * 10,
            "fecha": list(fechas) * 2,
            "viajes": list(range(1, 11)) + list(range(101, 111)),
        }
    )


class TestAgregarCalendario:
    @pytest.mark.parametrize(
        ("fecha", "dia_semana"),
        [("2026-06-01", 0), ("2026-06-06", 5), ("2026-06-07", 6)],
    )
    def test_dia_de_la_semana(self, fecha, dia_semana):
        """Verifica el día de la semana, con 0 = lunes."""
        df = agregar_calendario(pd.DataFrame({"fecha": pd.to_datetime([fecha])}), "fecha")

        assert df["dia_semana"].iloc[0] == dia_semana

    @pytest.mark.parametrize("festivo", FESTIVOS)
    def test_festivo_queda_marcado(self, festivo):
        """Caso límite: cada día de FESTIVOS se marca, incluso a media tarde, y el día siguiente no."""
        fechas = pd.Timestamp(festivo) + pd.to_timedelta([15, 24 + 15], unit="h")
        df = agregar_calendario(pd.DataFrame({"bloque": fechas}), "bloque")

        assert df["festivo"].tolist() == [1, 0]


class TestAgregarRezagos:
    def test_rezagos_por_estacion(self, serie_dos_estaciones):
        """Verifica el rezago de 1 y 7 días dentro de cada estación."""
        df = agregar_rezagos(serie_dos_estaciones, "viajes", [1, 7])
        estacion_2 = df[df["estacion_id"] == 2].reset_index(drop=True)

        assert estacion_2.loc[1, "viajes_lag_1"] == 101
        assert estacion_2.loc[7, "viajes_lag_7"] == 101
        assert estacion_2.loc[9, "viajes_lag_7"] == 103

    def test_primeros_dias_quedan_vacios_sin_mezclar_estaciones(self, serie_dos_estaciones):
        """Caso límite: los primeros días de cada estación no toman valores de otra estación."""
        df = agregar_rezagos(serie_dos_estaciones, "viajes", [1, 7], ventana=7)
        estacion_2 = df[df["estacion_id"] == 2].reset_index(drop=True)

        assert np.isnan(estacion_2.loc[0, "viajes_lag_1"])
        assert estacion_2.loc[:6, "viajes_lag_7"].isna().all()
        assert estacion_2.loc[:6, "viajes_media_7"].isna().all()

    def test_media_movil_no_usa_el_dia_que_se_predice(self, serie_dos_estaciones):
        """Caso límite: la media de 7 días solo usa los 7 días anteriores, nunca el mismo día.

        El día 8 tiene un valor enorme. Si la media lo incluyera, habría fuga de información.
        """
        serie_dos_estaciones.loc[7, "viajes"] = 1000
        df = agregar_rezagos(serie_dos_estaciones, "viajes", [1], ventana=7)

        assert df.loc[7, "viajes_media_7"] == pytest.approx(np.mean([1, 2, 3, 4, 5, 6, 7]))
        assert df.loc[8, "viajes_media_7"] == pytest.approx(np.mean([2, 3, 4, 5, 6, 7, 1000]))


class TestDemandaDiaria:
    def test_una_fila_por_estacion_y_dia(self, viajes, estaciones):
        """Verifica una fila por estación y día y que el conteo coincide con los viajes."""
        demanda = demanda_diaria(viajes, estaciones)

        assert not demanda.duplicated(["estacion_id", "fecha"]).any()
        fila = demanda.iloc[len(demanda) // 2]
        con_fechas = convertir_fechas(viajes)
        esperado = (
            (con_fechas["origen_id"] == fila["estacion_id"])
            & (con_fechas["inicio"].dt.normalize() == fila["fecha"])
        ).sum()
        assert fila["viajes"] == esperado

    def test_dia_sin_viajes_cuenta_como_cero(self, viajes, estaciones):
        """Caso límite: si una estación no tiene viajes un día, ese día aparece con 0."""
        dia = (pd.Timestamp(CORTE_PRUEBA) - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
        sin_ese_dia = viajes[~((viajes["origen_id"] == 1) & viajes["inicio"].str.startswith(dia))]

        demanda = demanda_diaria(sin_ese_dia, estaciones)
        fila = demanda[(demanda["estacion_id"] == 1) & (demanda["fecha"] == dia)]

        assert len(fila) == 1
        assert fila["viajes"].iloc[0] == 0

    def test_descarta_los_dias_sin_historia(self, viajes, estaciones):
        """Caso límite: los días sin 7 días previos no tienen rezagos completos y se descartan."""
        demanda = demanda_diaria(viajes, estaciones)
        primer_viaje = convertir_fechas(viajes).groupby("origen_id")["inicio"].min().dt.normalize()

        primera_fila = demanda.groupby("estacion_id")["fecha"].min()

        assert (primera_fila >= primer_viaje[primera_fila.index] + pd.Timedelta(days=7)).all()
        assert not demanda.isna().any().any()


def flujo_de_una_estacion(ultimo_flujo: int, capacidad: int = 10) -> pd.DataFrame:
    """Flujo de 60 bloques de una estación: 0 en todos y `ultimo_flujo` en el último."""
    flujo = pd.DataFrame(
        {
            "estacion_id": 1,
            "bloque": pd.date_range("2026-06-01", periods=60, freq="3h"),
            "salidas": 0.0,
            "llegadas": 0.0,
            "capacidad": capacidad,
        }
    )
    flujo.loc[59, "llegadas"] = max(ultimo_flujo, 0)
    flujo.loc[59, "salidas"] = max(-ultimo_flujo, 0)
    flujo["flujo_neto"] = flujo["llegadas"] - flujo["salidas"]
    return flujo


class TestEtiquetaRiesgo:
    def test_flujo_que_pasa_el_umbral_marca_riesgo(self, viajes, estaciones):
        """Verifica que riesgo es 1 cuando el flujo neto mueve al menos UMBRAL_RIESGO de la capacidad."""
        riesgo = etiqueta_riesgo(flujo_por_bloque(viajes, estaciones), umbral=UMBRAL_RIESGO)

        esperado = (riesgo["flujo_neto"].abs() >= UMBRAL_RIESGO * riesgo["capacidad"]).astype(int)
        pd.testing.assert_series_equal(riesgo["riesgo"], esperado, check_names=False)
        # El pico de cada lunes en el conftest deja bloques en riesgo
        assert riesgo["riesgo"].sum() > 0
        assert riesgo["riesgo"].isin([0, 1]).all()

    @pytest.mark.parametrize(
        ("ultimo_flujo", "riesgo"),
        [(4, 1), (-4, 1), (3, 0), (-3, 0), (0, 0)],
    )
    def test_flujo_justo_en_el_umbral(self, ultimo_flujo, riesgo):
        """Caso límite: con capacidad 10 y umbral 0.4, un flujo de 4 en cualquier sentido ya es riesgo."""
        resultado = etiqueta_riesgo(flujo_de_una_estacion(ultimo_flujo), umbral=0.4)

        assert resultado["riesgo"].iloc[-1] == riesgo

    def test_excluye_estaciones_sin_capacidad(self, viajes, estaciones, estacion_sin_capacidad):
        """Caso límite: una estación con capacidad 0 no tiene fracción que medir y queda fuera."""
        flujo = flujo_por_bloque(viajes, estaciones)
        assert estacion_sin_capacidad in set(flujo["estacion_id"])

        riesgo = etiqueta_riesgo(flujo, umbral=UMBRAL_RIESGO)

        assert estacion_sin_capacidad not in set(riesgo["estacion_id"])
        assert (riesgo["capacidad"] > 0).all()


class TestPerfilEstacion:
    def test_proporciones_por_hora_suman_uno(self, viajes, estaciones):
        """Verifica que la proporción de salidas por hora de cada estación suma 1."""
        perfil = perfil_estacion(viajes, estaciones)
        horas = [f"hora_{h:02d}" for h in range(24)]

        np.testing.assert_allclose(perfil[horas].sum(axis=1), 1.0)
        assert perfil["fin_de_semana"].between(0, 1).all()
        assert perfil["balance_neto"].between(-1, 1).all()

    @pytest.mark.parametrize(("min_dias", "incluida"), [(MIN_DIAS_PERFIL, False), (5, True)])
    def test_descarta_estaciones_con_pocos_dias(
        self, viajes, estaciones, estacion_nueva, min_dias, incluida
    ):
        """Caso límite: la estación con 10 días antes del corte solo entra si se piden menos días."""
        perfil = perfil_estacion(viajes, estaciones, min_dias=min_dias)

        assert (estacion_nueva in set(perfil["estacion_id"])) is incluida

    def test_no_usa_viajes_desde_el_corte_de_prueba(self, viajes, estaciones):
        """Caso límite: agregar o quitar viajes desde CORTE_PRUEBA no cambia el perfil."""
        antes = viajes[viajes["inicio"] < CORTE_PRUEBA]

        con_prueba = perfil_estacion(viajes, estaciones)
        sin_prueba = perfil_estacion(antes, estaciones)

        pd.testing.assert_frame_equal(con_prueba, sin_prueba)

    def test_no_tiene_columnas_por_viaje(self, viajes, estaciones):
        """Caso límite: el perfil es por estación y no trae género ni año de nacimiento por viaje."""
        perfil = perfil_estacion(viajes, estaciones)

        assert {"genero", "nacimiento", "inicio", "fin"}.isdisjoint(perfil.columns)
        assert set(VARIABLES_PERFIL_USO) <= set(perfil.columns)
        assert not perfil["estacion_id"].duplicated().any()
