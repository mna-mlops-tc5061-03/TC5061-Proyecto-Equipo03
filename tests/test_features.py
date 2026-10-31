import pandas as pd
import pytest

from mi_bici.config import VARIABLES_P1, VENTANA_TENDENCIA
from mi_bici.features import demanda_diaria

ESTACIONES = pd.DataFrame(
    {
        "estacion_id": [1],
        "nombre": ["(GDL-01) Estación sintética"],
        "latitud": [20.67],
        "longitud": [-103.35],
        "capacidad": [20],
    }
)


def viajes_por_dia(conteos: list[int], inicio: str = "2026-05-01") -> pd.DataFrame:
    """Viajes sintéticos de una estación con el formato de la ingesta: `conteos[i]` el día i."""
    filas = []
    for dia, n in zip(pd.date_range(inicio, periods=len(conteos), freq="D"), conteos):
        salida = dia + pd.Timedelta(hours=8)
        filas += [
            {
                "genero": "NE",
                "inicio": salida.strftime("%Y-%m-%d %H:%M"),
                "fin": (salida + pd.Timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M"),
                "origen_id": 1,
                "destino_id": 1,
                "nacimiento": 1990,
                "duracion_min": 15,
            }
        ] * n
    return pd.DataFrame(filas)


class TestTendencia:
    def test_variables_de_tendencia_en_p1(self):
        """Verifica que P1 usa la media larga y la tendencia."""
        assert f"viajes_media_{VENTANA_TENDENCIA}" in VARIABLES_P1
        assert "viajes_tendencia" in VARIABLES_P1

    def test_tendencia_no_usa_los_viajes_del_mismo_dia(self):
        """Un pico el día 40 no cambia la tendencia de ese día, solo la del día siguiente.

        Todos los días anteriores tienen 2 viajes, así que la media de 7 y la de 28 valen 2 y la
        tendencia del día del pico es 0. Si la tendencia usara los viajes del mismo día, saldría
        positiva y habría fuga de información.
        """
        conteos = [2] * 39 + [50, 2]
        demanda = demanda_diaria(viajes_por_dia(conteos), ESTACIONES).set_index("fecha")
        pico, siguiente = pd.Timestamp("2026-06-09"), pd.Timestamp("2026-06-10")

        assert demanda.loc[pico, "viajes"] == 50
        assert demanda.loc[pico, "viajes_tendencia"] == pytest.approx(0)
        assert demanda.loc[siguiente, "viajes_media_7"] == pytest.approx((6 * 2 + 50) / 7)
        assert demanda.loc[siguiente, "viajes_tendencia"] == pytest.approx(
            (6 * 2 + 50) / 7 - (27 * 2 + 50) / 28
        )

    @pytest.mark.parametrize(
        ("conteos", "signo"),
        [(list(range(1, 41)), 1), (list(range(40, 0, -1)), -1), ([5] * 40, 0)],
        ids=["subiendo", "bajando", "estable"],
    )
    def test_signo_de_la_tendencia(self, conteos, signo):
        """Verifica que la tendencia es positiva si la demanda sube, negativa si baja y 0 si no cambia."""
        demanda = demanda_diaria(viajes_por_dia(conteos), ESTACIONES)

        assert (demanda["viajes_tendencia"].apply(lambda v: (v > 0) - (v < 0)) == signo).all()

    def test_descarta_los_dias_sin_la_ventana_completa(self):
        """Caso límite: la primera fila es el día VENTANA_TENDENCIA, sin vacíos en las medias."""
        demanda = demanda_diaria(viajes_por_dia([3] * 35), ESTACIONES)

        assert demanda["fecha"].min() == pd.Timestamp("2026-05-01") + pd.Timedelta(
            days=VENTANA_TENDENCIA
        )
        assert len(demanda) == 35 - VENTANA_TENDENCIA
        assert not demanda[VARIABLES_P1].isna().any().any()
