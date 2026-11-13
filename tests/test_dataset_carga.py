import pandas as pd
import pytest

from mi_bici import dataset
from mi_bici.config import FECHA_FIN


@pytest.fixture
def viajes_dir(tmp_path, monkeypatch):
    """Escribe viajes sintéticos alrededor de `FECHA_FIN`, particionados como los procesados."""
    fin = pd.Timestamp(FECHA_FIN)
    inicios = [
        fin - pd.Timedelta(days=1),
        fin + pd.Timedelta(hours=23, minutes=59),
        fin + pd.Timedelta(days=1),
        fin + pd.Timedelta(days=15, hours=8),
    ]
    viajes = pd.DataFrame(
        {
            "inicio": [i.strftime("%Y-%m-%d %H:%M") for i in inicios],
            "origen_id": [1, 2, 3, 4],
        }
    )
    viajes["anio"] = viajes["inicio"].str[:4]
    viajes["mes"] = viajes["inicio"].str[5:7]
    carpeta = tmp_path / "viajes"
    viajes.to_parquet(carpeta, partition_cols=["anio", "mes"], index=False)
    monkeypatch.setattr(dataset, "VIAJES_DIR", carpeta)
    return carpeta


class TestCargaHasta:
    def test_por_defecto_nada_despues_de_fecha_fin(self, viajes_dir):
        """Los viajes posteriores a FECHA_FIN no llegan a los experimentos."""
        viajes = dataset.load_processed_travels()
        assert sorted(viajes["origen_id"]) == [1, 2]
        assert viajes["inicio"].max() < "2026-10-01"

    def test_incluye_todo_el_ultimo_dia(self, viajes_dir):
        """Caso límite: un viaje a las 23:59 del último día sí se incluye."""
        viajes = dataset.load_processed_travels(hasta=FECHA_FIN)
        assert (viajes["inicio"] == f"{FECHA_FIN} 23:59").any()

    def test_otro_tope_y_sin_tope(self, viajes_dir):
        """El monitoreo puede leer hasta otra fecha o todos los viajes."""
        assert len(dataset.load_processed_travels(hasta="2026-10-01")) == 3
        assert len(dataset.load_processed_travels(hasta=None)) == 4

    def test_filtra_aunque_no_se_pida_inicio(self, viajes_dir):
        """El tope aplica aunque `inicio` no esté entre las columnas que se leen."""
        viajes = dataset.load_processed_travels(columns=["origen_id"])
        assert list(viajes.columns) == ["origen_id"]
        assert sorted(viajes["origen_id"]) == [1, 2]
