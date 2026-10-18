import numpy as np
import pandas as pd
import pytest

from mi_bici.dataset import clean_data_stations, clean_data_travels


class TestCleaning:
    def test_clean_data_stations(self):
        """Verifica estandarización de columnas y tipos en datos de estaciones."""
        df_raw = pd.DataFrame(
            {
                "id": ["2", "3"],
                "name": ["Estacion A", "Estacion B"],
                "latitude": ["20.666", "20.667"],
                "longitude": ["-103.348", "-103.366"],
                "dpcapacity": ["15", "19"],
            }
        )
        cleaned = clean_data_stations(df_raw)

        expected_cols = ["estacion_id", "nombre", "latitud", "longitud", "capacidad"]
        assert list(cleaned.columns) == expected_cols
        assert cleaned["estacion_id"].dtype == "int64"
        assert cleaned["latitud"].dtype == "float64"
        assert cleaned["longitud"].dtype == "float64"
        assert cleaned["capacidad"].dtype == "int64"

    def test_clean_data_travels_duplicates_and_columns(self):
        """Verifica detección y eliminación de duplicados en viaje_id, y nuevos nombres origen_id / destino_id."""
        df_raw = pd.DataFrame(
            {
                "Viaje_Id": [101, 101, 102],  # Duplicado en 101
                "Usuario_Id": [1, 1, 2],
                "Genero": ["M", "M", "F"],
                "Año_de_nacimiento": [1990, 1990, np.nan],  # Nulo en nacimiento que debe preservarse
                "Inicio_del_viaje": [
                    "2025-10-01 08:00:00",
                    "2025-10-01 08:00:00",
                    "2025-10-01 09:00:00",
                ],
                "Fin_del_viaje": [
                    "2025-10-01 08:15:00",
                    "2025-10-01 08:15:00",
                    "2025-10-01 09:20:00",
                ],
                "Origen_Id": [10, 10, 20],
                "Destino_Id": [15, 15, 25],
            }
        )
        cleaned = clean_data_travels(df_raw)

        # Debe haber eliminado el duplicado de viaje_id 101
        assert len(cleaned) == 2
        # No deben estar viaje_id ni usuario_id
        assert "viaje_id" not in cleaned.columns
        assert "usuario_id" not in cleaned.columns
        # Deben estar origen_id y destino_id
        assert "origen_id" in cleaned.columns
        assert "destino_id" in cleaned.columns
        # Nacimiento nulo debe mantenerse
        assert cleaned["nacimiento"].isna().sum() == 1
        assert cleaned["nacimiento"].dtype == "Int64"
        assert cleaned["origen_id"].iloc[0] == 10
        assert cleaned["destino_id"].iloc[0] == 15

    def test_clean_data_travels_removes_trash_trips(self):
        """Verifica eliminación de viajes < 2 min con mismo origen y destino o duraciones negativas."""
        df_raw = pd.DataFrame(
            {
                "Viaje_Id": [1, 2, 3],
                "Usuario_Id": [1, 2, 3],
                "Genero": ["M", "F", "M"],
                "Año_de_nacimiento": [1985, 1992, 1999],
                "Inicio_del_viaje": [
                    "2025-10-01 08:00:00",
                    "2025-10-01 08:00:00",
                    "2025-10-01 08:00:00",
                ],
                "Fin_del_viaje": [
                    "2025-10-01 08:01:00",  # 1 min, origen == destino -> basura
                    "2025-10-01 08:01:00",  # 1 min, origen != destino -> válido
                    "2025-10-01 07:50:00",  # Negativo -> basura
                ],
                "Origen_Id": [10, 10, 30],
                "Destino_Id": [10, 20, 40],
            }
        )
        cleaned = clean_data_travels(df_raw)
        assert len(cleaned) == 1
        assert cleaned["origen_id"].iloc[0] == 10
        assert cleaned["destino_id"].iloc[0] == 20
