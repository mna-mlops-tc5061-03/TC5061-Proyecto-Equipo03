# pyrefly: ignore[missing-import]
import numpy as np
import pandas as pd
import pytest

from mi_bici.dataset import parse_datetime_series


class TestDateParsing:
    """Pruebas unitarias para la validación y parseo de fechas en múltiples formatos."""

    def test_iso_format_with_seconds(self):
        """Verifica parseo de formato ISO estándar con segundos (YYYY-MM-DD HH:MM:SS)."""
        series = pd.Series(["2025-10-01 00:00:26", "2025-10-15 14:30:45"])
        result = parse_datetime_series(series)

        assert pd.api.types.is_datetime64_any_dtype(result)
        assert result.iloc[0] == pd.Timestamp("2025-10-01 00:00:26")
        assert result.iloc[0].year == 2025
        assert result.iloc[0].month == 10
        assert result.iloc[0].day == 1
        assert result.iloc[0].second == 26

    def test_iso_format_without_seconds(self):
        """Verifica parseo de formato ISO sin segundos (YYYY-MM-DD HH:MM)."""
        series = pd.Series(["2025-10-01 00:00", "2025-10-15 14:30"])
        result = parse_datetime_series(series)

        assert pd.api.types.is_datetime64_any_dtype(result)
        assert result.iloc[0] == pd.Timestamp("2025-10-01 00:00:00")
        assert result.iloc[0].year == 2025
        assert result.iloc[0].month == 10
        assert result.iloc[0].day == 1
        assert result.iloc[0].hour == 0
        assert result.iloc[0].minute == 0
        assert result.iloc[0].second == 0

    def test_iso_format_with_t_separator(self):
        """Verifica parseo de formato ISO estándar con 'T' (YYYY-MM-DDTHH:MM:SS)."""
        series = pd.Series(["2025-10-01T08:15:30"])
        result = parse_datetime_series(series)

        assert result.iloc[0] == pd.Timestamp("2025-10-01 08:15:30")

    def test_slash_format_without_seconds(self):
        """Verifica parseo de formato con plecas sin segundos (DD/MM/YYYY H:MM)."""
        series = pd.Series(["01/11/2023 0:01", "15/11/2023 18:45"])
        result = parse_datetime_series(series)

        assert pd.api.types.is_datetime64_any_dtype(result)
        assert result.iloc[0] == pd.Timestamp("2023-11-01 00:01:00")
        assert result.iloc[0].year == 2023
        assert result.iloc[0].month == 11
        assert result.iloc[0].day == 1
        assert result.iloc[0].minute == 1

    def test_slash_format_with_seconds(self):
        """Verifica parseo de formato con plecas y segundos (DD/MM/YYYY HH:MM:SS)."""
        series = pd.Series(["01/11/2023 00:01:25", "15/11/2023 18:45:50"])
        result = parse_datetime_series(series)

        assert pd.api.types.is_datetime64_any_dtype(result)
        assert result.iloc[0] == pd.Timestamp("2023-11-01 00:01:25")
        assert result.iloc[0].second == 25

    def test_iso_does_not_swap_day_and_month(self):
        """Garantiza que días <= 12 en formato ISO (YYYY-MM-DD) no se confundan con mes."""
        # 2 de enero de 2020: en el bug anterior se interpretaba como 1 de febrero
        series = pd.Series(["2020-01-02 06:09:49", "2020-01-05 12:00:00"])
        result = parse_datetime_series(series)

        assert result.iloc[0].year == 2020
        assert result.iloc[0].month == 1   # Debe ser Enero
        assert result.iloc[0].day == 2     # Debe ser día 2

    def test_slash_respects_day_first(self):
        """Garantiza que formato con plecas interprete día primero (DD/MM/YYYY)."""
        # 2 de enero de 2023
        series = pd.Series(["02/01/2023 08:30"])
        result = parse_datetime_series(series)

        assert result.iloc[0].year == 2023
        assert result.iloc[0].month == 1   # Enero
        assert result.iloc[0].day == 2     # Día 2

    def test_null_values_handling(self):
        """Verifica que valores nulos o None se conviertan a pd.NaT sin fallar."""
        series = pd.Series(["2025-10-01 00:00:26", None, np.nan, "2025-10-02 10:00:00"])
        result = parse_datetime_series(series)

        assert len(result) == 4
        assert pd.notna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])
        assert pd.notna(result.iloc[3])

    def test_invalid_string_coerces_to_nat(self):
        """Verifica que strings que no representan fechas válidas se conviertan a NaT."""
        series = pd.Series(["2025-10-01 00:00:26", "fecha_invalida_123"])
        result = parse_datetime_series(series)

        assert pd.notna(result.iloc[0])
        assert pd.isna(result.iloc[1])

    def test_empty_series(self):
        """Verifica que una serie vacía no lance error."""
        series = pd.Series([], dtype="object")
        result = parse_datetime_series(series)
        assert len(result) == 0
