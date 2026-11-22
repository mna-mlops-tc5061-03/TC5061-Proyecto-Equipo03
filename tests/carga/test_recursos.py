"""Pruebas del parseo de CPU y memoria de contenedores (`mi_bici.carga.recursos`)."""

import pytest

from mi_bici.carga.recursos import a_mib, leer_stats, servicio


class TestConversionMemoria:
    @pytest.mark.parametrize(
        ("texto", "esperado_mib"),
        [
            ("150MiB / 7.6GiB", 150.0),
            ("1.2GiB / 8GiB", 1.2 * 1024),
            ("900KiB / 1GiB", 900.0 / 1024),
            ("512B / 1GiB", 512.0 / (1024**2)),
            # Unidades decimales que reporta Podman (MB / GB)
            ("150MB / 8GB", 150.0 * (1000**2) / (1024**2)),
        ],
    )
    def test_a_mib_convierte_unidades_docker_y_podman(self, texto, esperado_mib):
        """Verifica la conversión a MiB de MiB, GiB, KiB, B y unidades decimales MB."""
        assert a_mib(texto) == pytest.approx(esperado_mib, rel=1e-6)

    @pytest.mark.parametrize("texto_invalido", ["-- / --", "150TiB / 8GiB", "sin_numero"])
    def test_a_mib_rechaza_formatos_no_reconocidos(self, texto_invalido):
        """Verifica que valores inválidos o unidades desconocidas lancen `ValueError`."""
        with pytest.raises(ValueError, match="Memoria no reconocida"):
            a_mib(texto_invalido)


class TestLecturaStats:
    def test_servicio_reconoce_solo_contenedores_del_proyecto(self):
        """Verifica que `servicio` extrae el nombre de Compose y descarta otros proyectos."""
        assert servicio("mibici-api-1", proyecto="mibici") == "api"
        assert servicio("mibici_mlflow_1", proyecto="mibici") == "mlflow"
        assert servicio("otro-api-1", proyecto="mibici") is None
        assert servicio("api", proyecto="mibici") is None

    def test_leer_stats_convierte_cpu_y_memoria_y_filtra_proyecto(self):
        """Verifica el parseo completo de `docker stats` con varias unidades y filas inválidas."""
        salida = (
            "mibici-api-1\t12.5%\t150MiB / 7.6GiB\n"
            "mibici-mlflow-1\t3.25%\t1.2GiB / 8GiB\n"
            "mibici-worker-1\t0.50%\t900KiB / 1GiB\n"
            "otro-postgres-1\t40.0%\t500MiB / 8GiB\n"
            "mibici-arrancando-1\t--\t-- / --"
        )

        medidas = leer_stats(salida, proyecto="mibici")

        assert set(medidas.keys()) == {"mibici-api-1", "mibici-mlflow-1", "mibici-worker-1"}
        assert medidas["mibici-api-1"] == {
            "contenedor": "api",
            "cpu_pct": 12.5,
            "mem_mib": 150.0,
        }
        assert medidas["mibici-mlflow-1"] == {
            "contenedor": "mlflow",
            "cpu_pct": 3.25,
            "mem_mib": round(1.2 * 1024, 1),
        }
        assert medidas["mibici-worker-1"] == {
            "contenedor": "worker",
            "cpu_pct": 0.5,
            "mem_mib": round(900.0 / 1024, 1),
        }
