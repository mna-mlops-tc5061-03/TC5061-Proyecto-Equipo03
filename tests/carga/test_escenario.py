"""Pruebas del escenario de carga y del `LoadTestShape` sin levantar Locust contra la API."""

import json
import subprocess
import sys

import numpy as np
import pytest

from mi_bici.api.esquemas import SolicitudDemanda, SolicitudRiesgo
from mi_bici.carga.escenario import observaciones_demanda, observaciones_riesgo, usuarios_en
from mi_bici.config import ARRANQUE_CARGA, ESCALONES_CARGA, LOTE_CARGA, PROJ_ROOT

COLUMNAS_PERSONALES = {"genero", "nacimiento", "usuario_id"}


class TestEscenarioSolicitudes:
    def test_solicitudes_demanda_pasan_esquema_pydantic(self):
        """Verifica que tanto 1 estación como el lote completo validan con `SolicitudDemanda`."""
        rng = np.random.default_rng(42)

        puntual = SolicitudDemanda(observaciones=observaciones_demanda(rng, 1))
        lote = SolicitudDemanda(observaciones=observaciones_demanda(rng, LOTE_CARGA))

        assert len(puntual.observaciones) == 1
        assert len(lote.observaciones) == LOTE_CARGA

    def test_solicitudes_riesgo_pasan_esquema_pydantic_y_flujo_coherente(self):
        """Verifica `SolicitudRiesgo`, tamaño de lote y coherencia `flujo_neto = llegadas - salidas`."""
        rng = np.random.default_rng(42)

        puntual = SolicitudRiesgo(observaciones=observaciones_riesgo(rng, 1))
        lote = SolicitudRiesgo(observaciones=observaciones_riesgo(rng, LOTE_CARGA))

        assert len(puntual.observaciones) == 1
        assert len(lote.observaciones) == LOTE_CARGA
        for obs in lote.observaciones:
            assert obs.flujo_neto_lag_1 == pytest.approx(obs.llegadas_lag_1 - obs.salidas_lag_1)

    def test_observaciones_no_incluyen_columnas_personales(self):
        """Verifica que las solicitudes sintéticas no traen datos personales."""
        rng = np.random.default_rng(7)
        for obs in observaciones_demanda(rng, 10) + observaciones_riesgo(rng, 10):
            assert COLUMNAS_PERSONALES.isdisjoint(obs.keys())


class TestEscalonesYLocustfile:
    def test_usuarios_en_recorre_todos_los_escalones_y_termina_en_none(self):
        """Verifica las fronteras de cada escalón de `ESCALONES_CARGA` y el fin de la prueba."""
        inicio = 0
        for usuarios, duracion in ESCALONES_CARGA:
            assert usuarios_en(inicio) == usuarios
            assert usuarios_en(inicio + duracion / 2) == usuarios
            assert usuarios_en(inicio + duracion - 0.01) == usuarios
            inicio += duracion

        assert usuarios_en(inicio) is None
        assert usuarios_en(inicio + 30) is None

    def test_locustfile_load_test_shape_y_lotes_en_proceso_aislado(self):
        """Prueba `Escalones` y `OperadorMiBici` en un subproceso para no parchar pytest con gevent."""
        # Importar locustfile parcha socket vía gevent; aislarlo evita afectar el resto de la suite
        codigo = """
import json
from contextlib import contextmanager
from mi_bici.api.esquemas import SolicitudDemanda, SolicitudRiesgo
from mi_bici.carga.locustfile import Escalones, OperadorMiBici
from mi_bici.config import ARRANQUE_CARGA, ESCALONES_CARGA, LOTE_CARGA

forma = Escalones()
tiempos = []
t = 0
for usuarios, duracion in ESCALONES_CARGA:
    forma.get_run_time = lambda s=t: s
    tiempos.append((t, forma.tick()))
    forma.get_run_time = lambda s=t + duracion - 1: s
    tiempos.append((t + duracion - 1, forma.tick()))
    t += duracion
forma.get_run_time = lambda s=t: s
fin = forma.tick()

# Capturamos las peticiones que arman las tareas de OperadorMiBici sin red real
enviados = []

class _Resp:
    def __init__(self, n):
        self.status_code = 200
        self._n = n
        self.fallo = None
    def json(self):
        return {"predicciones": [0] * self._n}
    def failure(self, msg):
        self.fallo = msg

class _Cliente:
    @contextmanager
    def post(self, ruta, json=None, name=None, catch_response=False):
        enviados.append((ruta, name, len(json["observaciones"])))
        if ruta == "/predict/demanda":
            SolicitudDemanda(**json)
        elif ruta == "/predict/riesgo":
            SolicitudRiesgo(**json)
        yield _Resp(len(json["observaciones"]))
    def get(self, ruta):
        enviados.append((ruta, ruta, 0))

op = OperadorMiBici.__new__(OperadorMiBici)
op.client = _Cliente()
op.on_start()
op.demanda_estacion()
op.riesgo_estacion()
op.demanda_lote()
op.riesgo_lote()
op.salud()

print(json.dumps({"tiempos": tiempos, "fin": fin, "enviados": enviados}))
"""
        resultado = subprocess.run(
            [sys.executable, "-c", codigo],
            cwd=PROJ_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        salida = json.loads(resultado.stdout.strip().splitlines()[-1])

        esperados = []
        t = 0
        for usuarios, duracion in ESCALONES_CARGA:
            esperados.append([t, [usuarios, ARRANQUE_CARGA]])
            esperados.append([t + duracion - 1, [usuarios, ARRANQUE_CARGA]])
            t += duracion

        assert salida["tiempos"] == esperados
        assert salida["fin"] is None
        assert salida["enviados"] == [
            ["/predict/demanda", "/predict/demanda [1]", 1],
            ["/predict/riesgo", "/predict/riesgo [1]", 1],
            ["/predict/demanda", f"/predict/demanda [{LOTE_CARGA}]", LOTE_CARGA],
            ["/predict/riesgo", f"/predict/riesgo [{LOTE_CARGA}]", LOTE_CARGA],
            ["/health", "/health", 0],
        ]
