"""Escenario de Locust: cómo usaría la operación de MiBici la API.

Cada usuario simula a alguien de la operación que consulta una estación a la vez (lo más común),
corre los lotes de todas las estaciones (P1 una vez al día y P2 en cada bloque de 3 horas) y de
vez en cuando revisa `/health`. Los usuarios suben por los escalones de `ESCALONES_CARGA`, y
como cada uno hace en promedio una petición por segundo, el RPS deja de crecer con los usuarios
justo donde la API se satura.

    make carga    # con la API arriba (make sistema o make api)
"""

from itertools import count

from locust import HttpUser, LoadTestShape, between, task
import numpy as np

from mi_bici.carga.escenario import observaciones_demanda, observaciones_riesgo, usuarios_en
from mi_bici.config import ARRANQUE_CARGA, ESPERA_CARGA, HOST_CARGA, LOTE_CARGA, SEMILLA


class OperadorMiBici(HttpUser):
    """Usuario de la operación de MiBici que consulta la API."""

    host = HOST_CARGA
    wait_time = between(*ESPERA_CARGA)
    _numero = count()

    def on_start(self) -> None:
        """Le da a cada usuario su propio generador, con la semilla del proyecto y su número."""
        self.rng = np.random.default_rng([SEMILLA, next(self._numero)])

    def _predecir(self, pregunta: str, generar, n: int) -> None:
        """Pide `n` predicciones a `/predict/<pregunta>` y revisa que lleguen todas.

        La petición cuenta como falla si la respuesta no es 200, no trae `predicciones` o trae
        un número distinto de `n`. En las estadísticas, el nombre separa una estación del lote.
        """
        solicitud = {"observaciones": generar(self.rng, n)}
        with self.client.post(
            f"/predict/{pregunta}",
            json=solicitud,
            name=f"/predict/{pregunta} [{n}]",
            catch_response=True,
        ) as respuesta:
            if respuesta.status_code != 200:
                respuesta.failure(f"HTTP {respuesta.status_code}")
                return
            try:
                predicciones = respuesta.json()["predicciones"]
            except (ValueError, KeyError, TypeError):
                respuesta.failure("La respuesta no trae predicciones")
                return
            if len(predicciones) != n:
                respuesta.failure(f"{len(predicciones)} predicciones para {n} observaciones")

    @task(3)
    def demanda_estacion(self) -> None:
        """Consulta puntual: cuántos viajes saldrán mañana de una estación (P1)."""
        self._predecir("demanda", observaciones_demanda, 1)

    @task(3)
    def riesgo_estacion(self) -> None:
        """Consulta puntual: si una estación queda en riesgo en el siguiente bloque (P2)."""
        self._predecir("riesgo", observaciones_riesgo, 1)

    @task
    def demanda_lote(self) -> None:
        """Corrida diaria de P1 para todas las estaciones."""
        self._predecir("demanda", observaciones_demanda, LOTE_CARGA)

    @task
    def riesgo_lote(self) -> None:
        """Corrida de P2 de un bloque de 3 horas para todas las estaciones."""
        self._predecir("riesgo", observaciones_riesgo, LOTE_CARGA)

    @task
    def salud(self) -> None:
        """Revisión de que la API está arriba y con los modelos cargados."""
        self.client.get("/health")


class Escalones(LoadTestShape):
    """Sube los usuarios por los escalones de `ESCALONES_CARGA` y termina al acabar el último."""

    def tick(self) -> tuple[int, float] | None:
        """Devuelve los usuarios del escalón en curso y a qué ritmo arrancan, o `None` al final."""
        usuarios = usuarios_en(self.get_run_time())
        return None if usuarios is None else (usuarios, ARRANQUE_CARGA)
