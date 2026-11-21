"""Solicitudes y escalones del escenario de carga, sin depender de Locust.

Las solicitudes parten de `EJEMPLO_P1` y `EJEMPLO_P2` de la API y varían cada campo dentro de
lo que acepta, para que las peticiones no sean siempre las mismas. No traen datos personales.
Viven aparte del `locustfile` porque importar Locust parcha la red de Python (gevent), y así las
pruebas pueden revisarlas sin ese efecto.
"""

import numpy as np

from mi_bici.api.esquemas import EJEMPLO_P1, EJEMPLO_P2
from mi_bici.config import BLOQUE_HORAS, ESCALONES_CARGA

# Factor que escala los viajes de los ejemplos: de una estación con poca demanda a una con el doble
FACTOR_VIAJES = (0.2, 2.0)
# Anclajes de las estaciones de la nomenclatura: de 11 a 39
CAPACIDAD_ESTACIONES = (11, 39)
# Probabilidad de que el día sea festivo y de que el bloque de la semana anterior estuviera en riesgo
PROB_FESTIVO = 0.03
PROB_RIESGO_ANTERIOR = 0.1


def _escalar(
    ejemplos: list[dict], campo: str, indices: np.ndarray, factor: np.ndarray, decimales: int = 0
) -> np.ndarray:
    """Toma `campo` de los ejemplos elegidos por `indices`, lo escala por `factor` y lo redondea.

    Los viajes son conteos, así que por defecto quedan enteros. Sumar 0.0 cambia -0.0 por 0.0.
    """
    columna = np.array([ejemplo[campo] for ejemplo in ejemplos], dtype=float)[indices]
    return np.round(columna * factor, decimales) + 0.0


def _filas(columnas: dict[str, np.ndarray]) -> list[dict]:
    """Convierte columnas de numpy en una lista de observaciones con tipos de Python para JSON."""
    nombres = list(columnas)
    valores = zip(*(columna.tolist() for columna in columnas.values()))
    return [dict(zip(nombres, fila)) for fila in valores]


def _calendario(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Día de la semana, festivo y capacidad, comunes a las dos preguntas."""
    return {
        "dia_semana": rng.integers(0, 7, size=n),
        "festivo": rng.random(n) < PROB_FESTIVO,
        "capacidad": rng.integers(CAPACIDAD_ESTACIONES[0], CAPACIDAD_ESTACIONES[1] + 1, size=n),
    }


def observaciones_demanda(rng: np.random.Generator, n: int) -> list[dict]:
    """Arma `n` observaciones válidas para `/predict/demanda` (P1).

    Args:
        rng: Generador de números aleatorios.
        n: Número de observaciones (estaciones).

    Returns:
        Las observaciones, con los viajes de un ejemplo de la API escalados.
    """
    indices = rng.integers(len(EJEMPLO_P1), size=n)
    factor = rng.uniform(*FACTOR_VIAJES, size=n)
    columnas = {
        "viajes_lag_1": _escalar(EJEMPLO_P1, "viajes_lag_1", indices, factor),
        "viajes_lag_7": _escalar(EJEMPLO_P1, "viajes_lag_7", indices, factor),
        "viajes_media_7": _escalar(EJEMPLO_P1, "viajes_media_7", indices, factor, decimales=2),
    }
    return _filas(columnas | _calendario(rng, n))


def observaciones_riesgo(rng: np.random.Generator, n: int) -> list[dict]:
    """Arma `n` observaciones válidas para `/predict/riesgo` (P2).

    El flujo neto del bloque anterior se calcula como llegadas menos salidas, igual que en
    `features.py`, para que las observaciones sean coherentes.

    Args:
        rng: Generador de números aleatorios.
        n: Número de observaciones (estaciones).

    Returns:
        Las observaciones, con los viajes de un ejemplo de la API escalados.
    """
    indices = rng.integers(len(EJEMPLO_P2), size=n)
    factor = rng.uniform(*FACTOR_VIAJES, size=n)
    salidas = _escalar(EJEMPLO_P2, "salidas_lag_1", indices, factor)
    llegadas = _escalar(EJEMPLO_P2, "llegadas_lag_1", indices, factor)
    columnas = {
        "flujo_neto_lag_1": llegadas - salidas + 0.0,
        "flujo_neto_lag_2": _escalar(EJEMPLO_P2, "flujo_neto_lag_2", indices, factor),
        "salidas_lag_1": salidas,
        "llegadas_lag_1": llegadas,
        "riesgo_semana_anterior": (rng.random(n) < PROB_RIESGO_ANTERIOR).astype(float),
        "hora": rng.choice(np.arange(0, 24, BLOQUE_HORAS), size=n),
    }
    return _filas(columnas | _calendario(rng, n))


def usuarios_en(segundo: float, escalones: list[tuple[int, int]] = ESCALONES_CARGA) -> int | None:
    """Usuarios que debe tener la prueba en el `segundo` transcurrido.

    Args:
        segundo: Segundos desde que empezó la prueba.
        escalones: Pares de (usuarios, segundos que dura el escalón), en orden.

    Returns:
        Los usuarios del escalón en curso, o `None` cuando ya pasó el último y la prueba termina.
    """
    fin = 0
    for usuarios, duracion in escalones:
        fin += duracion
        if segundo < fin:
            return usuarios
    return None
