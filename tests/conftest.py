"""Datos sintéticos compartidos por las pruebas, con el mismo formato que deja la ingesta.

Son pocas estaciones y unos días alrededor de `CORTE_PRUEBA`, sin datos de personas reales, así
que las pruebas corren en el CI sin descargar MiBici. Incluyen los casos límite de las variables:
- Viajes antes y después de `CORTE_PRUEBA`, y uno que empieza antes y termina después.
- Una estación con menos de `MIN_DIAS_PERFIL` días con viajes antes del corte.
- Una estación con capacidad 0.
- Un bloque de cada lunes con suficientes viajes para que dos estaciones queden en riesgo.
"""

import pandas as pd
import pytest

from mi_bici.config import CORTE_PRUEBA, MIN_DIAS_PERFIL

ESTACION_NUEVA = 3
ESTACION_SIN_CAPACIDAD = 4
# Días antes y después del corte. Antes hay más de MIN_DIAS_PERFIL para las estaciones normales
DIAS_ANTES, DIAS_DESPUES = MIN_DIAS_PERFIL + 15, 7
HORAS_DE_VIAJE = (8, 13, 18)
DURACION_MIN = 10
# Viajes de la estación 2 a la 1 el lunes a las 7:00: 8 >= UMBRAL_RIESGO (0.4) x capacidad 20
VIAJES_PICO = 8


def _estaciones() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "estacion_id": [1, 2, ESTACION_NUEVA, ESTACION_SIN_CAPACIDAD],
            "nombre": [
                "(GDL-001) C. Uno / Av. Dos",
                "(GDL-002) C. Tres / Av. Cuatro",
                "(ZAP-001) C. Cinco / Av. Seis",
                "(TLQ-001) C. Siete / Av. Ocho",
            ],
            "latitud": [20.67, 20.68, 20.72, 20.64],
            "longitud": [-103.35, -103.36, -103.41, -103.31],
            "capacidad": [15, 20, 10, 0],
        }
    )


def _viajes() -> pd.DataFrame:
    corte = pd.Timestamp(CORTE_PRUEBA)
    dias = pd.date_range(corte - pd.Timedelta(days=DIAS_ANTES), periods=DIAS_ANTES + DIAS_DESPUES)
    # La estación nueva empieza 10 días antes del corte, menos que MIN_DIAS_PERFIL
    inicio_nueva = corte - pd.Timedelta(days=10)
    destino = {
        1: 2,
        2: ESTACION_NUEVA,
        ESTACION_NUEVA: ESTACION_SIN_CAPACIDAD,
        ESTACION_SIN_CAPACIDAD: 1,
    }

    filas = []
    for dia in dias:
        for origen, llegada in destino.items():
            if origen == ESTACION_NUEVA and dia < inicio_nueva:
                continue
            for hora in HORAS_DE_VIAJE:
                filas.append((dia + pd.Timedelta(hours=hora), origen, llegada))
        if dia.dayofweek == 0:
            filas += [(dia + pd.Timedelta(hours=7), 2, 1)] * VIAJES_PICO
    # Empieza antes del corte y termina después
    filas.append((corte - pd.Timedelta(minutes=5), 1, 2))

    inicio = pd.Series([fila[0] for fila in filas])
    return pd.DataFrame(
        {
            "genero": ["M", "F"] * (len(filas) // 2) + ["M"] * (len(filas) % 2),
            "inicio": inicio.dt.strftime("%Y-%m-%d %H:%M"),
            "fin": (inicio + pd.Timedelta(minutes=DURACION_MIN)).dt.strftime("%Y-%m-%d %H:%M"),
            "origen_id": [fila[1] for fila in filas],
            "destino_id": [fila[2] for fila in filas],
            "nacimiento": 1990,
            "duracion_min": DURACION_MIN,
        }
    )


@pytest.fixture
def estaciones() -> pd.DataFrame:
    """Estaciones con el formato de la ingesta. La 3 es nueva y la 4 tiene capacidad 0."""
    return _estaciones()


@pytest.fixture
def viajes() -> pd.DataFrame:
    """Viajes con el formato de la ingesta: `inicio` y `fin` como texto `YYYY-MM-DD HH:MM`."""
    return _viajes()


@pytest.fixture
def estacion_nueva() -> int:
    """Id de la estación con menos de `MIN_DIAS_PERFIL` días con viajes antes del corte."""
    return ESTACION_NUEVA


@pytest.fixture
def estacion_sin_capacidad() -> int:
    """Id de la estación con capacidad 0, que `etiqueta_riesgo` excluye."""
    return ESTACION_SIN_CAPACIDAD
