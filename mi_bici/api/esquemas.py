"""Define los esquemas de entrada y salida de la API.

Los nombres de los campos son los de `VARIABLES_P2`, que es la firma del modelo registrado.
"""

from pydantic import BaseModel


class ObservacionRiesgo(BaseModel):
    """Estado reciente de una estación en un bloque de horas."""

    flujo_neto_lag_1: float
    flujo_neto_lag_2: float
    salidas_lag_1: float
    llegadas_lag_1: float
    riesgo_semana_anterior: float
    hora: int
    dia_semana: int
    festivo: int
    capacidad: int


class SolicitudRiesgo(BaseModel):
    """Una o varias observaciones para predecir en una sola petición."""

    observaciones: list[ObservacionRiesgo]


class PrediccionRiesgo(BaseModel):
    """Predicción de una observación: si queda en riesgo y con qué probabilidad."""

    riesgo: int
    probabilidad: float


class RespuestaRiesgo(BaseModel):
    """Predicciones en el orden de la solicitud, con el umbral y la versión del modelo."""

    predicciones: list[PrediccionRiesgo]
    umbral: float
    modelo: str
    version: str | None


class Salud(BaseModel):
    """Estado del servicio y modelo que tiene cargado."""

    estado: str
    modelo: str
    version: str | None
    alias: str | None
