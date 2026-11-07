"""Define los esquemas de entrada y salida de la API.

Los nombres de los campos son los de `VARIABLES_P1` y `VARIABLES_P2`, que son las firmas de los
modelos registrados. Los rangos siguen la forma en que `features.py` calcula cada variable.
"""

from pydantic import BaseModel, ConfigDict, Field

from mi_bici.config import BLOQUE_HORAS, MAX_OBSERVACIONES

# Las filas del serving_input_example del champion de P1, para que /docs traiga un ejemplo que
# funciona tal cual
# TODO: reemplazar con las filas reales del serving_input_example.json de P1
EJEMPLO_P1 = [
    {
        "viajes_lag_1": 42.0,
        "viajes_lag_7": 38.0,
        "viajes_media_7": 40.3,
        "dia_semana": 2,
        "festivo": False,
        "capacidad": 15,
    },
    {
        "viajes_lag_1": 5.0,
        "viajes_lag_7": 0.0,
        "viajes_media_7": 3.1,
        "dia_semana": 6,
        "festivo": True,
        "capacidad": 12,
    },
]

# Las 3 filas del serving_input_example del champion de P2, para que /docs traiga un ejemplo que
# funciona tal cual
EJEMPLO_P2 = [
    {
        "flujo_neto_lag_1": -1.0,
        "flujo_neto_lag_2": 0.0,
        "salidas_lag_1": 1.0,
        "llegadas_lag_1": 0.0,
        "riesgo_semana_anterior": 0.0,
        "hora": 6,
        "dia_semana": 2,
        "festivo": False,
        "capacidad": 15,
    },
    {
        "flujo_neto_lag_1": 8.0,
        "flujo_neto_lag_2": -1.0,
        "salidas_lag_1": 13.0,
        "llegadas_lag_1": 21.0,
        "riesgo_semana_anterior": 0.0,
        "hora": 9,
        "dia_semana": 2,
        "festivo": False,
        "capacidad": 15,
    },
    {
        "flujo_neto_lag_1": -9.0,
        "flujo_neto_lag_2": 8.0,
        "salidas_lag_1": 17.0,
        "llegadas_lag_1": 8.0,
        "riesgo_semana_anterior": 0.0,
        "hora": 12,
        "dia_semana": 2,
        "festivo": False,
        "capacidad": 15,
    },
]


# P1: demanda diaria por estación


class ObservacionDemanda(BaseModel):
    """Historia reciente de una estación al empezar el día que se predice."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EJEMPLO_P1[0]]})

    viajes_lag_1: float = Field(ge=0, description="Viajes que salieron de la estación ayer")
    viajes_lag_7: float = Field(
        ge=0, description="Viajes que salieron el mismo día de la semana anterior"
    )
    viajes_media_7: float = Field(ge=0, description="Media de viajes diarios de los últimos 7 días")
    dia_semana: int = Field(ge=0, le=6, description="Día de la semana: 0 es lunes y 6 domingo")
    festivo: bool = Field(description="Si el día es de descanso obligatorio")
    capacidad: int = Field(gt=0, description="Anclajes de la estación")


class SolicitudDemanda(BaseModel):
    """Una o varias observaciones para predecir en una sola petición."""

    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"examples": [{"observaciones": EJEMPLO_P1}]}
    )

    observaciones: list[ObservacionDemanda] = Field(
        min_length=1,
        max_length=MAX_OBSERVACIONES,
        description=f"Entre 1 y {MAX_OBSERVACIONES} observaciones",
    )


class PrediccionDemanda(BaseModel):
    """Predicción de una observación: viajes esperados en el día."""

    viajes: float = Field(description="Viajes que se espera que salgan de la estación en el día")


class RespuestaDemanda(BaseModel):
    """Predicciones en el orden de la solicitud, con la versión del modelo."""

    predicciones: list[PrediccionDemanda]
    modelo: str = Field(description="Nombre del modelo en el Model Registry o ruta de la carpeta")
    version: str | None = Field(description="Versión del modelo en el Model Registry")


# P2: riesgo de quedarse sin bicis o sin anclajes


class ObservacionRiesgo(BaseModel):
    """Estado reciente de una estación al empezar un bloque de horas."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EJEMPLO_P2[0]]})

    flujo_neto_lag_1: float = Field(
        description="Llegadas menos salidas en el bloque anterior. Puede ser negativo"
    )
    flujo_neto_lag_2: float = Field(
        description="Llegadas menos salidas de hace dos bloques. Puede ser negativo"
    )
    salidas_lag_1: float = Field(ge=0, description="Viajes que salieron en el bloque anterior")
    llegadas_lag_1: float = Field(ge=0, description="Viajes que llegaron en el bloque anterior")
    riesgo_semana_anterior: float = Field(
        ge=0, le=1, description="1 si el mismo bloque de la semana anterior estuvo en riesgo"
    )
    hora: int = Field(
        ge=0, le=23, description=f"Hora en que empieza el bloque de {BLOQUE_HORAS} horas"
    )
    dia_semana: int = Field(ge=0, le=6, description="Día de la semana: 0 es lunes y 6 domingo")
    festivo: bool = Field(description="Si el día es de descanso obligatorio")
    capacidad: int = Field(gt=0, description="Anclajes de la estación")


class SolicitudRiesgo(BaseModel):
    """Una o varias observaciones para predecir en una sola petición."""

    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"examples": [{"observaciones": EJEMPLO_P2}]}
    )

    observaciones: list[ObservacionRiesgo] = Field(
        min_length=1,
        max_length=MAX_OBSERVACIONES,
        description=f"Entre 1 y {MAX_OBSERVACIONES} observaciones",
    )


class PrediccionRiesgo(BaseModel):
    """Predicción de una observación: si queda en riesgo y con qué probabilidad."""

    riesgo: bool = Field(description="Si la probabilidad llega al umbral del modelo")
    probabilidad: float = Field(ge=0, le=1, description="Probabilidad de riesgo")


class RespuestaRiesgo(BaseModel):
    """Predicciones en el orden de la solicitud, con el umbral y la versión del modelo."""

    predicciones: list[PrediccionRiesgo]
    umbral: float = Field(ge=0, le=1, description="Umbral de decisión que trae el modelo")
    modelo: str = Field(description="Nombre del modelo en el Model Registry o ruta de la carpeta")
    version: str | None = Field(description="Versión del modelo en el Model Registry")


# Salud del servicio


class ModeloCargado(BaseModel):
    """Modelo que sirve una pregunta."""

    modelo: str
    version: str | None
    alias: str | None


class Salud(BaseModel):
    """Estado del servicio y modelo que tiene cargado cada pregunta."""

    estado: str
    modelos: dict[str, ModeloCargado]
