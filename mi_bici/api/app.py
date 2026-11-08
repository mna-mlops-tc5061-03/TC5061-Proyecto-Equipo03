"""Sirve los modelos de P1 (demanda diaria) y P2 (riesgo por bloque) como API REST.

Se levanta con `uvicorn mi_bici.api.app:app`. Cada pregunta es una entrada de `PREGUNTAS`: el
modelo registrado, las variables de su firma, los esquemas de entrada y salida, y cómo se arma la
respuesta. Agregar una pregunta es agregar una entrada, no copiar un endpoint. Los modelos se cargan
al arrancar desde `MODELOS_URI`, que por defecto son los champions del Model Registry.
"""

from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Request
from loguru import logger
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
import pandas as pd
from pydantic import BaseModel

from mi_bici.api.esquemas import (
    ModeloCargado,
    PrediccionDemanda,
    PrediccionRiesgo,
    RespuestaDemanda,
    RespuestaRiesgo,
    Salud,
    SolicitudDemanda,
    SolicitudRiesgo,
)
from mi_bici.config import MODELOS_URI, VARIABLES_P1, VARIABLES_P2
from mi_bici.seguimiento import conectar_mlflow


def cargar_modelo(uri: str) -> dict:
    """Carga el modelo y lee los tipos de su firma y la versión que sirve.

    Si `uri` usa un alias (`models:/<nombre>@<alias>`), primero lo resuelve a una versión y carga
    esa versión, para que el alias no cambie entre la consulta y la carga. `uri` también puede ser
    una carpeta con un modelo de MLflow, por ejemplo si se empaqueta en la imagen.

    Args:
        uri: URI del modelo en el Model Registry o ruta a la carpeta del modelo.

    Returns:
        El modelo de pyfunc, el estimador de scikit-learn, los tipos de cada columna de la firma,
        y el nombre, la versión y el alias del modelo.
    """
    conectar_mlflow()
    nombre, version, alias = uri, None, None
    if uri.startswith("models:/"):
        referencia = uri.removeprefix("models:/")
        if "@" in referencia:
            nombre, alias = referencia.split("@", 1)
            version = MlflowClient().get_model_version_by_alias(nombre, alias).version
            uri = f"models:/{nombre}/{version}"
        else:
            nombre, version = referencia.rsplit("/", 1)
    modelo = mlflow.pyfunc.load_model(uri)
    # MLflow rechaza un int64 donde la firma pide int32, así que cada columna se convierte al tipo
    # de la firma antes de predecir
    tipos = {c.name: c.type.to_pandas() for c in modelo.metadata.get_input_schema().inputs}
    return {
        "modelo": modelo,
        "estimador": modelo.get_raw_model(),
        "tipos": tipos,
        "nombre": nombre,
        "version": version,
        "alias": alias,
    }


def responder_riesgo(servicio: dict, datos: pd.DataFrame) -> RespuestaRiesgo:
    """P2: riesgo y probabilidad de cada observación, con el umbral que trae el modelo."""
    riesgo = servicio["modelo"].predict(datos)
    probabilidad = servicio["estimador"].predict_proba(datos)[:, 1]
    return RespuestaRiesgo(
        predicciones=[
            PrediccionRiesgo(riesgo=bool(r), probabilidad=float(p))
            for r, p in zip(riesgo, probabilidad)
        ],
        umbral=float(servicio["estimador"].threshold),
        modelo=servicio["nombre"],
        version=servicio["version"],
    )


def responder_demanda(servicio: dict, datos: pd.DataFrame) -> RespuestaDemanda:
    """P1: viajes esperados en el día para cada observación."""
    viajes = servicio["modelo"].predict(datos)
    return RespuestaDemanda(
        predicciones=[PrediccionDemanda(viajes=float(v)) for v in viajes],
        modelo=servicio["nombre"],
        version=servicio["version"],
    )


@dataclass(frozen=True)
class Pregunta:
    """Lo que necesita el servicio para exponer un modelo en `POST /predict/<nombre>`."""

    uri: str
    variables: list[str]
    solicitud: type[BaseModel]
    respuesta: type[BaseModel]
    responder: Callable[[dict, pd.DataFrame], BaseModel]
    descripcion: str


PREGUNTAS = {
    "demanda": Pregunta(
        uri=MODELOS_URI["demanda"],
        variables=VARIABLES_P1,
        solicitud=SolicitudDemanda,
        respuesta=RespuestaDemanda,
        responder=responder_demanda,
        descripcion="P1: predice los viajes que saldrán de cada estación en el día.",
    ),
    "riesgo": Pregunta(
        uri=MODELOS_URI["riesgo"],
        variables=VARIABLES_P2,
        solicitud=SolicitudRiesgo,
        respuesta=RespuestaRiesgo,
        responder=responder_riesgo,
        descripcion="P2: predice el riesgo de cada observación con el umbral del modelo.",
    ),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga cada modelo al arrancar. Si uno falla, el servicio sigue arriba y esa ruta da 503."""
    app.state.servicios = {}
    for nombre, pregunta in PREGUNTAS.items():
        try:
            servicio = cargar_modelo(pregunta.uri)
            logger.success(f"Modelo cargado: {servicio['nombre']} versión {servicio['version']}")
        except (MlflowException, OSError) as error:  # sin servidor, sin modelo o sin la carpeta
            servicio = None
            logger.error(f"No se pudo cargar {pregunta.uri}: {error}")
        app.state.servicios[nombre] = servicio
    yield


app = FastAPI(
    title="MiBici: demanda y riesgo por estación",
    description=(
        "Predice los viajes del día por estación (P1) y si una estación se quedará sin bicis o "
        "sin anclajes en el próximo bloque (P2)."
    ),
    lifespan=lifespan,
)


def servicio_cargado(request: Request, nombre: str) -> dict:
    """Devuelve el modelo cargado de una pregunta o responde 503 si no se pudo cargar."""
    servicio = getattr(request.app.state, "servicios", {}).get(nombre)
    if servicio is None:
        raise HTTPException(
            status_code=503, detail=f"El modelo {PREGUNTAS[nombre].uri} no está cargado"
        )
    return servicio


@app.get("/health", response_model=Salud)
def salud(request: Request) -> Salud:
    """Indica el nombre y la versión de cada modelo. Responde 503 si alguno no está cargado."""
    servicios = getattr(request.app.state, "servicios", {})
    faltantes = [PREGUNTAS[n].uri for n in PREGUNTAS if servicios.get(n) is None]
    if faltantes:
        raise HTTPException(status_code=503, detail=f"Modelos sin cargar: {', '.join(faltantes)}")
    return Salud(
        estado="ok",
        modelos={
            n: ModeloCargado(modelo=s["nombre"], version=s["version"], alias=s["alias"])
            for n, s in servicios.items()
        },
    )


def agregar_ruta(nombre: str, pregunta: Pregunta) -> None:
    """Crea `POST /predict/<nombre>` con el mismo mecanismo de carga, tipos y respuesta."""

    def predecir(solicitud: pregunta.solicitud, request: Request):
        servicio = servicio_cargado(request, nombre)
        datos = pd.DataFrame([o.model_dump() for o in solicitud.observaciones])
        datos = datos[pregunta.variables].astype(servicio["tipos"])
        return pregunta.responder(servicio, datos)

    app.post(
        f"/predict/{nombre}",
        response_model=pregunta.respuesta,
        name=f"predecir_{nombre}",
        description=pregunta.descripcion,
    )(predecir)


for _nombre, _pregunta in PREGUNTAS.items():
    agregar_ruta(_nombre, _pregunta)
