"""Sirve el modelo de P2 (riesgo de quedarse sin bicis o sin anclajes) como API REST.

Se levanta con `uvicorn mi_bici.api.app:app`. El modelo se carga al arrancar desde `MODELO_URI`,
que por defecto es el champion de P2 en el Model Registry de `MLFLOW_TRACKING_URI`.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from loguru import logger
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
import pandas as pd

from mi_bici.api.esquemas import PrediccionRiesgo, RespuestaRiesgo, Salud, SolicitudRiesgo
from mi_bici.config import MODELO_URI, TRACKING_URI, VARIABLES_P2


def cargar_modelo(uri: str = MODELO_URI) -> dict:
    """Carga el modelo y lee los tipos de su firma y la versión que sirve.

    Si `uri` usa un alias (`models:/<nombre>@<alias>`), primero lo resuelve a una versión y carga
    esa versión, para que el alias no cambie entre la consulta y la carga. `uri` también puede ser
    una carpeta con un modelo de MLflow, por ejemplo si se empaqueta en la imagen.

    Args:
        uri: URI del modelo en el Model Registry o ruta a la carpeta del modelo.

    Returns:
        El modelo de pyfunc, el clasificador de scikit-learn, los tipos de cada columna de la firma,
        y el nombre, la versión y el alias del modelo.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
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
        "clasificador": modelo.get_raw_model(),
        "tipos": tipos,
        "nombre": nombre,
        "version": version,
        "alias": alias,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al arrancar. Si falla, el servicio sigue arriba y responde 503."""
    try:
        app.state.servicio = cargar_modelo()
        servicio = app.state.servicio
        logger.success(f"Modelo cargado: {servicio['nombre']} versión {servicio['version']}")
    except (MlflowException, OSError) as error:  # sin servidor, sin modelo o sin la carpeta
        app.state.servicio = None
        logger.error(f"No se pudo cargar {MODELO_URI}: {error}")
    yield


app = FastAPI(
    title="MiBici: riesgo por estación",
    description="Predice si una estación se quedará sin bicis o sin anclajes en el próximo bloque.",
    lifespan=lifespan,
)


def servicio_cargado(request: Request) -> dict:
    """Devuelve el modelo cargado o responde 503 si no se pudo cargar al arrancar."""
    servicio = getattr(request.app.state, "servicio", None)
    if servicio is None:
        raise HTTPException(status_code=503, detail=f"El modelo {MODELO_URI} no está cargado")
    return servicio


@app.get("/health", response_model=Salud)
def salud(request: Request) -> Salud:
    """Indica si el servicio tiene el modelo cargado y qué versión sirve."""
    servicio = servicio_cargado(request)
    return Salud(
        estado="ok",
        modelo=servicio["nombre"],
        version=servicio["version"],
        alias=servicio["alias"],
    )


@app.post("/predict/riesgo", response_model=RespuestaRiesgo)
def predecir_riesgo(solicitud: SolicitudRiesgo, request: Request) -> RespuestaRiesgo:
    """Predice el riesgo de cada observación con el umbral que trae el modelo registrado."""
    servicio = servicio_cargado(request)
    datos = pd.DataFrame([o.model_dump() for o in solicitud.observaciones])[VARIABLES_P2]
    datos = datos.astype(servicio["tipos"])
    riesgo = servicio["modelo"].predict(datos)
    probabilidad = servicio["clasificador"].predict_proba(datos)[:, 1]
    return RespuestaRiesgo(
        predicciones=[
            PrediccionRiesgo(riesgo=int(r), probabilidad=float(p))
            for r, p in zip(riesgo, probabilidad)
        ],
        umbral=float(servicio["clasificador"].threshold),
        modelo=servicio["nombre"],
        version=servicio["version"],
    )
