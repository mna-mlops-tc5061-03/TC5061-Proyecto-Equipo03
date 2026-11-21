"""Modelos y solicitudes de prueba para levantar la API en el CI sin datos ni modelos reales.

Se corre dentro de la imagen de la API, que ya trae MLflow, scikit-learn y `mi_bici`. Siempre deja
en la carpeta destino `solicitud.json` y `solicitud_p1.json` con los ejemplos de la API, y además:

- sin opciones, `modelo/` (P2) y `modelo_p1/` (P1), que la API carga como carpetas con
  `MODELO_URI` y `MODELO_URI_P1` (job `image`):

      docker run --rm -v /tmp/prueba:/tmp/prueba -v "$PWD/tests/ci:/ci:ro" mibici-api \\
          python /ci/modelos_prueba.py /tmp/prueba

- con `--registrar`, los registra en el MLflow de `MLFLOW_TRACKING_URI` con los nombres de
  `MODELOS_REGISTRO` y el alias de producción, para que la API los cargue del Registry como en
  producción (job `compose`, dentro de la red del compose):

      docker compose run --rm --no-deps -v "$PWD/tests/ci:/ci:ro" -v /tmp/prueba:/tmp/prueba \\
          api python /ci/modelos_prueba.py /tmp/prueba --registrar
"""

import argparse
import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.models import ModelSignature, infer_signature
import mlflow.sklearn
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import FixedThresholdClassifier

from mi_bici.api.esquemas import EJEMPLO_P1, EJEMPLO_P2
from mi_bici.config import ALIAS_PRODUCCION, MODELOS_REGISTRO, VARIABLES_P1, VARIABLES_P2

# Carpeta de cada modelo dentro del destino y nombre de su solicitud de ejemplo
CARPETAS = {"P2": "modelo", "P1": "modelo_p1"}
SOLICITUDES = {"P2": ("solicitud.json", EJEMPLO_P2), "P1": ("solicitud_p1.json", EJEMPLO_P1)}


def modelos_de_prueba() -> dict[str, tuple[BaseEstimator, ModelSignature]]:
    """Entrena un modelo chico por pregunta con los tipos de la firma real.

    Returns:
        Para `"P1"` y `"P2"`, el estimador y su firma.
    """
    # P2. Mismos tipos que la firma real de flujo_por_bloque (int32 en hora/dia_semana, int64 en festivo)
    x_p2 = pd.DataFrame(EJEMPLO_P2)[VARIABLES_P2].astype(
        {"hora": "int32", "dia_semana": "int32", "festivo": "int64"}
    )
    clasificador = LogisticRegression().fit(x_p2, [0, 0, 1])
    modelo_p2 = FixedThresholdClassifier(
        FrozenEstimator(clasificador), threshold=0.5, response_method="predict_proba"
    )

    # P1. Un regresor chico con las variables de VARIABLES_P1
    x_p1 = pd.DataFrame(EJEMPLO_P1)[VARIABLES_P1].astype(
        {"dia_semana": "int32", "festivo": "int64"}
    )
    modelo_p1 = LinearRegression().fit(x_p1, x_p1["viajes_media_7"])

    return {
        "P2": (modelo_p2, infer_signature(x_p2, modelo_p2.predict(x_p2))),
        "P1": (modelo_p1, infer_signature(x_p1, modelo_p1.predict(x_p1))),
    }


def guardar_modelos(destino: Path, modelos: dict) -> None:
    """Guarda cada modelo en su carpeta, para cargarlo con `MODELO_URI` sin un Registry.

    Args:
        destino: Carpeta donde quedan `modelo/` y `modelo_p1/`.
        modelos: Resultado de `modelos_de_prueba()`.
    """
    for pregunta, (modelo, firma) in modelos.items():
        mlflow.sklearn.save_model(modelo, destino / CARPETAS[pregunta], signature=firma)


def registrar_modelos(modelos: dict) -> None:
    """Registra cada modelo en el MLflow de `MLFLOW_TRACKING_URI` y le pone el alias de producción.

    Los artefactos suben por el proxy del servidor, así que esto también prueba que la red, el
    encabezado `Host` y `--artifacts-destination` del compose funcionan.

    Args:
        modelos: Resultado de `modelos_de_prueba()`.
    """
    cliente = MlflowClient()
    with mlflow.start_run(run_name="modelos-de-prueba-ci"):
        for pregunta, (modelo, firma) in modelos.items():
            nombre = MODELOS_REGISTRO[pregunta]
            info = mlflow.sklearn.log_model(
                modelo, name=CARPETAS[pregunta], signature=firma, registered_model_name=nombre
            )
            cliente.set_registered_model_alias(
                nombre, ALIAS_PRODUCCION, info.registered_model_version
            )
            print(f"{nombre} v{info.registered_model_version} @{ALIAS_PRODUCCION}")


def main() -> None:
    """Genera las solicitudes y guarda o registra los modelos según los argumentos."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("destino", type=Path, help="Carpeta de las solicitudes y los modelos")
    parser.add_argument(
        "--registrar",
        action="store_true",
        help="Registra los modelos en MLFLOW_TRACKING_URI en lugar de guardarlos en carpetas",
    )
    args = parser.parse_args()

    modelos = modelos_de_prueba()
    if args.registrar:
        registrar_modelos(modelos)
    else:
        guardar_modelos(args.destino, modelos)
    for archivo, ejemplo in SOLICITUDES.values():
        (args.destino / archivo).write_text(json.dumps({"observaciones": ejemplo}))


if __name__ == "__main__":
    main()
