"""Conexión con el servidor de tracking de MLflow.

La usan el entrenamiento, el Registry y la API, para que la URI se configure en un solo lugar.
"""

import mlflow

from mi_bici.config import TRACKING_URI


def conectar_mlflow(uri: str = TRACKING_URI) -> None:
    """Apunta MLflow al servidor de tracking del proyecto.

    Args:
        uri: URI del servidor. Por defecto, `TRACKING_URI` de `config.py`, que sale de
            `MLFLOW_TRACKING_URI` o es el SQLite local.
    """
    mlflow.set_tracking_uri(uri)
