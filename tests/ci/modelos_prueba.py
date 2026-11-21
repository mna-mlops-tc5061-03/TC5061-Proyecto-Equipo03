"""Modelos y solicitudes de prueba para levantar la API en el CI sin un Model Registry.

Se corre dentro de la imagen de la API, que ya trae MLflow, scikit-learn y `mi_bici`:

    docker run --rm -v /tmp/prueba:/tmp/prueba -v "$PWD/tests/ci:/ci:ro" mibici-api \
        python /ci/modelos_prueba.py /tmp/prueba

Deja en la carpeta destino `modelo/` (P2) y `modelo_p1/` (P1), que la API carga con `MODELO_URI` y
`MODELO_URI_P1`, y `solicitud.json` y `solicitud_p1.json` con los ejemplos de la API.
"""

import json
from pathlib import Path
import sys

from mlflow.models import infer_signature
import mlflow.sklearn
import pandas as pd
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import FixedThresholdClassifier

from mi_bici.api.esquemas import EJEMPLO_P1, EJEMPLO_P2
from mi_bici.config import VARIABLES_P1, VARIABLES_P2


def guardar_modelos(destino: Path) -> None:
    """Guarda un modelo chico por pregunta con su firma y una solicitud de ejemplo para cada uno.

    Args:
        destino: Carpeta donde quedan los modelos y las solicitudes.
    """
    # P2. Mismos tipos que la firma real de flujo_por_bloque (int32 en hora/dia_semana, int64 en festivo)
    x_p2 = pd.DataFrame(EJEMPLO_P2)[VARIABLES_P2].astype(
        {"hora": "int32", "dia_semana": "int32", "festivo": "int64"}
    )
    clasificador = LogisticRegression().fit(x_p2, [0, 0, 1])
    modelo_p2 = FixedThresholdClassifier(
        FrozenEstimator(clasificador), threshold=0.5, response_method="predict_proba"
    )
    mlflow.sklearn.save_model(
        modelo_p2, destino / "modelo", signature=infer_signature(x_p2, modelo_p2.predict(x_p2))
    )
    (destino / "solicitud.json").write_text(json.dumps({"observaciones": EJEMPLO_P2}))

    # P1. Un regresor chico con las variables de VARIABLES_P1
    x_p1 = pd.DataFrame(EJEMPLO_P1)[VARIABLES_P1].astype(
        {"dia_semana": "int32", "festivo": "int64"}
    )
    regresor = LinearRegression().fit(x_p1, x_p1["viajes_media_7"])
    mlflow.sklearn.save_model(
        regresor, destino / "modelo_p1", signature=infer_signature(x_p1, regresor.predict(x_p1))
    )
    (destino / "solicitud_p1.json").write_text(json.dumps({"observaciones": EJEMPLO_P1}))


if __name__ == "__main__":
    guardar_modelos(Path(sys.argv[1]))
