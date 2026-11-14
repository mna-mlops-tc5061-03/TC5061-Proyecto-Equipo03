"""Monitoreo de P1 con Evidently: drift por variable y calidad de los datos.

Compara el periodo de referencia (entrenamiento del champion de P1) con el periodo actual
(producción), incluyendo la predicción del champion y el objetivo `viajes`. Guarda el HTML y la
tabla de resultados en `reports/evidently/` y los registra en un run de MLflow.

python -m mi_bici.monitoreo
"""

from pathlib import Path

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from loguru import logger
import mlflow
import pandas as pd
import typer

from mi_bici.config import (
    ALIAS_PRODUCCION,
    CATEGORICAS_P1,
    COLUMNA_FECHA_P1,
    COLUMNA_PREDICCION,
    EQUIPO,
    EVIDENTLY_DIR,
    EXPERIMENT_NAME,
    MODELOS_REGISTRO,
    NUMERICAS_P1,
    PERIODO_ACTUAL,
    PERIODO_REFERENCIA,
    TRACKING_URI,
    VARIABLES_P1,
)
from mi_bici.dataset import load_processed_stations, load_processed_travels
from mi_bici.features import demanda_diaria
from mi_bici.modeling.base import estado_git, version_datos
from mi_bici.seguimiento import conectar_mlflow

app = typer.Typer()

HTML_FILE = EVIDENTLY_DIR / "p1_drift.html"
CSV_FILE = EVIDENTLY_DIR / "p1_drift.csv"
COLUMNAS_P1 = [*NUMERICAS_P1, *CATEGORICAS_P1]


def definicion_datos() -> DataDefinition:
    """Define las columnas numéricas y categóricas de P1 para Evidently."""
    return DataDefinition(
        numerical_columns=list(NUMERICAS_P1),
        categorical_columns=list(CATEGORICAS_P1),
    )


def conjuntos(modelo=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Arma la demanda diaria de P1 con la predicción del champion y separa referencia y actual.

    Calcula `demanda_diaria` una sola vez hasta `PERIODO_ACTUAL[1]` antes de separar, así los
    rezagos de los primeros días de agosto salen de finales de julio.

    Args:
        modelo: Modelo con método `predict`. Si es `None`, carga el champion de P1 desde MLflow.

    Returns:
        `(referencia, actual)` filtrados por `PERIODO_REFERENCIA` y `PERIODO_ACTUAL`.
    """
    viajes = load_processed_travels(hasta=PERIODO_ACTUAL[1])
    estaciones = load_processed_stations()
    demanda = demanda_diaria(viajes, estaciones)

    if modelo is None:
        conectar_mlflow(TRACKING_URI)
        uri = f"models:/{MODELOS_REGISTRO['P1']}@{ALIAS_PRODUCCION}"
        modelo = mlflow.pyfunc.load_model(uri)

    demanda[COLUMNA_PREDICCION] = modelo.predict(demanda[VARIABLES_P1])
    fechas = pd.to_datetime(demanda[COLUMNA_FECHA_P1])
    referencia = demanda[fechas.between(*PERIODO_REFERENCIA)].reset_index(drop=True)
    actual = demanda[fechas.between(*PERIODO_ACTUAL)].reset_index(drop=True)
    return referencia, actual


def _extraer_drift(snapshot) -> tuple[pd.DataFrame, float]:
    """Extrae de un `Snapshot` de Evidently la tabla de drift por variable y `drift_share`."""
    detectado_por_columna = {}
    for resultado, _ in getattr(snapshot.context, "_legacy_metrics", {}).values():
        por_columna = getattr(resultado, "drift_by_columns", None)
        if isinstance(por_columna, dict):
            for col, info in por_columna.items():
                detectado_por_columna[col] = bool(info.drift_detected)

    drift_share = 0.0
    filas = []
    for metrica in snapshot.dict().get("metrics", []):
        cfg = metrica.get("config", {})
        tipo = cfg.get("type", "")
        if tipo.endswith("DriftedColumnsCount"):
            drift_share = float(metrica["value"]["share"])
        elif tipo.endswith("ValueDrift"):
            variable = cfg["column"]
            metodo = str(cfg.get("method", ""))
            valor = float(metrica["value"])
            umbral = float(cfg["threshold"])
            if variable in detectado_por_columna:
                hay_drift = detectado_por_columna[variable]
            else:
                hay_drift = bool(valor < umbral if "p_value" in metodo else valor >= umbral)
            filas.append(
                {
                    "variable": variable,
                    "metodo": metodo,
                    "valor": valor,
                    "umbral": umbral,
                    "drift": hay_drift,
                }
            )
    return pd.DataFrame(filas), drift_share


def correr_reporte(
    actual: pd.DataFrame,
    referencia: pd.DataFrame,
    html: Path | None = HTML_FILE,
    csv: Path | None = CSV_FILE,
) -> tuple[pd.DataFrame, float]:
    """Corre `DataDriftPreset` y `DataSummaryPreset` de Evidently y guarda el HTML y el CSV.

    Args:
        actual: Datos del periodo actual (`PERIODO_ACTUAL`). Va primero en `report.run()`.
        referencia: Datos del periodo de referencia (`PERIODO_REFERENCIA`).
        html: Ruta donde guardar el HTML, o `None` para no guardarlo.
        csv: Ruta donde guardar la tabla CSV, o `None` para no guardarla.

    Returns:
        La tabla con `variable`, `metodo`, `valor`, `umbral` y `drift`, y la proporción de
        columnas con drift (`drift_share`).
    """
    definicion = definicion_datos()
    ds_actual = Dataset.from_pandas(actual[COLUMNAS_P1], data_definition=definicion)
    ds_referencia = Dataset.from_pandas(referencia[COLUMNAS_P1], data_definition=definicion)

    report = Report([DataDriftPreset(), DataSummaryPreset()])
    snapshot = report.run(ds_actual, ds_referencia)

    tabla, drift_share = _extraer_drift(snapshot)
    if html is not None:
        html.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html))
    if csv is not None:
        csv.parent.mkdir(parents=True, exist_ok=True)
        tabla.to_csv(csv, index=False)
    return tabla, drift_share


def registrar_monitoreo(
    referencia: pd.DataFrame,
    actual: pd.DataFrame,
    html: Path = HTML_FILE,
    csv: Path = CSV_FILE,
) -> tuple[str, pd.DataFrame, dict]:
    """Genera el reporte de Evidently y registra el run de monitoreo en MLflow.

    Args:
        referencia: Conjunto del periodo de referencia.
        actual: Conjunto del periodo actual.
        html: Ruta donde se guarda el reporte HTML.
        csv: Ruta donde se guarda la tabla CSV de drift.

    Returns:
        `(run_id, tabla, metricas)` del run registrado en MLflow.
    """
    tabla, drift_share = correr_reporte(actual, referencia, html=html, csv=csv)
    metricas = {
        "drift_share": drift_share,
        **{f"drift_{fila.variable}": float(fila.valor) for fila in tabla.itertuples()},
    }
    parametros = {
        "referencia_inicio": PERIODO_REFERENCIA[0],
        "referencia_fin": PERIODO_REFERENCIA[1],
        "actual_inicio": PERIODO_ACTUAL[0],
        "actual_fin": PERIODO_ACTUAL[1],
        "periodo_referencia": f"{PERIODO_REFERENCIA[0]}..{PERIODO_REFERENCIA[1]}",
        "periodo_actual": f"{PERIODO_ACTUAL[0]}..{PERIODO_ACTUAL[1]}",
        "filas_referencia": len(referencia),
        "filas_actual": len(actual),
        "data_version": version_datos(),
    }
    etiquetas = {
        "equipo": EQUIPO,
        "question": "p1",
        "stage_of_work": "monitoring",
        **estado_git(),
    }

    conectar_mlflow(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="P1-monitoreo") as run:
        mlflow.set_tags(etiquetas)
        mlflow.log_params(parametros)
        mlflow.log_metrics(metricas)
        mlflow.log_artifact(str(html), artifact_path="evidently")
        mlflow.log_artifact(str(csv), artifact_path="evidently")
    return run.info.run_id, tabla, metricas


@app.command()
def main() -> None:
    """Genera el reporte de drift y calidad de datos de P1 y lo registra en MLflow."""
    referencia, actual = conjuntos()
    logger.info(
        f"Monitoreo P1: referencia {PERIODO_REFERENCIA[0]}..{PERIODO_REFERENCIA[1]} "
        f"({len(referencia):,} filas) contra actual {PERIODO_ACTUAL[0]}..{PERIODO_ACTUAL[1]} "
        f"({len(actual):,} filas)"
    )
    run_id, tabla, metricas = registrar_monitoreo(referencia, actual)
    con_drift = int(tabla["drift"].sum())
    logger.success(
        f"Reporte guardado en {HTML_FILE} y {CSV_FILE} (run {run_id}: "
        f"{con_drift}/{len(tabla)} columnas con drift, drift_share={metricas['drift_share']:.3f})"
    )


if __name__ == "__main__":
    app()
