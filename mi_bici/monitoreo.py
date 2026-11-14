"""Monitoreo de P1 con Evidently: drift por variable, calidad de datos y evolución mensual.

Compara el periodo de referencia (entrenamiento del champion de P1) con el periodo actual
(producción), incluyendo la predicción del champion y el objetivo `viajes`. Además evalúa mes a mes
(`2026-08`, `2026-09` y `2026-10`) cómo evolucionan el `drift_share` y el MAE del champion frente
al baseline para sustentar el disparador de reentrenamiento. Guarda el HTML, las tablas y la
gráfica en `reports/evidently/` y los registra en un run de MLflow.

python -m mi_bici.monitoreo
"""

from pathlib import Path

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from loguru import logger
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error
import typer

from mi_bici import plots
from mi_bici.config import (
    ALIAS_PRODUCCION,
    BASELINE_P1,
    CATEGORICAS_P1,
    COLUMNA_FECHA_P1,
    COLUMNA_PREDICCION,
    EQUIPO,
    EVIDENTLY_DIR,
    EXPERIMENT_NAME,
    MODELOS_REGISTRO,
    NUMERICAS_P1,
    OBJETIVO_P1,
    PERIODO_ACTUAL,
    PERIODO_REFERENCIA,
    TRACKING_URI,
    UMBRAL_DRIFT_SHARE,
    VARIABLES_P1,
)
from mi_bici.dataset import load_processed_stations, load_processed_travels
from mi_bici.features import demanda_diaria
from mi_bici.modeling.base import estado_git, version_datos
from mi_bici.seguimiento import conectar_mlflow

app = typer.Typer()

HTML_FILE = EVIDENTLY_DIR / "p1_drift.html"
CSV_FILE = EVIDENTLY_DIR / "p1_drift.csv"
POR_MES_CSV_FILE = EVIDENTLY_DIR / "p1_por_mes.csv"
POR_MES_PNG_FILE = EVIDENTLY_DIR / "p1_por_mes.png"
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

    presets = [DataDriftPreset(), DataSummaryPreset()] if html is not None else [DataDriftPreset()]
    report = Report(presets)
    snapshot = report.run(ds_actual, ds_referencia)

    tabla, drift_share = _extraer_drift(snapshot)
    if html is not None:
        html.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html))
    if csv is not None:
        csv.parent.mkdir(parents=True, exist_ok=True)
        tabla.to_csv(csv, index=False)
    return tabla, drift_share


def evaluar_por_mes(
    actual: pd.DataFrame,
    referencia: pd.DataFrame,
    csv: Path | None = POR_MES_CSV_FILE,
    png: Path | None = POR_MES_PNG_FILE,
) -> pd.DataFrame:
    """Calcula `drift_share` contra la referencia y el MAE del champion y del baseline por mes.

    Nota sobre artefactos vs cambios reales (Chip Huyen, cap. 8):
    - `festivo` no es un cambio real: agosto y octubre no tienen días de descanso obligatorio en
      `FESTIVOS`, por lo que su distancia de Jensen-Shannon sube (~0.083) solo por el calendario.
    - `capacidad` tampoco es un cambio de comportamiento: refleja la actualización de la
      nomenclatura vigente de estaciones (#132) y las estaciones activas en el mes (~0.084–0.091).
    - El drift real ocurre en las 5 columnas de demanda (`viajes_lag_1`, `viajes_lag_7`,
      `viajes_media_7`, `viajes` y `prediccion`), que en octubre triplican el umbral de 0.1 y
      elevan el MAE del champion de 4.67 (agosto) y 6.42 (septiembre) a 7.48 (octubre).

    Args:
        actual: Conjunto del periodo actual con `fecha`, variables, `viajes` y `prediccion`.
        referencia: Conjunto del periodo de referencia.
        csv: Ruta donde guardar la tabla mensual CSV, o `None` para no guardarla.
        png: Ruta donde guardar la gráfica de las dos series, o `None` para no guardarla.

    Returns:
        DataFrame con una fila por mes (`mes`, `filas`, `drift_share`, `columnas_con_drift`,
        `mae_champion`, `mae_baseline` y `reentrenar`).
    """
    meses = pd.to_datetime(actual[COLUMNA_FECHA_P1]).dt.strftime("%Y-%m")
    filas = []
    for mes, datos_mes in actual.groupby(meses, sort=True):
        tabla_mes, drift_share = correr_reporte(datos_mes, referencia, html=None, csv=None)
        mae_champion = float(
            mean_absolute_error(datos_mes[OBJETIVO_P1], datos_mes[COLUMNA_PREDICCION])
        )
        mae_baseline = float(mean_absolute_error(datos_mes[OBJETIVO_P1], datos_mes[BASELINE_P1]))
        filas.append(
            {
                "mes": str(mes),
                "filas": len(datos_mes),
                "drift_share": drift_share,
                "columnas_con_drift": int(tabla_mes["drift"].sum()),
                "mae_champion": mae_champion,
                "mae_baseline": mae_baseline,
                "reentrenar": bool(drift_share >= UMBRAL_DRIFT_SHARE),
            }
        )
    por_mes = pd.DataFrame(filas)
    if csv is not None:
        csv.parent.mkdir(parents=True, exist_ok=True)
        por_mes.to_csv(csv, index=False)
    if png is not None:
        png.parent.mkdir(parents=True, exist_ok=True)
        plots.drift_y_error_por_mes(por_mes, UMBRAL_DRIFT_SHARE, png)
    return por_mes


def registrar_monitoreo(
    referencia: pd.DataFrame,
    actual: pd.DataFrame,
    html: Path = HTML_FILE,
    csv: Path = CSV_FILE,
    por_mes_csv: Path = POR_MES_CSV_FILE,
    por_mes_png: Path = POR_MES_PNG_FILE,
) -> tuple[str, pd.DataFrame, dict]:
    """Genera el reporte de Evidently, la evolución mensual y registra el run en MLflow.

    Args:
        referencia: Conjunto del periodo de referencia.
        actual: Conjunto del periodo actual.
        html: Ruta donde se guarda el reporte HTML.
        csv: Ruta donde se guarda la tabla CSV de drift.
        por_mes_csv: Ruta donde se guarda la tabla CSV mensual.
        por_mes_png: Ruta donde se guarda la gráfica mensual de drift y MAE.

    Returns:
        `(run_id, tabla, metricas)` del run registrado en MLflow.
    """
    tabla, drift_share = correr_reporte(actual, referencia, html=html, csv=csv)
    por_mes = evaluar_por_mes(actual, referencia, csv=por_mes_csv, png=por_mes_png)
    metricas = {
        "drift_share": drift_share,
        **{f"drift_{fila.variable}": float(fila.valor) for fila in tabla.itertuples()},
    }
    for fila in por_mes.itertuples():
        sufijo = fila.mes.replace("-", "_")
        metricas[f"drift_share_{sufijo}"] = float(fila.drift_share)
        metricas[f"mae_champion_{sufijo}"] = float(fila.mae_champion)
        metricas[f"mae_baseline_{sufijo}"] = float(fila.mae_baseline)

    parametros = {
        "referencia_inicio": PERIODO_REFERENCIA[0],
        "referencia_fin": PERIODO_REFERENCIA[1],
        "actual_inicio": PERIODO_ACTUAL[0],
        "actual_fin": PERIODO_ACTUAL[1],
        "periodo_referencia": f"{PERIODO_REFERENCIA[0]}..{PERIODO_REFERENCIA[1]}",
        "periodo_actual": f"{PERIODO_ACTUAL[0]}..{PERIODO_ACTUAL[1]}",
        "filas_referencia": len(referencia),
        "filas_actual": len(actual),
        "umbral_drift_share": UMBRAL_DRIFT_SHARE,
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
        for archivo in (html, csv, por_mes_csv, por_mes_png):
            if archivo is not None and archivo.exists():
                mlflow.log_artifact(str(archivo), artifact_path="evidently")
    return run.info.run_id, tabla, metricas


@app.command()
def main() -> None:
    """Genera el reporte de drift, la evolución por mes de P1 y los registra en MLflow."""
    referencia, actual = conjuntos()
    logger.info(
        f"Monitoreo P1: referencia {PERIODO_REFERENCIA[0]}..{PERIODO_REFERENCIA[1]} "
        f"({len(referencia):,} filas) contra actual {PERIODO_ACTUAL[0]}..{PERIODO_ACTUAL[1]} "
        f"({len(actual):,} filas)"
    )
    run_id, tabla, metricas = registrar_monitoreo(referencia, actual)
    con_drift = int(tabla["drift"].sum())
    logger.success(
        f"Reporte guardado en {HTML_FILE}, {CSV_FILE}, {POR_MES_CSV_FILE} y {POR_MES_PNG_FILE} "
        f"(run {run_id}: {con_drift}/{len(tabla)} columnas con drift, "
        f"drift_share={metricas['drift_share']:.3f})"
    )


if __name__ == "__main__":
    app()
