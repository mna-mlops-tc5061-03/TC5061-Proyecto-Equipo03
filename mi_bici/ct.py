"""Ciclo de entrenamiento continuo (CT) de P1: monitoreo -> reentrenar -> A/B -> promoción.

Encadena los módulos existentes siguiendo las reglas de `config.py`:
1. Corre el monitoreo de P1 (`mi_bici.monitoreo`) y evalúa si algún mes supera el umbral
   `MAE > (1 + TOLERANCIA_MAE_P1) × mae_referencia`.
2. Si el disparador se activa, reentrena `@champion` hasta el día antes de `CORTE_AB` y registra la
   nueva versión como `@challenger` (`mi_bici.modeling.reentrenar`).
3. Corre el A/B offline sobre el periodo desde `CORTE_AB` (`mi_bici.ab_testing`).
4. Solo si el A/B aprueba al challenger **y** se pasó `--promover` (`make ct PROMOVER=1`), mueve
   `@champion` a la nueva versión y deja la anterior en `@challenger`.

Cada etapa deja su propio run en MLflow y al final se registra un run `P1-ct` con la decisión, el
motivo y los identificadores de todos los runs ejecutados.

    make ct
    make ct PROMOVER=1
"""

import json
from pathlib import Path
from typing import Annotated

from loguru import logger
import mlflow
from mlflow import MlflowClient
import pandas as pd
from sklearn.metrics import mean_absolute_error
import typer

from mi_bici import ab_testing, monitoreo
from mi_bici.config import (
    ALIAS_PRODUCCION,
    ALIAS_RESPALDO,
    COLUMNA_PREDICCION,
    EQUIPO,
    EVIDENTLY_DIR,
    EXPERIMENT_NAME,
    MODELOS_REGISTRO,
    OBJETIVO_P1,
    TOLERANCIA_MAE_P1,
    TRACKING_URI,
)
from mi_bici.modeling import reentrenar
from mi_bici.modeling.base import estado_git, version_datos
from mi_bici.seguimiento import conectar_mlflow

app = typer.Typer()


def evaluar_disparador(
    por_mes: pd.DataFrame,
    mae_referencia: float,
    tolerancia: float = TOLERANCIA_MAE_P1,
) -> dict:
    """Evalúa si algún mes supera la regla `MAE > (1 + tolerancia) × mae_referencia`.

    Args:
        por_mes: Tabla mensual de monitoreo con `mes` y `mae_champion`.
        mae_referencia: MAE del champion en el periodo de referencia.
        tolerancia: Margen permitido sobre `mae_referencia` (`TOLERANCIA_MAE_P1` de `config.py`).

    Returns:
        Diccionario con `disparado`, `meses_disparados`, `mae_referencia`, `umbral_mae`,
        `mae_mes_max` y `mes_peor`.
    """
    if por_mes.empty:
        raise ValueError("La tabla mensual de monitoreo está vacía; no se puede evaluar CT.")

    umbral_mae = mae_referencia * (1 + tolerancia)
    disparados = por_mes[por_mes["mae_champion"] > umbral_mae]
    peor = por_mes.loc[por_mes["mae_champion"].idxmax()]
    return {
        "disparado": bool(not disparados.empty),
        "meses_disparados": disparados["mes"].astype(str).tolist(),
        "mae_referencia": float(mae_referencia),
        "umbral_mae": float(umbral_mae),
        "mae_mes_max": float(peor["mae_champion"]),
        "mes_peor": str(peor["mes"]),
    }


def decidir_ct(
    disparador: dict,
    resultado_ab: dict | None,
    promover: bool,
    version_inicial: str,
    version_challenger: str | None,
    tolerancia: float = TOLERANCIA_MAE_P1,
) -> tuple[str, str, bool]:
    """Determina la decisión final del ciclo CT y redacta el motivo auditable.

    Returns:
        Tupla `(decision, motivo, promovido)` donde `decision` es una de:
        `"sin_reentrenamiento"`, `"mantener"`, `"aprobado_sin_promover"` o `"promovido"`.
    """
    factor = 1 + tolerancia
    if not disparador["disparado"]:
        motivo = (
            f"Ningún mes superó el umbral de MAE ({disparador['umbral_mae']:.3f} = "
            f"{factor:.2f} × {disparador['mae_referencia']:.3f}); máximo observado: "
            f"{disparador['mae_mes_max']:.3f} en {disparador['mes_peor']}. "
            f"@{ALIAS_PRODUCCION} permanece en v{version_inicial}."
        )
        return "sin_reentrenamiento", motivo, False

    meses = ", ".join(disparador["meses_disparados"])
    if resultado_ab is None:
        raise ValueError("Si el disparador se activó, `resultado_ab` no puede ser None.")
    resumen_ab = (
        f"ΔMAE={resultado_ab['delta_mae']:+.3f}, "
        f"IC 95 %=[{resultado_ab['ic_bajo']:+.3f}, {resultado_ab['ic_alto']:+.3f}], "
        f"p={resultado_ab['p_valor']:.3g}"
    )
    if not resultado_ab["promover"]:
        motivo = (
            f"Disparador activo en {meses} (MAE máx {disparador['mae_mes_max']:.3f} > "
            f"{disparador['umbral_mae']:.3f}); se reentrenó v{version_challenger}, pero el A/B "
            f"no la aprobó ({resumen_ab}). @{ALIAS_PRODUCCION} permanece en v{version_inicial}."
        )
        return "mantener", motivo, False

    if not promover:
        motivo = (
            f"Disparador activo en {meses} y A/B favorable a v{version_challenger} ({resumen_ab}), "
            f"pero PROMOVER no se activó (--promover). @{ALIAS_PRODUCCION} permanece en "
            f"v{version_inicial}."
        )
        return "aprobado_sin_promover", motivo, False

    motivo = (
        f"Disparador activo en {meses}, A/B favorable ({resumen_ab}) y PROMOVER=1: "
        f"@{ALIAS_PRODUCCION} pasó de v{version_inicial} a v{version_challenger} y "
        f"@{ALIAS_RESPALDO} quedó en v{version_inicial}."
    )
    return "promovido", motivo, True


def _alias_opcional(cliente: MlflowClient, nombre: str, alias: str) -> str | None:
    """Devuelve la versión asociada a `alias`, o `None` si el alias todavía no existe."""
    try:
        return str(cliente.get_model_version_by_alias(nombre, alias).version)
    except mlflow.exceptions.MlflowException:
        return None


def ciclo_ct(
    cliente: MlflowClient | None = None,
    nombre: str = MODELOS_REGISTRO["P1"],
    promover: bool = False,
    evidently_dir: Path = EVIDENTLY_DIR,
) -> dict:
    """Ejecuta el ciclo CT completo de P1 y registra el run final de decisión en MLflow.

    Args:
        cliente: Cliente de MLflow. Si es `None`, conecta con `TRACKING_URI` y crea uno.
        nombre: Nombre del modelo registrado de P1.
        promover: Si es `True`, mueve `@champion` cuando el A/B favorece al challenger.
        evidently_dir: Carpeta donde se guardan las salidas de monitoreo de Evidently.

    Returns:
        Resumen del ciclo con `decision`, `motivo`, `disparado`, `promovido`, versiones inicial y
        final de los aliases y los `run_id` de cada etapa.
    """
    if cliente is None:
        conectar_mlflow(TRACKING_URI)
        cliente = MlflowClient()

    version_inicial = str(cliente.get_model_version_by_alias(nombre, ALIAS_PRODUCCION).version)

    # 1. Monitoreo (deja su run stage_of_work=monitoring en MLflow)
    referencia, actual = monitoreo.conjuntos()
    por_mes_csv = evidently_dir / monitoreo.POR_MES_CSV_FILE.name
    run_monitoreo, _, _ = monitoreo.registrar_monitoreo(
        referencia,
        actual,
        html=evidently_dir / monitoreo.HTML_FILE.name,
        csv=evidently_dir / monitoreo.CSV_FILE.name,
        por_mes_csv=por_mes_csv,
        por_mes_png=evidently_dir / monitoreo.POR_MES_PNG_FILE.name,
    )
    mae_referencia = float(
        mean_absolute_error(referencia[OBJETIVO_P1], referencia[COLUMNA_PREDICCION])
    )
    por_mes = pd.read_csv(por_mes_csv)
    disparador = evaluar_disparador(por_mes, mae_referencia)

    # 2 y 3. Reentrenamiento + A/B solo si el disparador se cumplió
    resumen_reentreno: dict | None = None
    resultado_ab: dict | None = None
    version_challenger: str | None = None
    if disparador["disparado"]:
        resumen_reentreno = reentrenar.reentrenar(cliente, nombre=nombre)
        version_challenger = resumen_reentreno["challenger"]
        _, resultado_ab = ab_testing.ab_test(cliente, nombre=nombre, mover=promover)

    decision, motivo, promovido = decidir_ct(
        disparador,
        resultado_ab,
        promover=promover,
        version_inicial=version_inicial,
        version_challenger=version_challenger,
    )
    champion_final = str(cliente.get_model_version_by_alias(nombre, ALIAS_PRODUCCION).version)
    challenger_final = _alias_opcional(cliente, nombre, ALIAS_RESPALDO)

    runs = {
        "monitoreo": run_monitoreo,
        "reentrenamiento": resumen_reentreno["run_id"] if resumen_reentreno else None,
        "ab_test": resultado_ab["run_id"] if resultado_ab else None,
    }
    resumen = {
        "modelo": nombre,
        "decision": decision,
        "motivo": motivo,
        "disparado": disparador["disparado"],
        "promover_solicitado": bool(promover),
        "promovido": promovido,
        "meses_disparados": disparador["meses_disparados"],
        "version_champion_inicial": version_inicial,
        "version_champion_final": champion_final,
        "version_challenger_final": challenger_final,
        "runs": runs,
    }

    # 4. Run final de CT con la decisión, el motivo y el linaje de todos los pasos
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="P1-ct") as run_ct:
        mlflow.set_tags(
            {
                "equipo": EQUIPO,
                "question": "p1",
                "stage_of_work": "ct",
                "decision": decision,
                "motivo": motivo,
                **estado_git(),
            }
        )
        parametros = {
            "modelo_registro": nombre,
            "tolerancia_mae_p1": TOLERANCIA_MAE_P1,
            "disparador_activo": disparador["disparado"],
            "promover_solicitado": bool(promover),
            "promovido": promovido,
            "meses_disparados": ",".join(disparador["meses_disparados"]) or "ninguno",
            "version_champion_inicial": version_inicial,
            "version_champion_final": champion_final,
            "version_challenger_final": challenger_final or "ninguna",
            "run_monitoreo": run_monitoreo,
            "run_reentrenamiento": runs["reentrenamiento"] or "no_ejecutado",
            "run_ab_test": runs["ab_test"] or "no_ejecutado",
            "data_version": version_datos(),
        }
        mlflow.log_params(parametros)

        metricas = {
            "mae_referencia": disparador["mae_referencia"],
            "umbral_mae": disparador["umbral_mae"],
            "mae_mes_max": disparador["mae_mes_max"],
        }
        if resultado_ab is not None:
            metricas.update(
                {
                    "ab_delta_mae": float(resultado_ab["delta_mae"]),
                    "ab_ic_bajo": float(resultado_ab["ic_bajo"]),
                    "ab_ic_alto": float(resultado_ab["ic_alto"]),
                    "ab_p_valor": float(resultado_ab["p_valor"]),
                }
            )
        mlflow.log_metrics(metricas)
        mlflow.log_text(json.dumps(resumen, indent=2, ensure_ascii=False), "data/decision_ct.json")

    resumen["run_id"] = run_ct.info.run_id
    return resumen


@app.command()
def main(
    promover: Annotated[
        bool,
        typer.Option(
            "--promover",
            help="Mueve @champion al challenger si el disparador y el A/B lo aprueban",
        ),
    ] = False,
) -> None:
    """Corre el ciclo CT de P1 (monitoreo -> reentrenar -> A/B) sin promover por defecto."""
    conectar_mlflow(TRACKING_URI)
    resumen = ciclo_ct(MlflowClient(), promover=promover)
    logger.success(f"CT P1 [{resumen['decision']}] (run {resumen['run_id']}): {resumen['motivo']}")
    if resumen["decision"] == "aprobado_sin_promover":
        logger.info("Para aplicar la promoción en el Registry: make ct PROMOVER=1")


if __name__ == "__main__":
    app()
