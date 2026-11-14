"""Compara offline el champion y el challenger de P1 con un mes que ninguno de los dos vio.

Las dos versiones predicen las mismas filas, de `CORTE_AB` al fin de `PERIODO_ACTUAL`. Se miden
con las métricas de siempre y el baseline como referencia, y la diferencia se prueba con un
Wilcoxon pareado sobre los errores absolutos y un intervalo bootstrap que remuestrea estaciones.
La regla de decisión está fija en `config.py`. Solo con `--promover` se mueven los aliases.

python -m mi_bici.ab_testing
python -m mi_bici.ab_testing --promover
"""

import json

from loguru import logger
import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import typer

from mi_bici.config import (
    AB_DIR,
    ALFA_AB,
    ALIAS_PRODUCCION,
    ALIAS_RESPALDO,
    BASELINE_P1,
    COLUMNA_FECHA_P1,
    CORTE_AB,
    EQUIPO,
    EXPERIMENT_NAME,
    MODELOS_REGISTRO,
    N_BOOTSTRAP,
    OBJETIVO_P1,
    PERIODO_ACTUAL,
    SEMILLA,
    VARIABLES_P1,
)
from mi_bici.dataset import load_processed_stations, load_processed_travels
from mi_bici.features import demanda_diaria
from mi_bici.modeling.base import estado_git, version_datos
from mi_bici.modeling.evaluate import metricas_regresion
from mi_bici.seguimiento import conectar_mlflow

app = typer.Typer()

# Sábado y domingo en `dia_semana` (0 = lunes)
FIN_DE_SEMANA = (5, 6)


def datos_ab() -> pd.DataFrame:
    """Construye la demanda diaria del periodo del A/B, de `CORTE_AB` al fin de `PERIODO_ACTUAL`.

    Los rezagos del primer día salen de los días anteriores, que no se evalúan.

    Returns:
        Las filas que se evalúan, con las variables de P1, el objetivo y la estación.
    """
    demanda = demanda_diaria(
        load_processed_travels(hasta=PERIODO_ACTUAL[1]), load_processed_stations()
    )
    return demanda[demanda[COLUMNA_FECHA_P1] >= CORTE_AB].reset_index(drop=True)


def intervalo_bootstrap(
    diferencias: np.ndarray,
    grupos: np.ndarray,
    n: int = N_BOOTSTRAP,
    semilla: int = SEMILLA,
    nivel: float = 0.95,
) -> tuple[float, float]:
    """Calcula el intervalo de la diferencia media remuestreando grupos completos.

    Cada réplica elige estaciones con reemplazo y promedia todas sus filas, así el intervalo
    respeta que los días de una misma estación se parecen entre sí.

    Args:
        diferencias: Diferencia por fila, por ejemplo |error de A| - |error de B|.
        grupos: Grupo de cada fila (la estación).
        n: Número de réplicas.
        semilla: Semilla de las réplicas.
        nivel: Nivel de confianza.

    Returns:
        Los límites inferior y superior del intervalo.
    """
    tabla = pd.DataFrame({"d": diferencias, "g": grupos}).groupby("g")["d"].agg(["sum", "size"])
    sumas, tamanos = tabla["sum"].to_numpy(), tabla["size"].to_numpy()
    generador = np.random.default_rng(semilla)
    elegidos = generador.integers(0, len(tabla), size=(n, len(tabla)))
    replicas = sumas[elegidos].sum(axis=1) / tamanos[elegidos].sum(axis=1)
    cola = (1 - nivel) / 2
    return float(np.quantile(replicas, cola)), float(np.quantile(replicas, 1 - cola))


def comparar(datos: pd.DataFrame, pred_a, pred_b, alfa: float = ALFA_AB) -> dict:
    """Compara dos predicciones de las mismas filas y aplica la regla de decisión.

    La diferencia es MAE de A menos MAE de B, así que es positiva cuando B se equivoca menos.

    Args:
        datos: Filas evaluadas, con el objetivo, `estacion_id` y `dia_semana`.
        pred_a: Predicciones del champion.
        pred_b: Predicciones del challenger.
        alfa: Nivel de significancia del Wilcoxon.

    Returns:
        La diferencia de MAE, su intervalo de 95 %, el p-valor, la diferencia en días hábiles y
        en fin de semana, y si B se promueve.
    """
    y = datos[OBJETIVO_P1].to_numpy()
    error_a, error_b = np.abs(y - np.asarray(pred_a)), np.abs(y - np.asarray(pred_b))
    diferencias = error_a - error_b
    bajo, alto = intervalo_bootstrap(diferencias, datos["estacion_id"].to_numpy())
    # Sin ninguna diferencia (dos modelos iguales) Wilcoxon no está definido: no hay evidencia
    p_valor = float(wilcoxon(error_a, error_b).pvalue) if diferencias.any() else 1.0
    fin_de_semana = datos["dia_semana"].isin(FIN_DE_SEMANA).to_numpy()
    return {
        "delta_mae": float(diferencias.mean()),
        "ic_bajo": bajo,
        "ic_alto": alto,
        "p_valor": p_valor,
        "delta_mae_habiles": float(diferencias[~fin_de_semana].mean()),
        "delta_mae_fin_de_semana": float(diferencias[fin_de_semana].mean()),
        "promover": bool(bajo > 0 and p_valor < alfa),
    }


def tabla_metricas(datos: pd.DataFrame, predicciones: dict) -> pd.DataFrame:
    """Arma la tabla de MAE, RMSE y R² de cada versión y del baseline.

    Args:
        datos: Filas evaluadas.
        predicciones: Nombre de la fila -> predicciones.

    Returns:
        Una fila por versión con `mae`, `rmse` y `r2`.
    """
    filas = {
        nombre: metricas_regresion(datos[OBJETIVO_P1], pred, prefijo="ab")
        for nombre, pred in predicciones.items()
    }
    tabla = pd.DataFrame(filas).T
    tabla.columns = [c.removeprefix("ab_") for c in tabla.columns]
    return tabla.rename_axis("version").reset_index()


def promover(cliente: MlflowClient, nombre: str, champion: str, challenger: str) -> None:
    """Pasa el challenger a champion y deja la versión anterior como challenger.

    Args:
        cliente: Cliente de MLflow.
        nombre: Nombre del modelo registrado.
        champion: Versión que deja de estar en producción.
        challenger: Versión que entra a producción.
    """
    cliente.set_registered_model_alias(nombre, ALIAS_PRODUCCION, challenger)
    cliente.set_registered_model_alias(nombre, ALIAS_RESPALDO, champion)


def ab_test(cliente: MlflowClient, nombre: str = MODELOS_REGISTRO["P1"], mover: bool = False):
    """Corre el A/B de P1, lo registra en MLflow y guarda la tabla.

    Args:
        cliente: Cliente de MLflow.
        nombre: Nombre del modelo registrado.
        mover: Si se mueven los aliases cuando la regla favorece al challenger.

    Returns:
        La tabla de métricas y el resultado de la comparación, con las versiones y el Run ID.
    """
    versiones = {
        alias: str(cliente.get_model_version_by_alias(nombre, alias).version)
        for alias in (ALIAS_PRODUCCION, ALIAS_RESPALDO)
    }
    datos = datos_ab()
    predicciones = {
        f"{alias} (v{version})": mlflow.pyfunc.load_model(f"models:/{nombre}@{alias}").predict(
            datos[VARIABLES_P1]
        )
        for alias, version in versiones.items()
    }
    pred_a, pred_b = predicciones.values()
    predicciones[f"baseline ({BASELINE_P1})"] = datos[BASELINE_P1].to_numpy()
    tabla = tabla_metricas(datos, predicciones)
    resultado = comparar(datos, pred_a, pred_b)
    logger.info(f"A/B de {nombre} con {len(datos)} filas:\n{tabla.to_string(index=False)}")

    AB_DIR.mkdir(parents=True, exist_ok=True)
    ruta = AB_DIR / "p1_ab.csv"
    tabla.to_csv(ruta, index=False)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="P1-ab-test") as run:
        mlflow.set_tags(
            {
                "equipo": EQUIPO,
                "question": "p1",
                "stage_of_work": "ab_test",
                "decision": "promover" if resultado["promover"] else "mantener",
                **estado_git(),
            }
        )
        mlflow.log_params(
            {
                "version_a": versiones[ALIAS_PRODUCCION],
                "version_b": versiones[ALIAS_RESPALDO],
                "desde": CORTE_AB,
                "hasta": PERIODO_ACTUAL[1],
                "filas": len(datos),
                "estaciones": datos["estacion_id"].nunique(),
                "alfa": ALFA_AB,
                "n_bootstrap": N_BOOTSTRAP,
                "semilla": SEMILLA,
                "data_version": version_datos(),
                "promover_solicitado": mover,
            }
        )
        mlflow.log_metrics({k: v for k, v in resultado.items() if k != "promover"})
        for _, fila in tabla.iterrows():
            prefijo = fila["version"].split(" ")[0]
            mlflow.log_metrics({f"{prefijo}_{m}": fila[m] for m in ("mae", "rmse", "r2")})
        mlflow.log_artifact(str(ruta), artifact_path="data")
        mlflow.log_text(json.dumps(resultado, indent=2), "data/decision.json")

    if mover and resultado["promover"]:
        promover(cliente, nombre, versiones[ALIAS_PRODUCCION], versiones[ALIAS_RESPALDO])
    return tabla, {**resultado, **versiones, "run_id": run.info.run_id}


@app.command()
def main(
    promover_challenger: bool = typer.Option(
        False, "--promover", help="Mueve champion al challenger si la regla lo favorece"
    ),
) -> None:
    """Corre el A/B de P1 y, solo con `--promover`, aplica la decisión en el Registry."""
    conectar_mlflow()
    _, resultado = ab_test(MlflowClient(), mover=promover_challenger)
    decision = "promover el challenger" if resultado["promover"] else "mantener el champion"
    logger.success(
        f"ΔMAE (A - B) = {resultado['delta_mae']:+.3f} viajes, IC 95 % "
        f"[{resultado['ic_bajo']:+.3f}, {resultado['ic_alto']:+.3f}], p = "
        f"{resultado['p_valor']:.2g}: la regla dice {decision}"
    )
    if resultado["promover"] and not promover_challenger:
        logger.info("Los aliases no se movieron. Para aplicarlo: make ab-test ARGS=--promover")
    elif resultado["promover"]:
        logger.success(f"@{ALIAS_PRODUCCION} ahora es la v{resultado[ALIAS_RESPALDO]}")


if __name__ == "__main__":
    app()
