"""Análisis de desempeño de la prueba de carga y costo por inferencia (FinOps).

Lee las salidas de `make carga` (`reports/carga/locust_stats_history.csv` y
`reports/carga/recursos.csv`), resume cada escalón de `ESCALONES_CARGA`, calcula el RPS sostenible
que cumple `LATENCIA_OBJETIVO_MS` y `TASA_ERROR_MAX`, estima el costo por inferencia (y por 1,000)
para peticiones de una estación y de lote, evalúa la escalabilidad frente a la demanda real de
MiBici y genera las gráficas en `reports/figures/semana10/`.

    make finops
"""

import math
from pathlib import Path
from typing import Annotated

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mi_bici import plots
from mi_bici.carga.escenario import usuarios_en
from mi_bici.config import (
    BLOQUE_HORAS,
    CARGA_DIR,
    DIAS_MES,
    ESCALONES_CARGA,
    FIGURAS_SEMANA10_DIR,
    HORAS_MES,
    INSTANCIA_REFERENCIA,
    LATENCIA_OBJETIVO_MS,
    LOTE_CARGA,
    PRECIO_HORA_USD,
    PROJ_ROOT,
    TASA_ERROR_MAX,
    VENTANA_PICO_SEG,
)

HISTORIA_CSV = CARGA_DIR / "locust_stats_history.csv"
RECURSOS_CSV = CARGA_DIR / "recursos.csv"
FINOPS_CSV = CARGA_DIR / "finops.csv"
ESCALONES_CSV = CARGA_DIR / "escalones.csv"

FIG_LATENCIA = FIGURAS_SEMANA10_DIR / "latencia_carga.png"
FIG_RPS_ERROR = FIGURAS_SEMANA10_DIR / "rps_y_error_carga.png"
FIG_RECURSOS = FIGURAS_SEMANA10_DIR / "recursos_carga.png"

SEGUNDOS_POR_HORA = 3600

app = typer.Typer()


def _a_numerico(serie: pd.Series) -> pd.Series:
    """Convierte columnas de Locust (que pueden traer `'N/A'`) a flotante."""
    return pd.to_numeric(serie, errors="coerce")


def _serie_por_modo(historia: pd.DataFrame, modo: str) -> pd.DataFrame:
    """Filtra y agrega `locust_stats_history.csv` por segundo para un modo de petición.

    Modos soportados:
    - `"agregado"`: fila `Aggregated` de Locust (todas las rutas combinadas).
    - `"una_estacion"`: endpoints terminados en `"[1]"` (1 predicción por petición).
    - `"lote"`: endpoints terminados en `f"[{LOTE_CARGA}]"` (`LOTE_CARGA` predicciones por petición).
    """
    if modo == "agregado":
        sub = historia[historia["Name"] == "Aggregated"].copy()
        if sub.empty:
            return pd.DataFrame()
        df = pd.DataFrame(
            {
                "Timestamp": _a_numerico(sub["Timestamp"]),
                "rps": _a_numerico(sub["Requests/s"]).fillna(0.0),
                "fallos_s": _a_numerico(sub["Failures/s"]).fillna(0.0),
                "p50_ms": _a_numerico(sub["50%"]),
                "p95_ms": _a_numerico(sub["95%"]),
                "total_req": _a_numerico(sub["Total Request Count"]).fillna(0.0),
                "total_fail": _a_numerico(sub["Total Failure Count"]).fillna(0.0),
            }
        )
        return df.sort_values("Timestamp").reset_index(drop=True)

    sufijo = "[1]" if modo == "una_estacion" else f"[{LOTE_CARGA}]"
    sub = historia[historia["Name"].astype(str).str.endswith(sufijo)].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["Timestamp"] = _a_numerico(sub["Timestamp"])
    sub["rps"] = _a_numerico(sub["Requests/s"]).fillna(0.0)
    sub["fallos_s"] = _a_numerico(sub["Failures/s"]).fillna(0.0)
    sub["p50_ms"] = _a_numerico(sub["50%"])
    sub["p95_ms"] = _a_numerico(sub["95%"])
    sub["total_req"] = _a_numerico(sub["Total Request Count"]).fillna(0.0)
    sub["total_fail"] = _a_numerico(sub["Total Failure Count"]).fillna(0.0)

    # Por marca de tiempo: los RPS y conteos se suman entre P1 y P2; en p95 se toma el máximo
    # para exigir que las dos preguntas cumplan la latencia objetivo
    agrupado = (
        sub.groupby("Timestamp", as_index=False)
        .agg(
            rps=("rps", "sum"),
            fallos_s=("fallos_s", "sum"),
            p50_ms=("p50_ms", "median"),
            p95_ms=("p95_ms", "max"),
            total_req=("total_req", "sum"),
            total_fail=("total_fail", "sum"),
        )
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )
    return agrupado


def serie_temporal_carga(historia: pd.DataFrame) -> pd.DataFrame:
    """Construye la serie agregada por segundo transcurrido para las gráficas de Locust."""
    serie = _serie_por_modo(historia, "agregado")
    if serie.empty:
        raise ValueError("El archivo de historial de Locust no contiene filas 'Aggregated'.")
    t0 = float(serie["Timestamp"].min())
    serie["segundo"] = serie["Timestamp"] - t0
    serie["tasa_error_pct"] = np.where(
        serie["rps"] > 0, (serie["fallos_s"] / serie["rps"]) * 100.0, 0.0
    )
    return serie


def preparar_recursos(recursos: pd.DataFrame, t0: float) -> pd.DataFrame:
    """Alinea las muestras de `recursos.csv` al inicio `t0` de la prueba de Locust."""
    if recursos.empty:
        return pd.DataFrame(columns=["epoch", "segundo", "contenedor", "cpu_pct", "mem_mib"])
    df = recursos.copy()
    df["epoch"] = _a_numerico(df["epoch"])
    df["cpu_pct"] = _a_numerico(df["cpu_pct"])
    df["mem_mib"] = _a_numerico(df["mem_mib"])
    df["segundo"] = df["epoch"] - t0
    return df.dropna(subset=["epoch"]).sort_values("epoch").reset_index(drop=True)


def resumen_por_escalon(
    historia: pd.DataFrame,
    recursos: pd.DataFrame,
    escalones: list[tuple[int, int]] = ESCALONES_CARGA,
    latencia_objetivo_ms: float = LATENCIA_OBJETIVO_MS,
    tasa_error_max: float = TASA_ERROR_MAX,
    modo: str = "agregado",
) -> pd.DataFrame:
    """Resume desempeño y recursos de la API para cada escalón de la prueba de carga.

    Args:
        historia: DataFrame leído de `locust_stats_history.csv`.
        recursos: DataFrame leído de `recursos.csv` (puede estar vacío).
        escalones: Lista de pares `(usuarios, duracion_seg)`.
        latencia_objetivo_ms: Umbral objetivo de p95 en milisegundos.
        tasa_error_max: Proporción máxima de errores permitida (p. ej. 0.01 = 1 %).
        modo: `"agregado"`, `"una_estacion"` o `"lote"`.

    Returns:
        Tabla con una fila por escalón (`usuarios`, `rps`, `p50_ms`, `p95_ms`, `tasa_error`,
        `cumple_slo`, `cpu_media_pct`, `cpu_max_pct`, `mem_max_mib`).
    """
    serie = _serie_por_modo(historia, modo)
    if serie.empty:
        return pd.DataFrame()

    t0 = float(_a_numerico(historia["Timestamp"]).min())
    serie["segundo"] = serie["Timestamp"] - t0
    serie["usuarios"] = serie["segundo"].apply(lambda s: usuarios_en(s, escalones))

    rec_api = preparar_recursos(recursos, t0)
    if not rec_api.empty:
        rec_api = rec_api[rec_api["contenedor"] == "api"].copy()
        rec_api["usuarios"] = rec_api["segundo"].apply(lambda s: usuarios_en(s, escalones))

    filas = []
    req_prev = 0.0
    fail_prev = 0.0
    for usuarios, _ in escalones:
        bloque = serie[serie["usuarios"] == usuarios]
        if bloque.empty:
            continue

        rps = float(bloque["rps"].mean())
        p50_ms = float(bloque["p50_ms"].median()) if bloque["p50_ms"].notna().any() else math.nan
        p95_ms = float(bloque["p95_ms"].median()) if bloque["p95_ms"].notna().any() else math.nan

        req_fin = float(bloque["total_req"].max())
        fail_fin = float(bloque["total_fail"].max())
        delta_req = req_fin - req_prev
        delta_fail = max(0.0, fail_fin - fail_prev)
        if delta_req > 0:
            tasa_error = delta_fail / delta_req
        else:
            suma_rps = float(bloque["rps"].sum())
            tasa_error = float(bloque["fallos_s"].sum()) / suma_rps if suma_rps > 0 else 0.0
        req_prev = max(req_prev, req_fin)
        fail_prev = max(fail_prev, fail_fin)

        cumple = (
            rps > 0
            and not math.isnan(p95_ms)
            and p95_ms <= latencia_objetivo_ms
            and tasa_error <= tasa_error_max
        )

        rec_bloque = rec_api[rec_api["usuarios"] == usuarios] if not rec_api.empty else rec_api
        cpu_media = float(rec_bloque["cpu_pct"].mean()) if not rec_bloque.empty else math.nan
        cpu_max = float(rec_bloque["cpu_pct"].max()) if not rec_bloque.empty else math.nan
        mem_max = float(rec_bloque["mem_mib"].max()) if not rec_bloque.empty else math.nan

        filas.append(
            {
                "usuarios": int(usuarios),
                "rps": round(rps, 3),
                "p50_ms": round(p50_ms, 2) if not math.isnan(p50_ms) else math.nan,
                "p95_ms": round(p95_ms, 2) if not math.isnan(p95_ms) else math.nan,
                "tasa_error": round(tasa_error, 6),
                "cumple_slo": bool(cumple),
                "cpu_media_pct": round(cpu_media, 2) if not math.isnan(cpu_media) else math.nan,
                "cpu_max_pct": round(cpu_max, 2) if not math.isnan(cpu_max) else math.nan,
                "mem_max_mib": round(mem_max, 1) if not math.isnan(mem_max) else math.nan,
            }
        )
    return pd.DataFrame(filas)


def escalon_sostenible(
    tabla_escalones: pd.DataFrame,
    latencia_objetivo_ms: float = LATENCIA_OBJETIVO_MS,
    tasa_error_max: float = TASA_ERROR_MAX,
    etiqueta: str = "agregado",
) -> pd.Series:
    """Devuelve la fila del escalón con mayor RPS que cumple latencia p95 y tasa de error.

    Raises:
        ValueError: Si ningún escalón cumple los objetivos de servicio.
    """
    if tabla_escalones.empty:
        raise ValueError(f"No hay datos de escalones para '{etiqueta}'.")
    validos = tabla_escalones[tabla_escalones["cumple_slo"] & (tabla_escalones["rps"] > 0)]
    if validos.empty:
        p95_min = tabla_escalones["p95_ms"].min(skipna=True)
        err_min = tabla_escalones["tasa_error"].min(skipna=True) * 100
        raise ValueError(
            f"Ningún escalón de '{etiqueta}' cumplió el objetivo de servicio "
            f"(p95 <= {latencia_objetivo_ms:.0f} ms y error <= {tasa_error_max * 100:.1f} %). "
            f"Mejor observado: p95={p95_min:.1f} ms, error={err_min:.2f} %. "
            "No se puede calcular el costo por inferencia sin dividir entre cero."
        )
    indice_mejor = validos["rps"].idxmax()
    return validos.loc[indice_mejor]


def costo_por_inferencia(
    rps_sostenible: float,
    precio_hora_usd: float = PRECIO_HORA_USD,
    predicciones_por_peticion: int = 1,
) -> tuple[float, float]:
    """Calcula el costo por inferencia y por 1,000 inferencias en USD.

    Fórmula: `PRECIO_HORA_USD / (rps_sostenible * predicciones_por_peticion * 3600)`.

    Args:
        rps_sostenible: Peticiones por segundo sostenibles de la instancia.
        precio_hora_usd: Costo por hora de la instancia de referencia en USD.
        predicciones_por_peticion: 1 para consultas puntuales, `LOTE_CARGA` (470) para lotes.

    Returns:
        Tupla `(costo_por_inferencia_usd, costo_por_1000_usd)`.
    """
    if rps_sostenible <= 0:
        raise ValueError("El RPS sostenible debe ser mayor que cero.")
    if predicciones_por_peticion <= 0:
        raise ValueError("`predicciones_por_peticion` debe ser mayor que cero.")
    predicciones_por_hora = rps_sostenible * predicciones_por_peticion * SEGUNDOS_POR_HORA
    costo_unitario = precio_hora_usd / predicciones_por_hora
    return costo_unitario, costo_unitario * 1000.0


def escalabilidad_mibici(
    rps_sostenible: float,
    predicciones_por_peticion: int = 1,
    precio_hora_usd: float = PRECIO_HORA_USD,
    estaciones_lote: int = LOTE_CARGA,
    bloque_horas: int = BLOQUE_HORAS,
    dias_mes: int = DIAS_MES,
    ventana_pico_seg: int = VENTANA_PICO_SEG,
) -> dict[str, float | int]:
    """Compara la demanda real mensual de MiBici contra la capacidad de una instancia.

    Demanda real diaria:
    - P1: `estaciones_lote` (470) predicciones al día.
    - P2: `estaciones_lote` (470) predicciones por cada bloque de `bloque_horas` (3 h → 8 bloques/día).
    Pico simultáneo: arranque de un bloque donde coinciden P1 y P2 (`2 * estaciones_lote` = 940
    predicciones) atendido dentro de `ventana_pico_seg` (60 s por defecto).
    """
    if rps_sostenible <= 0:
        raise ValueError("El RPS sostenible debe ser mayor que cero.")

    bloques_por_dia = 24 // bloque_horas
    pred_dia_p1 = estaciones_lote
    pred_dia_p2 = estaciones_lote * bloques_por_dia
    demanda_mensual_pred = (pred_dia_p1 + pred_dia_p2) * dias_mes

    pred_por_segundo = rps_sostenible * predicciones_por_peticion
    horas_mes = dias_mes * 24
    capacidad_mensual_pred = pred_por_segundo * SEGUNDOS_POR_HORA * horas_mes
    utilizacion_pct = (demanda_mensual_pred / capacidad_mensual_pred) * 100.0

    costo_unitario, _ = costo_por_inferencia(
        rps_sostenible, precio_hora_usd, predicciones_por_peticion
    )
    costo_mensual_instancia_usd = precio_hora_usd * horas_mes
    costo_mensual_demanda_usd = demanda_mensual_pred * costo_unitario

    # Pico: P1 + P2 simultáneos (940 predicciones = 2 lotes) dentro de ventana_pico_seg
    pred_pico = 2 * estaciones_lote
    demanda_pico_pred_s = pred_pico / ventana_pico_seg
    instancias_pico = max(1, math.ceil(demanda_pico_pred_s / pred_por_segundo))

    return {
        "demanda_mensual_pred": int(demanda_mensual_pred),
        "capacidad_mensual_pred": round(capacidad_mensual_pred, 1),
        "utilizacion_pct": round(utilizacion_pct, 6),
        "costo_mensual_instancia_usd": round(costo_mensual_instancia_usd, 2),
        "costo_mensual_demanda_usd": costo_mensual_demanda_usd,
        "instancias_pico": int(instancias_pico),
    }


def analizar_finops(
    historia: pd.DataFrame,
    recursos: pd.DataFrame,
    escalones: list[tuple[int, int]] = ESCALONES_CARGA,
    precio_hora_usd: float = PRECIO_HORA_USD,
    latencia_objetivo_ms: float = LATENCIA_OBJETIVO_MS,
    tasa_error_max: float = TASA_ERROR_MAX,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta el análisis por escalón y calcula costo por inferencia y escalabilidad por modo.

    Returns:
        `(tabla_escalones, tabla_finops)` donde `tabla_escalones` resume cada escalón agregado y
        `tabla_finops` trae una fila por modo (`agregado`, `una_estacion`, `lote`).
    """
    tabla_escalones = resumen_por_escalon(
        historia,
        recursos,
        escalones=escalones,
        latencia_objetivo_ms=latencia_objetivo_ms,
        tasa_error_max=tasa_error_max,
        modo="agregado",
    )
    mejor_agregado = escalon_sostenible(
        tabla_escalones, latencia_objetivo_ms, tasa_error_max, etiqueta="agregado"
    )

    modos = [
        ("agregado", 1, tabla_escalones, mejor_agregado),
        ("una_estacion", 1, None, None),
        ("lote", LOTE_CARGA, None, None),
    ]

    filas_finops = []
    for modo, pred_por_req, tabla_modo, mejor_modo in modos:
        if tabla_modo is None:
            tabla_modo = resumen_por_escalon(
                historia,
                recursos,
                escalones=escalones,
                latencia_objetivo_ms=latencia_objetivo_ms,
                tasa_error_max=tasa_error_max,
                modo=modo,
            )
            if tabla_modo.empty:
                # Si el historial no traía filas desagregadas por ruta, se usa el RPS agregado
                mejor_modo = mejor_agregado
            else:
                mejor_modo = escalon_sostenible(
                    tabla_modo, latencia_objetivo_ms, tasa_error_max, etiqueta=modo
                )

        rps = float(mejor_modo["rps"])
        costo_unit, costo_1000 = costo_por_inferencia(rps, precio_hora_usd, pred_por_req)
        esc = escalabilidad_mibici(rps, pred_por_req, precio_hora_usd)
        filas_finops.append(
            {
                "modo": modo,
                "instancia_referencia": INSTANCIA_REFERENCIA,
                "precio_hora_usd": precio_hora_usd,
                "escalon_usuarios": int(mejor_modo["usuarios"]),
                "rps_sostenible": rps,
                "p50_ms": mejor_modo["p50_ms"],
                "p95_ms": mejor_modo["p95_ms"],
                "tasa_error": mejor_modo["tasa_error"],
                "cpu_media_pct": mejor_modo["cpu_media_pct"],
                "cpu_max_pct": mejor_modo["cpu_max_pct"],
                "mem_max_mib": mejor_modo["mem_max_mib"],
                "predicciones_por_peticion": pred_por_req,
                "predicciones_por_segundo": round(rps * pred_por_req, 3),
                "costo_inferencia_usd": costo_unit,
                "costo_1000_inferencias_usd": costo_1000,
                **esc,
            }
        )

    return tabla_escalones, pd.DataFrame(filas_finops)


def _relativa(ruta: Path) -> Path:
    return ruta.relative_to(PROJ_ROOT) if ruta.is_relative_to(PROJ_ROOT) else ruta


@app.command()
def main(
    historia_csv: Annotated[
        Path, typer.Option(help="Historial de Locust por segundo")
    ] = HISTORIA_CSV,
    recursos_csv: Annotated[Path, typer.Option(help="Muestras de CPU y memoria")] = RECURSOS_CSV,
    salida_finops: Annotated[
        Path, typer.Option(help="Tabla de costos y escalabilidad")
    ] = FINOPS_CSV,
    salida_escalones: Annotated[
        Path, typer.Option(help="Tabla de métricas por escalón")
    ] = ESCALONES_CSV,
    figuras_dir: Annotated[
        Path, typer.Option(help="Carpeta para las gráficas PNG")
    ] = FIGURAS_SEMANA10_DIR,
) -> None:
    """Genera las tablas FinOps y las gráficas de desempeño a partir de `make carga`."""
    if not historia_csv.exists():
        logger.error(
            f"No existe {historia_csv}. Corre primero `make carga` con la API levantada "
            "(`make sistema`) para generar los CSV de Locust."
        )
        raise typer.Exit(code=1)

    historia = pd.read_csv(historia_csv)
    recursos = pd.read_csv(recursos_csv) if recursos_csv.exists() else pd.DataFrame()

    try:
        serie = serie_temporal_carga(historia)
        tabla_escalones, tabla_finops = analizar_finops(historia, recursos)
    except ValueError as error:
        logger.error(str(error))
        raise typer.Exit(code=1) from error

    salida_finops.parent.mkdir(parents=True, exist_ok=True)
    salida_escalones.parent.mkdir(parents=True, exist_ok=True)
    figuras_dir.mkdir(parents=True, exist_ok=True)

    tabla_escalones.to_csv(salida_escalones, index=False)
    tabla_finops.to_csv(salida_finops, index=False)

    rps_sost = float(
        tabla_finops.loc[tabla_finops["modo"] == "agregado", "rps_sostenible"].iloc[0]
    )
    t0 = float(_a_numerico(historia["Timestamp"]).min())
    rec_alineado = preparar_recursos(recursos, t0)

    rutas_figuras = [
        plots.latencia_carga(
            serie, ESCALONES_CARGA, LATENCIA_OBJETIVO_MS, figuras_dir / FIG_LATENCIA.name
        ),
        plots.rps_y_error_carga(
            serie,
            ESCALONES_CARGA,
            TASA_ERROR_MAX,
            figuras_dir / FIG_RPS_ERROR.name,
            rps_sostenible=rps_sost,
        ),
        plots.recursos_carga(rec_alineado, ESCALONES_CARGA, figuras_dir / FIG_RECURSOS.name),
    ]

    logger.info(
        f"\n=== Métricas por escalón ({INSTANCIA_REFERENCIA}) ===\n"
        f"{tabla_escalones.to_string(index=False)}"
    )
    cols_resumen = [
        "modo",
        "escalon_usuarios",
        "rps_sostenible",
        "predicciones_por_segundo",
        "costo_1000_inferencias_usd",
        "demanda_mensual_pred",
        "utilizacion_pct",
        "costo_mensual_instancia_usd",
        "instancias_pico",
    ]
    logger.info(
        f"\n=== Resumen FinOps ({HORAS_MES} h/mes) ===\n"
        f"{tabla_finops[cols_resumen].to_string(index=False)}"
    )

    logger.success(
        f"Tablas guardadas en {_relativa(salida_finops)} y {_relativa(salida_escalones)}; "
        f"figuras en {', '.join(str(_relativa(r)) for r in rutas_figuras)}"
    )


if __name__ == "__main__":
    app()
