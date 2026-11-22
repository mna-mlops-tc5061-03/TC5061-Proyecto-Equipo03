"""Genera las gráficas que se registran como artefactos de cada run en MLflow."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay


def _guardar(fig, ruta: Path) -> Path:
    """Guarda una figura y la cierra para no acumular memoria entre runs.

    Args:
        fig: Figura de matplotlib.
        ruta: Ruta del archivo PNG.

    Returns:
        La misma ruta.
    """
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)
    return ruta


def real_contra_predicho(fechas, y, pred, ruta: Path) -> Path:
    """Grafica los viajes diarios del sistema en prueba, reales contra predichos (P1).

    Args:
        fechas: Fecha de cada fila de prueba.
        y: Viajes reales.
        pred: Viajes predichos.
        ruta: Ruta del archivo PNG.

    Returns:
        La ruta de la gráfica.
    """
    total = pd.DataFrame({"fecha": fechas, "real": y, "predicho": pred}).groupby("fecha").sum()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(total.index, total["real"], label="real", color="#1f77b4")
    ax.plot(total.index, total["predicho"], label="predicho", color="#ff7f0e", linestyle="--")
    ax.set(
        xlabel="fecha",
        ylabel="viajes iniciados en el sistema",
        title="Prueba: viajes por día, real contra predicho",
    )
    ax.legend()
    fig.autofmt_xdate()
    return _guardar(fig, ruta)


def matriz_confusion(y, pred, ruta: Path) -> Path:
    """Grafica los aciertos y errores por clase en prueba (P2).

    Args:
        y: Etiquetas reales.
        pred: Etiquetas predichas.
        ruta: Ruta del archivo PNG.

    Returns:
        La ruta de la gráfica.
    """
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, display_labels=["sin riesgo", "riesgo"], ax=ax, colorbar=False, values_format=","
    )
    return _guardar(fig, ruta)


def curva_pr(y, puntaje, ruta: Path) -> Path:
    """Grafica la precisión contra el recall para todos los umbrales del modelo (P2).

    Args:
        y: Etiquetas reales.
        puntaje: Probabilidad de la clase positiva.
        ruta: Ruta del archivo PNG.

    Returns:
        La ruta de la gráfica.
    """
    fig, ax = plt.subplots(figsize=(5, 4.5))
    PrecisionRecallDisplay.from_predictions(y, puntaje, ax=ax, name="candidato")
    ax.axhline(pd.Series(y).mean(), color="gray", linestyle=":", label="proporción de riesgo")
    ax.legend()
    return _guardar(fig, ruta)


def mapa_grupos(perfil: pd.DataFrame, columna: str, ruta: Path) -> Path:
    """Grafica la ubicación de cada estación coloreada por su grupo (P3).

    Args:
        perfil: Estaciones con `latitud`, `longitud` y la columna del grupo.
        columna: Nombre de la columna del grupo.
        ruta: Ruta del archivo PNG.

    Returns:
        La ruta del mapa.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    for grupo, datos in perfil.groupby(columna):
        ax.scatter(datos["longitud"], datos["latitud"], s=14, label=f"{grupo} ({len(datos)})")
    ax.set(xlabel="longitud", ylabel="latitud", title="Grupos de estaciones")
    ax.set_aspect("equal")
    ax.legend(title="grupo (estaciones)", fontsize=8)
    return _guardar(fig, ruta)


def perfil_horario(perfil: pd.DataFrame, columna: str, ruta: Path) -> Path:
    """Grafica la proporción promedio de salidas por hora de cada grupo, para interpretarlos (P3).

    Args:
        perfil: Estaciones con las columnas `hora_00` a `hora_23` y la columna del grupo.
        columna: Nombre de la columna del grupo.
        ruta: Ruta del archivo PNG.

    Returns:
        La ruta de la gráfica.
    """
    horas = [c for c in perfil.columns if c.startswith("hora_")]
    promedio = perfil.groupby(columna)[horas].mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    for grupo, fila in promedio.iterrows():
        ax.plot(range(24), fila.values, marker="o", markersize=3, label=str(grupo))
    ax.set(
        xlabel="hora del día",
        ylabel="proporción de salidas",
        xticks=range(0, 24, 2),
        title="Perfil horario promedio por grupo",
    )
    ax.legend(title="grupo", fontsize=8)
    return _guardar(fig, ruta)


def drift_y_error_por_mes(
    por_mes: pd.DataFrame,
    umbral: float,
    ruta: Path,
    umbral_mae: float | None = None,
) -> Path:
    """Grafica la evolución mensual del `drift_share` y del MAE del champion contra el baseline (P1).

    Args:
        por_mes: Tabla mensual con `mes`, `drift_share`, `mae_champion` y `mae_baseline`.
        umbral: Umbral de `drift_share` propuesto como alerta temprana.
        ruta: Ruta del archivo PNG.
        umbral_mae: Límite máximo de MAE (`mae_referencia * (1 + TOLERANCIA_MAE_P1)`), opcional.

    Returns:
        La ruta de la gráfica.
    """
    fig, (ax_drift, ax_mae) = plt.subplots(1, 2, figsize=(10, 4))
    meses = por_mes["mes"].astype(str)

    ax_drift.plot(meses, por_mes["drift_share"], marker="o", color="#1f77b4", label="drift_share")
    ax_drift.axhline(umbral, color="#d62728", linestyle="--", label=f"alerta ({umbral:.2f})")
    ax_drift.set(
        xlabel="mes",
        ylabel="proporción de columnas con drift",
        ylim=(-0.05, 1.05),
        title="Drift contra la referencia por mes",
    )
    ax_drift.legend()

    ax_mae.plot(meses, por_mes["mae_champion"], marker="o", color="#1f77b4", label="champion")
    ax_mae.plot(
        meses,
        por_mes["mae_baseline"],
        marker="s",
        color="#ff7f0e",
        linestyle="--",
        label="baseline",
    )
    if umbral_mae is not None:
        ax_mae.axhline(
            umbral_mae, color="#d62728", linestyle=":", label=f"límite MAE ({umbral_mae:.2f})"
        )
    ax_mae.set(
        xlabel="mes",
        ylabel="MAE (viajes por estación y día)",
        title="Error de predicción por mes (P1)",
    )
    ax_mae.legend()
    return _guardar(fig, ruta)


def _marcar_escalones(ax, escalones: list[tuple[int, int]], con_etiqueta: bool = True) -> None:
    """Traza líneas verticales en los cambios de escalón y rotula arriba del marco."""
    inicio = 0
    for indice, (usuarios, duracion) in enumerate(escalones):
        if indice > 0:
            ax.axvline(inicio, color="gray", linestyle=":", linewidth=0.9, alpha=0.8)
        if con_etiqueta:
            ax.text(
                inicio + duracion / 2,
                1.015,
                f"{usuarios} u",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8,
                color="dimgray",
            )
        inicio += duracion


def latencia_carga(
    serie: pd.DataFrame,
    escalones: list[tuple[int, int]],
    latencia_objetivo_ms: float,
    ruta: Path,
) -> Path:
    """Grafica latencia p50 y p95 en el tiempo con los escalones y el objetivo de servicio.

    Args:
        serie: Serie temporal con `segundo`, `p50_ms` y `p95_ms`.
        escalones: Lista de pares `(usuarios, duracion_seg)`.
        latencia_objetivo_ms: Umbral objetivo de latencia p95 en milisegundos.
        ruta: Ruta del archivo PNG.

    Returns:
        La ruta de la gráfica.
    """
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.plot(serie["segundo"], serie["p50_ms"], color="#1f77b4", label="p50")
    ax.plot(serie["segundo"], serie["p95_ms"], color="#ff7f0e", label="p95")
    ax.axhline(
        latencia_objetivo_ms,
        color="#d62728",
        linestyle="--",
        label=f"objetivo p95 ({latencia_objetivo_ms:.0f} ms)",
    )
    _marcar_escalones(ax, escalones)
    ax.set(xlabel="tiempo transcurrido (s)", ylabel="latencia (ms)")
    ax.set_title("Latencia p50 y p95 durante la prueba de carga", pad=18)
    ax.legend(loc="center left")
    return _guardar(fig, ruta)


def rps_y_error_carga(
    serie: pd.DataFrame,
    escalones: list[tuple[int, int]],
    tasa_error_max: float,
    ruta: Path,
    rps_sostenible: float | None = None,
) -> Path:
    """Grafica RPS y tasa de error en el tiempo con los escalones marcados.

    Args:
        serie: Serie temporal con `segundo`, `rps` y `tasa_error_pct`.
        escalones: Lista de pares `(usuarios, duracion_seg)`.
        tasa_error_max: Tasa máxima de error tolerada (proporción, p. ej. 0.01 = 1 %).
        ruta: Ruta del archivo PNG.
        rps_sostenible: RPS sostenible encontrado, opcional para trazar la referencia.

    Returns:
        La ruta de la gráfica.
    """
    fig, (ax_rps, ax_err) = plt.subplots(1, 2, figsize=(11, 4.4))

    ax_rps.plot(serie["segundo"], serie["rps"], color="#1f77b4", label="RPS")
    if rps_sostenible is not None:
        ax_rps.axhline(
            rps_sostenible,
            color="#2ca02c",
            linestyle="--",
            label=f"RPS sostenible ({rps_sostenible:.1f})",
        )
    _marcar_escalones(ax_rps, escalones)
    ax_rps.set(xlabel="tiempo transcurrido (s)", ylabel="peticiones por segundo")
    ax_rps.set_title("Rendimiento (RPS) en el tiempo", pad=18)
    ax_rps.legend(loc="lower right")

    limite_pct = tasa_error_max * 100
    max_err = float(serie["tasa_error_pct"].fillna(0.0).max()) if not serie.empty else 0.0
    ax_err.plot(serie["segundo"], serie["tasa_error_pct"], color="#d62728", label="tasa de error")
    ax_err.axhline(
        limite_pct,
        color="gray",
        linestyle="--",
        label=f"límite ({limite_pct:.1f} %)",
    )
    _marcar_escalones(ax_err, escalones)
    ax_err.set(
        xlabel="tiempo transcurrido (s)",
        ylabel="errores (%)",
        ylim=(-0.2, max(limite_pct * 2, max_err * 1.15)),
    )
    ax_err.set_title("Tasa de error en el tiempo", pad=18)
    ax_err.legend(loc="upper right")
    return _guardar(fig, ruta)


def recursos_carga(
    recursos: pd.DataFrame,
    escalones: list[tuple[int, int]],
    ruta: Path,
    limite_cpu_pct: float | None = None,
) -> Path:
    """Grafica CPU (%) y memoria (MiB) de los contenedores `api` y `mlflow` durante la carga.

    Args:
        recursos: Tabla con `segundo`, `contenedor`, `cpu_pct` y `mem_mib`.
        escalones: Lista de pares `(usuarios, duracion_seg)`.
        ruta: Ruta del archivo PNG.
        limite_cpu_pct: Límite de CPU del contenedor `api` en % (p. ej. 200 para 2 vCPU), opcional.

    Returns:
        La ruta de la gráfica.
    """
    fig, (ax_cpu, ax_mem) = plt.subplots(1, 2, figsize=(11, 4.4))
    colores = {"api": "#1f77b4", "mlflow": "#ff7f0e"}

    if recursos.empty:
        for ax in (ax_cpu, ax_mem):
            ax.text(0.5, 0.5, "Sin muestras en recursos.csv", ha="center", va="center")
    else:
        for contenedor in ("api", "mlflow"):
            sub = recursos[recursos["contenedor"] == contenedor]
            if sub.empty:
                continue
            color = colores.get(contenedor)
            ax_cpu.plot(sub["segundo"], sub["cpu_pct"], label=contenedor, color=color)
            ax_mem.plot(sub["segundo"], sub["mem_mib"], label=contenedor, color=color)
        for contenedor, sub in recursos.groupby("contenedor"):
            if contenedor in colores:
                continue
            ax_cpu.plot(sub["segundo"], sub["cpu_pct"], label=str(contenedor))
            ax_mem.plot(sub["segundo"], sub["mem_mib"], label=str(contenedor))

    if limite_cpu_pct is not None:
        ax_cpu.axhline(
            limite_cpu_pct,
            color="#d62728",
            linestyle="--",
            label=f"límite api ({limite_cpu_pct:.0f} %)",
        )
        max_cpu = float(recursos["cpu_pct"].fillna(0.0).max()) if not recursos.empty else 0.0
        ax_cpu.set_ylim(bottom=0, top=max(limite_cpu_pct, max_cpu) * 1.08)

    if not recursos.empty or limite_cpu_pct is not None:
        ax_cpu.legend(loc="center left")
    if not recursos.empty:
        ax_mem.legend(loc="center right")

    _marcar_escalones(ax_cpu, escalones)
    _marcar_escalones(ax_mem, escalones)
    ax_cpu.set(xlabel="tiempo transcurrido (s)", ylabel="CPU (%)")
    ax_cpu.set_title("Uso de CPU por contenedor", pad=18)
    ax_mem.set(xlabel="tiempo transcurrido (s)", ylabel="memoria (MiB)")
    ax_mem.set_title("Uso de memoria por contenedor", pad=18)
    return _guardar(fig, ruta)
