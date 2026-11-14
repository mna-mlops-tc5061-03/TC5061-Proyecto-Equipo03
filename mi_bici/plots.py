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


def drift_y_error_por_mes(por_mes: pd.DataFrame, umbral: float, ruta: Path) -> Path:
    """Grafica la evolución mensual del `drift_share` y del MAE del champion contra el baseline (P1).

    Args:
        por_mes: Tabla mensual con `mes`, `drift_share`, `mae_champion` y `mae_baseline`.
        umbral: Umbral de `drift_share` propuesto como disparador de reentrenamiento.
        ruta: Ruta del archivo PNG.

    Returns:
        La ruta de la gráfica.
    """
    fig, (ax_drift, ax_mae) = plt.subplots(1, 2, figsize=(10, 4))
    meses = por_mes["mes"].astype(str)

    ax_drift.plot(meses, por_mes["drift_share"], marker="o", color="#1f77b4", label="drift_share")
    ax_drift.axhline(umbral, color="#d62728", linestyle="--", label=f"umbral ({umbral:.2f})")
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
    ax_mae.set(
        xlabel="mes",
        ylabel="MAE (viajes por estación y día)",
        title="Error de predicción por mes (P1)",
    )
    ax_mae.legend()
    return _guardar(fig, ruta)
