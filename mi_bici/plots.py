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
