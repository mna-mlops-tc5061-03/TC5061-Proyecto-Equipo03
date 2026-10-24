"""Calcula las métricas estándar de cada tipo de tarea.

Todas devuelven un diccionario con el conjunto en el nombre de cada métrica (`val_`, `test_` o
`train_`).
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)


def metricas_regresion(y, pred, prefijo: str = "test") -> dict:
    """Calcula las métricas de regresión de P1.

    El MAE se lee en viajes, que es lo que el operador entiende. No se usa MAPE porque hay días
    con cero viajes.

    Args:
        y: Valores reales del conjunto evaluado.
        pred: Valores predichos.
        prefijo: Conjunto evaluado, `"val"` o `"test"`.

    Returns:
        `<prefijo>_mae`, `<prefijo>_rmse` y `<prefijo>_r2`.
    """
    return {
        f"{prefijo}_mae": mean_absolute_error(y, pred),
        f"{prefijo}_rmse": float(np.sqrt(mean_squared_error(y, pred))),
        f"{prefijo}_r2": r2_score(y, pred),
    }


def metricas_clasificacion(y, pred, puntaje, prefijo: str = "test") -> dict:
    """Calcula las métricas de clasificación de P2.

    Con pocos bloques en riesgo, el recall dice cuántos se detectan y la precisión cuántas visitas
    del camión serían en vano. La precisión promedio resume ambas sin fijar un umbral.

    Args:
        y: Etiquetas reales del conjunto evaluado.
        pred: Etiquetas predichas.
        puntaje: Probabilidad o puntaje de la clase positiva.
        prefijo: Conjunto evaluado, `"val"` o `"test"`.

    Returns:
        `<prefijo>_recall`, `<prefijo>_precision`, `<prefijo>_f1` y
        `<prefijo>_average_precision`.
    """
    return {
        f"{prefijo}_recall": recall_score(y, pred, zero_division=0),
        f"{prefijo}_precision": precision_score(y, pred, zero_division=0),
        f"{prefijo}_f1": f1_score(y, pred, zero_division=0),
        f"{prefijo}_average_precision": average_precision_score(y, puntaje),
    }


def metricas_clustering(X, grupos, prefijo: str = "train") -> dict:
    """Calcula las métricas de clustering de P3.

    Sin etiquetas reales, se mide qué tan compactos y separados quedan los grupos. En la silueta y
    en Calinski-Harabasz más es mejor, y en Davies-Bouldin menos es mejor.

    Args:
        X: Variables escaladas de las estaciones.
        grupos: Grupo asignado a cada estación.
        prefijo: Conjunto evaluado. En P3 es `"train"`, porque no hay prueba.

    Returns:
        `<prefijo>_silhouette`, `<prefijo>_davies_bouldin` y `<prefijo>_calinski_harabasz`.
    """
    return {
        f"{prefijo}_silhouette": silhouette_score(X, grupos),
        f"{prefijo}_davies_bouldin": davies_bouldin_score(X, grupos),
        f"{prefijo}_calinski_harabasz": calinski_harabasz_score(X, grupos),
    }
