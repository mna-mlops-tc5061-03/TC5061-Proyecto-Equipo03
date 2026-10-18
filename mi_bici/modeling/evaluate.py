"""Calcula las métricas estándar de cada tipo de tarea.

Todas devuelven un diccionario con el conjunto en el nombre de cada métrica (`test_` o `train_`).
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


def metricas_regresion(y, pred) -> dict:
    """Calcula las métricas de regresión de P1.

    El MAE se lee en viajes, que es lo que el operador entiende. No se usa MAPE porque hay días
    con cero viajes.

    Args:
        y: Valores reales del conjunto de prueba.
        pred: Valores predichos.

    Returns:
        `test_mae`, `test_rmse` y `test_r2`.
    """
    return {
        "test_mae": mean_absolute_error(y, pred),
        "test_rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "test_r2": r2_score(y, pred),
    }


def metricas_clasificacion(y, pred, puntaje) -> dict:
    """Calcula las métricas de clasificación de P2.

    Con pocos bloques en riesgo, el recall dice cuántos se detectan y la precisión cuántas visitas
    del camión serían en vano. PR-AUC resume ambas sin fijar un umbral.

    Args:
        y: Etiquetas reales del conjunto de prueba.
        pred: Etiquetas predichas.
        puntaje: Probabilidad o puntaje de la clase positiva.

    Returns:
        `test_recall`, `test_precision`, `test_f1` y `test_pr_auc`.
    """
    return {
        "test_recall": recall_score(y, pred, zero_division=0),
        "test_precision": precision_score(y, pred, zero_division=0),
        "test_f1": f1_score(y, pred, zero_division=0),
        "test_pr_auc": average_precision_score(y, puntaje),
    }


def metricas_clustering(X, grupos) -> dict:
    """Calcula las métricas de clustering de P3.

    Sin etiquetas reales, se mide qué tan compactos y separados quedan los grupos. En la silueta y
    en Calinski-Harabasz más es mejor, y en Davies-Bouldin menos es mejor.

    Args:
        X: Variables escaladas de las estaciones.
        grupos: Grupo asignado a cada estación.

    Returns:
        `train_silhouette`, `train_davies_bouldin` y `train_calinski_harabasz`.
    """
    return {
        "train_silhouette": silhouette_score(X, grupos),
        "train_davies_bouldin": davies_bouldin_score(X, grupos),
        "train_calinski_harabasz": calinski_harabasz_score(X, grupos),
    }
