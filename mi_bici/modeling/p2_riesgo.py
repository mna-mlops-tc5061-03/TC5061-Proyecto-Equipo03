"""Define el experimento de P2: estaciones en riesgo de quedarse sin bicis o sin anclajes."""

from pathlib import Path
from typing import ClassVar

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import FixedThresholdClassifier

from mi_bici import plots
from mi_bici.config import (
    BASELINE_P2,
    BLOQUE_HORAS,
    COLUMNA_FECHA_P2,
    CORTE_PRUEBA,
    OBJETIVO_P2,
    SEMILLA,
    UMBRAL_DECISION_P2,
    UMBRAL_RIESGO,
    VARIABLES_P2,
)
from mi_bici.dataset import load_processed_stations, load_processed_travels
from mi_bici.features import etiqueta_riesgo, flujo_por_bloque
from mi_bici.modeling.base import ARBOLES_HGB, ARBOLES_RF, ExperimentoBase, Modelo
from mi_bici.modeling.evaluate import metricas_clasificacion


class ExperimentoRiesgo(ExperimentoBase):
    """Predice si una estación estará en riesgo de vaciarse o llenarse en el siguiente bloque (P2)."""

    pregunta = "P2"
    # Solo cerca de 5 % de los bloques están en riesgo, así que se da más peso a esa clase
    catalogo: ClassVar[dict[str, Modelo]] = {
        "baseline": Modelo("semana_anterior"),
        "hgb": Modelo(
            "gradient_boosting",
            HistGradientBoostingClassifier,
            por_defecto={"class_weight": "balanced"},
            cuadricula={"learning_rate": [0.05, 0.1], "max_leaf_nodes": [31, 63, 127]},
        ),
        "logreg": Modelo(
            "lineal",
            LogisticRegression,
            por_defecto={"class_weight": "balanced", "max_iter": 1000},
            cuadricula={"C": [0.01, 0.1, 1.0, 10.0]},
            escalar=True,
        ),
        "rf": Modelo(
            "random_forest",
            RandomForestClassifier,
            por_defecto={
                "class_weight": "balanced",
                "n_estimators": 100,
                "max_depth": 15,
                "n_jobs": -1,
            },
            cuadricula={"n_estimators": [50, 100], "max_depth": [10, 20]},
        ),
    }
    tipos_confiables = (*ARBOLES_HGB, *ARBOLES_RF)
    columna_fecha = COLUMNA_FECHA_P2

    def __init__(
        self, modelo: str = "baseline", params: dict | None = None, semilla: int = SEMILLA
    ):
        super().__init__(modelo, params, semilla)
        self.umbral_decision: float = UMBRAL_DECISION_P2

    def preparar(self) -> None:
        """Construye el flujo por bloque y la etiqueta de riesgo, y separa por fecha."""
        flujo = flujo_por_bloque(load_processed_travels(), load_processed_stations())
        riesgo = etiqueta_riesgo(flujo, UMBRAL_RIESGO)
        prueba = riesgo[COLUMNA_FECHA_P2] >= CORTE_PRUEBA
        self.train, self.test = riesgo[~prueba], riesgo[prueba]
        self.X_train = self.train[VARIABLES_P2]

    def entrenar(self, datos) -> None:
        """Ajusta el modelo del catálogo con `datos`. El baseline no se entrena."""
        if self.tipo == "candidato":
            self.modelo = self.crear_modelo()
            self.modelo.fit(datos[VARIABLES_P2], datos[OBJETIVO_P2])

    def evaluar(self, datos, prefijo: str) -> dict:
        """Evalúa con `datos`. El baseline repite el riesgo del mismo bloque la semana anterior.

        Returns:
            `<prefijo>_recall`, `<prefijo>_precision`, `<prefijo>_f1`,
            `<prefijo>_average_precision` y la proporción de bloques en riesgo
            (`train_tasa_riesgo`, `<prefijo>_tasa_riesgo`).
        """
        if self.modelo is None:
            self.pred = self.puntaje = datos[BASELINE_P2].to_numpy()
        else:
            self.puntaje = self.modelo.predict_proba(datos[VARIABLES_P2])[:, 1]
            if prefijo == "val":
                precisions, recalls, thresholds = precision_recall_curve(
                    datos[OBJETIVO_P2], self.puntaje
                )
                denominador = precisions[:-1] + recalls[:-1]
                f1_scores = np.divide(
                    2 * precisions[:-1] * recalls[:-1],
                    denominador,
                    out=np.zeros_like(thresholds),
                    where=denominador > 0,
                )
                if len(f1_scores) > 0 and np.max(f1_scores) > 0:
                    self.umbral_decision = float(thresholds[np.argmax(f1_scores)])
                else:
                    self.umbral_decision = UMBRAL_DECISION_P2
            self.pred = (self.puntaje >= self.umbral_decision).astype(int)
        metricas = metricas_clasificacion(datos[OBJETIVO_P2], self.pred, self.puntaje, prefijo)
        # La tasa real de riesgo justifica el umbral y el uso de average precision
        metricas["train_tasa_riesgo"] = self.train[OBJETIVO_P2].mean()
        metricas[f"{prefijo}_tasa_riesgo"] = datos[OBJETIVO_P2].mean()
        if self.modelo is not None:
            metricas[f"{prefijo}_umbral_decision"] = round(self.umbral_decision, 4)
        return metricas

    def artefactos(self, carpeta: Path) -> list[Path]:
        """Guarda la matriz de confusión y, en el candidato, la curva de precisión y recall.

        Args:
            carpeta: Carpeta temporal del run.

        Returns:
            Las rutas de las gráficas.
        """
        rutas = [
            plots.matriz_confusion(
                self.test[OBJETIVO_P2], self.pred, carpeta / "matriz_confusion.png"
            )
        ]
        if self.modelo is not None:
            rutas.append(
                plots.curva_pr(self.test[OBJETIVO_P2], self.puntaje, carpeta / "curva_pr.png")
            )
        return rutas

    def modelo_a_guardar(self):
        """Envuelve el clasificador con `umbral_decision` para que `predict` aplique ese corte."""
        if self.modelo is None:
            return None
        return FixedThresholdClassifier(
            FrozenEstimator(self.modelo),
            threshold=self.umbral_decision,
            response_method="predict_proba",
        )

    def hiperparametros(self) -> dict:
        """Devuelve el modelo, el bloque y el umbral que definen la etiqueta, y sus hiperparámetros."""
        datos = {
            **super().hiperparametros(),
            "umbral_riesgo": UMBRAL_RIESGO,
            "bloque_horas": BLOQUE_HORAS,
        }
        if self.modelo is not None:
            datos["variables"] = ",".join(VARIABLES_P2)
            datos["umbral_decision"] = round(self.umbral_decision, 4)
        return datos
