"""Define el experimento de P2: estaciones en riesgo de quedarse sin bicis o sin anclajes."""

from pathlib import Path
from typing import ClassVar

from sklearn.ensemble import HistGradientBoostingClassifier

from mi_bici import plots
from mi_bici.config import (
    BASELINE_P2,
    BLOQUE_HORAS,
    COLUMNA_FECHA_P2,
    CORTE_PRUEBA,
    OBJETIVO_P2,
    UMBRAL_RIESGO,
    VARIABLES_P2,
)
from mi_bici.dataset import load_processed_stations, load_processed_travels
from mi_bici.features import etiqueta_riesgo, flujo_por_bloque
from mi_bici.modeling.base import ARBOLES_HGB, ExperimentoBase, Modelo
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
        ),
    }
    tipos_confiables = ARBOLES_HGB
    columna_fecha = COLUMNA_FECHA_P2

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
            self.pred = self.modelo.predict(datos[VARIABLES_P2])
            self.puntaje = self.modelo.predict_proba(datos[VARIABLES_P2])[:, 1]
        metricas = metricas_clasificacion(datos[OBJETIVO_P2], self.pred, self.puntaje, prefijo)
        # La tasa real de riesgo justifica el umbral y el uso de average precision
        metricas["train_tasa_riesgo"] = self.train[OBJETIVO_P2].mean()
        metricas[f"{prefijo}_tasa_riesgo"] = datos[OBJETIVO_P2].mean()
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

    def hiperparametros(self) -> dict:
        """Devuelve el modelo, el bloque y el umbral que definen la etiqueta, y sus hiperparámetros."""
        datos = {
            **super().hiperparametros(),
            "umbral_riesgo": UMBRAL_RIESGO,
            "bloque_horas": BLOQUE_HORAS,
        }
        if self.modelo is not None:
            datos["variables"] = ",".join(VARIABLES_P2)
        return datos
