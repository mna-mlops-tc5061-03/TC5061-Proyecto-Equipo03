"""Define el experimento de P1: viajes que saldrán de cada estación al día siguiente."""

from pathlib import Path
from typing import ClassVar

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from mi_bici import plots
from mi_bici.config import (
    BASELINE_P1,
    COLUMNA_FECHA_P1,
    CORTE_PRUEBA,
    OBJETIVO_P1,
    VARIABLES_P1,
)
from mi_bici.dataset import load_processed_stations, load_processed_travels
from mi_bici.features import demanda_diaria
from mi_bici.modeling.base import ARBOLES_HGB, ARBOLES_RF, ExperimentoBase, Modelo
from mi_bici.modeling.evaluate import metricas_regresion


class ExperimentoDemanda(ExperimentoBase):
    """Pronostica los viajes que iniciarán mañana en cada estación (P1)."""

    pregunta = "P1"
    catalogo: ClassVar[dict[str, Modelo]] = {
        "baseline": Modelo("semana_anterior"),
        "hgb": Modelo("gradient_boosting", HistGradientBoostingRegressor),
        "rf": Modelo(
            "random_forest",
            RandomForestRegressor,
            por_defecto={"n_estimators": 100, "max_depth": 15, "n_jobs": -1},
        ),
        "ridge": Modelo(
            "lineal",
            Ridge,
            por_defecto={"alpha": 1.0},
            escalar=True,
        ),
    }
    tipos_confiables = (*ARBOLES_HGB, *ARBOLES_RF)
    columna_fecha = COLUMNA_FECHA_P1

    def preparar(self) -> None:
        """Construye la demanda diaria y la separa en entrenamiento y prueba por fecha."""
        demanda = demanda_diaria(load_processed_travels(), load_processed_stations())
        prueba = demanda[COLUMNA_FECHA_P1] >= CORTE_PRUEBA
        self.train, self.test = demanda[~prueba], demanda[prueba]
        self.X_train = self.train[VARIABLES_P1]

    def entrenar(self, datos) -> None:
        """Ajusta el modelo del catálogo con `datos`. El baseline no se entrena."""
        if self.tipo == "candidato":
            self.modelo = self.crear_modelo()
            self.modelo.fit(datos[VARIABLES_P1], datos[OBJETIVO_P1])

    def evaluar(self, datos, prefijo: str) -> dict:
        """Evalúa con `datos`. El baseline predice los viajes del mismo día de la semana anterior.

        Returns:
            `<prefijo>_mae`, `<prefijo>_rmse` y `<prefijo>_r2`.
        """
        if self.modelo is None:
            self.pred = datos[BASELINE_P1].to_numpy()
        else:
            self.pred = self.modelo.predict(datos[VARIABLES_P1])
        return metricas_regresion(datos[OBJETIVO_P1], self.pred, prefijo)

    def artefactos(self, carpeta: Path) -> list[Path]:
        """Guarda la gráfica de viajes reales contra predichos.

        Args:
            carpeta: Carpeta temporal del run.

        Returns:
            La ruta de la gráfica.
        """
        ruta = carpeta / "real_contra_predicho.png"
        return [
            plots.real_contra_predicho(
                self.test[COLUMNA_FECHA_P1], self.test[OBJETIVO_P1], self.pred, ruta
            )
        ]

    def hiperparametros(self) -> dict:
        """Devuelve el modelo, sus hiperparámetros y, en los candidatos, sus variables."""
        if self.modelo is None:
            return super().hiperparametros()
        return {**super().hiperparametros(), "variables": ",".join(VARIABLES_P1)}
