"""Corre los experimentos base de P1, P2 y P3 y los registra en MLflow.

python -m mi_bici.modeling.train
python -m mi_bici.modeling.train --pregunta P2 --modelo candidato
"""

from pathlib import Path
from typing import ClassVar
import warnings

from loguru import logger
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import typer

from mi_bici import plots
from mi_bici.config import (
    BLOQUE_HORAS,
    CORTE_PRUEBA,
    K_CANDIDATOS,
    UMBRAL_RIESGO,
    VARIABLES_P1,
    VARIABLES_P2,
)
from mi_bici.dataset import load_processed_stations, load_processed_travels
from mi_bici.features import (
    VARIABLES_PERFIL_USO,
    demanda_diaria,
    etiqueta_riesgo,
    flujo_por_bloque,
    perfil_estacion,
)
from mi_bici.modeling.base import ExperimentoBase
from mi_bici.modeling.evaluate import (
    metricas_clasificacion,
    metricas_clustering,
    metricas_regresion,
)

# Las variables enteras (rezagos, hora, día) no tienen faltantes, así que el aviso de MLflow
# sobre columnas enteras en la firma del modelo no aplica
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")

app = typer.Typer()

# Tipos internos de los árboles de HistGradientBoosting que skops debe aceptar al guardar
ARBOLES_HGB = ("sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",)
HIPERPARAMETROS_HGB = ("learning_rate", "max_iter", "max_leaf_nodes")


class ExperimentoDemanda(ExperimentoBase):
    """Pronostica los viajes que iniciarán mañana en cada estación (P1)."""

    pregunta = "P1"
    nombres: ClassVar[dict[str, str]] = {
        "baseline": "P1-baseline-semana-anterior",
        "candidato": "P1-hgb-rezagos",
    }
    tipos_confiables = ARBOLES_HGB

    def preparar(self) -> None:
        """Construye la demanda diaria y la separa en entrenamiento y prueba por fecha."""
        demanda = demanda_diaria(load_processed_travels(), load_processed_stations())
        prueba = demanda["fecha"] >= CORTE_PRUEBA
        self.train, self.test = demanda[~prueba], demanda[prueba]
        self.X_train, self.X_test = self.train[VARIABLES_P1], self.test[VARIABLES_P1]

    def entrenar(self) -> None:
        """Ajusta el HistGradientBoostingRegressor. El baseline no se entrena."""
        if self.tipo == "candidato":
            self.modelo = HistGradientBoostingRegressor(random_state=self.semilla)
            self.modelo.fit(self.X_train, self.train["viajes"])

    def evaluar(self) -> dict:
        """Evalúa en prueba. El baseline predice los viajes del mismo día de la semana anterior.

        Returns:
            `test_mae`, `test_rmse` y `test_r2`.
        """
        if self.modelo is None:
            self.pred = self.test["viajes_lag_7"].to_numpy()
        else:
            self.pred = self.modelo.predict(self.X_test)
        return metricas_regresion(self.test["viajes"], self.pred)

    def artefactos(self, carpeta: Path) -> list[Path]:
        """Guarda la gráfica de viajes reales contra predichos.

        Args:
            carpeta: Carpeta temporal del run.

        Returns:
            La ruta de la gráfica.
        """
        ruta = carpeta / "real_contra_predicho.png"
        return [
            plots.real_contra_predicho(self.test["fecha"], self.test["viajes"], self.pred, ruta)
        ]

    def hiperparametros(self) -> dict:
        """Devuelve el modelo, sus variables e hiperparámetros."""
        if self.modelo is None:
            return {"modelo": "viajes_lag_7"}
        parametros = self.modelo.get_params()
        return {
            "modelo": "HistGradientBoostingRegressor",
            "variables": ",".join(VARIABLES_P1),
            **{k: parametros[k] for k in HIPERPARAMETROS_HGB},
        }


class ExperimentoRiesgo(ExperimentoBase):
    """Predice si una estación estará en riesgo de vaciarse o llenarse en el siguiente bloque (P2)."""

    pregunta = "P2"
    nombres: ClassVar[dict[str, str]] = {
        "baseline": "P2-baseline-semana-anterior",
        "candidato": "P2-hgb-balanced",
    }
    tipos_confiables = ARBOLES_HGB

    def preparar(self) -> None:
        """Construye el flujo por bloque y la etiqueta de riesgo, y separa por fecha."""
        flujo = flujo_por_bloque(load_processed_travels(), load_processed_stations())
        riesgo = etiqueta_riesgo(flujo, UMBRAL_RIESGO)
        prueba = riesgo["bloque"] >= CORTE_PRUEBA
        self.train, self.test = riesgo[~prueba], riesgo[prueba]
        self.X_train, self.X_test = self.train[VARIABLES_P2], self.test[VARIABLES_P2]

    def entrenar(self) -> None:
        """Ajusta el HistGradientBoostingClassifier con clases balanceadas.

        Solo cerca de 5 % de los bloques están en riesgo, así que se da más peso a esa clase.
        """
        if self.tipo == "candidato":
            self.modelo = HistGradientBoostingClassifier(
                class_weight="balanced", random_state=self.semilla
            )
            self.modelo.fit(self.X_train, self.train["riesgo"])

    def evaluar(self) -> dict:
        """Evalúa en prueba. El baseline repite el riesgo del mismo bloque la semana anterior.

        Returns:
            `test_recall`, `test_precision`, `test_f1`, `test_average_precision` y la proporción
            de bloques en riesgo (`train_tasa_riesgo`, `test_tasa_riesgo`).
        """
        if self.modelo is None:
            self.pred = self.puntaje = self.test["riesgo_semana_anterior"].to_numpy()
        else:
            self.pred = self.modelo.predict(self.X_test)
            self.puntaje = self.modelo.predict_proba(self.X_test)[:, 1]
        metricas = metricas_clasificacion(self.test["riesgo"], self.pred, self.puntaje)
        # La tasa real de riesgo justifica el umbral y el uso de average precision
        metricas["train_tasa_riesgo"] = self.train["riesgo"].mean()
        metricas["test_tasa_riesgo"] = self.test["riesgo"].mean()
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
                self.test["riesgo"], self.pred, carpeta / "matriz_confusion.png"
            )
        ]
        if self.modelo is not None:
            rutas.append(
                plots.curva_pr(self.test["riesgo"], self.puntaje, carpeta / "curva_pr.png")
            )
        return rutas

    def hiperparametros(self) -> dict:
        """Devuelve el modelo, el bloque y el umbral que definen la etiqueta, y sus hiperparámetros."""
        comunes = {"umbral_riesgo": UMBRAL_RIESGO, "bloque_horas": BLOQUE_HORAS}
        if self.modelo is None:
            return {"modelo": "riesgo_semana_anterior", **comunes}
        parametros = self.modelo.get_params()
        return {
            "modelo": "HistGradientBoostingClassifier",
            "variables": ",".join(VARIABLES_P2),
            "class_weight": "balanced",
            **comunes,
            **{k: parametros[k] for k in HIPERPARAMETROS_HGB},
        }


class ExperimentoGrupos(ExperimentoBase):
    """Agrupa las estaciones con patrones de uso parecidos (P3)."""

    pregunta = "P3"
    nombres: ClassVar[dict[str, str]] = {
        "baseline": "P3-baseline-municipio",
        "candidato": "P3-kmeans-perfil-horario",
    }

    def preparar(self) -> None:
        """Construye el perfil de uso de cada estación con el periodo de entrenamiento."""
        self.perfil = perfil_estacion(load_processed_travels(), load_processed_stations())
        municipio = self.perfil["nombre"].str.extract(r"\((\w+)-", expand=False)
        # Una estación sin prefijo en el nombre queda en su propio grupo en lugar de NaN
        self.perfil["municipio"] = municipio.fillna("SIN")
        self.X_train = self.perfil[VARIABLES_PERFIL_USO]
        # Mismo espacio escalado para medir baseline y candidato con la misma vara
        self.X_escalado = StandardScaler().fit_transform(self.X_train)

    def entrenar(self) -> None:
        """Prueba KMeans con cada k de `K_CANDIDATOS` y se queda con el de mayor silueta."""
        if self.tipo == "baseline":
            return
        ajustes, self.silueta_por_k = {}, {}
        for k in K_CANDIDATOS:
            ajustes[k] = Pipeline(
                [
                    ("escalado", StandardScaler()),
                    ("modelo", KMeans(n_clusters=k, n_init=10, random_state=self.semilla)),
                ]
            )
            grupos = ajustes[k].fit_predict(self.X_train)
            self.silueta_por_k[k] = silhouette_score(self.X_escalado, grupos)
        self.k = max(self.silueta_por_k, key=self.silueta_por_k.get)
        self.modelo = ajustes[self.k]

    def evaluar(self) -> dict:
        """Asigna los grupos y mide su separación. El baseline agrupa por municipio.

        Returns:
            `train_silhouette`, `train_davies_bouldin` y `train_calinski_harabasz`.
        """
        if self.modelo is None:
            self.perfil["grupo"] = self.perfil["municipio"]
        else:
            self.perfil["grupo"] = self.modelo.predict(self.X_train)
        return metricas_clustering(self.X_escalado, self.perfil["grupo"])

    def artefactos(self, carpeta: Path) -> list[Path]:
        """Guarda el mapa, el perfil horario, el grupo de cada estación y el resumen por grupo.

        El resumen incluye el % de usuarias y la edad mediana promedio de cada grupo, solo para
        describirlos. No hay datos de viajes individuales.

        Args:
            carpeta: Carpeta temporal del run.

        Returns:
            Las rutas de las gráficas y tablas.
        """
        estaciones = carpeta / "grupos_estaciones.csv"
        self.perfil[["estacion_id", "nombre", "latitud", "longitud", "grupo"]].to_csv(
            estaciones, index=False
        )
        resumen = carpeta / "resumen_grupos.csv"
        self.perfil.groupby("grupo").agg(
            estaciones=("estacion_id", "count"),
            pct_usuarias=("pct_usuarias", "mean"),
            edad_mediana=("edad_mediana", "mean"),
            viajes_por_dia_log=("viajes_por_dia_log", "mean"),
        ).round(3).to_csv(resumen)
        rutas = [
            plots.mapa_grupos(self.perfil, "grupo", carpeta / "mapa_grupos.png"),
            plots.perfil_horario(self.perfil, "grupo", carpeta / "perfil_horario.png"),
            estaciones,
            resumen,
        ]
        if self.modelo is not None:
            silueta = carpeta / "silueta_por_k.csv"
            pd.Series(self.silueta_por_k, name="silueta").rename_axis("k").to_csv(silueta)
            rutas.append(silueta)
        return rutas

    def hiperparametros(self) -> dict:
        """Devuelve el modelo, el k elegido y los k probados."""
        if self.modelo is None:
            return {"modelo": "municipio"}
        return {
            "modelo": "KMeans",
            "k": self.k,
            "k_probados": ",".join(map(str, K_CANDIDATOS)),
            "n_init": 10,
        }


EXPERIMENTOS = {"P1": ExperimentoDemanda, "P2": ExperimentoRiesgo, "P3": ExperimentoGrupos}


@app.command()
def main(
    pregunta: str = typer.Option("todas", help="P1, P2, P3 o todas"),
    modelo: str = typer.Option("ambos", help="baseline, candidato o ambos"),
) -> None:
    """Corre los experimentos pedidos y registra cada uno como un run en MLflow."""
    if pregunta not in (*EXPERIMENTOS, "todas"):
        raise typer.BadParameter(f"pregunta debe ser una de {[*EXPERIMENTOS, 'todas']}")
    if modelo not in ("baseline", "candidato", "ambos"):
        raise typer.BadParameter("modelo debe ser baseline, candidato o ambos")

    preguntas = list(EXPERIMENTOS) if pregunta == "todas" else [pregunta]
    tipos = ["baseline", "candidato"] if modelo == "ambos" else [modelo]
    for p in preguntas:
        for tipo in tipos:
            experimento = EXPERIMENTOS[p](tipo)
            run_id, metricas = experimento.ejecutar()
            resumen = "  ".join(f"{k}={v:.3f}" for k, v in metricas.items())
            logger.success(f"{experimento.nombres[tipo]}: {resumen} (run {run_id})")


if __name__ == "__main__":
    app()
