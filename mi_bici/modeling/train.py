"""Corre los experimentos de P1, P2 y P3 y los registra en MLflow.

Cada pregunta tiene un catálogo de modelos. Un run es una pregunta y un modelo del catálogo.

python -m mi_bici.modeling.train
python -m mi_bici.modeling.train --question p2 --model hgb --params "learning_rate=0.05"
"""

import ast
from pathlib import Path
from typing import ClassVar
import warnings

from loguru import logger
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import silhouette_score
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
from mi_bici.modeling.base import ExperimentoBase, Modelo
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


class ExperimentoDemanda(ExperimentoBase):
    """Pronostica los viajes que iniciarán mañana en cada estación (P1)."""

    pregunta = "P1"
    catalogo: ClassVar[dict[str, Modelo]] = {
        "baseline": Modelo("semana_anterior"),
        "hgb": Modelo("gradient_boosting", HistGradientBoostingRegressor),
    }
    tipos_confiables = ARBOLES_HGB
    columna_fecha = "fecha"

    def preparar(self) -> None:
        """Construye la demanda diaria y la separa en entrenamiento y prueba por fecha."""
        demanda = demanda_diaria(load_processed_travels(), load_processed_stations())
        prueba = demanda["fecha"] >= CORTE_PRUEBA
        self.train, self.test = demanda[~prueba], demanda[prueba]
        self.X_train = self.train[VARIABLES_P1]

    def entrenar(self, datos) -> None:
        """Ajusta el modelo del catálogo con `datos`. El baseline no se entrena."""
        if self.tipo == "candidato":
            self.modelo = self.crear_modelo()
            self.modelo.fit(datos[VARIABLES_P1], datos["viajes"])

    def evaluar(self, datos, prefijo: str) -> dict:
        """Evalúa con `datos`. El baseline predice los viajes del mismo día de la semana anterior.

        Returns:
            `<prefijo>_mae`, `<prefijo>_rmse` y `<prefijo>_r2`.
        """
        if self.modelo is None:
            self.pred = datos["viajes_lag_7"].to_numpy()
        else:
            self.pred = self.modelo.predict(datos[VARIABLES_P1])
        return metricas_regresion(datos["viajes"], self.pred, prefijo)

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
        """Devuelve el modelo, sus hiperparámetros y, en los candidatos, sus variables."""
        if self.modelo is None:
            return super().hiperparametros()
        return {**super().hiperparametros(), "variables": ",".join(VARIABLES_P1)}


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
    columna_fecha = "bloque"

    def preparar(self) -> None:
        """Construye el flujo por bloque y la etiqueta de riesgo, y separa por fecha."""
        flujo = flujo_por_bloque(load_processed_travels(), load_processed_stations())
        riesgo = etiqueta_riesgo(flujo, UMBRAL_RIESGO)
        prueba = riesgo["bloque"] >= CORTE_PRUEBA
        self.train, self.test = riesgo[~prueba], riesgo[prueba]
        self.X_train = self.train[VARIABLES_P2]

    def entrenar(self, datos) -> None:
        """Ajusta el modelo del catálogo con `datos`. El baseline no se entrena."""
        if self.tipo == "candidato":
            self.modelo = self.crear_modelo()
            self.modelo.fit(datos[VARIABLES_P2], datos["riesgo"])

    def evaluar(self, datos, prefijo: str) -> dict:
        """Evalúa con `datos`. El baseline repite el riesgo del mismo bloque la semana anterior.

        Returns:
            `<prefijo>_recall`, `<prefijo>_precision`, `<prefijo>_f1`,
            `<prefijo>_average_precision` y la proporción de bloques en riesgo
            (`train_tasa_riesgo`, `<prefijo>_tasa_riesgo`).
        """
        if self.modelo is None:
            self.pred = self.puntaje = datos["riesgo_semana_anterior"].to_numpy()
        else:
            self.pred = self.modelo.predict(datos[VARIABLES_P2])
            self.puntaje = self.modelo.predict_proba(datos[VARIABLES_P2])[:, 1]
        metricas = metricas_clasificacion(datos["riesgo"], self.pred, self.puntaje, prefijo)
        # La tasa real de riesgo justifica el umbral y el uso de average precision
        metricas["train_tasa_riesgo"] = self.train["riesgo"].mean()
        metricas[f"{prefijo}_tasa_riesgo"] = datos["riesgo"].mean()
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
        datos = {
            **super().hiperparametros(),
            "umbral_riesgo": UMBRAL_RIESGO,
            "bloque_horas": BLOQUE_HORAS,
        }
        if self.modelo is not None:
            datos["variables"] = ",".join(VARIABLES_P2)
        return datos


class ExperimentoGrupos(ExperimentoBase):
    """Agrupa las estaciones con patrones de uso parecidos (P3)."""

    pregunta = "P3"
    catalogo: ClassVar[dict[str, Modelo]] = {
        "baseline": Modelo("municipio"),
        "kmeans": Modelo("kmeans", KMeans, por_defecto={"n_init": 10}, escalar=True),
    }

    def preparar(self) -> None:
        """Construye el perfil de uso de cada estación con el periodo de entrenamiento."""
        self.perfil = perfil_estacion(load_processed_travels(), load_processed_stations())
        municipio = self.perfil["nombre"].str.extract(r"\((\w+)-", expand=False)
        # Una estación sin prefijo en el nombre queda en su propio grupo en lugar de NaN
        self.perfil["municipio"] = municipio.fillna("SIN")
        self.train = self.perfil
        self.X_train = self.perfil[VARIABLES_PERFIL_USO]
        # Mismo espacio escalado para medir baseline y candidato con la misma vara
        self.X_escalado = StandardScaler().fit_transform(self.X_train)

    def entrenar(self, datos) -> None:
        """Ajusta el modelo con cada k y se queda con el de mayor silueta.

        Prueba los k de `K_CANDIDATOS`, o solo `n_clusters` si el run lo indica.
        """
        if self.tipo == "baseline":
            return
        self.k_probados = (
            [self.params["n_clusters"]] if "n_clusters" in self.params else K_CANDIDATOS
        )
        ajustes, self.silueta_por_k = {}, {}
        for k in self.k_probados:
            ajustes[k] = self.crear_modelo(n_clusters=k)
            grupos = ajustes[k].fit_predict(datos[VARIABLES_PERFIL_USO])
            self.silueta_por_k[k] = silhouette_score(self.X_escalado, grupos)
        self.k = max(self.silueta_por_k, key=self.silueta_por_k.get)
        self.modelo = ajustes[self.k]

    def evaluar(self, datos, prefijo: str) -> dict:
        """Asigna los grupos y mide su separación. El baseline agrupa por municipio.

        Returns:
            `<prefijo>_silhouette`, `<prefijo>_davies_bouldin` y `<prefijo>_calinski_harabasz`.
        """
        if self.modelo is None:
            self.perfil["grupo"] = datos["municipio"]
        else:
            self.perfil["grupo"] = self.modelo.predict(datos[VARIABLES_PERFIL_USO])
        return metricas_clustering(self.X_escalado, self.perfil["grupo"], prefijo)

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
        """Devuelve el modelo, sus hiperparámetros, el k elegido y los k probados."""
        if self.modelo is None:
            return super().hiperparametros()
        return {
            **super().hiperparametros(),
            "k": self.k,
            "k_probados": ",".join(map(str, self.k_probados)),
        }


EXPERIMENTOS = {"P1": ExperimentoDemanda, "P2": ExperimentoRiesgo, "P3": ExperimentoGrupos}


def leer_params(texto: str) -> dict:
    """Convierte `"k=v,..."` en un diccionario de hiperparámetros.

    Los números, booleanos y `None` se leen con su tipo, y el resto queda como texto.

    Args:
        texto: Hiperparámetros separados por comas. Vacío si no se indica ninguno.

    Returns:
        Diccionario con los hiperparámetros.
    """
    params = {}
    for par in filter(None, texto.split(",")):
        clave, valor = (parte.strip() for parte in par.split("=", 1))
        try:
            params[clave] = ast.literal_eval(valor)
        except (ValueError, SyntaxError):
            params[clave] = valor
    return params


@app.command()
def main(
    question: str = typer.Option("todas", help="p1, p2, p3 o todas"),
    model: str = typer.Option("todos", help="Clave del catálogo de la pregunta o todos"),
    params: str = typer.Option("", help='Hiperparámetros "k=v,..." que reemplazan al catálogo'),
) -> None:
    """Corre los modelos pedidos y registra cada uno como un run en MLflow."""
    pregunta = question.upper()
    if pregunta not in (*EXPERIMENTOS, "TODAS"):
        raise typer.BadParameter(f"question debe ser una de {[*EXPERIMENTOS, 'todas']}")
    if pregunta == "TODAS" and (model != "todos" or params):
        raise typer.BadParameter("model y params requieren una sola pregunta")

    for p in EXPERIMENTOS if pregunta == "TODAS" else [pregunta]:
        clase = EXPERIMENTOS[p]
        claves = list(clase.catalogo) if model == "todos" else [model]
        for clave in claves:
            experimento = clase(clave, leer_params(params))
            run_id, metricas = experimento.ejecutar()
            resumen = "  ".join(f"{k}={v:.3f}" for k, v in metricas.items())
            logger.success(f"{p}-{clave}: {resumen} (run {run_id})")


if __name__ == "__main__":
    app()
