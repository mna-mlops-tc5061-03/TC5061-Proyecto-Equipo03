"""Define el experimento de P3: grupos de estaciones con patrones de uso parecidos."""

from pathlib import Path
from typing import ClassVar

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.cluster import Birch, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from mi_bici import plots
from mi_bici.config import K_CANDIDATOS, SEMILLAS_ESTABILIDAD
from mi_bici.dataset import load_processed_stations, load_processed_travels
from mi_bici.features import VARIABLES_PERFIL_USO, perfil_estacion
from mi_bici.modeling.base import ExperimentoBase, Modelo
from mi_bici.modeling.evaluate import metricas_clustering


class ExperimentoGrupos(ExperimentoBase):
    """Agrupa las estaciones con patrones de uso parecidos (P3)."""

    pregunta = "P3"
    catalogo: ClassVar[dict[str, Modelo]] = {
        "baseline": Modelo("municipio"),
        "kmeans": Modelo(
            "kmeans",
            KMeans,
            por_defecto={"n_init": 10},
            escalar=True,
            cuadricula={"n_clusters": K_CANDIDATOS},
        ),
        "birch": Modelo(
            "birch",
            Birch,
            escalar=True,
            cuadricula={"n_clusters": K_CANDIDATOS},
            serializacion="cloudpickle",
        ),
        "gmm": Modelo(
            "mezcla",
            GaussianMixture,
            escalar=True,
            cuadricula={"n_components": K_CANDIDATOS},
        ),
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

    def _param_k(self) -> str:
        """Devuelve el nombre del parámetro del número de grupos según el estimador."""
        if self.entrada.clase is not None and "n_components" in self.entrada.clase().get_params():
            return "n_components"
        return "n_clusters"

    def crear_modelo(self, **extra):
        """Crea el estimador del catálogo traduciendo parámetros si es necesario."""
        param_k = self._param_k()
        if param_k == "n_components":
            if "n_clusters" in extra:
                extra["n_components"] = extra.pop("n_clusters")
            if "n_clusters" in self.params:
                self.params["n_components"] = self.params.pop("n_clusters")
        return super().crear_modelo(**extra)

    def entrenar(self, datos) -> None:
        """Ajusta el modelo con cada k y se queda con el de mayor silueta.

        Prueba los k de `K_CANDIDATOS`, o solo el k indicado si el run lo define.
        """
        if self.tipo == "baseline":
            return
        param_k = self._param_k()
        k_pedido = (
            self.params.get(param_k)
            or self.params.get("n_clusters")
            or self.params.get("n_components")
        )
        self.k_probados = [k_pedido] if k_pedido else K_CANDIDATOS
        ajustes, self.silueta_por_k = {}, {}
        for k in self.k_probados:
            ajustes[k] = self.crear_modelo(**{param_k: k})
            grupos = ajustes[k].fit_predict(datos[VARIABLES_PERFIL_USO])
            self.silueta_por_k[k] = silhouette_score(self.X_escalado, grupos)
        self.k = max(self.silueta_por_k, key=self.silueta_por_k.get)
        self.modelo = ajustes[self.k]

        # Detectar si el k elegido cayó en el borde del rango de candidatos
        self.k_en_borde = self.k in (min(K_CANDIDATOS), max(K_CANDIDATOS))
        if self.k_en_borde:
            logger.warning(
                f"P3-{self.clave}: el k elegido ({self.k}) está en el borde de "
                f"K_CANDIDATOS ({min(K_CANDIDATOS)}..{max(K_CANDIDATOS)}). "
                "Considera ampliar el rango de búsqueda."
            )

    def evaluar(self, datos, prefijo: str) -> dict:
        """Asigna los grupos y mide su separación. El baseline agrupa por municipio.

        En los candidatos (salvo los hijos de la búsqueda) mide además la estabilidad de la
        silueta y el tamaño del grupo menor al ajustar con `SEMILLAS_ESTABILIDAD`.

        Returns:
            `<prefijo>_silhouette`, `<prefijo>_davies_bouldin`, `<prefijo>_calinski_harabasz` y,
            en los candidatos, `<prefijo>_silhouette_std` y `<prefijo>_grupo_menor_min`.
        """
        if not hasattr(self, "perfil") or self.perfil is None:
            self.perfil = datos.copy()
        if self.modelo is None:
            self.perfil["grupo"] = datos["municipio"]
        else:
            self.perfil["grupo"] = self.modelo.predict(datos[VARIABLES_PERFIL_USO])
        metricas = metricas_clustering(self.X_escalado, self.perfil["grupo"], prefijo)
        if self.modelo is not None and self.etapa != "search":
            metricas |= self.estabilidad(datos, prefijo)
        return metricas

    def estabilidad(self, datos, prefijo: str = "train") -> dict:
        """Mide la estabilidad con `SEMILLAS_ESTABILIDAD` para el `k` elegido.

        Birch no usa semilla, así que su desviación es `0.0`.
        """
        usa_semilla = (
            self.entrada.clase is not None and "random_state" in self.entrada.clase().get_params()
        )
        if not usa_semilla:
            grupos = self.perfil["grupo"]
            _, conteos = np.unique(grupos, return_counts=True)
            return {
                f"{prefijo}_silhouette_std": 0.0,
                f"{prefijo}_grupo_menor_min": float(conteos.min() / len(grupos)),
            }

        param_k = self._param_k()
        siluetas, menores = [], []
        for semilla in SEMILLAS_ESTABILIDAD:
            modelo = self.crear_modelo(**{param_k: self.k, "random_state": semilla})
            grupos = modelo.fit_predict(datos[VARIABLES_PERFIL_USO])
            siluetas.append(silhouette_score(self.X_escalado, grupos))
            _, conteos = np.unique(grupos, return_counts=True)
            menores.append(conteos.min() / len(grupos))
        return {
            f"{prefijo}_silhouette_std": float(np.std(siluetas)),
            f"{prefijo}_grupo_menor_min": float(min(menores)),
        }

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

    def etiquetas(self) -> dict:
        """Agrega la etiqueta k_en_borde si el k elegido cayó en el extremo del catálogo."""
        tags = super().etiquetas()
        if self.modelo is not None and hasattr(self, "k_en_borde"):
            tags["k_en_borde"] = str(self.k_en_borde)
        return tags
