"""Registra en MLflow los experimentos de P1, P2 y P3 con un formato común.

El registro es el mismo para las tres preguntas, así que vive aquí una sola vez y cada pregunta
solo define sus datos, su modelo y su evaluación.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import ClassVar

from loguru import logger
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mi_bici.config import (
    CORTE_PRUEBA,
    CORTE_VALIDACION,
    EQUIPO,
    EXPERIMENT_NAME,
    FECHA_FIN,
    FECHA_INICIO,
    MANIFEST_FILE,
    METRICA_SELECCION,
    PROJ_ROOT,
    SEMILLA,
    TRACKING_URI,
)


@dataclass(frozen=True)
class Modelo:
    """Representa una entrada del catálogo de modelos de una pregunta.

    Args:
        familia: Familia del modelo, o la regla que sigue el baseline.
        clase: Estimador de scikit-learn. `None` en el baseline, que es una regla y no se entrena.
        por_defecto: Hiperparámetros que se usan si el run no indica otros.
        escalar: Si el estimador va después de un `StandardScaler`, como los modelos lineales.
    """

    familia: str
    clase: type | None = None
    por_defecto: dict = field(default_factory=dict)
    escalar: bool = False


def _sin_fechas(valor):
    """Quita los campos `created_at` de un manifiesto, en todos sus niveles.

    Esos campos cambian en cada ejecución aunque los datos sean los mismos.

    Args:
        valor: Contenido del manifiesto o una parte de él.

    Returns:
        El mismo contenido sin los campos `created_at`.
    """
    if isinstance(valor, dict):
        return {k: _sin_fechas(v) for k, v in valor.items() if k != "created_at"}
    if isinstance(valor, list):
        return [_sin_fechas(v) for v in valor]
    return valor


def version_datos(ruta: Path = MANIFEST_FILE) -> str:
    """Calcula la versión de los datos a partir del manifiesto.

    Dos ingestas con los mismos datos dan la misma versión, así cada run queda ligado a los datos
    exactos con que se entrenó.

    Args:
        ruta: Ruta del manifiesto de datos.

    Returns:
        Los primeros 12 caracteres del hash del manifiesto, o `"sin_manifiesto"` si no existe.
    """
    if not ruta.exists():
        logger.warning(
            f"No existe {ruta.name}: el run se registra con data_version=sin_manifiesto"
        )
        return "sin_manifiesto"
    contenido = _sin_fechas(json.loads(ruta.read_text(encoding="utf-8")))
    texto = json.dumps(contenido, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:12]


def estado_git() -> dict:
    """Obtiene el commit actual y si hay cambios sin guardar.

    Returns:
        Diccionario con `git_commit` (o `"sin_git"`) y `git_dirty` (`"True"` o `"False"`).
    """

    def git(*args):
        return subprocess.run(
            ["git", *args], check=False, cwd=PROJ_ROOT, capture_output=True, text=True
        ).stdout.strip()

    return {
        "git_commit": git("rev-parse", "HEAD") or "sin_git",
        "git_dirty": str(bool(git("status", "--porcelain"))),
    }


class ExperimentoBase(ABC):
    """Representa un run de MLflow para una pregunta y un modelo de su catálogo.

    Cada pregunta hereda de esta clase, declara su catálogo y define cómo prepara sus datos,
    entrena, evalúa y qué artefactos guarda. `ejecutar()` hace el resto y registra todo en
    MLflow siempre igual. Agregar un modelo es agregar una entrada al catálogo.

    Args:
        modelo: Clave del catálogo, por ejemplo `"baseline"` o `"hgb"`.
        params: Hiperparámetros que reemplazan a los del catálogo.
        semilla: Semilla para que el run sea reproducible.
    """

    pregunta: str
    catalogo: ClassVar[dict[str, Modelo]]
    # Columna de fecha con que se separa la validación. P3 no tiene y se mide con todo el periodo
    columna_fecha: ClassVar[str | None] = None
    # Tipos internos del modelo que skops (formato de MLflow) debe aceptar. Son seguros porque
    # el modelo lo genera este mismo código
    tipos_confiables: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self, modelo: str = "baseline", params: dict | None = None, semilla: int = SEMILLA
    ):
        if modelo not in self.catalogo:
            raise ValueError(f"modelo debe ser uno de {list(self.catalogo)}")
        self.clave = modelo
        self.entrada = self.catalogo[modelo]
        self.params = {**self.entrada.por_defecto, **(params or {})}
        self.tipo = "baseline" if self.entrada.clase is None else "candidato"
        self.semilla = semilla
        self.modelo = None  # estimador de scikit-learn. El baseline es una regla y no tiene
        self.test = None  # conjunto de prueba. P3 no tiene

    def crear_modelo(self, **extra):
        """Crea el estimador del catálogo con la semilla y los hiperparámetros del run.

        Args:
            **extra: Hiperparámetros que se fijan solo en esta llamada, como el k de P3.

        Returns:
            El estimador sin entrenar, dentro de un `Pipeline` si la entrada pide escalar.
        """
        params = {**self.params, **extra}
        if "random_state" in self.entrada.clase().get_params():
            params["random_state"] = self.semilla
        estimador = self.entrada.clase(**params)
        if self.entrada.escalar:
            return Pipeline([("escalado", StandardScaler()), ("modelo", estimador)])
        return estimador

    @abstractmethod
    def preparar(self) -> None:
        """Lee los datos procesados, construye las variables y separa entrenamiento y prueba.

        Debe dejar `self.train`, `self.test` (o `None`) y `self.X_train` con las variables de
        entrenamiento. Sus primeras filas se
        guardan como ejemplo del modelo, así que no debe traer columnas de viajes individuales
        (como `genero` o `nacimiento`), solo variables agregadas.
        """

    @abstractmethod
    def entrenar(self, datos) -> None:
        """Ajusta el modelo candidato con `datos`. El baseline no se entrena.

        Args:
            datos: Filas de entrenamiento, antes de la validación o de la prueba.
        """

    @abstractmethod
    def evaluar(self, datos, prefijo: str) -> dict:
        """Evalúa el modelo o la regla del baseline con `datos`.

        Args:
            datos: Filas que se evalúan.
            prefijo: Conjunto evaluado: `"val"`, `"test"` o `"train"`.

        Returns:
            Métricas con el conjunto en el nombre, por ejemplo `val_mae` o `train_silhouette`.
        """

    @abstractmethod
    def artefactos(self, carpeta: Path) -> list[Path]:
        """Guarda las gráficas o tablas del run.

        Args:
            carpeta: Carpeta temporal donde se guardan los archivos.

        Returns:
            Rutas de los archivos que se registran como artefactos.
        """

    def hiperparametros(self) -> dict:
        """Devuelve la clave del catálogo, la familia y los hiperparámetros del estimador.

        Cada pregunta puede agregar los suyos llamando a este método.
        """
        datos = {"modelo": self.clave, "familia": self.entrada.familia}
        if self.modelo is not None:
            estimador = self.modelo[-1] if isinstance(self.modelo, Pipeline) else self.modelo
            datos |= {"clase": type(estimador).__name__, **estimador.get_params()}
        return datos

    def parametros(self) -> dict:
        """Reúne todo lo necesario para reproducir el run.

        Returns:
            Rango de fechas, cortes de validación y prueba, métrica de selección, versión de los
            datos, semilla, filas de entrenamiento, modelo e hiperparámetros.
        """
        return {
            "fecha_inicio": FECHA_INICIO,
            "fecha_fin": FECHA_FIN,
            "corte_validacion": CORTE_VALIDACION,
            "corte_prueba": CORTE_PRUEBA,
            "metrica_seleccion": METRICA_SELECCION[self.pregunta][0],
            "data_version": version_datos(),
            "semilla": self.semilla,
            "filas_train": len(self.X_train),
            **self.hiperparametros(),
        }

    def ejecutar(self) -> tuple[str, dict]:
        """Corre el experimento completo y lo registra en MLflow.

        Primero ajusta con los meses antes de `CORTE_VALIDACION` y mide en validación, que es con
        lo que se comparan los modelos. Después reentrena con todo el periodo de entrenamiento y
        mide en prueba, que no se usa para elegir nada.

        Returns:
            El Run ID y las métricas del run.
        """
        self.preparar()
        metricas = {}
        if self.columna_fecha:
            antes = self.train[self.columna_fecha] < CORTE_VALIDACION
            self.entrenar(self.train[antes])
            metricas |= self.evaluar(self.train[~antes], "val")
        self.entrenar(self.train)
        if self.test is None:
            metricas |= self.evaluar(self.train, "train")
        else:
            metricas |= self.evaluar(self.test, "test")

        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"{self.pregunta}-{self.clave}") as run:
            mlflow.set_tags(
                {"equipo": EQUIPO, "pregunta": self.pregunta, "tipo": self.tipo, **estado_git()}
            )
            mlflow.log_params(self.parametros())
            mlflow.log_metrics(metricas)
            with tempfile.TemporaryDirectory() as carpeta:
                for ruta in self.artefactos(Path(carpeta)):
                    mlflow.log_artifact(str(ruta))
            if self.modelo is not None:
                mlflow.sklearn.log_model(
                    self.modelo,
                    name="model",
                    input_example=self.X_train.head(3),
                    skops_trusted_types=list(self.tipos_confiables),
                )
        return run.info.run_id, metricas
