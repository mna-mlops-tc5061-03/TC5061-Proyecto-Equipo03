"""Registra en MLflow los experimentos de P1, P2 y P3 con un formato común.

El registro es el mismo para las tres preguntas, así que vive aquí una sola vez y cada pregunta
solo define sus datos, su modelo y su evaluación.
"""

from abc import ABC, abstractmethod
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import ClassVar

import mlflow
import mlflow.sklearn

from mi_bici.config import (
    CORTE_PRUEBA,
    EQUIPO,
    EXPERIMENT_NAME,
    FECHA_FIN,
    FECHA_INICIO,
    MANIFEST_FILE,
    PROJ_ROOT,
    SEMILLA,
    TRACKING_URI,
)

TIPOS = ("baseline", "candidato")


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
    """Representa un run de MLflow para una pregunta y un tipo de modelo.

    Cada pregunta hereda de esta clase y define cómo prepara sus datos, entrena, evalúa y qué
    artefactos guarda. `ejecutar()` hace el resto y registra todo en MLflow siempre igual.

    Args:
        tipo: `"baseline"` o `"candidato"`.
        semilla: Semilla para que el run sea reproducible.
    """

    pregunta: str
    nombres: dict[str, str]  # nombre descriptivo del run para cada tipo
    # Tipos internos del modelo que skops (formato de MLflow) debe aceptar. Son seguros porque
    # el modelo lo genera este mismo código
    tipos_confiables: ClassVar[list[str]] = []

    def __init__(self, tipo: str = "candidato", semilla: int = SEMILLA):
        if tipo not in TIPOS:
            raise ValueError(f"tipo debe ser uno de {TIPOS}")
        self.tipo = tipo
        self.semilla = semilla
        self.modelo = None  # estimador de scikit-learn. El baseline es una regla y no tiene

    @abstractmethod
    def preparar(self) -> None:
        """Lee los datos procesados, construye las variables y separa entrenamiento y prueba.

        Debe dejar `self.X_train` con las variables de entrenamiento.
        """

    @abstractmethod
    def entrenar(self) -> None:
        """Ajusta el modelo candidato. El baseline no se entrena."""

    @abstractmethod
    def evaluar(self) -> dict:
        """Evalúa el modelo o la regla del baseline.

        Returns:
            Métricas con el conjunto en el nombre, por ejemplo `test_mae` o `train_silhouette`.
        """

    @abstractmethod
    def artefactos(self, carpeta: Path) -> list[Path]:
        """Guarda las gráficas o tablas del run.

        Args:
            carpeta: Carpeta temporal donde se guardan los archivos.

        Returns:
            Rutas de los archivos que se registran como artefactos.
        """

    @abstractmethod
    def hiperparametros(self) -> dict:
        """Devuelve el nombre del modelo y sus hiperparámetros para registrarlos."""

    def parametros(self) -> dict:
        """Reúne todo lo necesario para reproducir el run.

        Returns:
            Rango de fechas, corte de prueba, versión de los datos, semilla, filas de
            entrenamiento, modelo e hiperparámetros.
        """
        return {
            "fecha_inicio": FECHA_INICIO,
            "fecha_fin": FECHA_FIN,
            "corte_prueba": CORTE_PRUEBA,
            "data_version": version_datos(),
            "semilla": self.semilla,
            "filas_train": len(self.X_train),
            **self.hiperparametros(),
        }

    def ejecutar(self) -> tuple[str, dict]:
        """Corre el experimento completo y lo registra en MLflow.

        Returns:
            El Run ID y las métricas del run.
        """
        self.preparar()
        self.entrenar()
        metricas = self.evaluar()

        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=self.nombres[self.tipo]) as run:
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
                    skops_trusted_types=self.tipos_confiables,
                )
        return run.info.run_id, metricas
