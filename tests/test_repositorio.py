"""Revisa que el repositorio no exponga credenciales ni datos.

Lee los archivos versionados con `git ls-files`, así que la prueba falla si alguien agrega al
índice un `.env`, una base de MLflow, datos o un modelo, aunque todavía no haga commit.
"""

from pathlib import PurePosixPath
import re
import subprocess

import pytest

from mi_bici.config import PROJ_ROOT

# Archivos que nunca se versionan: secretos, bases de MLflow, datos y modelos serializados
EXTENSIONES_PROHIBIDAS = (".db", ".sqlite", ".parquet", ".csv", ".pkl", ".joblib", ".skops")
CARPETAS_PROHIBIDAS = ("mlruns", "mlartifacts")
# Lo único de data/ y models/ que sí va a Git
PERMITIDOS = {"data/manifest.json", "models/.gitkeep"}

# Rutas de ejemplo que el .gitignore debe ignorar y las que debe dejar pasar
IGNORADAS = (
    ".env",
    ".env.produccion",
    "data/raw/datos_abiertos_2026_09.csv",
    "data/processed/estaciones.parquet",
    "mlflow.db",
    "mlruns/0/meta.yaml",
    "mlartifacts/1/modelo/model.skops",
    "models/modelo.pkl",
    "reports/deepchecks/resumen.json",
)
NO_IGNORADAS = ("data/manifest.json", ".env.example", "models/.gitkeep")

# Patrones de credenciales que no deben aparecer en ningún archivo versionado. Sin \b al inicio
# y con \w* después, para que detecten nombres con prefijo como MLFLOW_TRACKING_PASSWORD
PATRONES_CREDENCIALES = {
    "clave": r"(?i)(password|passwd|pwd)\w*\s*[:=]\s*['\"]?[^\s'\"]{4,}",
    "token": r"(?i)(token|secret|api_?key)\w*\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{8,}",
    "llave de AWS": r"\bAKIA[0-9A-Z]{16}\b",
    "token de GitHub": r"\bgh[pousr]_[A-Za-z0-9]{36,}\b",
    "llave privada": r"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
}


def git(*args: str) -> subprocess.CompletedProcess:
    """Corre un comando de git en la raíz del proyecto."""
    return subprocess.run(  # nosec B603 B607: comando y argumentos fijos, sin entrada externa
        ["git", *args], cwd=PROJ_ROOT, capture_output=True, text=True, check=False
    )


@pytest.fixture(scope="module")
def versionados() -> list[str]:
    """Devuelve las rutas de los archivos versionados, incluidos los que están en el índice."""
    resultado = git("ls-files")
    if resultado.returncode != 0:
        pytest.skip("El proyecto no está en un repositorio de git")
    return resultado.stdout.splitlines()


def es_texto(ruta: str) -> bool:
    """Indica si un archivo es de texto: los binarios (imágenes, PDF) traen bytes nulos."""
    with open(PROJ_ROOT / ruta, "rb") as archivo:
        return b"\0" not in archivo.read(8192)


class TestArchivosVersionados:
    def test_no_hay_archivos_env(self, versionados):
        """Verifica que solo se versiona `.env.example` y ningún otro `.env`."""
        envs = [r for r in versionados if PurePosixPath(r).name.startswith(".env")]
        assert envs == [".env.example"]

    @pytest.mark.parametrize("extension", EXTENSIONES_PROHIBIDAS)
    def test_no_hay_datos_bases_ni_modelos(self, versionados, extension):
        """Verifica que no se versionan datos, bases de MLflow ni modelos serializados."""
        encontrados = [r for r in versionados if r.endswith(extension)]
        assert encontrados == []

    def test_no_hay_carpetas_de_mlflow(self, versionados):
        """Verifica que no se versionan las corridas ni los artefactos locales de MLflow."""
        encontrados = [r for r in versionados if r.split("/")[0] in CARPETAS_PROHIBIDAS]
        assert encontrados == []

    def test_de_data_y_models_solo_va_lo_permitido(self, versionados):
        """Verifica que de `data/` y `models/` solo se versionan el manifiesto y `.gitkeep`."""
        encontrados = [
            r for r in versionados if r.startswith(("data/", "models/")) and r not in PERMITIDOS
        ]
        assert encontrados == []


class TestGitignore:
    @pytest.mark.parametrize("ruta", IGNORADAS)
    def test_ignora_secretos_datos_y_salidas(self, ruta):
        """Verifica que el `.gitignore` ignora `ruta` aunque el archivo no exista."""
        assert git("check-ignore", "--no-index", "-q", ruta).returncode == 0

    @pytest.mark.parametrize("ruta", NO_IGNORADAS)
    def test_no_ignora_lo_que_si_se_versiona(self, ruta):
        """Verifica que el `.gitignore` deja pasar el manifiesto, `.env.example` y `.gitkeep`."""
        assert git("check-ignore", "--no-index", "-q", ruta).returncode == 1


class TestCredenciales:
    def test_env_example_solo_trae_valores_de_ejemplo(self):
        """Verifica que `.env.example` solo apunta al servidor local y no trae secretos."""
        lineas = (PROJ_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        variables = dict(
            linea.split("=", 1) for linea in lineas if linea.strip() and not linea.startswith("#")
        )
        assert variables == {"MLFLOW_TRACKING_URI": "http://127.0.0.1:5001"}

    @pytest.mark.parametrize("nombre", PATRONES_CREDENCIALES)
    def test_ningun_archivo_trae_credenciales(self, versionados, nombre):
        """Verifica que ningún archivo de texto versionado tiene el patrón `nombre`."""
        patron = re.compile(PATRONES_CREDENCIALES[nombre])
        encontrados = [
            ruta
            for ruta in versionados
            if (PROJ_ROOT / ruta).is_file()
            and es_texto(ruta)
            and patron.search((PROJ_ROOT / ruta).read_text(encoding="utf-8", errors="ignore"))
        ]
        assert encontrados == []
