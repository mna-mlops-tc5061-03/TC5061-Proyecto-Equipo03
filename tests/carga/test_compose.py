"""Pruebas de configuración y seguridad sobre `docker-compose.yml` sin levantar contenedores."""

import re

import pytest
from tests.test_repositorio import PATRONES_CREDENCIALES
import yaml

from mi_bici.config import PROJ_ROOT

COMPOSE_FILE = PROJ_ROOT / "docker-compose.yml"
PALABRAS_SECRETO = ("password", "passwd", "pwd", "secret", "token", "api_key", "apikey")


@pytest.fixture(scope="module")
def compose() -> dict:
    """Carga `docker-compose.yml` con `yaml.safe_load`."""
    return yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))


class TestDockerCompose:
    def test_existen_servicios_mlflow_y_api(self, compose):
        """Verifica que el archivo define los servicios `mlflow` y `api`."""
        servicios = compose.get("services", {})
        assert "mlflow" in servicios
        assert "api" in servicios

    def test_api_depende_de_mlflow_con_service_healthy(self, compose):
        """Verifica que `api` espera a que `mlflow` esté sano y que `mlflow` tiene healthcheck."""
        servicios = compose["services"]
        assert "healthcheck" in servicios["mlflow"]
        assert "test" in servicios["mlflow"]["healthcheck"]

        dependencias = servicios["api"].get("depends_on", {})
        assert isinstance(dependencias, dict), "`depends_on` debe usar forma larga con `condition`"
        assert dependencias.get("mlflow", {}).get("condition") == "service_healthy"

    def test_puertos_publicados_solo_en_loopback(self, compose):
        """Verifica que ningún servicio expone puertos fuera de `127.0.0.1`."""
        puertos_revisados = 0
        for nombre, definicion in compose["services"].items():
            for puerto in definicion.get("ports", []):
                puertos_revisados += 1
                if isinstance(puerto, str):
                    assert puerto.startswith("127.0.0.1:"), (
                        f"El servicio `{nombre}` publica el puerto `{puerto}` fuera de 127.0.0.1"
                    )
                elif isinstance(puerto, dict):
                    assert puerto.get("host_ip") == "127.0.0.1", (
                        f"El servicio `{nombre}` no restringe `host_ip` a 127.0.0.1"
                    )
        assert puertos_revisados >= 2

    def test_compose_no_contiene_secretos(self, compose):
        """Verifica que `docker-compose.yml` no expone credenciales en texto ni en `environment`."""
        texto = COMPOSE_FILE.read_text(encoding="utf-8")
        for nombre, regex in PATRONES_CREDENCIALES.items():
            assert not re.search(regex, texto), (
                f"`docker-compose.yml` coincide con el patrón de credencial `{nombre}`"
            )

        for servicio, definicion in compose["services"].items():
            env = definicion.get("environment") or {}
            pares = (
                env.items()
                if isinstance(env, dict)
                else (item.split("=", 1) for item in env if "=" in item)
            )
            for clave, valor in pares:
                if valor is None or str(valor).startswith("${"):
                    continue
                clave_lower = str(clave).lower()
                assert not any(palabra in clave_lower for palabra in PALABRAS_SECRETO), (
                    f"Posible secreto literal en `{servicio}.environment.{clave}`"
                )
