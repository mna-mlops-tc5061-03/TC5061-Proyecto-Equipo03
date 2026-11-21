"""CPU y memoria de los contenedores del sistema mientras corre la prueba de carga.

Toma muestras seguidas con `docker stats --no-stream` (o `podman stats`) de los contenedores del
proyecto de Compose y las agrega a `reports/carga/recursos.csv`, con la misma marca de tiempo
(epoch) que `locust_stats_history.csv`. Cada muestra tarda unos 2 s, lo que Docker necesita
para calcular el % de CPU entre dos lecturas. Al terminar guarda en `maquina.json` el hardware de la
máquina, el del motor de contenedores y los límites de cada contenedor, porque el costo se
extrapola a otra instancia. No guarda el usuario ni el nombre de la máquina.

make carga lo corre en segundo plano y lo detiene al terminar Locust. A mano:

    python -m mi_bici.carga.recursos --duracion 300
"""

import csv
import json
import os
import platform
import re
import signal
import subprocess
import time

from loguru import logger
import typer

from mi_bici.config import (
    CARGA_DIR,
    ESCALONES_CARGA,
    INTERVALO_RECURSOS,
    PROJ_ROOT,
    PROYECTO_COMPOSE,
)

RECURSOS_CSV = CARGA_DIR / "recursos.csv"
MAQUINA_JSON = CARGA_DIR / "maquina.json"
COLUMNAS = ["epoch", "contenedor", "cpu_pct", "mem_mib"]
MOTORES = ("docker", "podman")
# Plantilla que entienden docker stats y podman stats: nombre, CPU y memoria
FORMATO_STATS = "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
# Docker reporta la memoria en unidades binarias (MiB) y Podman en decimales (MB)
BYTES_POR_UNIDAD = {
    "b": 1,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
}
MIB = 1024**2
GIB = 1024**3

app = typer.Typer()


def a_mib(memoria: str) -> float:
    """Convierte la memoria usada de `MemUsage` ("75.3MiB / 3.8GiB" o "75.3MB / 8.2GB") a MiB."""
    usada = memoria.split("/")[0].strip()
    partes = re.fullmatch(r"([\d.]+)\s*([A-Za-z]+)", usada)
    if partes is None or partes.group(2).lower() not in BYTES_POR_UNIDAD:
        raise ValueError(f"Memoria no reconocida: {memoria!r}")
    return float(partes.group(1)) * BYTES_POR_UNIDAD[partes.group(2).lower()] / MIB


def servicio(nombre: str, proyecto: str = PROYECTO_COMPOSE) -> str | None:
    """Servicio de Compose del contenedor (`mibici-api-1` → `api`), o None si es de otro proyecto."""
    partes = re.fullmatch(rf"{re.escape(proyecto)}[-_](.+)[-_]\d+", nombre)
    return partes.group(1) if partes else None


def leer_stats(salida: str, proyecto: str = PROYECTO_COMPOSE) -> dict[str, dict]:
    """Por contenedor del proyecto: servicio, % de CPU (100 = un núcleo) y MiB de memoria."""
    medidas = {}
    for linea in salida.splitlines():
        try:
            nombre, cpu, memoria = (campo.strip() for campo in linea.split("\t"))
            nombre_servicio = servicio(nombre, proyecto)
            if nombre_servicio is not None:
                medidas[nombre] = {
                    "contenedor": nombre_servicio,
                    "cpu_pct": float(cpu.rstrip("%")),
                    "mem_mib": round(a_mib(memoria), 1),
                }
        except ValueError:
            # Un contenedor que arranca o se detiene puede reportar "--"
            continue
    return medidas


def _correr(*comando: str) -> str:
    """Salida de un comando, o "" si falla (por ejemplo, si Docker no está corriendo)."""
    # bandit avisa por subprocess, pero se llama sin shell y con argumentos del código
    try:
        resultado = subprocess.run(  # nosec B603
            list(comando), check=False, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return resultado.stdout if resultado.returncode == 0 else ""


def limites(motor: str, nombres: list[str]) -> dict[str, dict]:
    """CPU y memoria máximas de cada contenedor (None si no tiene límite)."""
    formato = "{{.Name}}\t{{.HostConfig.NanoCpus}}\t{{.HostConfig.Memory}}"
    resultado = {}
    for linea in _correr(motor, "inspect", "--format", formato, *nombres).splitlines():
        nombre, nano_cpus, memoria = linea.split("\t")
        nombre_servicio = servicio(nombre.lstrip("/"))
        if nombre_servicio is not None:
            resultado[nombre_servicio] = {
                "cpus": int(nano_cpus) / 1e9 or None,
                "mem_gib": round(int(memoria) / GIB, 2) or None,
            }
    return resultado


def _modelo_cpu() -> str:
    if platform.system() == "Darwin":
        return _correr("sysctl", "-n", "machdep.cpu.brand_string").strip()
    try:
        with open("/proc/cpuinfo") as cpuinfo:
            for linea in cpuinfo:
                if linea.startswith("model name"):
                    return linea.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def maquina(motor: str, limites_contenedores: dict) -> dict:
    """Hardware de la máquina y del motor de contenedores, sin usuario ni nombre de la máquina."""
    info = {
        "sistema": platform.system(),
        "arquitectura": platform.machine(),
        "cpu": _modelo_cpu(),
        "nucleos": os.cpu_count(),
        "ram_gib": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GIB, 1),
    }
    # En macOS los contenedores corren en una máquina virtual con sus propios CPU y memoria
    if motor == "docker":
        formato = "{{.NCPU}}\t{{.MemTotal}}\t{{.ServerVersion}}"
    else:
        formato = "{{.Host.CPUs}}\t{{.Host.MemTotal}}\t{{.Version.Version}}"
    salida = _correr(motor, "info", "--format", formato).strip()
    if salida:
        cpus, memoria, version = salida.split("\t")
        info["motor"] = {
            "nombre": motor,
            "version": version,
            "cpus": int(cpus),
            "ram_gib": round(int(memoria) / GIB, 1),
        }
    info["contenedores"] = limites_contenedores
    return info


@app.command()
def main(
    motor: str = typer.Option("docker", help="docker o podman"),
    duracion: float = typer.Option(
        sum(segundos for _, segundos in ESCALONES_CARGA) + 60,
        help="Segundos máximos de muestreo, por si nadie lo detiene",
    ),
) -> None:
    """Muestrea CPU y memoria de los contenedores hasta que lo detengan o pase la duración."""
    if motor not in MOTORES:
        raise typer.BadParameter(f"El motor debe ser uno de {MOTORES}")

    # make carga lo detiene con SIGTERM; a mano, con Ctrl+C. Termina la muestra en curso y sale
    detener = False

    def _detener(*_):
        nonlocal detener
        detener = True

    signal.signal(signal.SIGTERM, _detener)
    signal.signal(signal.SIGINT, _detener)

    CARGA_DIR.mkdir(parents=True, exist_ok=True)
    ruta = RECURSOS_CSV.relative_to(PROJ_ROOT)
    logger.info(f"Muestreo de {motor} stats en {ruta} (máximo {duracion:.0f} s)")
    fin = time.monotonic() + duracion
    muestras, servicios, limites_contenedores = 0, set(), {}
    with open(RECURSOS_CSV, "w", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS)
        escritor.writeheader()
        while not detener and time.monotonic() < fin:
            inicio = time.monotonic()
            medidas = leer_stats(_correr(motor, "stats", "--no-stream", "--format", FORMATO_STATS))
            epoch = int(time.time())
            for medida in medidas.values():
                escritor.writerow({"epoch": epoch, **medida})
                servicios.add(medida["contenedor"])
            archivo.flush()
            if medidas:
                muestras += 1
                if not limites_contenedores:
                    limites_contenedores = limites(motor, list(medidas))
            time.sleep(max(0.0, INTERVALO_RECURSOS - (time.monotonic() - inicio)))

    MAQUINA_JSON.write_text(
        json.dumps(maquina(motor, limites_contenedores), indent=2, ensure_ascii=False) + "\n"
    )
    if servicios:
        logger.success(f"{muestras} muestras de {', '.join(sorted(servicios))} en {ruta}")
    else:
        logger.warning(
            f"No hubo contenedores de {PROYECTO_COMPOSE} ({motor} stats), así que {ruta} quedó "
            "vacío. Para medir los contenedores, levanta la API con make sistema"
        )


if __name__ == "__main__":
    app()
