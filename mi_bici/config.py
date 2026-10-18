import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Salidas de la ingesta. El manifiesto es lo único de data/ que se versiona en Git
VIAJES_DIR = PROCESSED_DATA_DIR / "viajes"
ESTACIONES_FILE = PROCESSED_DATA_DIR / "estaciones.parquet"
MANIFEST_FILE = DATA_DIR / "manifest.json"

# Diccionario de columnas para limpieza
COLUMN_PATTERNS_REGEX = {
    "viaje": r"^viaje",
    "usuario": r"^usuario",
    "genero": r"^genero",
    "inicio": r"^inicio",
    "fin": r"^fin",
    "origen": r"^origen",
    "destino": r"^destino",
    "nacimiento": r"naci",
}
TRAVEL_COLUMNS_TO_KEEP = [
    "genero",
    "inicio",
    "fin",
    "origen",
    "destino",
    "nacimiento",
]

# Portal de datos abiertos de MiBici. Cada archivo tiene un ID propio en la URL, por eso se
# fijan aquí en lugar de leer la página, que puede cambiar
URL_BASE = "https://mibici.net/site/assets/files"
URLS_VIAJES = {
    "2025-10": f"{URL_BASE}/115104/datos_abiertos_2025_10.csv",
    "2025-11": f"{URL_BASE}/116773/datos_abiertos_2025_11.csv",
    "2025-12": f"{URL_BASE}/118153/datos_abiertos_2025_12.csv",
    "2026-01": f"{URL_BASE}/120163/datos_abiertos_2026_01.csv",
    "2026-02": f"{URL_BASE}/121761/datos_abiertos_2026_02.csv",
    "2026-03": f"{URL_BASE}/123236/datos_abiertos_2026_03.csv",
    "2026-04": f"{URL_BASE}/124927/datos_abiertos_2026_04.csv",
    "2026-05": f"{URL_BASE}/125943/datos_abiertos_2026_05.csv",
    "2026-06": f"{URL_BASE}/127734/datos_abiertos_2026_06.csv",
    "2026-07": f"{URL_BASE}/129006/datos_abiertos_2026_07.csv",
    "2026-08": f"{URL_BASE}/130362/datos_abiertos_2026_08.csv",
    "2026-09": f"{URL_BASE}/131825/datos_abiertos_2026_09.csv",
}
URL_NOMENCLATURA = f"{URL_BASE}/1118/nomenclatura_2026_09.csv"

# Periodo: los 12 meses completos más recientes. Prueba en los dos últimos para no usar
# información del futuro al entrenar
FECHA_INICIO = "2025-10-01"
FECHA_FIN = "2026-09-30"
CORTE_PRUEBA = "2026-08-01"

# Limpieza: viajes de menos de 2 min que salen y regresan a la misma estación son errores al
# sacar la bici. Más de 12 h se considera irregular en los términos de MiBici
DURACION_MINIMA_MIN = 2
DURACION_MAXIMA_HORAS = 12

# Maximo umbral de filas a remover, en caso de ser mayor se deja un log error para su atencion
UMBRAL_FILAS_A_REMOVER = 0.05

# Días de descanso obligatorio (art. 74 de la Ley Federal del Trabajo) dentro del periodo
FESTIVOS = [
    "2025-11-17",
    "2025-12-25",
    "2026-01-01",
    "2026-02-02",
    "2026-03-16",
    "2026-05-01",
    "2026-09-16",
]

# P2: una estación está en riesgo si en un bloque el flujo neto mueve al menos esta fracción
# de su capacidad
BLOQUE_HORAS = 3
UMBRAL_RIESGO = 0.4

# P3: valores de k que se prueban y días mínimos con viajes para que el perfil sea estable
K_CANDIDATOS = list(range(3, 9))
MIN_DIAS_PERFIL = 30

SEMILLA = 42

# MLflow: local por defecto. Para un servidor compartido basta con MLFLOW_TRACKING_URI en .env
EQUIPO = "equipo03"
EXPERIMENT_NAME = f"mibici-{EQUIPO}"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJ_ROOT / 'mlflow.db'}")

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
