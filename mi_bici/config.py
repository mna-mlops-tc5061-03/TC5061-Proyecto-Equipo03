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
ESTACIONES_METADATA_FILE = PROCESSED_DATA_DIR / "estaciones_metadata.json"
VIAJES_METADATA_FILE = PROCESSED_DATA_DIR / "viajes_metadata.json"

# Diccionario de columnas para limpieza de viajes
COLUMN_PATTERNS_REGEX = {
    "viaje_id": r"^viaje",
    "usuario_id": r"^usuario",
    "genero": r"^genero",
    "inicio": r"^inicio",
    "fin": r"^fin",
    "origen_id": r"^origen",
    "destino_id": r"^destino",
    "nacimiento": r"naci",
}
TRAVEL_COLUMNS_TO_KEEP = [
    "genero",
    "inicio",
    "fin",
    "origen_id",
    "destino_id",
    "nacimiento",
]

# Diccionario de columnas para limpieza de estaciones
STATION_PATTERNS_REGEX = {
    "estacion_id": r"^(id|estacion)",
    "nombre": r"^(name|nombre)",
    "latitud": r"^(lat|latitude|latitud)",
    "longitud": r"^(lon|long|longitude|longitud)",
    "capacidad": r"^(dpcapacity|capacidad|capacity)",
}
STATION_COLUMNS_TO_KEEP = [
    "estacion_id",
    "nombre",
    "latitud",
    "longitud",
    "capacidad",
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
URL_STATIONS = f"{URL_BASE}/1118/nomenclatura_2026_09.csv"
URL_NOMENCLATURA = URL_STATIONS

# Periodo: los 12 meses completos más recientes. Prueba en los dos últimos para no usar
# información del futuro al entrenar
FECHA_INICIO = "2025-10-01"
FECHA_FIN = "2026-09-30"
CORTE_PRUEBA = "2026-08-01"
# Validación: los dos meses antes de la prueba. Los modelos se comparan y eligen con ella, y la
# prueba solo confirma el resultado
CORTE_VALIDACION = "2026-06-01"

# Métrica con que se elige el modelo de cada pregunta y si mayor es mejor. Un candidato se acepta
# si le gana al baseline en esta métrica en validación y también en prueba. P3 no tiene prueba y
# se mide con el periodo de entrenamiento
METRICA_SELECCION = {
    "P1": ("val_mae", False),
    "P2": ("val_average_precision", True),
    "P3": ("train_silhouette", True),
}

# Limpieza: viajes de menos de 2 min que salen y regresan a la misma estación son errores al
# sacar la bici. Más de 12 h se considera irregular en los términos de MiBici
DURACION_MINIMA_MIN = 2
DURACION_MAXIMA_MIN = 12 * 60

# Limpieza: Valor para rellenar los nulos en genero: NE=No especificado
GENERO_NULL_VALUE = "NE"

# Limpieza: Valor mínimo teorico a aceptar en anio de nacimiento
ANIO_NACIMIENTO_MINIMO = 1920

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
# Supuesto: un camión de rebalanceo alcanza a visitar 10 estaciones en un bloque de 3 horas
# (~18 min por estación entre traslado y carga o descarga de bicicletas)
CAPACIDAD_REBALANCEO = 10
# Umbral de decisión inicial del clasificador en P2 (se calibra en validación con CAPACIDAD_REBALANCEO)
UMBRAL_DECISION_P2 = 0.5

# P3: valores de k que se prueban y días mínimos con viajes para que el perfil sea estable
K_CANDIDATOS = list(range(2, 9))
MIN_DIAS_PERFIL = 30

# Variables que usa cada modelo. Todas vienen de periodos anteriores al que se predice
VARIABLES_P1 = [
    "viajes_lag_1",
    "viajes_lag_7",
    "viajes_media_7",
    "dia_semana",
    "festivo",
    "capacidad",
]
VARIABLES_P2 = [
    "flujo_neto_lag_1",
    "flujo_neto_lag_2",
    "salidas_lag_1",
    "llegadas_lag_1",
    "riesgo_semana_anterior",
    "hora",
    "dia_semana",
    "festivo",
    "capacidad",
]

# Columnas que crea features.py y que leen los experimentos: la fecha que separa los conjuntos,
# el objetivo y la que el baseline usa como predicción
COLUMNA_FECHA_P1 = "fecha"
OBJETIVO_P1 = "viajes"
BASELINE_P1 = "viajes_lag_7"
COLUMNA_FECHA_P2 = "bloque"
OBJETIVO_P2 = "riesgo"
BASELINE_P2 = "riesgo_semana_anterior"

SEMILLA = 42

# MLflow: local por defecto. Para un servidor compartido basta con MLFLOW_TRACKING_URI en .env
EQUIPO = "equipo03"
EXPERIMENT_NAME = f"mibici-{EQUIPO}"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJ_ROOT / 'mlflow.db'}")

# Model Registry: un modelo registrado por pregunta. El alias champion marca la versión en
# producción y challenger la siguiente mejor
MODELOS_REGISTRO = {
    "P1": f"mibici-{EQUIPO}-p1-demanda",
    "P2": f"mibici-{EQUIPO}-p2-riesgo",
    "P3": f"mibici-{EQUIPO}-p3-grupos",
}
ALIAS_PRODUCCION = "champion"
ALIAS_RESPALDO = "challenger"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
