from pathlib import Path
import re

from loguru import logger
import pandas as pd
import requests
from tqdm import tqdm
import typer

from mi_bici.config import (
    COLUMN_PATTERNS_REGEX,
    ESTACIONES_FILE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    URL_NOMENCLATURA,
    URLS_VIAJES,
    VIAJES_DIR,
)

app = typer.Typer()


def download_data() -> None:
    """Descarga de datos CSV directa del sitio usando nombres mapeados."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "curl/7.81.0", "Accept": "*/*"}

    # Descarga nomenclatura
    nom_filename = Path(URL_NOMENCLATURA).name
    nom_dest = RAW_DATA_DIR / nom_filename
    if nom_dest.exists() and nom_dest.stat().st_size > 0:
        logger.info(f"Nomenclatura ya existe en {nom_dest.name}, omitiendo descarga.")
    else:
        logger.info(f"Descargando nomenclatura desde {URL_NOMENCLATURA}...")
        response = requests.get(URL_NOMENCLATURA, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with (
            open(nom_dest, "wb") as f,
            tqdm(
                desc=nom_filename,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                leave=False,
            ) as bar,
        ):
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        logger.success(f"Nomenclatura guardada en {nom_dest.name}")

    # Descarga de viajes
    for periodo, url in URLS_VIAJES.items():
        viaje_filename = Path(url).name
        viaje_dest = RAW_DATA_DIR / viaje_filename
        if viaje_dest.exists() and viaje_dest.stat().st_size > 0:
            logger.info(f"Archivo {viaje_filename} ({periodo}) ya existe, omitiendo descarga.")
            continue

        logger.info(f"Descargando viajes {periodo} desde {url}...")
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with (
            open(viaje_dest, "wb") as f,
            tqdm(
                desc=viaje_filename,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                leave=False,
            ) as bar,
        ):
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        logger.success(f"Archivo {viaje_filename} guardado exitosamente.")


def load_data_travels(month: int, year: int) -> pd.DataFrame:
    """Carga de CSV de un año y mes para su procesamiento."""
    file_path = RAW_DATA_DIR / f"datos_abiertos_{year}_{month:02d}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo de viajes no encontrado: {file_path}")

    logger.info(f"Cargando archivo: {file_path.name}")
    try:
        return pd.read_csv(file_path, encoding="iso-8859-1")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="utf-8")


def load_data_noms() -> pd.DataFrame:
    """Carga de CSV de lugares/nomenclatura para su procesamiento."""
    file_path = RAW_DATA_DIR / Path(URL_NOMENCLATURA).name
    if not file_path.exists():
        matches = sorted(RAW_DATA_DIR.glob("nomenclatura*.csv"))
        if matches:
            file_path = matches[-1]
        else:
            raise FileNotFoundError(f"Archivo de nomenclatura no encontrado en: {RAW_DATA_DIR}")

    logger.info(f"Cargando nomenclatura desde: {file_path.name}")
    try:
        return pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="iso-8859-1")


def clean_data_noms(df: pd.DataFrame) -> pd.DataFrame:
    """Proceso de limpieza de datos de CSV nominales."""
    # TODO: Implementar reglas de limpieza específicas para nomenclatura si se requieren
    return df


def clean_data_travels(df: pd.DataFrame) -> pd.DataFrame:
    """Proceso de limpieza de datos de CSV viajes."""
    total_filas_inicial = len(df)
    cols = df.columns

    # Paso 1: Parsear las columnas a un nombre común debido a diversos nombres de columnas en distintos datasets
    cols_detectadas = {
        clave: next((c for c in cols if re.search(patron, c, re.IGNORECASE)), None)
        for clave, patron in COLUMN_PATTERNS_REGEX.items()
    }

    mapa_renombrado = {
        col_real: nombre_estandar
        for nombre_estandar, col_real in cols_detectadas.items()
        if col_real is not None
    }

    df = df.rename(columns=mapa_renombrado)

    # Paso 2: Eliminar columnas innecesarias para P1, P2, P3
    df = df.drop(columns=["viaje", "usuario"], errors="ignore")

    # Paso 3: Eliminar columnas con fechas de nacimiento corruptas
    df = df[df["nacimiento"].notna()]
    if "origen" in df.columns:
        df = df[df["origen"].notna()]
    if "destino" in df.columns:
        df = df[df["destino"].notna()]

    # Paso 4: Parsear como ints columnas en float
    df["nacimiento"] = df["nacimiento"].astype(int)
    df["origen"] = df["origen"].astype(int)
    df["destino"] = df["destino"].astype(int)

    # Paso 5: Normalizar columnas de fecha y hora. Nueva columna de minutos
    df["inicio"] = pd.to_datetime(df["inicio"], format="mixed", errors="coerce")
    df["fin"] = pd.to_datetime(df["fin"], format="mixed", errors="coerce")

    # Filtrar registros donde no se pudo parsear fecha
    df = df[df["inicio"].notna() & df["fin"].notna()]

    df["duracion_min"] = ((df["fin"] - df["inicio"]).dt.total_seconds() // 60).astype("Int64")

    df["inicio"] = df["inicio"].dt.strftime("%Y-%m-%d %H:%M")
    df["fin"] = df["fin"].dt.strftime("%Y-%m-%d %H:%M")

    # Paso 6: Genero rellenar los null con No Binario
    df["genero"] = df["genero"].fillna("NB")

    # Paso 7: Eliminar rows basura que son posibles errores que no aportan nada
    # Escenario: Si hay viajes que duran menos de 2 minutos y son originales y destino, eliminarlos
    df = df.drop(df[(df["duracion_min"] < 2) & (df["origen"] == df["destino"])].index)
    # Escenario: Si hay viajes con duracion negativa, elimnarlos
    df = df.drop(df[df["duracion_min"] < 0].index)

    filas_eliminadas = total_filas_inicial - len(df)
    if total_filas_inicial > 0 and (filas_eliminadas / total_filas_inicial) > 0.05:
        logger.error(
            f"Se eliminaron {filas_eliminadas} filas ({filas_eliminadas / total_filas_inicial:.2%}) "
            f"de un total de {total_filas_inicial}, superando el umbral del 5%."
        )

    # Reset de index
    df = df.reset_index(drop=True)

    return df


@app.command()
def main() -> None:
    logger.info("Descargar datos de viajes de sitio Mi Bici")
    logger.info("Descargar datos de nomenclatura de sitio Mi Bici")
    download_data()

    logger.info("Cargar datos nomenclatura")
    df_noms = load_data_noms()

    logger.info("Limpieza datos nomenclatura")
    df_noms = clean_data_noms(df_noms)

    logger.info("Guardado de archivo parquet nomenclatura")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_noms.to_parquet(ESTACIONES_FILE, engine="pyarrow", index=False)
    logger.success(f"Archivo de nomenclatura guardado en: {ESTACIONES_FILE}")

    logger.info("Comienza carga de datos de viajes")
    # TODO: Carga de archivos de forma cronologica, sort por fecha usando nombre de archivo
    archivos_viajes = sorted(RAW_DATA_DIR.glob("datos_abiertos_*.csv"))

    VIAJES_DIR.mkdir(parents=True, exist_ok=True)

    for archivo in archivos_viajes:
        match = re.search(r"datos_abiertos_(\d{4})_(\d{2})\.csv", archivo.name)
        if not match:
            continue
        anio = int(match.group(1))
        mes = int(match.group(2))

        logger.info(f"Cargando datos de viajes de archivo {archivo.name}")
        df_viajes = load_data_travels(month=mes, year=anio)

        logger.info(f"Limpieza datos viajes de archivo {archivo.name}")
        df_viajes = clean_data_travels(df_viajes)

        # Siguiendo como el notebook hizo los archivos parquet:
        df_viajes["anio"] = df_viajes["inicio"].str[:4]
        df_viajes["mes"] = df_viajes["inicio"].str[5:7]

        logger.info(f"Guardado en parquet de particion de viajes {anio}-{mes:02d}")
        df_viajes.to_parquet(
            VIAJES_DIR,
            engine="pyarrow",
            partition_cols=["anio", "mes"],
            index=False,
        )

    logger.success("Completado proceso de dataset")


if __name__ == "__main__":
    app()
