import hashlib
import json
from pathlib import Path
import re
import shutil

from loguru import logger
import pandas as pd
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

# pyrefly: ignore[missing-import]
import typer

from mi_bici.config import (
    ANIO_NACIMIENTO_MINIMO,
    COLUMN_PATTERNS_REGEX,
    DURACION_MAXIMA_MIN,
    DURACION_MINIMA_MIN,
    ESTACIONES_FILE,
    ESTACIONES_METADATA_FILE,
    GENERO_NULL_VALUE,
    MANIFEST_FILE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    STATION_COLUMNS_TO_KEEP,
    STATION_PATTERNS_REGEX,
    TRAVEL_COLUMNS_TO_KEEP,
    UMBRAL_FILAS_A_REMOVER,
    URL_STATIONS,
    URLS_VIAJES,
    VIAJES_DIR,
    VIAJES_METADATA_FILE,
)

app = typer.Typer()


def download_data_stations(force: bool = False) -> Path:
    """Descarga de datos de estaciones (nomenclatura en origen) en CSV a raw."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "curl/7.81.0", "Accept": "*/*"}

    station_filename = Path(URL_STATIONS).name
    station_dest = RAW_DATA_DIR / station_filename
    if station_dest.exists() and station_dest.stat().st_size > 0 and not force:
        logger.info(f"Archivo de estaciones ya existe en {station_dest.name}, omitiendo descarga.")
        return station_dest

    logger.info(f"Descargando estaciones desde {URL_STATIONS}...")
    response = requests.get(URL_STATIONS, headers=headers, stream=True, timeout=30)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    with (
        open(station_dest, "wb") as f,
        tqdm(
            desc=station_filename,
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
    logger.success(f"Archivo de estaciones guardado en {station_dest.name}")
    return station_dest


def download_data_travel(
    period: str | None = None,
    month: int | None = None,
    year: int | None = None,
    force: bool = False,
) -> Path:
    """Descarga un CSV específico de viajes de acuerdo a su periodo ('YYYY-MM') o mes y año."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "curl/7.81.0", "Accept": "*/*"}

    if period is None:
        if month is None or year is None:
            raise ValueError(
                "Se debe proporcionar 'period' (ej. '2025-10') o ambos 'month' y 'year'."
            )
        period = f"{year}-{month:02d}"
    else:
        period = period.replace("_", "-")

    if period not in URLS_VIAJES:
        raise KeyError(
            f"Periodo '{period}' no encontrado en URLS_VIAJES. Periodos disponibles: {list(URLS_VIAJES.keys())}"
        )

    url = URLS_VIAJES[period]
    viaje_filename = Path(url).name
    viaje_dest = RAW_DATA_DIR / viaje_filename

    if viaje_dest.exists() and viaje_dest.stat().st_size > 0 and not force:
        logger.info(f"Archivo {viaje_filename} ({period}) ya existe, omitiendo descarga.")
        return viaje_dest

    logger.info(f"Descargando viajes {period} desde {url}...")
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
    return viaje_dest


def download_data(period: str | None = None, force: bool = False) -> None:
    """Descarga de datos CSV del sitio. Si se especifica period, descarga solo ese periodo."""
    download_data_stations(force=force)

    if period is not None:
        download_data_travel(period=period, force=force)
    else:
        for p in URLS_VIAJES:
            download_data_travel(period=p, force=force)


def load_data_travels(
    month: int | None = None,
    year: int | None = None,
    period: str | None = None,
    download_if_missing: bool = False,
) -> pd.DataFrame:
    """Carga de un CSV específico de viajes (por periodo 'YYYY-MM' o mes y año)."""
    if period is not None:
        period_norm = period.replace("-", "_")
        parts = period_norm.split("_")
        if len(parts) == 2:
            year, month = int(parts[0]), int(parts[1])
        else:
            raise ValueError(
                f"Formato de periodo no reconocido: '{period}'. Use 'YYYY-MM' o 'YYYY_MM'."
            )
    elif month is None or year is None:
        raise ValueError("Se debe proporcionar 'period' (ej. '2025-10') o ambos 'month' y 'year'.")

    file_path = RAW_DATA_DIR / f"datos_abiertos_{year}_{month:02d}.csv"
    if not file_path.exists():
        if download_if_missing:
            logger.info(f"Archivo {file_path.name} no encontrado localmente. Descargando...")
            download_data_travel(year=year, month=month)
        else:
            raise FileNotFoundError(f"Archivo de viajes no encontrado: {file_path}")

    logger.info(f"Cargando archivo: {file_path.name}")
    try:
        return pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="latin-1")


def load_data_stations(download_if_missing: bool = False) -> pd.DataFrame:
    """Carga de CSV de estaciones (nomenclatura en origen) para su procesamiento."""
    file_path = RAW_DATA_DIR / Path(URL_STATIONS).name
    if not file_path.exists():
        matches = sorted(RAW_DATA_DIR.glob("nomenclatura*.csv"))
        if matches:
            file_path = matches[-1]
        elif download_if_missing:
            file_path = download_data_stations()
        else:
            raise FileNotFoundError(f"Archivo de estaciones no encontrado en: {RAW_DATA_DIR}")

    logger.info(f"Cargando estaciones desde: {file_path.name}")
    try:
        return pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="iso-8859-1")


def load_data_from_url(
    url: str,
    dest_dir: Path = RAW_DATA_DIR,
    force: bool = False,
) -> pd.DataFrame:
    """Descarga (si no existe) y carga un dataset CSV desde una URL directa para exploración en notebooks.

    Aplica la misma estrategia de verificación de existencia local, descarga con headers compatibles
    y decodificación con fallback de encodings (iso-8859-1 / utf-8).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(url).name
    dest_file = dest_dir / filename

    if dest_file.exists() and dest_file.stat().st_size > 0 and not force:
        logger.info(f"El archivo {filename} ya existe localmente. Omitiendo descarga.")
    else:
        headers = {"User-Agent": "curl/7.81.0", "Accept": "*/*"}
        logger.info(f"Descargando dataset desde {url}...")
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with (
            open(dest_file, "wb") as f,
            tqdm(
                desc=filename,
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
        logger.success(f"Archivo guardado en: {dest_file}")

    logger.info(f"Cargando dataset desde {dest_file.name}...")
    try:
        return pd.read_csv(dest_file, encoding="iso-8859-1")
    except UnicodeDecodeError:
        return pd.read_csv(dest_file, encoding="utf-8")


def load_processed_travels(columns: list[str] | None = None) -> pd.DataFrame:
    """Carga el dataset completo de viajes procesados en formato parquet."""
    if not VIAJES_DIR.exists():
        raise FileNotFoundError(f"No se encontró el directorio de viajes procesados: {VIAJES_DIR}")
    # Limpiar metadata.json residual dentro del directorio particionado si existiera
    legacy_meta = VIAJES_DIR / "metadata.json"
    if legacy_meta.exists():
        legacy_meta.unlink()
    return pd.read_parquet(VIAJES_DIR, columns=columns)


def load_processed_stations() -> pd.DataFrame:
    """Carga el dataset de estaciones procesado en formato parquet."""
    if not ESTACIONES_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de estaciones procesado: {ESTACIONES_FILE}"
        )
    return pd.read_parquet(ESTACIONES_FILE)


def clean_data_stations(df: pd.DataFrame) -> pd.DataFrame:
    """Proceso de limpieza y estandarización de columnas de CSV de estaciones."""
    cols = df.columns

    cols_detectadas = {
        clave: next((c for c in cols if re.search(patron, c, re.IGNORECASE)), None)
        for clave, patron in STATION_PATTERNS_REGEX.items()
    }

    mapa_renombrado = {
        col_real: nombre_estandar
        for nombre_estandar, col_real in cols_detectadas.items()
        if col_real is not None
    }

    df = df.rename(columns=mapa_renombrado)

    # Validar y parsear tipos de datos si las columnas están presentes
    if "estacion_id" in df.columns:
        df["estacion_id"] = pd.to_numeric(df["estacion_id"], errors="coerce").astype(int)
    if "nombre" in df.columns:
        df["nombre"] = df["nombre"].astype(str)
    if "latitud" in df.columns:
        df["latitud"] = pd.to_numeric(df["latitud"], errors="coerce").astype(float)
    if "longitud" in df.columns:
        df["longitud"] = pd.to_numeric(df["longitud"], errors="coerce").astype(float)
    if "capacidad" in df.columns:
        df["capacidad"] = pd.to_numeric(df["capacidad"], errors="coerce").astype(int)

    # Filtrar orden de columnas deseadas
    cols_finales = [c for c in STATION_COLUMNS_TO_KEEP if c in df.columns]
    if cols_finales:
        df = df[cols_finales]

    df = df.reset_index(drop=True)
    return df


def parse_datetime_series(series: pd.Series) -> pd.Series:
    """Parsea una serie de fechas detectando automáticamente el formato (ISO o DD/MM/YYYY con o sin segundos)."""
    valido = series.dropna()
    if valido.empty:
        return pd.to_datetime(series)

    primer_valor = str(valido.iloc[0]).strip()
    if "-" in primer_valor:
        # Formato ISO (YYYY-MM-DD HH:MM:SS, YYYY-MM-DD HH:MM, o ISO con T) a velocidad nativa en C
        return pd.to_datetime(series, format="ISO8601", errors="coerce")
    elif "/" in primer_valor:
        # Formato día/mes/año (DD/MM/YYYY HH:MM:SS o DD/MM/YYYY HH:MM)
        fmt = "%d/%m/%Y %H:%M:%S" if primer_valor.count(":") >= 2 else "%d/%m/%Y %H:%M"
        return pd.to_datetime(series, format=fmt, errors="coerce")
    else:
        # Fallback de seguridad
        return pd.to_datetime(series, format="mixed", dayfirst=True, errors="coerce")


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

    # Paso 2: Detectar viajes Id duplicados si existen antes de eliminar la columna
    if "viaje_id" in df.columns:
        duplicados = int(df["viaje_id"].duplicated().sum())
        if duplicados > 0:
            logger.warning(
                f"Se detectaron {duplicados} registros con viaje_id duplicado. Eliminando duplicados..."
            )
            df = df.drop_duplicates(subset=["viaje_id"], keep="first")

    # Paso 3: Eliminar columnas innecesarias para P1, P2, P3
    df = df.drop(columns=["viaje_id", "usuario_id", "usuario", "viaje"], errors="ignore")

    # Paso 4: Eliminar filas con nulos en estaciones de origen y destino (no en nacimiento)
    mask_notna = pd.Series(True, index=df.index)
    if "origen_id" in df.columns:
        mask_notna &= df["origen_id"].notna()
    if "destino_id" in df.columns:
        mask_notna &= df["destino_id"].notna()
    df = df[mask_notna]

    # Paso 5: Parsear tipos numéricos (nacimiento nullable Int64 para preservar nulos)
    if "nacimiento" in df.columns:
        df["nacimiento"] = pd.to_numeric(df["nacimiento"], errors="coerce").astype("Int64")
    if "origen_id" in df.columns:
        df["origen_id"] = pd.to_numeric(df["origen_id"], errors="coerce").astype(int)
    if "destino_id" in df.columns:
        df["destino_id"] = pd.to_numeric(df["destino_id"], errors="coerce").astype(int)

    # Paso 6: Eliminar años de nacimiento inválidos (debajo del mínimo). Los nulos se conservan
    df = df[df["nacimiento"].isna() | (df["nacimiento"] >= ANIO_NACIMIENTO_MINIMO)]

    # Paso 7: Normalizar columnas de fecha y hora. Nueva columna de minutos
    df["inicio"] = parse_datetime_series(df["inicio"])
    df["fin"] = parse_datetime_series(df["fin"])

    # Filtrar registros donde no se pudo parsear fecha
    df = df[df["inicio"].notna() & df["fin"].notna()]

    df["duracion_min"] = ((df["fin"] - df["inicio"]).dt.total_seconds() // 60).astype("Int64")

    df["inicio"] = df["inicio"].dt.strftime("%Y-%m-%d %H:%M")
    df["fin"] = df["fin"].dt.strftime("%Y-%m-%d %H:%M")

    # Paso 8: Genero rellenar los null con valor de configuracion
    if "genero" in df.columns:
        df["genero"] = df["genero"].fillna(GENERO_NULL_VALUE)

    # Paso 9: Eliminar viajes que son errores: menos de DURACION_MINIMA_MIN en la misma estación
    # (no se sacó bien la bici), más de DURACION_MAXIMA_MIN (irregular según MiBici) o negativos.
    # Cada condición se evalúa por separado: basta con que se cumpla una para quitar el viaje
    muy_corto = (df["duracion_min"] < DURACION_MINIMA_MIN) & (df["origen_id"] == df["destino_id"])
    muy_largo = df["duracion_min"] > DURACION_MAXIMA_MIN
    negativo = df["duracion_min"] < 0
    df = df[~(muy_corto | muy_largo | negativo)]

    filas_eliminadas = total_filas_inicial - len(df)
    if (
        total_filas_inicial > 0
        and (filas_eliminadas / total_filas_inicial) > UMBRAL_FILAS_A_REMOVER
    ):
        logger.error(
            f"Se eliminaron {filas_eliminadas} filas ({filas_eliminadas / total_filas_inicial:.2%}) "
            f"de un total de {total_filas_inicial}, superando el umbral del 5%."
        )

    # Paso 10: Ordenar columnas finales
    cols_finales = [c for c in TRAVEL_COLUMNS_TO_KEEP + ["duracion_min"] if c in df.columns]
    if cols_finales:
        df = df[cols_finales]

    # Reset de index
    df = df.reset_index(drop=True)

    return df


def compute_file_md5(file_path: Path) -> str:
    """Calcula el hash MD5 de un archivo leyendo en bloques binarios."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_parquet_metadata(target: Path, output_json: Path | None = None) -> dict:
    """Calcula el hash MD5 y metadatos de un archivo o carpeta parquet y crea un JSON de metadatos asociado."""
    if not target.exists():
        raise FileNotFoundError(f"Ruta objetivo no encontrada: {target}")

    if target.is_file():
        md5_hash = compute_file_md5(target)
        size_bytes = target.stat().st_size
        pf = pq.ParquetFile(target)
        num_rows = pf.metadata.num_rows
        columns = pf.schema.names

        metadata = {
            "name": target.stem,
            "file_name": target.name,
            "md5": md5_hash,
            "size_bytes": size_bytes,
            "num_rows": num_rows,
            "columns": columns,
        }
        if output_json is None:
            output_json = target.parent / f"{target.stem}_metadata.json"

    else:
        # Asegurar que no quede un metadata.json dentro de la carpeta particionada
        legacy_meta = target / "metadata.json"
        if legacy_meta.exists():
            legacy_meta.unlink()

        parquet_files = sorted(target.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No se encontraron archivos parquet en {target}")

        files_info = []
        total_size = 0
        total_rows = 0
        columns = []
        combined_hasher = hashlib.md5()

        for pfile in parquet_files:
            rel_path = str(pfile.relative_to(target))
            f_md5 = compute_file_md5(pfile)
            f_size = pfile.stat().st_size
            pf = pq.ParquetFile(pfile)
            f_rows = pf.metadata.num_rows
            if not columns:
                columns = pf.schema.names

            total_size += f_size
            total_rows += f_rows
            combined_hasher.update(f_md5.encode("utf-8"))

            files_info.append(
                {
                    "path": rel_path,
                    "md5": f_md5,
                    "size_bytes": f_size,
                    "num_rows": f_rows,
                }
            )

        metadata = {
            "name": target.name,
            "directory": target.name,
            "md5": combined_hasher.hexdigest(),
            "total_files": len(parquet_files),
            "total_size_bytes": total_size,
            "total_rows": total_rows,
            "columns": columns,
            "files": files_info,
        }
        if output_json is None:
            output_json = target.parent / f"{target.name}_metadata.json"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.success(f"Metadatos guardados en: {output_json}")
    return metadata


def generate_manifest(manifest_path: Path = MANIFEST_FILE) -> dict:
    """Genera o actualiza el archivo manifest.json consolidando metadatos de estaciones y viajes."""
    estaciones_meta = generate_parquet_metadata(ESTACIONES_FILE, ESTACIONES_METADATA_FILE)
    viajes_meta = generate_parquet_metadata(VIAJES_DIR, VIAJES_METADATA_FILE)

    manifest = {
        "version": "1.0",
        "estaciones": {
            "file": estaciones_meta["file_name"],
            "md5": estaciones_meta["md5"],
            "size_bytes": estaciones_meta["size_bytes"],
            "num_rows": estaciones_meta["num_rows"],
            "columns": estaciones_meta["columns"],
        },
        "viajes": {
            "directory": viajes_meta["directory"],
            "md5": viajes_meta["md5"],
            "total_files": viajes_meta["total_files"],
            "total_size_bytes": viajes_meta["total_size_bytes"],
            "total_rows": viajes_meta["total_rows"],
            "columns": viajes_meta["columns"],
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.success(f"Manifiesto global generado exitosamente en: {manifest_path}")
    return manifest


@app.command()
def main() -> None:
    logger.info("Descargar datos de viajes de sitio Mi Bici")
    logger.info("Descargar datos de estaciones de sitio Mi Bici")
    download_data()

    logger.info("Cargar datos de estaciones")
    df_stations = load_data_stations()

    logger.info("Limpieza datos estaciones")
    df_stations = clean_data_stations(df_stations)

    logger.info("Guardado de archivo parquet estaciones")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_stations.to_parquet(ESTACIONES_FILE, engine="pyarrow", index=False)
    logger.success(f"Archivo de estaciones guardado en: {ESTACIONES_FILE}")
    generate_parquet_metadata(ESTACIONES_FILE, ESTACIONES_METADATA_FILE)

    logger.info("Comienza carga de datos de viajes (exclusivamente periodos de URLS_VIAJES)")
    # Limpiar VIAJES_DIR si ya existe para evitar particiones de datasets exploratorios antiguos (ej. 2020, 2023)
    if VIAJES_DIR.exists():
        shutil.rmtree(VIAJES_DIR)
    VIAJES_DIR.mkdir(parents=True, exist_ok=True)

    for period in sorted(URLS_VIAJES.keys()):
        parts = period.split("-")
        anio = int(parts[0])
        mes = int(parts[1])
        archivo_nombre = f"datos_abiertos_{anio}_{mes:02d}.csv"
        archivo_path = RAW_DATA_DIR / archivo_nombre

        if not archivo_path.exists():
            logger.warning(
                f"Archivo {archivo_nombre} de {period} no encontrado en raw. Descargando..."
            )
            download_data_travel(period=period)

        logger.info(f"Cargando datos de viajes de periodo {period} ({archivo_nombre})")
        df_viajes = load_data_travels(period=period)

        logger.info(f"Limpieza datos viajes de periodo {period}")
        df_viajes = clean_data_travels(df_viajes)

        # Siguiendo como el notebook hizo los archivos parquet:
        df_viajes["anio"] = df_viajes["inicio"].str[:4]
        df_viajes["mes"] = df_viajes["inicio"].str[5:7]

        logger.info(f"Guardado en parquet de particion de viajes {period}")
        df_viajes.to_parquet(
            VIAJES_DIR,
            engine="pyarrow",
            partition_cols=["anio", "mes"],
            index=False,
        )

    logger.info("Generando metadatos y manifest de datos")
    generate_parquet_metadata(VIAJES_DIR, VIAJES_METADATA_FILE)
    generate_manifest(MANIFEST_FILE)

    logger.success("Completado proceso de dataset")


if __name__ == "__main__":
    app()
