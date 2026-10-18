"""Construye las variables de cada pregunta a partir de los viajes y estaciones procesados.

Convención para P1 y P2: cada fila es el periodo que se quiere predecir y todas sus variables
vienen de periodos anteriores (`shift` de al menos 1). Así ninguna usa información del futuro.
"""

import pandas as pd

from mi_bici.config import FESTIVOS
from mi_bici.dataset import parse_datetime_series


def convertir_fechas(viajes: pd.DataFrame) -> pd.DataFrame:
    """Convierte `inicio` y `fin` de texto a fecha y hora.

    La ingesta guarda las fechas como texto (`YYYY-MM-DD HH:MM`) y las variables necesitan
    operarlas como fechas. Si ya vienen convertidas, las deja igual.

    Args:
        viajes: Viajes procesados por la ingesta.

    Returns:
        Copia de los viajes con `inicio` y `fin` como fecha y hora.
    """
    viajes = viajes.copy()
    for columna in ("inicio", "fin"):
        if not pd.api.types.is_datetime64_any_dtype(viajes[columna]):
            viajes[columna] = parse_datetime_series(viajes[columna])
    return viajes


def _malla_completa(conteos: pd.DataFrame, columna_tiempo: str, frecuencia: str) -> pd.DataFrame:
    """Rellena con 0 los periodos sin viajes, desde la primera aparición de cada estación.

    Un periodo sin viajes es información (demanda cero), no un dato faltante. Empezar en la
    primera aparición evita inventar ceros antes de que la estación existiera.

    Args:
        conteos: Conteos por `estacion_id` y periodo.
        columna_tiempo: Nombre de la columna con el periodo.
        frecuencia: Frecuencia de pandas del periodo, por ejemplo `"D"` o `"3h"`.

    Returns:
        Conteos con todos los periodos de cada estación, ordenados por estación y periodo.
    """
    fin = conteos[columna_tiempo].max()
    partes = []
    for estacion, grupo in conteos.groupby("estacion_id"):
        periodos = pd.date_range(grupo[columna_tiempo].min(), fin, freq=frecuencia)
        partes.append(pd.DataFrame({"estacion_id": estacion, columna_tiempo: periodos}))
    malla = pd.concat(partes, ignore_index=True)
    return malla.merge(conteos, on=["estacion_id", columna_tiempo], how="left").fillna(0)


def agregar_calendario(df: pd.DataFrame, columna_tiempo: str) -> pd.DataFrame:
    """Agrega el día de la semana y si es día festivo.

    La demanda cambia entre días laborales, fines de semana y días de descanso obligatorio.
    Ambas variables se conocen de antemano, así que no usan información del futuro.

    Args:
        df: Datos con una columna de fecha.
        columna_tiempo: Nombre de la columna de fecha.

    Returns:
        Los mismos datos con `dia_semana` (0 = lunes) y `festivo` (1 o 0).
    """
    fechas = df[columna_tiempo]
    df["dia_semana"] = fechas.dt.dayofweek
    df["festivo"] = fechas.dt.normalize().isin(pd.to_datetime(FESTIVOS)).astype(int)
    return df


def agregar_rezagos(
    df: pd.DataFrame, columna: str, rezagos: list[int], ventana: int | None = None
) -> pd.DataFrame:
    """Agrega valores pasados de una columna por estación y, si se pide, su media móvil.

    La mejor pista del uso de una estación es su propio uso reciente. El `shift` va antes de la
    ventana, así la media solo incluye periodos ya observados. Espera los datos ordenados por
    estación y tiempo, como los deja `_malla_completa`.

    Args:
        df: Datos por `estacion_id` y periodo.
        columna: Columna de la que se toman los valores pasados.
        rezagos: Cuántos periodos hacia atrás, por ejemplo `[1, 7]`.
        ventana: Tamaño de la media móvil. Si es `None` no se calcula.

    Returns:
        Los mismos datos con `{columna}_lag_{k}` y, si aplica, `{columna}_media_{ventana}`.
    """
    por_estacion = df.groupby("estacion_id")[columna]
    for k in rezagos:
        df[f"{columna}_lag_{k}"] = por_estacion.shift(k)
    if ventana:
        df[f"{columna}_media_{ventana}"] = por_estacion.transform(
            lambda serie: serie.shift(1).rolling(ventana).mean()
        )
    return df


def demanda_diaria(viajes: pd.DataFrame, estaciones: pd.DataFrame) -> pd.DataFrame:
    """Construye las variables de P1: viajes iniciados por estación y día.

    El rezago de 7 días captura el patrón semanal (un lunes se parece al lunes anterior) y es a
    la vez el baseline de P1. La capacidad indica el tamaño de la estación. Solo quedan las
    estaciones de la nomenclatura, porque de las demás no se conoce la capacidad.

    Args:
        viajes: Viajes procesados por la ingesta.
        estaciones: Estaciones procesadas por la ingesta.

    Returns:
        Una fila por estación y día con `viajes` (objetivo), `viajes_lag_1`, `viajes_lag_7`,
        `viajes_media_7`, `dia_semana`, `festivo` y `capacidad`. Se descartan los primeros días
        de cada estación, que no tienen suficiente historia para los rezagos.
    """
    viajes = convertir_fechas(viajes)
    conteos = (
        viajes.assign(fecha=viajes["inicio"].dt.normalize())
        .groupby(["origen_id", "fecha"])
        .size()
        .rename("viajes")
        .reset_index()
        .rename(columns={"origen_id": "estacion_id"})
    )
    demanda = _malla_completa(conteos, "fecha", "D")
    demanda = agregar_rezagos(demanda, "viajes", [1, 7], ventana=7)
    demanda = agregar_calendario(demanda, "fecha")
    demanda = demanda.merge(estaciones[["estacion_id", "capacidad"]], on="estacion_id")
    return demanda.dropna().reset_index(drop=True)
