"""Construye las variables de cada pregunta a partir de los viajes y estaciones procesados.

Convención para P1 y P2: cada fila es el periodo que se quiere predecir y todas sus variables
vienen de periodos anteriores (`shift` de al menos 1). Así ninguna usa información del futuro.
"""

import numpy as np
import pandas as pd

from mi_bici.config import BLOQUE_HORAS, CORTE_PRUEBA, FESTIVOS, MIN_DIAS_PERFIL
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


def flujo_por_bloque(
    viajes: pd.DataFrame, estaciones: pd.DataFrame, horas: int = BLOQUE_HORAS
) -> pd.DataFrame:
    """Construye salidas, llegadas y flujo neto por estación y bloque de horas.

    MiBici no publica cuántas bicicletas hay en cada estación. El flujo neto (llegadas menos
    salidas) es la mejor aproximación: si en un bloque salen muchas más bicis de las que llegan,
    la estación se vacía, y si llegan muchas más, se llena.

    Args:
        viajes: Viajes procesados por la ingesta.
        estaciones: Estaciones procesadas por la ingesta.
        horas: Tamaño del bloque en horas.

    Returns:
        Una fila por estación y bloque con `salidas`, `llegadas`, `flujo_neto` y `capacidad`.
        Los bloques sin viajes quedan en 0.
    """
    viajes = convertir_fechas(viajes)
    frecuencia = f"{horas}h"
    salidas = (
        viajes.assign(bloque=viajes["inicio"].dt.floor(frecuencia))
        .groupby(["origen_id", "bloque"])
        .size()
        .rename("salidas")
    )
    llegadas = (
        viajes.assign(bloque=viajes["fin"].dt.floor(frecuencia))
        .groupby(["destino_id", "bloque"])
        .size()
        .rename("llegadas")
    )
    salidas.index.names = llegadas.index.names = ["estacion_id", "bloque"]
    conteos = pd.concat([salidas, llegadas], axis=1).fillna(0).reset_index()
    flujo = _malla_completa(conteos, "bloque", frecuencia)
    flujo["flujo_neto"] = flujo["llegadas"] - flujo["salidas"]
    return flujo.merge(estaciones[["estacion_id", "capacidad"]], on="estacion_id")


def etiqueta_riesgo(flujo: pd.DataFrame, umbral: float, horas: int = BLOQUE_HORAS) -> pd.DataFrame:
    """Construye la etiqueta de riesgo de P2 y sus variables.

    Una estación está en riesgo si en el bloque gana o pierde, en neto, al menos `umbral` veces
    su capacidad. Las variables vienen solo de bloques anteriores: el flujo de los dos bloques
    previos, las salidas y llegadas del bloque previo, y el riesgo del mismo bloque una semana
    antes, que también es el baseline. Se excluyen las estaciones con capacidad 0, porque no se
    puede medir la fracción.

    Args:
        flujo: Resultado de `flujo_por_bloque`.
        umbral: Fracción de la capacidad que define el riesgo, por ejemplo `UMBRAL_RIESGO`.
        horas: Tamaño del bloque en horas, el mismo que en `flujo_por_bloque`.

    Returns:
        Una fila por estación y bloque con `riesgo` (objetivo), `flujo_neto_lag_1`,
        `flujo_neto_lag_2`, `salidas_lag_1`, `llegadas_lag_1`, `riesgo_semana_anterior`,
        `hora`, `dia_semana`, `festivo` y `capacidad`.
    """
    flujo = flujo[flujo["capacidad"] > 0].copy()
    flujo["riesgo"] = (flujo["flujo_neto"].abs() >= umbral * flujo["capacidad"]).astype(int)
    bloques_semana = 7 * 24 // horas
    flujo = agregar_rezagos(flujo, "flujo_neto", [1, 2])
    flujo = agregar_rezagos(flujo, "salidas", [1])
    flujo = agregar_rezagos(flujo, "llegadas", [1])
    flujo = agregar_rezagos(flujo, "riesgo", [bloques_semana])
    flujo = flujo.rename(columns={f"riesgo_lag_{bloques_semana}": "riesgo_semana_anterior"})
    flujo["hora"] = flujo["bloque"].dt.hour
    flujo = agregar_calendario(flujo, "bloque")
    return flujo.dropna().reset_index(drop=True)


# Variables del perfil de uso que entran al clustering de P3. Las demográficas solo describen
# los grupos: no se usan para formarlos
VARIABLES_PERFIL_USO = [f"hora_{h:02d}" for h in range(24)] + [
    "fin_de_semana",
    "balance_neto",
    "viajes_por_dia_log",
]


def perfil_estacion(
    viajes: pd.DataFrame,
    estaciones: pd.DataFrame,
    corte: str = CORTE_PRUEBA,
    min_dias: int = MIN_DIAS_PERFIL,
) -> pd.DataFrame:
    """Construye el perfil de uso de cada estación para agruparlas en P3.

    Solo usa viajes anteriores a `corte`, el periodo de entrenamiento. Deja fuera las estaciones
    con menos de `min_dias` días con viajes, porque su perfil todavía no es representativo.

    El perfil de uso, que es lo que agrupa el clustering, tiene:
    - La proporción de salidas por hora del día (24 columnas). Distingue estaciones de zonas de
      trabajo (picos de tarde), residenciales (picos de mañana) y de ocio.
    - La proporción de salidas en fin de semana.
    - El balance neto relativo, (llegadas - salidas) / total, que separa las estaciones que se
      vacían de las que se llenan.
    - Los viajes por día en escala logarítmica, para que el volumen no domine a las proporciones.

    Además calcula `pct_usuarias` y `edad_mediana`, agregados por estación. Solo sirven para
    describir los grupos y no entran al clustering.

    Args:
        viajes: Viajes procesados por la ingesta.
        estaciones: Estaciones procesadas por la ingesta.
        corte: Fecha de inicio del periodo de prueba. Los viajes desde esa fecha no se usan.
        min_dias: Días mínimos con viajes para incluir una estación.

    Returns:
        Una fila por estación con `VARIABLES_PERFIL_USO`, `pct_usuarias`, `edad_mediana`,
        `nombre`, `latitud` y `longitud`.
    """
    viajes = convertir_fechas(viajes)
    viajes = viajes[viajes["inicio"] < corte]
    dias_activa = viajes.groupby("origen_id")["inicio"].agg(lambda s: s.dt.normalize().nunique())
    viajes = viajes[viajes["origen_id"].isin(dias_activa[dias_activa >= min_dias].index)]

    por_hora = pd.crosstab(viajes["origen_id"], viajes["inicio"].dt.hour, normalize="index")
    por_hora = por_hora.reindex(columns=range(24), fill_value=0)
    por_hora.columns = [f"hora_{h:02d}" for h in por_hora.columns]
    salidas = viajes.groupby("origen_id").size()
    llegadas = viajes.groupby("destino_id").size().reindex(salidas.index, fill_value=0)
    dias = viajes["inicio"].dt.normalize().nunique()

    conocido = viajes[viajes["genero"].isin(["M", "F"])]
    edad = viajes["inicio"].dt.year - viajes["nacimiento"].astype("float")

    perfil = por_hora.assign(
        fin_de_semana=(viajes["inicio"].dt.dayofweek >= 5).groupby(viajes["origen_id"]).mean(),
        balance_neto=(llegadas - salidas) / (llegadas + salidas),
        viajes_por_dia_log=np.log1p(salidas / dias),
        pct_usuarias=(conocido["genero"] == "F").groupby(conocido["origen_id"]).mean(),
        edad_mediana=edad.groupby(viajes["origen_id"]).median(),
    )
    perfil.index.name = "estacion_id"
    return perfil.reset_index().merge(
        estaciones[["estacion_id", "nombre", "latitud", "longitud"]], on="estacion_id"
    )
