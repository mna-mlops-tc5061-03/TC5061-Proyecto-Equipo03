"""Valida la calidad de los datos procesados con DeepChecks.

Integridad: corre la suite `data_integrity` sobre los viajes y las estaciones procesados, con
condiciones explícitas para nulos, duplicados y tipos mezclados. Guarda el reporte HTML de cada
tabla y un resumen sin filas en `reports/deepchecks/resumen.json`.

python -m mi_bici.validacion integridad
"""

import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mi_bici.config import (
    DEEPCHECKS_DIR,
    RANGO_TIPOS_MEZCLADOS,
    UMBRAL_CORRELACION,
    UMBRAL_DUPLICADOS,
    UMBRAL_NULOS,
)

app = typer.Typer()

RESUMEN_FILE = DEEPCHECKS_DIR / "resumen.json"

# Columnas categóricas de cada tabla procesada. Los id de estación son números, pero no miden una
# cantidad, así que DeepChecks debe tratarlos como categorías
CATEGORICAS = {"viajes": ["genero", "origen_id", "destino_id"]}
# Columna que identifica cada fila. No se valida como variable: un id distinto en cada fila
# "explica" cualquier otra columna y la correlación saldría en 1
IDENTIFICADORES = {"estaciones": "estacion_id"}

# Chequeos de data_integrity que se quitan:
# - Los que necesitan una etiqueta. Los datos procesados todavía no la tienen: el objetivo de P1 y
#   P2 aparece hasta las tablas de variables.
# - OutlierSampleDetection, que llama a np.quantile(interpolation=...), un argumento que numpy 2
#   ya no acepta. Siempre termina en error, así que no aporta nada al reporte.
CHEQUEOS_QUE_SE_QUITAN = (
    "ConflictingLabels",
    "FeatureLabelCorrelation",
    "IdentifierLabelCorrelation",
    "OutlierSampleDetection",
)


def preparar_deepchecks():
    """Importa DeepChecks después de ajustar lo que le falta con nuestras versiones.

    DeepChecks 0.19.1 es de diciembre de 2024 y no soporta numpy 2 ni scikit-learn 1.9:

    - Usa `np.Inf` y `np.NaN`, que numpy 2 quitó (ahora solo existen `np.inf` y `np.nan`). Sin
      los alias, el import falla con `AttributeError`.
    - Al importarse arma su lista de métricas con `get_scorer("max_error")`, que scikit-learn 1.9
      ya no registra por nombre. Sin registrarlo, el import falla con `ValueError`.

    Los dos ajustes solo agregan nombres que faltan y se hacen antes del import. Además, DeepChecks
    usa `pkg_resources`, que setuptools quitó en la versión 81; por eso requirements.txt fija
    `setuptools<81`.

    Returns:
        El módulo `deepchecks.tabular`.
    """
    from sklearn.metrics import _scorer, make_scorer, max_error

    _scorer._SCORERS.setdefault("max_error", make_scorer(max_error, greater_is_better=False))
    for alias, valor in (("Inf", np.inf), ("NaN", np.nan)):
        if not hasattr(np, alias):
            setattr(np, alias, valor)

    import deepchecks.tabular as dc_tabular

    return dc_tabular


def condicion_pares_correlacionados(umbral: float):
    """Condición de `FeatureFeatureCorrelation` que no marca pares por error con pandas 3.

    La condición de DeepChecks busca los pares con `resultado[resultado > umbral].stack()`. En
    pandas 3, `stack()` ya no quita los valores vacíos, así que devuelve todos los pares y la
    condición falla aunque ninguna correlación pase el umbral. Esta versión lee la matriz celda
    por celda.

    Args:
        umbral: Correlación a partir de la cual un par se considera redundante.

    Returns:
        Función que recibe la matriz de correlaciones y regresa un `ConditionResult`.
    """
    from deepchecks.core import ConditionCategory, ConditionResult

    def condicion(matriz: pd.DataFrame):
        valores = matriz.apply(pd.to_numeric, errors="coerce")
        pares = [
            (fila, columna)
            for fila in valores.index
            for columna in valores.columns
            if str(fila) < str(columna) and valores.loc[fila, columna] > umbral
        ]
        if pares:
            return ConditionResult(
                ConditionCategory.FAIL, f"Correlation is greater than {umbral} for pairs {pares}"
            )
        return ConditionResult(ConditionCategory.PASS, f"All correlations are less than {umbral}")

    return condicion


def suite_integridad():
    """Arma la suite `data_integrity` con las condiciones del proyecto.

    Cambia las condiciones por defecto de duplicados, tipos mezclados y correlación por las de
    `config.py`, agrega el porcentaje de nulos, que la suite no trae, y quita los chequeos de
    `CHEQUEOS_QUE_SE_QUITAN`.

    Returns:
        La suite lista para correr sobre un `Dataset` de DeepChecks.
    """
    preparar_deepchecks()
    from deepchecks.tabular.checks import (
        DataDuplicates,
        FeatureFeatureCorrelation,
        MixedDataTypes,
        PercentOfNulls,
    )
    from deepchecks.tabular.suites import data_integrity

    suite = data_integrity()
    for indice, chequeo in list(suite.checks.items()):
        if type(chequeo).__name__ in CHEQUEOS_QUE_SE_QUITAN:
            suite.remove(indice)
        elif isinstance(chequeo, FeatureFeatureCorrelation):
            chequeo.clean_conditions()
            chequeo.add_condition(
                f"Not more than 0 pairs are correlated above {UMBRAL_CORRELACION}",
                condicion_pares_correlacionados(UMBRAL_CORRELACION),
            )
        elif isinstance(chequeo, DataDuplicates):
            chequeo.clean_conditions()
            chequeo.add_condition_ratio_less_or_equal(UMBRAL_DUPLICADOS)
        elif isinstance(chequeo, MixedDataTypes):
            chequeo.clean_conditions()
            chequeo.add_condition_rare_type_ratio_not_in_range(RANGO_TIPOS_MEZCLADOS)
    # Con 10 millones de filas cubre la tabla completa, como los demás chequeos de la suite
    nulos = PercentOfNulls(n_samples=10_000_000)
    suite.add(nulos.add_condition_percent_of_nulls_not_greater_than(UMBRAL_NULOS))
    return suite


def resumir(resultado, nombre: str, filas: int, columnas: list[str]) -> dict:
    """Resume el resultado de una suite sin copiar ninguna fila de los datos.

    Args:
        resultado: Resultado de `suite.run()`.
        nombre: Nombre de la tabla validada.
        filas: Número de filas de la tabla.
        columnas: Columnas de la tabla.

    Returns:
        Diccionario con cada condición (pasa, advertencia o falla y su detalle), los chequeos que
        no se pudieron correr y la lista de condiciones que no pasaron.
    """
    # bandit confunde "pasa" con una contraseña escrita en el código
    estados = {"PASS": "pasa", "WARN": "advertencia", "FAIL": "falla", "ERROR": "error"}  # nosec B105
    condiciones, no_corrieron = [], []
    for chequeo in resultado.results:
        if not hasattr(chequeo, "conditions_results"):
            # CheckFailure: el chequeo no corrió con estos datos
            no_corrieron.append({"chequeo": chequeo.header, "motivo": str(chequeo.exception)})
            continue
        for condicion in chequeo.conditions_results:
            condiciones.append(
                {
                    "chequeo": chequeo.get_header(),
                    "condicion": condicion.name,
                    "estado": estados.get(condicion.category.name, condicion.category.name),
                    "detalle": condicion.details,
                }
            )
    return {
        "tabla": nombre,
        "filas": filas,
        "columnas": columnas,
        "paso": all(c["estado"] == "pasa" for c in condiciones),
        "no_pasaron": [c for c in condiciones if c["estado"] != "pasa"],
        "condiciones": condiciones,
        "no_corrieron": no_corrieron,
    }


def validar_integridad(df: pd.DataFrame, nombre: str, html: Path | None = None) -> dict:
    """Valida la integridad de una tabla con DeepChecks.

    Revisa nulos, duplicados, tipos mezclados, columnas con un solo valor, caracteres especiales,
    distintas formas de nulo, variantes de un mismo texto, longitudes de texto atípicas y
    correlación entre columnas.

    Args:
        df: Tabla a validar.
        nombre: Nombre de la tabla. Si está en `CATEGORICAS` o `IDENTIFICADORES`, se usan sus
            columnas categóricas y su identificador.
        html: Ruta donde guardar el reporte HTML. Si es `None` no se guarda.

    Returns:
        El resumen de `resumir`: qué condiciones pasaron, cuáles no y por qué.
    """
    dc_tabular = preparar_deepchecks()
    categoricas = [c for c in CATEGORICAS.get(nombre, []) if c in df.columns]
    identificador = IDENTIFICADORES.get(nombre)
    if identificador not in df.columns:
        identificador = None
    dataset = dc_tabular.Dataset(
        df,
        cat_features=categoricas,
        index_name=identificador,
        set_index_from_dataframe_index=False,
    )
    resultado = suite_integridad().run(dataset)
    if html is not None:
        html.parent.mkdir(parents=True, exist_ok=True)
        resultado.save_as_html(str(html), as_widget=False)
    return resumir(resultado, nombre, len(df), list(df.columns))


def guardar_resumen(seccion: str, contenido: dict, ruta: Path = RESUMEN_FILE) -> None:
    """Guarda una sección del resumen y conserva las demás que ya estén en el archivo."""
    resumen = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    resumen[seccion] = contenido
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")


@app.callback()
def main() -> None:
    """Validación de los datos procesados con DeepChecks."""


@app.command()
def integridad() -> None:
    """Valida los viajes y las estaciones procesados y guarda el HTML y el resumen."""
    from mi_bici.dataset import load_processed_stations, load_processed_travels

    tablas = {"viajes": load_processed_travels(), "estaciones": load_processed_stations()}
    resultados = {}
    for nombre, df in tablas.items():
        logger.info(f"Validando la integridad de {nombre} ({len(df):,} filas)")
        resultados[nombre] = validar_integridad(
            df, nombre, html=DEEPCHECKS_DIR / f"integridad_{nombre}.html"
        )
        for condicion in resultados[nombre]["no_pasaron"]:
            logger.warning(f"{nombre}: {condicion['condicion']} -> {condicion['detalle']}")
    guardar_resumen("integridad", resultados)
    logger.success(f"Resumen guardado en {RESUMEN_FILE}")


if __name__ == "__main__":
    app()
