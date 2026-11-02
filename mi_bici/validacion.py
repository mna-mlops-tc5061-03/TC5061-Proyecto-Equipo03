"""Valida la calidad de los datos procesados con DeepChecks.

Integridad: corre la suite `data_integrity` sobre los viajes y las estaciones procesados, con
condiciones explícitas para nulos, duplicados y tipos mezclados.

Drift: corre la suite `train_test_validation` sobre las tablas de variables de P1 y P2 y compara
el periodo de entrenamiento con el de prueba (desde `CORTE_PRUEBA`).

Cada comando guarda el reporte HTML de cada tabla y un resumen sin filas en
`reports/deepchecks/resumen.json`.

python -m mi_bici.validacion integridad
python -m mi_bici.validacion drift
"""

import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mi_bici.config import (
    COLUMNA_FECHA_P1,
    COLUMNA_FECHA_P2,
    CORTE_PRUEBA,
    DEEPCHECKS_DIR,
    OBJETIVO_P1,
    OBJETIVO_P2,
    RANGO_TIPOS_MEZCLADOS,
    UMBRAL_CORRELACION,
    UMBRAL_DRIFT_OBJETIVO,
    UMBRAL_DRIFT_VARIABLES,
    UMBRAL_DUPLICADOS,
    UMBRAL_NULOS,
    UMBRAL_RIESGO,
    VARIABLES_P1,
    VARIABLES_P2,
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


# Chequeos de train_test_validation que se quitan:
# - TrainTestSamplesMix usa DataFrame.applymap, que pandas 3 quitó. Siempre termina en error.
# - MultivariateDrift entrena un modelo con una clase propia que no define __sklearn_tags__, que
#   scikit-learn 1.9 exige. También termina siempre en error.
# - IndexTrainTestLeakage necesita un índice por fila y las tablas de variables no lo tienen: la
#   misma estación aparece en entrenamiento y en prueba a propósito, en otras fechas.
CHEQUEOS_DRIFT_QUE_SE_QUITAN = (
    "TrainTestSamplesMix",
    "MultivariateDrift",
    "IndexTrainTestLeakage",
)

# Tablas de variables que se revisan por drift: columnas, objetivo, tipo de objetivo, columna de
# fecha y variables categóricas (el día, la hora y las banderas no miden una cantidad)
TABLAS_DRIFT = {
    "P1": {
        "variables": VARIABLES_P1,
        "objetivo": OBJETIVO_P1,
        "tipo": "regression",
        "fecha": COLUMNA_FECHA_P1,
        "categoricas": ["dia_semana", "festivo"],
    },
    "P2": {
        "variables": VARIABLES_P2,
        "objetivo": OBJETIVO_P2,
        "tipo": "binary",
        "fecha": COLUMNA_FECHA_P2,
        "categoricas": ["hora", "dia_semana", "festivo", "riesgo_semana_anterior"],
    },
}


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


def separar_por_fecha(
    tabla: pd.DataFrame, columna_fecha: str, corte: str = CORTE_PRUEBA
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa una tabla de variables en entrenamiento (antes de `corte`) y prueba (desde `corte`).

    Es la misma partición que usan los experimentos: el entrenamiento incluye el periodo de
    validación, porque los dos se conocen antes de la prueba.
    """
    fechas = pd.to_datetime(tabla[columna_fecha])
    return tabla[fechas < corte], tabla[fechas >= corte]


def suite_drift():
    """Arma la suite `train_test_validation` con las condiciones de drift del proyecto.

    Cambia las condiciones por defecto de `FeatureDrift` y `LabelDrift` por los umbrales de
    `config.py` y quita los chequeos de `CHEQUEOS_DRIFT_QUE_SE_QUITAN`. Los demás (tamaños,
    categorías y etiquetas nuevas, fuga por fechas y cambio en la relación de cada variable con el
    objetivo) se quedan con sus condiciones por defecto.

    Returns:
        La suite lista para correr sobre dos `Dataset` de DeepChecks.
    """
    preparar_deepchecks()
    from deepchecks.tabular.checks import FeatureDrift, LabelDrift
    from deepchecks.tabular.suites import train_test_validation

    suite = train_test_validation()
    for indice, chequeo in list(suite.checks.items()):
        if type(chequeo).__name__ in CHEQUEOS_DRIFT_QUE_SE_QUITAN:
            suite.remove(indice)
        elif isinstance(chequeo, FeatureDrift):
            chequeo.clean_conditions()
            chequeo.add_condition_drift_score_less_than(
                max_allowed_categorical_score=UMBRAL_DRIFT_VARIABLES,
                max_allowed_numeric_score=UMBRAL_DRIFT_VARIABLES,
            )
        elif isinstance(chequeo, LabelDrift):
            chequeo.clean_conditions()
            chequeo.add_condition_drift_score_less_than(UMBRAL_DRIFT_OBJETIVO)
    return suite


def validar_drift(
    train: pd.DataFrame,
    test: pd.DataFrame,
    etiqueta: str,
    categoricas: list[str] | None = None,
    tipo_etiqueta: str | None = None,
    fecha: str | None = None,
    html: Path | None = None,
    nombre: str | None = None,
) -> dict:
    """Revisa si los datos de prueba se parecen a los de entrenamiento.

    Args:
        train: Tabla de entrenamiento con las variables y el objetivo.
        test: Tabla de prueba con las mismas columnas.
        etiqueta: Nombre del objetivo. Su cambio lo mide `LabelDrift`; el de las demás columnas,
            `FeatureDrift`.
        categoricas: Variables categóricas. Si es `None`, DeepChecks las infiere.
        tipo_etiqueta: `"regression"`, `"binary"` o `"multiclass"`. Si es `None`, se infiere.
        fecha: Columna de fecha. Sirve para revisar que la prueba sea posterior al entrenamiento.
        html: Ruta donde guardar el reporte HTML. Si es `None` no se guarda.
        nombre: Nombre de la tabla en el resumen, como `P1`. Si es `None`, se usa `etiqueta`.

    Returns:
        El resumen de `resumir`, más el puntaje de drift de cada variable y del objetivo, y la
        media del objetivo en cada conjunto (para P2 es la tasa de riesgo).
    """
    dc_tabular = preparar_deepchecks()

    def dataset(tabla: pd.DataFrame):
        return dc_tabular.Dataset(
            tabla,
            label=etiqueta,
            label_type=tipo_etiqueta,
            cat_features=categoricas,
            datetime_name=fecha,
            set_index_from_dataframe_index=False,
        )

    resultado = suite_drift().run(train_dataset=dataset(train), test_dataset=dataset(test))
    if html is not None:
        html.parent.mkdir(parents=True, exist_ok=True)
        resultado.save_as_html(str(html), as_widget=False)

    resumen = resumir(resultado, nombre or etiqueta, len(train) + len(test), list(train.columns))
    resumen["filas_entrenamiento"], resumen["filas_prueba"] = len(train), len(test)
    resumen["objetivo"] = {
        "nombre": etiqueta,
        "media_entrenamiento": float(train[etiqueta].mean()),
        "media_prueba": float(test[etiqueta].mean()),
    }
    resumen["drift"] = {}
    for chequeo in resultado.results:
        valor = getattr(chequeo, "value", None)
        if chequeo.get_header() == "Feature Drift" and isinstance(valor, dict):
            for columna, datos in valor.items():
                resumen["drift"][columna] = _puntaje(datos)
        elif chequeo.get_header() == "Label Drift" and isinstance(valor, dict):
            resumen["drift"][etiqueta] = _puntaje(valor)
    return resumen


def _puntaje(datos: dict) -> dict:
    """Puntaje y método de drift de una columna, redondeados para el resumen."""
    puntaje = datos.get("Drift score")
    return {
        "puntaje": None if puntaje is None else round(float(puntaje), 4),
        "metodo": datos.get("Method"),
    }


def tablas_de_variables() -> dict[str, pd.DataFrame]:
    """Construye las tablas de variables de P1 y P2 con los datos procesados."""
    from mi_bici.dataset import load_processed_stations, load_processed_travels
    from mi_bici.features import demanda_diaria, etiqueta_riesgo, flujo_por_bloque

    viajes, estaciones = load_processed_travels(), load_processed_stations()
    return {
        "P1": demanda_diaria(viajes, estaciones),
        "P2": etiqueta_riesgo(flujo_por_bloque(viajes, estaciones), UMBRAL_RIESGO),
    }


def guardar_resumen(seccion: str, contenido: dict, ruta: Path = RESUMEN_FILE) -> None:
    """Guarda una sección del resumen y conserva las demás que ya estén en el archivo."""
    resumen = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    resumen[seccion] = contenido
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")


def reportar(nombre: str, resumen: dict) -> bool:
    """Escribe cuántas condiciones pasan, cuáles no y qué chequeos no corrieron.

    Returns:
        `True` si pasan todas las condiciones de la tabla.
    """
    total = len(resumen["condiciones"])
    logger.info(f"{nombre}: {total - len(resumen['no_pasaron'])} de {total} condiciones pasan")
    for condicion in resumen["no_pasaron"]:
        logger.warning(f"{nombre}: {condicion['condicion']} -> {condicion['detalle']}")
    for chequeo in resumen["no_corrieron"]:
        logger.warning(f"{nombre}: {chequeo['chequeo']} no corrió -> {chequeo['motivo']}")
    return resumen["paso"]


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
    pasan = [reportar(nombre, resumen) for nombre, resumen in resultados.items()]
    guardar_resumen("integridad", resultados)
    logger.success(f"Resumen guardado en {RESUMEN_FILE}")
    if not all(pasan):
        raise typer.Exit(1)


@app.command()
def drift() -> None:
    """Compara entrenamiento y prueba en las tablas de P1 y P2 y guarda el HTML y el resumen."""
    resultados = {}
    for pregunta, tabla in tablas_de_variables().items():
        cfg = TABLAS_DRIFT[pregunta]
        columnas = [cfg["fecha"], *cfg["variables"], cfg["objetivo"]]
        train, test = separar_por_fecha(tabla[columnas], cfg["fecha"])
        logger.info(f"Revisando drift de {pregunta}: {len(train):,} filas contra {len(test):,}")
        resultados[pregunta] = validar_drift(
            train,
            test,
            cfg["objetivo"],
            categoricas=cfg["categoricas"],
            tipo_etiqueta=cfg["tipo"],
            fecha=cfg["fecha"],
            html=DEEPCHECKS_DIR / f"drift_{pregunta.lower()}.html",
            nombre=pregunta,
        )
        objetivo = resultados[pregunta]["objetivo"]
        logger.info(
            f"{pregunta}: media de {objetivo['nombre']} {objetivo['media_entrenamiento']:.4f} en "
            f"entrenamiento y {objetivo['media_prueba']:.4f} en prueba"
        )
    pasan = [reportar(pregunta, resumen) for pregunta, resumen in resultados.items()]
    guardar_resumen("drift", resultados)
    logger.success(f"Resumen guardado en {RESUMEN_FILE}")
    if not all(pasan):
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
