"""Lleva al Model Registry el mejor modelo de cada pregunta y lo marca como champion.

Compara los candidatos de cada pregunta contra su baseline con la métrica de selección de
`config.py`. El mejor de los que le ganan queda con el alias `champion`, que es la versión en
producción, y el segundo con `challenger`. Cada versión guarda su linaje: experimento, run de
origen, commit y versión de los datos.

python -m mi_bici.modeling.registry
python -m mi_bici.modeling.registry --question p2
"""

from loguru import logger
import mlflow
from mlflow import MlflowClient
import pandas as pd
import typer

from mi_bici.config import (
    ALIAS_PRODUCCION,
    ALIAS_RESPALDO,
    BLOQUE_HORAS,
    CRITERIOS_ESTABILIDAD,
    EXPERIMENT_NAME,
    METRICA_SELECCION,
    MODELOS_REGISTRO,
    TRACKING_URI,
)

app = typer.Typer()

# Etiquetas con que cada run indica su pregunta y su etapa (baseline, candidate o search)
ETIQUETA_PREGUNTA = "tags.question"
ETIQUETA_ETAPA = "tags.stage_of_work"
# Lo que un candidato debe compartir con el baseline para que la comparación sea válida
PARTICION = ["params.data_version", "params.corte_validacion", "params.corte_prueba"]

DESCRIPCIONES = {
    "P1": "Pronostica cuántos viajes saldrán de cada estación al día siguiente (regresión).",
    "P2": (
        "Predice si una estación quedará sin bicis o sin anclajes en el siguiente bloque de "
        f"{BLOQUE_HORAS} horas (clasificación)."
    ),
    "P3": "Asigna a cada estación un grupo según su patrón de uso por hora (clustering).",
}


def metrica_prueba(metrica: str) -> str | None:
    """Devuelve la métrica de prueba que corresponde a la de selección.

    Args:
        metrica: Métrica de selección, por ejemplo `"val_mae"`.

    Returns:
        La misma métrica en prueba (`"test_mae"`), o `None` si la pregunta se mide en
        entrenamiento, como P3.
    """
    if metrica.startswith("val_"):
        return "test_" + metrica.removeprefix("val_")
    return None


def mejora(valor: float, base: float, mayor_es_mejor: bool) -> float:
    """Calcula cuánto mejora un valor sobre el del baseline, como fracción del baseline.

    Positivo siempre es mejor, sin importar si la métrica sube o baja.

    Args:
        valor: Métrica del candidato.
        base: La misma métrica en el baseline.
        mayor_es_mejor: Si la métrica es mejor mientras más alta.

    Returns:
        La mejora relativa, o `nan` si el baseline vale cero.
    """
    diferencia = valor - base if mayor_es_mejor else base - valor
    return diferencia / abs(base) if base else float("nan")


def buscar_runs() -> pd.DataFrame:
    """Lee los runs terminados del experimento del equipo, del más reciente al más antiguo."""
    return mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME], filter_string="attributes.status = 'FINISHED'"
    )


def elegir(runs: pd.DataFrame, pregunta: str) -> tuple[pd.Series, pd.DataFrame]:
    """Compara los candidatos de una pregunta contra su baseline.

    Solo cuentan los runs sin cambios sin guardar (`git_dirty=False`), porque su commit basta para
    reproducirlos, y de los candidatos solo los que tienen los mismos datos y la misma partición
    que el baseline más reciente. Un candidato se acepta si le gana al baseline en la métrica de
    selección y, si la pregunta tiene prueba, también en prueba.

    Args:
        runs: Runs del experimento, como los devuelve `buscar_runs()`.
        pregunta: `"P1"`, `"P2"` o `"P3"`.

    Returns:
        El baseline y los candidatos aceptados, del mejor al peor.

    Raises:
        ValueError: Si la pregunta no tiene un baseline sin cambios sin guardar.
    """
    metrica, mayor_es_mejor = METRICA_SELECCION[pregunta]
    metricas = [f"metrics.{m}" for m in (metrica, metrica_prueba(metrica)) if m]
    criterios = CRITERIOS_ESTABILIDAD.get(pregunta, {})
    columnas_estabilidad = [f"metrics.{m}" for m in criterios]
    columnas = [
        ETIQUETA_PREGUNTA,
        ETIQUETA_ETAPA,
        "tags.git_dirty",
        *PARTICION,
        *metricas,
        *columnas_estabilidad,
    ]
    runs = runs.reindex(columns=[*runs.columns, *(c for c in columnas if c not in runs)])
    propios = runs[
        (runs[ETIQUETA_PREGUNTA].fillna("").astype(str).str.upper() == pregunta)
        & (runs["tags.git_dirty"] == "False")
    ]

    baselines = propios[propios[ETIQUETA_ETAPA] == "baseline"]
    if baselines.empty:
        raise ValueError(
            f"{pregunta}: no hay un baseline sin cambios sin guardar en {EXPERIMENT_NAME}"
        )
    base = baselines.iloc[0]
    candidatos = propios[
        (propios[ETIQUETA_ETAPA] == "candidate")
        & (propios[PARTICION] == base[PARTICION]).all(axis=1)
    ]

    signo = 1 if mayor_es_mejor else -1
    acepta = pd.Series(True, index=candidatos.index)
    for columna in metricas:
        acepta &= signo * (candidatos[columna] - base[columna]) > 0
    for m, (cota, mayor) in criterios.items():
        col = candidatos[f"metrics.{m}"]
        acepta &= (col >= cota) if mayor else (col <= cota)
    aceptados = candidatos[acepta].sort_values(metricas[0], ascending=not mayor_es_mejor)
    return base, aceptados


def modelo_del_run(cliente: MlflowClient, run_id: str) -> str:
    """Devuelve la URI del modelo que guardó un run, con el formato `models:/m-...` de MLflow 3.

    Args:
        cliente: Cliente de MLflow.
        run_id: Run que guardó el modelo.

    Returns:
        La URI del modelo.

    Raises:
        ValueError: Si el run no guardó un modelo.
    """
    salidas = cliente.get_run(run_id).outputs
    if not salidas or not salidas.model_outputs:
        raise ValueError(f"El run {run_id} no guardó un modelo")
    return f"models:/{salidas.model_outputs[0].model_id}"


def registrar_version(cliente: MlflowClient, nombre: str, run: pd.Series) -> str:
    """Registra el modelo de un run con su linaje, o reutiliza la versión si ya lo estaba.

    Así correr `make registry` otra vez no crea versiones repetidas.

    Args:
        cliente: Cliente de MLflow.
        nombre: Nombre del modelo registrado.
        run: Run elegido, como fila de `buscar_runs()`.

    Returns:
        El número de versión.
    """
    existentes = cliente.search_model_versions(f"name = '{nombre}' and run_id = '{run.run_id}'")
    if existentes:
        version = existentes[0].version
    else:
        version = mlflow.register_model(modelo_del_run(cliente, run.run_id), nombre).version
    linaje = {
        "experiment": EXPERIMENT_NAME,
        "source_run": run.run_id,
        "git_commit": run["tags.git_commit"],
        "data_version": run["params.data_version"],
    }
    for clave, valor in linaje.items():
        cliente.set_model_version_tag(nombre, version, clave, valor)
    return version


def describir_version(run: pd.Series, base: pd.Series, pregunta: str) -> str:
    """Explica por qué se eligió una versión: sus métricas contra las del baseline y su origen.

    Args:
        run: Run de la versión.
        base: Run del baseline con que se comparó.
        pregunta: `"P1"`, `"P2"` o `"P3"`.

    Returns:
        La descripción en Markdown.
    """
    metrica, mayor_es_mejor = METRICA_SELECCION[pregunta]
    comparaciones = []
    for m in filter(None, (metrica, metrica_prueba(metrica))):
        valor, referencia = run[f"metrics.{m}"], base[f"metrics.{m}"]
        cambio = mejora(valor, referencia, mayor_es_mejor)
        comparaciones.append(
            f"`{m}` = {valor:.3f} contra {referencia:.3f} del baseline ({cambio:+.1%})"
        )
    criterios = CRITERIOS_ESTABILIDAD.get(pregunta, {})
    if criterios:
        detalle = ", ".join(
            f"`{m}` = {run[f'metrics.{m}']:.3f} ({'>=' if mayor else '<='} {cota})"
            for m, (cota, mayor) in criterios.items()
        )
        comparaciones.append(f"pasó el criterio de estabilidad ({detalle})")
    return (
        f"Elegida por {'; '.join(comparaciones)}. Run `{run.run_id}` de `{EXPERIMENT_NAME}`, "
        f"commit `{run['tags.git_commit'][:7]}`, datos `{run['params.data_version']}`."
    )


def registrar(pregunta: str, runs: pd.DataFrame, cliente: MlflowClient) -> dict:
    """Registra el mejor candidato de una pregunta como champion y el segundo como challenger.

    Args:
        pregunta: `"P1"`, `"P2"` o `"P3"`.
        runs: Runs del experimento, como los devuelve `buscar_runs()`.
        cliente: Cliente de MLflow.

    Returns:
        Nombre y versión del champion, su métrica de selección, la del baseline y la mejora.

    Raises:
        ValueError: Si no hay baseline o ningún candidato le gana.
    """
    metrica, mayor_es_mejor = METRICA_SELECCION[pregunta]
    base, aceptados = elegir(runs, pregunta)
    if aceptados.empty:
        raise ValueError(f"{pregunta}: ningún candidato le gana al baseline en {metrica}")

    nombre = MODELOS_REGISTRO[pregunta]
    versiones = {}
    for alias, (_, run) in zip((ALIAS_PRODUCCION, ALIAS_RESPALDO), aceptados.head(2).iterrows()):
        versiones[alias] = registrar_version(cliente, nombre, run)
        cliente.update_model_version(
            nombre, versiones[alias], description=describir_version(run, base, pregunta)
        )
        cliente.set_registered_model_alias(nombre, alias, versiones[alias])
    # Si ya no hay segundo candidato, el challenger anterior deja de aplicar
    vigentes = cliente.get_registered_model(nombre).aliases
    if ALIAS_RESPALDO not in versiones and ALIAS_RESPALDO in vigentes:
        cliente.delete_registered_model_alias(nombre, ALIAS_RESPALDO)

    sentido = "más" if mayor_es_mejor else "menos"
    cliente.update_registered_model(
        nombre,
        description=(
            f"{DESCRIPCIONES[pregunta]} Se elige con `{metrica}` ({sentido} es mejor) contra el "
            f"baseline. `@{ALIAS_PRODUCCION}` es la versión en producción y `@{ALIAS_RESPALDO}` "
            f'la siguiente mejor. Se carga con `mlflow.pyfunc.load_model("models:/{nombre}@'
            f'{ALIAS_PRODUCCION}")`.'
        ),
    )
    cliente.set_registered_model_tag(nombre, "question", pregunta)
    cliente.set_registered_model_tag(nombre, "metrica_seleccion", metrica)

    champion = aceptados.iloc[0]
    valor, referencia = champion[f"metrics.{metrica}"], base[f"metrics.{metrica}"]
    return {
        "nombre": nombre,
        "version": versiones[ALIAS_PRODUCCION],
        "metrica": metrica,
        "valor": valor,
        "baseline": referencia,
        "mejora": mejora(valor, referencia, mayor_es_mejor),
    }


def comprobar_carga(nombre: str) -> int:
    """Carga el champion por su nombre y alias y predice sobre su ejemplo de entrada.

    Es la misma carga que usarán las siguientes etapas del proyecto.

    Args:
        nombre: Nombre del modelo registrado.

    Returns:
        El número de predicciones.

    Raises:
        RuntimeError: Si el modelo no trae ejemplo de entrada o no da una predicción por fila.
    """
    modelo = mlflow.pyfunc.load_model(f"models:/{nombre}@{ALIAS_PRODUCCION}")
    ejemplo = modelo.input_example
    if ejemplo is None:
        raise RuntimeError(f"{nombre}@{ALIAS_PRODUCCION} no tiene ejemplo de entrada")
    prediccion = modelo.predict(ejemplo)
    if len(prediccion) != len(ejemplo):
        raise RuntimeError(f"{nombre}@{ALIAS_PRODUCCION} no da una predicción por fila")
    return len(prediccion)


@app.command()
def main(question: str = typer.Option("todas", help="p1, p2, p3 o todas")) -> None:
    """Registra el mejor modelo de cada pregunta y comprueba que su champion se puede cargar."""
    pregunta = question.upper()
    if pregunta not in (*MODELOS_REGISTRO, "TODAS"):
        raise typer.BadParameter(f"question debe ser una de {[*MODELOS_REGISTRO, 'todas']}")

    mlflow.set_tracking_uri(TRACKING_URI)
    runs, cliente = buscar_runs(), MlflowClient()
    fallas = []
    for p in MODELOS_REGISTRO if pregunta == "TODAS" else [pregunta]:
        try:
            resumen = registrar(p, runs, cliente)
            filas = comprobar_carga(resumen["nombre"])
        except (ValueError, RuntimeError) as error:
            logger.error(error)
            fallas.append(p)
            continue
        logger.success(
            f"{p}: {resumen['nombre']} v{resumen['version']} @{ALIAS_PRODUCCION} "
            f"({resumen['metrica']} = {resumen['valor']:.3f} contra {resumen['baseline']:.3f} "
            f"del baseline, {resumen['mejora']:+.1%}); predice {filas} filas de su ejemplo"
        )
    if fallas:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
