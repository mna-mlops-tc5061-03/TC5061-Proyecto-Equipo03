import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import get_scorer

from mi_bici.validacion import (
    condicion_pares_correlacionados,
    guardar_resumen,
    preparar_deepchecks,
    reportar,
    separar_por_fecha,
    validar_drift,
    validar_integridad,
)


@pytest.fixture
def tabla_limpia():
    """Tabla sintética sin nulos, sin duplicados y con columnas independientes."""
    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame(
        {
            "duracion": rng.normal(15, 5, n).round(2),
            "capacidad": rng.integers(10, 40, n),
            "genero": rng.choice(["M", "F", "NE"], n),
        }
    )


def chequeos_que_no_pasaron(resumen: dict) -> set[str]:
    return {condicion["chequeo"] for condicion in resumen["no_pasaron"]}


class TestValidacionIntegridad:
    def test_deepchecks_se_importa_con_nuestras_versiones(self):
        """Verifica que tras preparar DeepChecks existen los alias de numpy y el scorer max_error."""
        dc_tabular = preparar_deepchecks()

        assert hasattr(dc_tabular, "Dataset")
        assert np.Inf == np.inf
        assert np.isnan(np.NaN)
        assert get_scorer("max_error") is not None

    def test_tabla_limpia_pasa(self, tabla_limpia):
        """Verifica que una tabla sin problemas pasa todas las condiciones."""
        resumen = validar_integridad(tabla_limpia, "sintetica")

        assert resumen["paso"], resumen["no_pasaron"]
        assert resumen["filas"] == len(tabla_limpia)
        assert resumen["no_corrieron"] == []

    def test_detecta_nulos(self, tabla_limpia):
        """Verifica que una columna con 20 % de nulos no pasa la condición de nulos (máximo 5 %)."""
        tabla_limpia.loc[tabla_limpia.sample(frac=0.2, random_state=0).index, "duracion"] = np.nan

        resumen = validar_integridad(tabla_limpia, "sintetica")

        assert not resumen["paso"]
        assert "PercentOfNulls" in chequeos_que_no_pasaron(resumen)

    def test_detecta_duplicados(self, tabla_limpia):
        """Verifica que 10 % de filas repetidas no pasa la condición de duplicados (máximo 1 %)."""
        con_duplicados = pd.concat([tabla_limpia, tabla_limpia.head(30)], ignore_index=True)

        resumen = validar_integridad(con_duplicados, "sintetica")

        assert not resumen["paso"]
        assert "Data Duplicates" in chequeos_que_no_pasaron(resumen)

    def test_reportar_dice_si_pasan_todas(self, tabla_limpia):
        """Verifica que `reportar` devuelve si pasa la tabla, para que la CLI salga con error."""
        assert reportar("limpia", validar_integridad(tabla_limpia, "sintetica"))

        tabla_limpia.loc[tabla_limpia.sample(frac=0.2, random_state=0).index, "duracion"] = np.nan
        assert not reportar("con_nulos", validar_integridad(tabla_limpia, "sintetica"))

    def test_un_poco_de_nulos_si_pasa(self, tabla_limpia):
        """Caso límite: 1 % de nulos queda debajo del umbral y no se marca."""
        tabla_limpia.loc[tabla_limpia.index[:3], "duracion"] = np.nan

        resumen = validar_integridad(tabla_limpia, "sintetica")

        assert "PercentOfNulls" not in chequeos_que_no_pasaron(resumen)

    def test_resumen_no_copia_filas(self, tabla_limpia, tmp_path):
        """Verifica que el resumen guardado no incluye valores de las filas."""
        tabla_limpia["comentario"] = "texto_privado_" + tabla_limpia.index.astype(str)
        resumen = validar_integridad(tabla_limpia, "sintetica")

        ruta = tmp_path / "resumen.json"
        guardar_resumen("integridad", {"sintetica": resumen}, ruta)

        assert "texto_privado" not in ruta.read_text(encoding="utf-8")
        assert (
            json.loads(ruta.read_text(encoding="utf-8"))["integridad"]["sintetica"]["filas"] == 300
        )


class TestCondicionCorrelacion:
    def test_no_marca_columnas_independientes(self):
        """Con pandas 3 la condición original marcaba todos los pares; la nuestra no."""
        matriz = pd.DataFrame(
            [[1.0, 0.1, np.nan], [0.1, 1.0, 0.2], [np.nan, 0.2, 1.0]],
            index=["a", "b", "c"],
            columns=["a", "b", "c"],
            dtype=object,
        )

        resultado = condicion_pares_correlacionados(0.9)(matriz)

        assert resultado.is_pass()

    def test_marca_columnas_redundantes(self):
        """Verifica que un par con correlación mayor al umbral se reporta una sola vez."""
        matriz = pd.DataFrame(
            [[1.0, 0.95], [0.95, 1.0]], index=["a", "b"], columns=["a", "b"], dtype=object
        )

        resultado = condicion_pares_correlacionados(0.9)(matriz)

        assert not resultado.is_pass()
        assert "('a', 'b')" in resultado.details
        assert "('b', 'a')" not in resultado.details


@pytest.fixture
def entrenamiento_y_prueba():
    """Dos conjuntos sintéticos tomados de la misma distribución, en fechas consecutivas."""
    rng = np.random.default_rng(1)

    def tabla(n: int, inicio: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "fecha": pd.date_range(inicio, periods=n, freq="h"),
                "viajes_lag_1": rng.poisson(20, n).astype(float),
                "dia_semana": rng.integers(0, 7, n),
                "capacidad": rng.integers(10, 40, n),
                "riesgo": rng.binomial(1, 0.1, n),
            }
        )

    return tabla(1500, "2026-01-01"), tabla(1500, "2026-08-01")


def validar(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    return validar_drift(
        train,
        test,
        "riesgo",
        categoricas=["dia_semana"],
        tipo_etiqueta="binary",
        fecha="fecha",
    )


def condiciones_de(resumen: dict, chequeo: str) -> list[dict]:
    return [c for c in resumen["condiciones"] if c["chequeo"] == chequeo]


class TestValidacionDrift:
    def test_misma_distribucion_no_marca_drift(self, entrenamiento_y_prueba):
        """Verifica que dos muestras de la misma distribución no marcan drift."""
        resumen = validar(*entrenamiento_y_prueba)

        for chequeo in ("Feature Drift", "Label Drift"):
            assert condiciones_de(resumen, chequeo), f"{chequeo} no corrió"
            assert all(c["estado"] == "pasa" for c in condiciones_de(resumen, chequeo))

    def test_detecta_variable_desplazada(self, entrenamiento_y_prueba):
        """Verifica que desplazar una variable en prueba marca drift en esa variable."""
        train, test = entrenamiento_y_prueba
        test = test.assign(viajes_lag_1=test["viajes_lag_1"] + 15)

        resumen = validar(train, test)

        assert any(c["estado"] != "pasa" for c in condiciones_de(resumen, "Feature Drift"))
        assert resumen["drift"]["viajes_lag_1"]["puntaje"] > 0.2
        assert resumen["drift"]["capacidad"]["puntaje"] < 0.2

    def test_detecta_cambio_en_la_tasa_del_objetivo(self, entrenamiento_y_prueba):
        """Caso límite: si la tasa de riesgo sube de 10 % a 60 %, LabelDrift lo marca."""
        train, test = entrenamiento_y_prueba
        test = test.assign(riesgo=np.random.default_rng(2).binomial(1, 0.6, len(test)))

        resumen = validar(train, test)

        assert any(c["estado"] != "pasa" for c in condiciones_de(resumen, "Label Drift"))
        assert resumen["objetivo"]["media_prueba"] > resumen["objetivo"]["media_entrenamiento"]

    def test_resumen_usa_el_nombre_de_la_tabla(self, entrenamiento_y_prueba):
        """Verifica que el resumen se identifica con el nombre de la tabla y no con el objetivo."""
        resumen = validar_drift(
            *entrenamiento_y_prueba, "riesgo", tipo_etiqueta="binary", nombre="P2"
        )

        assert resumen["tabla"] == "P2"
        assert resumen["objetivo"]["nombre"] == "riesgo"

    def test_separar_por_fecha(self, entrenamiento_y_prueba):
        """Verifica que el corte deja el entrenamiento antes y la prueba desde la fecha de corte."""
        tabla = pd.concat(entrenamiento_y_prueba, ignore_index=True)

        train, test = separar_por_fecha(tabla, "fecha", "2026-08-01")

        assert train["fecha"].max() < pd.Timestamp("2026-08-01") <= test["fecha"].min()
        assert len(train) + len(test) == len(tabla)
