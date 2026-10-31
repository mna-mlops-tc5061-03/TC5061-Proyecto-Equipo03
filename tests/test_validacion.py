import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import get_scorer

from mi_bici.validacion import (
    condicion_pares_correlacionados,
    guardar_resumen,
    preparar_deepchecks,
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
