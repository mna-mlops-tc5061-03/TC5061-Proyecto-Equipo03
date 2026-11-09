"""Pruebas de POST /predict/demanda (P1) con un modelo falso (ver conftest.py)."""

import copy

import numpy as np
import pytest

from mi_bici.api.esquemas import EJEMPLO_P1
from mi_bici.config import MAX_OBSERVACIONES, VARIABLES_P1

RUTA = "/predict/demanda"


def observacion(**cambios) -> dict:
    """Una observación válida, con los campos que se quieran cambiar."""
    return {**copy.deepcopy(EJEMPLO_P1[0]), **cambios}


def lote(n: int) -> dict:
    return {"observaciones": [observacion() for _ in range(n)]}


class TestCasosNormales:
    def test_una_observacion(self, cliente):
        """El modelo falso predice viajes_media_7 + 1."""
        respuesta = cliente.post(RUTA, json={"observaciones": [observacion(viajes_media_7=20.0)]})

        assert respuesta.status_code == 200
        assert respuesta.json() == {
            "predicciones": [{"viajes": 21.0}],
            "modelo": "mibici-equipo03-p1-demanda",
            "version": "3",
        }

    def test_responde_con_las_filas_del_serving_input_example(self, cliente):
        respuesta = cliente.post(RUTA, json={"observaciones": EJEMPLO_P1})

        assert respuesta.status_code == 200
        viajes = [p["viajes"] for p in respuesta.json()["predicciones"]]
        assert viajes == pytest.approx([fila["viajes_media_7"] + 1 for fila in EJEMPLO_P1])


class TestCasosLimite:
    @pytest.mark.parametrize("dia_semana", [0, 6])
    def test_dias_extremos_son_validos(self, cliente, dia_semana):
        respuesta = cliente.post(
            RUTA, json={"observaciones": [observacion(dia_semana=dia_semana)]}
        )

        assert respuesta.status_code == 200

    def test_ceros_en_lags_y_media_son_validos(self, cliente):
        cero = observacion(viajes_lag_1=0.0, viajes_lag_7=0.0, viajes_media_7=0.0)

        assert cliente.post(RUTA, json={"observaciones": [cero]}).status_code == 200

    def test_lote_del_tamano_maximo(self, cliente):
        respuesta = cliente.post(RUTA, json=lote(MAX_OBSERVACIONES))

        assert respuesta.status_code == 200
        assert len(respuesta.json()["predicciones"]) == MAX_OBSERVACIONES

    @pytest.mark.parametrize(
        "cambios",
        [
            {"viajes_lag_1": -1.0},
            {"viajes_lag_7": -1.0},
            {"viajes_media_7": -0.5},
            {"dia_semana": -1},
            {"dia_semana": 7},
            {"capacidad": 0},
            {"festivo": "quizás"},
            {"dia_semana": "martes"},
        ],
        ids=lambda c: "-".join(f"{k}={v}" for k, v in c.items()),
    )
    def test_valores_invalidos_dan_422(self, cliente, cambios):
        respuesta = cliente.post(RUTA, json={"observaciones": [observacion(**cambios)]})

        assert respuesta.status_code == 422

    @pytest.mark.parametrize("campo", VARIABLES_P1)
    def test_campo_faltante_da_422(self, cliente, campo):
        incompleta = observacion()
        del incompleta[campo]

        respuesta = cliente.post(RUTA, json={"observaciones": [incompleta]})

        assert respuesta.status_code == 422

    def test_campo_de_p2_da_422(self, cliente):
        """Un campo de otra pregunta es un campo extra."""
        respuesta = cliente.post(RUTA, json={"observaciones": [observacion(hora=6)]})

        assert respuesta.status_code == 422

    def test_lote_vacio_da_422(self, cliente):
        assert cliente.post(RUTA, json=lote(0)).status_code == 422

    def test_lote_demasiado_grande_da_422(self, cliente):
        assert cliente.post(RUTA, json=lote(MAX_OBSERVACIONES + 1)).status_code == 422

    def test_el_modelo_no_se_llama_si_la_entrada_es_invalida(self, cliente, servicio_demanda):
        respuesta = cliente.post(RUTA, json={"observaciones": [observacion(dia_semana=7)]})

        assert respuesta.status_code == 422
        assert servicio_demanda["modelo"].ultimos_datos is None


class TestSinModelo:
    def test_predict_da_503(self, cliente_sin_modelo):
        assert cliente_sin_modelo.post(RUTA, json=lote(1)).status_code == 503


class TestConversionDeTipos:
    def test_columnas_llegan_en_el_orden_de_la_firma(self, cliente, servicio_demanda):
        cliente.post(RUTA, json=lote(2))

        assert list(servicio_demanda["modelo"].ultimos_datos.columns) == VARIABLES_P1

    def test_columnas_llegan_con_los_tipos_de_la_firma(self, cliente, servicio_demanda):
        cliente.post(RUTA, json=lote(2))

        datos = servicio_demanda["modelo"].ultimos_datos
        assert datos["dia_semana"].dtype == np.int32
        assert datos["capacidad"].dtype == np.int64
        assert datos["festivo"].dtype == np.int64
        assert datos["viajes_media_7"].dtype == np.float64
