"""Pruebas de POST /predict/riesgo (P2) con un modelo falso (ver conftest.py)."""

import copy

import numpy as np
import pytest

from mi_bici.api.esquemas import EJEMPLO_P2
from mi_bici.config import MAX_OBSERVACIONES, VARIABLES_P2

RUTA = "/predict/riesgo"


def observacion(**cambios) -> dict:
    """Una observación válida, con los campos que se quieran cambiar."""
    return {**copy.deepcopy(EJEMPLO_P2[0]), **cambios}


def lote(n: int) -> dict:
    return {"observaciones": [observacion() for _ in range(n)]}


class TestCasosNormales:
    def test_una_observacion(self, cliente):
        respuesta = cliente.post(RUTA, json={"observaciones": [observacion(salidas_lag_1=18.0)]})

        assert respuesta.status_code == 200
        cuerpo = respuesta.json()
        assert cuerpo["predicciones"] == [{"riesgo": True, "probabilidad": pytest.approx(0.9)}]
        assert cuerpo["umbral"] == 0.5
        assert cuerpo["modelo"] == "mibici-equipo03-p2-riesgo"
        assert cuerpo["version"] == "7"

    def test_lote_respeta_el_orden(self, cliente):
        """El ejemplo de /docs tiene 1, 13 y 17 salidas: probabilidades 0.05, 0.65 y 0.85."""
        respuesta = cliente.post(RUTA, json={"observaciones": EJEMPLO_P2})

        assert respuesta.status_code == 200
        predicciones = respuesta.json()["predicciones"]
        assert [p["riesgo"] for p in predicciones] == [False, True, True]
        assert [p["probabilidad"] for p in predicciones] == pytest.approx([0.05, 0.65, 0.85])


class TestCasosLimite:
    @pytest.mark.parametrize("hora", [0, 23])
    def test_horas_extremas_son_validas(self, cliente, hora):
        respuesta = cliente.post(RUTA, json={"observaciones": [observacion(hora=hora)]})

        assert respuesta.status_code == 200

    def test_lote_del_tamano_maximo(self, cliente):
        respuesta = cliente.post(RUTA, json=lote(MAX_OBSERVACIONES))

        assert respuesta.status_code == 200
        assert len(respuesta.json()["predicciones"]) == MAX_OBSERVACIONES

    @pytest.mark.parametrize(
        "cambios",
        [
            {"hora": -1},
            {"hora": 24},
            {"dia_semana": -1},
            {"dia_semana": 7},
            {"salidas_lag_1": -1.0},
            {"llegadas_lag_1": -1.0},
            {"riesgo_semana_anterior": -0.1},
            {"riesgo_semana_anterior": 1.1},
            {"capacidad": 0},
            {"hora": "seis"},
            {"festivo": "quizás"},
        ],
        ids=lambda c: "-".join(f"{k}={v}" for k, v in c.items()),
    )
    def test_valores_invalidos_dan_422(self, cliente, cambios):
        respuesta = cliente.post(RUTA, json={"observaciones": [observacion(**cambios)]})

        assert respuesta.status_code == 422

    @pytest.mark.parametrize("campo", VARIABLES_P2)
    def test_campo_faltante_da_422(self, cliente, campo):
        incompleta = observacion()
        del incompleta[campo]

        respuesta = cliente.post(RUTA, json={"observaciones": [incompleta]})

        assert respuesta.status_code == 422

    def test_campo_extra_da_422(self, cliente):
        respuesta = cliente.post(RUTA, json={"observaciones": [observacion(estacion_id=1)]})

        assert respuesta.status_code == 422

    def test_campo_extra_en_la_solicitud_da_422(self, cliente):
        respuesta = cliente.post(RUTA, json={**lote(1), "umbral": 0.1})

        assert respuesta.status_code == 422

    def test_lote_vacio_da_422(self, cliente):
        respuesta = cliente.post(RUTA, json=lote(0))

        assert respuesta.status_code == 422

    def test_lote_demasiado_grande_da_422(self, cliente):
        respuesta = cliente.post(RUTA, json=lote(MAX_OBSERVACIONES + 1))

        assert respuesta.status_code == 422

    def test_el_modelo_no_se_llama_si_la_entrada_es_invalida(self, cliente, servicio_riesgo):
        cliente.post(RUTA, json={"observaciones": [observacion(hora=24)]})

        assert servicio_riesgo["modelo"].ultimos_datos is None


class TestSinModelo:
    def test_predict_da_503(self, cliente_sin_modelo):
        respuesta = cliente_sin_modelo.post(RUTA, json=lote(1))

        assert respuesta.status_code == 503


class TestConversionDeTipos:
    def test_columnas_llegan_en_el_orden_de_la_firma(self, cliente, servicio_riesgo):
        cliente.post(RUTA, json=lote(2))

        assert list(servicio_riesgo["modelo"].ultimos_datos.columns) == VARIABLES_P2

    def test_columnas_llegan_con_los_tipos_de_la_firma(self, cliente, servicio_riesgo):
        cliente.post(RUTA, json=lote(2))

        datos = servicio_riesgo["modelo"].ultimos_datos
        assert datos["hora"].dtype == np.int32
        assert datos["dia_semana"].dtype == np.int32
        assert datos["capacidad"].dtype == np.int32
        assert datos["festivo"].dtype == np.bool_
        assert datos["flujo_neto_lag_1"].dtype == np.float64
