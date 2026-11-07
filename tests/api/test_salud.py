"""Pruebas de /health y de que cada pregunta carga su modelo por separado."""

import copy

from mi_bici.api.app import PREGUNTAS
from mi_bici.api.esquemas import EJEMPLO_P1, EJEMPLO_P2


class TestSalud:
    def test_muestra_nombre_y_version_de_cada_modelo(self, cliente):
        respuesta = cliente.get("/health")

        assert respuesta.status_code == 200
        assert respuesta.json() == {
            "estado": "ok",
            "modelos": {
                "demanda": {
                    "modelo": "mibici-equipo03-p1-demanda",
                    "version": "3",
                    "alias": "champion",
                },
                "riesgo": {
                    "modelo": "mibici-equipo03-p2-riesgo",
                    "version": "7",
                    "alias": "champion",
                },
            },
        }

    def test_da_503_sin_modelos(self, cliente_sin_modelo):
        assert cliente_sin_modelo.get("/health").status_code == 503

    def test_da_503_si_falta_uno_y_dice_cual(self, cliente_solo_riesgo):
        respuesta = cliente_solo_riesgo.get("/health")

        assert respuesta.status_code == 503
        assert PREGUNTAS["demanda"].uri in respuesta.json()["detail"]
        assert PREGUNTAS["riesgo"].uri not in respuesta.json()["detail"]


class TestPreguntasIndependientes:
    def test_si_p1_no_carga_p2_sigue_respondiendo(self, cliente_solo_riesgo):
        demanda = cliente_solo_riesgo.post(
            "/predict/demanda", json={"observaciones": copy.deepcopy(EJEMPLO_P1)}
        )
        riesgo = cliente_solo_riesgo.post(
            "/predict/riesgo", json={"observaciones": copy.deepcopy(EJEMPLO_P2)}
        )

        assert demanda.status_code == 503
        assert riesgo.status_code == 200

    def test_cada_ruta_usa_su_propio_modelo(self, cliente, servicio_demanda, servicio_riesgo):
        cliente.post("/predict/demanda", json={"observaciones": copy.deepcopy(EJEMPLO_P1)})

        assert servicio_demanda["modelo"].ultimos_datos is not None
        assert servicio_riesgo["modelo"].ultimos_datos is None

    def test_hay_una_ruta_por_pregunta(self, cliente):
        rutas = {r.path for r in cliente.app.routes}

        assert {f"/predict/{nombre}" for nombre in PREGUNTAS} <= rutas
