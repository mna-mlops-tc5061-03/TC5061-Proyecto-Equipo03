from mi_bici.api.esquemas import (
    EJEMPLO_P1,
    EJEMPLO_P2,
    ObservacionDemanda,
    ObservacionRiesgo,
    SolicitudDemanda,
    SolicitudRiesgo,
)
from mi_bici.config import VARIABLES_P1, VARIABLES_P2


class TestEsquemas:
    def test_campos_son_la_firma_del_modelo(self):
        """Verifica que los campos de la observación son exactamente VARIABLES_P2 y en su orden."""
        assert list(ObservacionRiesgo.model_fields) == VARIABLES_P2

    def test_ejemplo_de_docs_es_valido(self):
        """Verifica que el ejemplo que muestra /docs pasa la validación tal cual."""
        solicitud = SolicitudRiesgo.model_validate({"observaciones": EJEMPLO_P2})

        assert len(solicitud.observaciones) == len(EJEMPLO_P2)

    def test_campos_de_p1_son_la_firma_del_modelo(self):
        """Verifica que los campos de la observación de P1 son exactamente VARIABLES_P1."""
        assert list(ObservacionDemanda.model_fields) == VARIABLES_P1

    def test_ejemplo_de_docs_de_p1_es_valido(self):
        """Verifica que el ejemplo de P1 que muestra /docs pasa la validación tal cual."""
        solicitud = SolicitudDemanda.model_validate({"observaciones": EJEMPLO_P1})

        assert len(solicitud.observaciones) == len(EJEMPLO_P1)
