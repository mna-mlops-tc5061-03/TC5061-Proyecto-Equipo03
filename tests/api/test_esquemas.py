from mi_bici.api.esquemas import EJEMPLO_P2, ObservacionRiesgo, SolicitudRiesgo
from mi_bici.config import VARIABLES_P2


class TestEsquemas:
    def test_campos_son_la_firma_del_modelo(self):
        """Verifica que los campos de la observación son exactamente VARIABLES_P2 y en su orden."""
        assert list(ObservacionRiesgo.model_fields) == VARIABLES_P2

    def test_ejemplo_de_docs_es_valido(self):
        """Verifica que el ejemplo que muestra /docs pasa la validación tal cual."""
        solicitud = SolicitudRiesgo.model_validate({"observaciones": EJEMPLO_P2})

        assert len(solicitud.observaciones) == len(EJEMPLO_P2)
