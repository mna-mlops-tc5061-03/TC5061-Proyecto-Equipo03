import subprocess

import pytest

from mi_bici.modeling import base
from mi_bici.modeling.p2_riesgo import ExperimentoRiesgo
from mi_bici.modeling.train import EXPERIMENTOS, leer_params


class TestTrainBase:
    def test_leer_params_convierte_tipos(self):
        """Verifica que los números, booleanos y None conservan su tipo y el resto queda como texto."""
        params = leer_params(
            "learning_rate=0.05, max_iter=200,early_stopping=False,class_weight=balanced,max_depth=None"
        )

        assert params == {
            "learning_rate": 0.05,
            "max_iter": 200,
            "early_stopping": False,
            "class_weight": "balanced",
            "max_depth": None,
        }
        assert leer_params("") == {}

    def test_catalogo_tiene_baseline_y_candidato(self):
        """Verifica que cada pregunta tiene un baseline sin estimador y al menos un candidato."""
        for clase in EXPERIMENTOS.values():
            assert clase.catalogo["baseline"].clase is None
            assert any(m.clase is not None for m in clase.catalogo.values())

    def test_params_del_run_reemplazan_al_catalogo(self):
        """Verifica que los hiperparámetros del run se suman a los del catálogo y llegan al estimador."""
        experimento = ExperimentoRiesgo("hgb", {"learning_rate": 0.05}, semilla=7)
        estimador = experimento.crear_modelo()

        assert experimento.tipo == "candidato"
        assert estimador.class_weight == "balanced"
        assert estimador.learning_rate == 0.05
        assert estimador.random_state == 7

    def test_modelo_desconocido_falla(self):
        """Verifica que una clave fuera del catálogo se rechaza con un mensaje claro."""
        with pytest.raises(ValueError, match="modelo debe ser uno de"):
            ExperimentoRiesgo("xgboost")

    def test_p1_y_p3_guardan_su_modelo_sin_envolver(self):
        """Verifica que P1 y P3 devuelven self.modelo tal cual en modelo_a_guardar."""
        for clave in ("P1", "P3"):
            experimento = EXPERIMENTOS[clave]("baseline")
            assert experimento.modelo_a_guardar() is None
            modelo = next(k for k in EXPERIMENTOS[clave].catalogo if k != "baseline")
            candidato = EXPERIMENTOS[clave](modelo)
            candidato.modelo = candidato.crear_modelo()
            assert candidato.modelo_a_guardar() is candidato.modelo


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Crea un repositorio con código y un reporte versionados, y apunta `estado_git` a él."""
    (tmp_path / "mi_bici").mkdir()
    (tmp_path / "mi_bici" / "config.py").write_text("SEMILLA = 42\n", encoding="utf-8")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "reporte.md").write_text("versión 1\n", encoding="utf-8")
    identidad = ["-c", "user.name=Prueba", "-c", "user.email=prueba@example.com"]
    for args in (
        ["init", "-q"],
        ["add", "."],
        [*identidad, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "inicial"],
    ):
        subprocess.run(  # nosec B603 B607: comando y argumentos fijos, sin entrada externa
            ["git", *args], cwd=tmp_path, check=True, capture_output=True
        )
    monkeypatch.setattr(base, "PROJ_ROOT", tmp_path)
    return tmp_path


class TestEstadoGit:
    def test_repositorio_limpio(self, repo):
        """Verifica que sin cambios el run queda limpio y con el commit actual."""
        estado = base.estado_git()

        assert estado["git_dirty"] == "False"
        assert len(estado["git_commit"]) == 40

    def test_salidas_en_reports_no_ensucian(self, repo):
        """Verifica que un reporte nuevo y uno versionado que se regenera no marcan el run."""
        (repo / "reports" / "evidently").mkdir()
        (repo / "reports" / "evidently" / "p1_drift.csv").write_text("a\n", encoding="utf-8")
        (repo / "reports" / "reporte.md").write_text("versión 2\n", encoding="utf-8")

        assert base.estado_git()["git_dirty"] == "False"

    @pytest.mark.parametrize("ruta", ["mi_bici/config.py", "mi_bici/nuevo.py", "reports.py"])
    def test_cambios_fuera_de_reports_si_ensucian(self, repo, ruta):
        """Verifica que el código cambiado o nuevo sí marca el run, aunque empiece con `reports`."""
        (repo / ruta).write_text("SEMILLA = 7\n", encoding="utf-8")

        assert base.estado_git()["git_dirty"] == "True"
