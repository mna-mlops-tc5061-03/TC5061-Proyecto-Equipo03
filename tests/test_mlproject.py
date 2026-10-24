from mlflow.projects import load_project

from mi_bici.config import PROJ_ROOT


class TestMLproject:
    def test_mlproject_declara_entorno_e_ingest(self):
        """Verifica que el MLproject carga, usa python_env.yaml y expone la entrada ingest."""
        proyecto = load_project(str(PROJ_ROOT))

        assert proyecto.env_type == "python_env"
        assert proyecto.env_config_path.endswith("python_env.yaml")
        assert proyecto.get_entry_point("ingest").command == "python -m mi_bici.dataset"
