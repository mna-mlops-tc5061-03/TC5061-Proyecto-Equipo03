from mlflow.projects import load_project

from mi_bici.config import PROJ_ROOT


class TestMLproject:
    def test_mlproject_declara_entorno_e_ingest(self):
        """Verifica que el MLproject carga, usa python_env.yaml y expone la entrada ingest."""
        proyecto = load_project(str(PROJ_ROOT))

        assert proyecto.env_type == "python_env"
        assert proyecto.env_config_path.endswith("python_env.yaml")
        assert proyecto.get_entry_point("ingest").command == "python -m mi_bici.dataset"

    def test_entrada_train_tiene_parametros_con_valor_por_defecto(self):
        """Verifica que la entrada train recibe question, model y params con valor por defecto."""
        entrada = load_project(str(PROJ_ROOT)).get_entry_point("train")

        assert set(entrada.parameters) == {"question", "model", "params"}
        assert all(p.default is not None for p in entrada.parameters.values())

    def test_entrada_search_tiene_parametros_con_valor_por_defecto(self):
        """Verifica que la entrada search recibe question, model y grid con valor por defecto."""
        entrada = load_project(str(PROJ_ROOT)).get_entry_point("search")

        assert set(entrada.parameters) == {"question", "model", "grid"}
        assert all(p.default is not None for p in entrada.parameters.values())
