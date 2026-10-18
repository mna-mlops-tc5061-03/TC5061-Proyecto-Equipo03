# Referencias

Fuentes que respaldan las decisiones del proyecto y que se citan en la propuesta. Formato APA 7.

## Lista de referencias

Conventional Commits. (s.f.). *Conventional Commits 1.0.0*. Recuperado el 17 de octubre de 2026, de https://www.conventionalcommits.org/es/v1.0.0/

Dorard, L. (2015). *The Machine Learning Canvas* (versión 1.2) [Plantilla]. OWNML. https://www.ownml.co/machine-learning-canvas

Dorard, L. (2018, 9 de marzo). *From data to AI with the Machine Learning Canvas (Part I)*. Own Machine Learning. https://medium.com/louis-dorard/from-data-to-ai-with-the-machine-learning-canvas-part-i-d171b867b047

DrivenData. (s.f.-a). *Cookiecutter Data Science*. Recuperado el 10 de octubre de 2026, de https://cookiecutter-data-science.drivendata.org/

DrivenData. (s.f.-b). *Opinions*. Cookiecutter Data Science. Recuperado el 10 de octubre de 2026, de https://cookiecutter-data-science.drivendata.org/opinions/

DrivenData. (s.f.-c). *Using the template*. Cookiecutter Data Science. Recuperado el 10 de octubre de 2026, de https://cookiecutter-data-science.drivendata.org/using-the-template/

DrivenData. (s.f.-d). *Why ccds?* Cookiecutter Data Science. Recuperado el 10 de octubre de 2026, de https://cookiecutter-data-science.drivendata.org/why/

GitHub. (s.f.-a). *About Git*. GitHub Docs. Recuperado el 17 de octubre de 2026, de https://docs.github.com/en/get-started/using-git/about-git

GitHub. (s.f.-b). *About protected branches*. GitHub Docs. Recuperado el 17 de octubre de 2026, de https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches

GitHub. (s.f.-c). *About pull request reviews*. GitHub Docs. Recuperado el 17 de octubre de 2026, de https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/about-pull-request-reviews

GitHub. (s.f.-d). *GitHub flow*. GitHub Docs. Recuperado el 17 de octubre de 2026, de https://docs.github.com/en/get-started/using-github/github-flow

Lauchande, N. (2021). *Machine learning engineering with MLflow: Manage the end-to-end machine learning life cycle with MLflow*. Packt Publishing.

Maddula, S. (2025, 19 de marzo). *Feature engineering in machine learning: A practical guide*. DataCamp. https://www.datacamp.com/tutorial/feature-engineering

MiBici. (s.f.). *Datos abiertos* [Conjunto de datos]. Sistema de Bicicleta Pública del Área Metropolitana de Guadalajara. Recuperado el 17 de octubre de 2026, de https://mibici.net/es/datos-abiertos/

MLflow. (s.f.). *MLflow Tracking* [Documentación]. Recuperado el 17 de octubre de 2026, de https://mlflow.org/docs/latest/ml/tracking/

The pandas development team. (s.f.). *IO tools (text, CSV, HDF5, …): Parquet*. pandas documentation. Recuperado el 17 de octubre de 2026, de https://pandas.pydata.org/docs/user_guide/io.html#parquet

Python Software Foundation. (s.f.). *hashlib — Secure hashes and message digests*. Python 3 documentation. Recuperado el 17 de octubre de 2026, de https://docs.python.org/3/library/hashlib.html

scikit-learn developers. (s.f.-a). *Clustering*. scikit-learn user guide. Recuperado el 17 de octubre de 2026, de https://scikit-learn.org/stable/modules/clustering.html

scikit-learn developers. (s.f.-b). *Metrics and scoring: Quantifying the quality of predictions*. scikit-learn user guide. Recuperado el 17 de octubre de 2026, de https://scikit-learn.org/stable/modules/model_evaluation.html

scikit-learn developers. (s.f.-c). *Preprocessing data*. scikit-learn user guide. Recuperado el 17 de octubre de 2026, de https://scikit-learn.org/stable/modules/preprocessing.html

scikit-learn developers. (s.f.-d). *Supervised learning*. scikit-learn user guide. Recuperado el 17 de octubre de 2026, de https://scikit-learn.org/stable/supervised_learning.html

Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J.-F., & Dennison, D. (2015). Hidden technical debt in machine learning systems. En C. Cortes, N. Lawrence, D. Lee, M. Sugiyama y R. Garnett (Eds.), *Advances in Neural Information Processing Systems 28* (pp. 2503–2511). Curran Associates.

Treveil, M., Omont, N., Stenac, C., Lefevre, K., Phan, D., Zentici, J., Lavoillotte, A., Miyazaki, M., & Heidmann, L. (2020). *Introducing MLOps: How to scale machine learning in the enterprise*. O'Reilly Media.

Wilson, B. (2022). *Machine learning engineering in action*. Manning Publications.

## Qué decisión respalda cada fuente

| Decisión del proyecto | Fuentes |
|---|---|
| Formular P1, P2 y P3 con el ML Canvas | Dorard (2015, 2018) |
| Datos de viajes y estaciones | MiBici (s.f.) |
| Estructura del repositorio, datos crudos inmutables y fuera de Git, notebooks solo para explorar, secretos en `.env` | DrivenData (s.f.-a, s.f.-b, s.f.-c, s.f.-d) |
| Código en módulos en lugar de notebooks; evitar deuda técnica | Sculley et al. (2015); Wilson (2022) |
| Reproducibilidad y versión de los datos con un manifiesto con hash | Treveil et al. (2020); Python Software Foundation (s.f.) |
| Datos limpios en Parquet | The pandas development team (s.f.) |
| Registro de experimentos | MLflow (s.f.); Lauchande (2021) |
| Variables sin fuga de información | Maddula (2025); scikit-learn developers (s.f.-c) |
| Modelos y métricas por tipo de tarea | scikit-learn developers (s.f.-a, s.f.-b, s.f.-d) |
| Flujo de trabajo con ramas, Pull Requests y rama protegida | GitHub (s.f.-a, s.f.-b, s.f.-c, s.f.-d); Conventional Commits (s.f.) |

Los diagramas del proyecto están en `references/arquitectura.svg` y `references/pipeline.svg`, con su versión PNG para documentos.
