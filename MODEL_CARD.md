# Model Card: modelos de MiBici del Equipo 03

Ficha de los modelos que el proyecto tiene en producción, con la estructura de Mitchell et al. (2019). Documenta qué hace cada modelo, con qué datos se entrenó, qué tan bien funciona y dónde no conviene usarlo. Las cifras salen del Model Registry oficial y de los reportes de cada entrega, y cada una indica su fuente.

| | Modelo en el Registry | `@champion` | `@challenger` | Lo sirve la API |
|---|---|---|---|---|
| **P1** Demanda | `mibici-equipo03-p1-demanda` | v1 | v2 (v3 en la corrida de la entrega 5) | `POST /predict/demanda` |
| **P2** Riesgo | `mibici-equipo03-p2-riesgo` | v1 | v2 | `POST /predict/riesgo` |
| **P3** Grupos | `mibici-equipo03-p3-grupos` | v1 | v2 | no |

## 1. Detalles comunes

- **Equipo:**
  - Anaí Lael Barrera Magaña;
  - Gerardo González Martínez;
  - Miguel Ángel Ramírez Castillo;
  - Sebastián Estrada García.

  Es el proyecto integrador de TC5061 Operaciones de aprendizaje automático, del Tecnológico de Monterrey (noviembre de 2026).
- **Linaje:** todas las versiones del Registry se entrenaron desde el commit `46b18e1`, con `git_dirty=False` y los datos `data_version = 4ab2182e7493`. Cada versión guarda su `source_run`, su `git_commit` y su `data_version` como etiquetas, además de una descripción con su métrica contra el baseline.
- **Estado de producción:**
  - `@champion` es la versión que se usa;
  - `@challenger` es la siguiente candidata;
  - no se usan etapas (`Staging`, `Production`), que están obsoletas en MLflow.

  La API carga `models:/<modelo>@champion` al arrancar (ver la sección [Model Registry](README.md#model-registry) del README).
- **Cómo se elige:**
  - `make registry` compara cada candidato contra su baseline con la métrica de selección de `config.py`, medida en validación (de junio a julio de 2026);
  - el periodo de prueba (agosto y septiembre de 2026) no se usa para elegir nada;
  - solo entran runs sin cambios sin commit, con los mismos datos y los mismos cortes.
- **Periodos:**

  | Conjunto | Periodo |
  |---|---|
  | Entrenamiento final | octubre de 2025 a julio de 2026 |
  | Validación (para elegir) | junio y julio de 2026 |
  | Prueba | agosto y septiembre de 2026 |
  | Monitoreo y A/B | agosto a octubre de 2026 |

  La validación se ajusta primero con los meses anteriores a junio. Después, el modelo elegido se reentrena con todo el entrenamiento.

## 2. P1: demanda por estación

**Detalles.**
- `HistGradientBoostingRegressor` de scikit-learn;
- `learning_rate` 0.1, `max_iter` 100, `max_leaf_nodes` 63, `loss` squared_error, semilla 42;
- run `1259eb65aa2443adabccc41a6b32a308`.

**Uso previsto.** Pronostica cuántos viajes saldrán de cada estación al día siguiente, para que el equipo de operación de MiBici planee la redistribución de bicicletas.

**Fuera de alcance:**
- decisiones sobre personas;
- tarifas;
- estaciones nuevas sin al menos 7 días de historia;
- días con eventos que el modelo no conoce (cierres, obras, eventos masivos).

**Variables.** `viajes_lag_1`, `viajes_lag_7`, `viajes_media_7`, `dia_semana`, `festivo` y `capacidad`. Todas se conocen antes del día que se predice.

**Métricas** (Registry oficial, descripción de la v1):

| Conjunto | MAE (viajes) | RMSE | R² | MAE del baseline (`viajes_lag_7`) |
|---|---|---|---|---|
| Validación | 5.83 | 10.36 | 0.892 | 7.84 |
| Prueba (ago–sep 2026) | 5.54 | 10.36 | 0.910 | 6.91 |

**Monitoreo** (entrega 5, `reports/evidently/p1_por_mes.csv`):

| Mes | `drift_share` | MAE del champion | MAE del baseline | ¿Reentrenar? |
|---|---|---|---|---|
| Agosto 2026 | 0.625 | 4.67 | 5.81 | no |
| Septiembre 2026 | 0.000 | 6.42 | 8.02 | sí |
| Octubre 2026 | 0.625 | 7.48 | 8.57 | sí |

- El drift y el error no se mueven juntos. Por eso el drift (Evidently, `drift_share` ≥ 0.5) es solo una alerta temprana.
- La decisión de reentrenar se toma con el error: cuando el MAE del mes supera en 15 % al de referencia (`TOLERANCIA_MAE_P1`).
- El champion le gana al baseline todos los meses.

**A/B** (entrega 5, `reports/ab_testing/p1_ab.csv`):
- Se comparó contra la v3, reentrenada con dos meses más, sobre las 14,540 filas de octubre de 2026.
- La v3 mejora el MAE en 0.09 viajes (1.2 %), con un IC 95 % de [+0.040, +0.156].
- Pero el Wilcoxon por estación da p = 0.198, así que la regla fijada antes (IC a favor y p < 0.05) dice **mantener la v1**.
- La mejora se concentra en el 25 % de estaciones con más demanda.

**Limitaciones.**
- Puede dar predicciones negativas, que no tienen sentido al contar viajes: el mínimo fue de −30.44 en el periodo de referencia y de −5.08 en el actual (entrega 5). Quien lo use debe tomar el máximo entre la predicción y 0.
- No conoce el rebalanceo, las estaciones cerradas ni los incidentes, porque esos datos no se publican.
- El error crece en los meses de más demanda (septiembre y octubre).

## 3. P2: riesgo de quedarse sin bicis o sin anclajes

**Detalles.**
- `HistGradientBoostingClassifier` con `class_weight` balanced;
- `learning_rate` 0.05, `max_iter` 100, semilla 42;
- run `a8438298e9e84b38a7d8a35f2254b830`.

**Uso previsto.** Marca las estaciones en riesgo de quedarse sin bicis o sin anclajes en el siguiente bloque de 3 horas, para decidir a cuáles manda el camión de rebalanceo.

**Fuera de alcance:**
- usarlo como garantía de disponibilidad para los usuarios;
- bloques de menos de 3 horas.

**Variables.** `flujo_neto_lag_1`, `flujo_neto_lag_2`, `salidas_lag_1`, `llegadas_lag_1`, `riesgo_semana_anterior`, `hora`, `dia_semana`, `festivo` y `capacidad`.

**Umbral de decisión: 0.889, guardado dentro del modelo.**
- No es 0.5. Se eligió en validación como el más bajo que deja en promedio a lo más 10 alertas por bloque (`CAPACIDAD_REBALANCEO`): las estaciones que un camión alcanza a visitar en 3 horas.
- La API lo devuelve en cada respuesta.

**Métricas** (Registry oficial, v1):

| Conjunto | Average precision | Precisión | Recall | F1 | Alertas por bloque | Baseline (AP) |
|---|---|---|---|---|---|---|
| Validación | 0.329 | 0.429 | 0.279 | 0.338 | 10.0 | 0.133 |
| Prueba | 0.386 | 0.490 | 0.322 | 0.389 | 13.3 | 0.168 |

- Solo el 4.4 % de los bloques de prueba son de riesgo, así que una AP de 0.386 es más del doble que la del baseline.

**Limitaciones.**
- **El modelo prioriza la precisión sobre el recall** para no saturar al camión: deja pasar cerca de dos de cada tres bloques en riesgo. Debe complementar la supervisión del operador, no reemplazarla.
- **En prueba da 13.3 alertas por bloque, más que las 10 de validación.** Si la demanda sigue creciendo, conviene recalibrar el umbral.
- **Tiene los mismos huecos de datos que P1:** rebalanceo, cierres e incidentes.

## 4. P3: grupos de estaciones (no se sirve)

- **El modelo:** `KMeans` con k = 2 (se probaron k de 2 a 8), run `2e73f734ad334461bc1e07c59b967b9d`.
- **Para qué sirve:** agrupa las 426 estaciones con al menos 30 días de uso según su patrón por hora. Se usa para análisis, no para decisiones en línea.
- **Cómo se eligió:**
  - por silueta (0.110 contra 0.017 del baseline);
  - además, pasó el criterio de estabilidad de `make registry` con 10 semillas: desviación de la silueta 0.000 (máximo 0.01) y el grupo más chico con el 47.7 % de las estaciones (mínimo 5 %).
- **Limitaciones:**
  - la silueta es baja, así que los grupos se traslapan;
  - k = 2 está en el borde del rango probado, y por debajo de 2 no hay grupos.

## 5. Datos, privacidad y consideraciones éticas

- **Fuente:** los [datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/), con los viajes de octubre de 2025 a octubre de 2026 y el catálogo de 470 estaciones.
- **Datos personales:**
  - los viajes traen identificador de usuario, género y año de nacimiento;
  - la limpieza quita los identificadores;
  - **ningún modelo usa género ni edad**: las variables son agregados por estación y por día o bloque;
  - las predicciones son por estación, nunca por persona.
- **Nada de esto se versiona:** ni los datos ni los modelos ni la base de MLflow. `data/manifest.json` guarda solo el número de filas y el hash. `tests/test_repositorio.py` verifica que no se suban datos ni credenciales.
- **Riesgos de sesgo:**
  - los modelos aprenden de dónde ya hay más viajes. Si el rebalanceo sigue solo sus alertas, las zonas con menos uso podrían recibir menos atención;
  - las estaciones nuevas no tienen historia y el modelo no las cubre bien.

  Se recomienda revisar las alertas por zona y no usar el modelo para decidir dónde abrir o cerrar estaciones.
- **Impacto de un error:**
  - un falso positivo de P2 manda el camión a una estación que no lo necesitaba (costo operativo);
  - un falso negativo deja a usuarios sin bici o sin anclaje.

  Nadie queda excluido de un servicio por una predicción.

## 6. Operación, monitoreo y gobernanza

- **Servicio:**
  - API FastAPI en contenedor, que se levanta con MLflow mediante Docker Compose (`make sistema`);
  - el CI de GitHub Actions corre lint, pruebas y seguridad, construye la imagen, levanta el sistema completo y publica la imagen en GHCR con cada merge a `main`.
- **Desempeño y costo** (entrega 6, `reports/carga/`):
  - con 200 usuarios simultáneos, la API sostiene 189.6 peticiones por segundo con p95 de 70 ms y 0 errores en 20,782 peticiones;
  - en una AWS EC2 t3.medium ($0.0416 USD/h), cuesta $5.78 × 10⁻⁷ USD por 1,000 predicciones a plena capacidad;
  - con la demanda real de MiBici, una sola instancia basta.
- **Monitoreo:**
  - `make drift`: reporte de Evidently por periodo y por mes;
  - `make ab-test`: comparación champion contra challenger;
  - los dos registran su run en MLflow.
- **Quién decide un cambio de versión:**
  - la regla del A/B está fija en `config.py` antes de correrlo;
  - los aliases solo se mueven con `--promover` y si la regla se cumple;
  - todo cambio de código o de configuración entra por pull request con la revisión de otro integrante y el CI en verde.
- **Reentrenamiento:**
  - se recomienda mensualmente, cuando MiBici publica un mes nuevo y el MAE pasa la tolerancia;
  - `make reentrenar` registra la nueva versión como `@challenger` con su linaje y sin tocar al champion.

## Referencias

- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. En *Proceedings of the Conference on Fairness, Accountability, and Transparency* (pp. 220–229). ACM. https://doi.org/10.1145/3287560.3287596
- Reportes del equipo en `reports/`:
  - `TC5061_Reporte_MLFlow_Equipo03.pdf` (entrega 2);
  - `TC5061_Drift_ABTest_Equipo03.pdf` (entrega 5);
  - `TC5061_Operacion_FinOps_Equipo03.pdf` (entrega 6).
