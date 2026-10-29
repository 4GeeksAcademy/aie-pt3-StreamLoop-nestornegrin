# Informe de ajuste · Clasificador de cancelación de StreamLoop

**Modelo:** `RandomForestClassifier` dentro de un `Pipeline` con imputación, escalado y one-hot encoding.
**Datos:** 7.043 clientes (26,5 % cancelan), divididos 80 / 20 de forma estratificada antes de cualquier otro paso.
**Detalle completo y reproducible:** [`churn_tuning.ipynb`](churn_tuning.ipynb).

## Resultado en una línea

El modelo ajustado detecta **295 de los 374 clientes que cancelan en test (78,9 %)**, frente a **178 (47,6 %)** de la
línea base: **117 cancelaciones más detectadas**, a cambio de emitir más alertas de retención.

## Línea base frente a modelo ajustado (test set, mismas métricas)

| Métrica | Línea base (por defecto) | Modelo ajustado | Diferencia |
|---|---|---|---|
| **F2 (métrica de optimización)** | 0,4986 | **0,7174** | **+0,2188** |
| Recall | 0,4759 | **0,7888** | +0,3128 |
| Precisión | 0,6159 | 0,5268 | −0,0891 |
| F1 | 0,5370 | 0,6317 | +0,0947 |
| Accuracy | 0,7821 | 0,7559 | −0,0263 |
| ROC-AUC | 0,8176 | 0,8443 | +0,0267 |
| Churners detectados | 178 de 374 | **295 de 374** | +117 |
| Alertas emitidas | 289 | 560 | +271 |

El test set se usó **exactamente dos veces**: una para la línea base y otra para el modelo final.

## Hiperparámetros finales

| Hiperparámetro | Valor | Por defecto |
|---|---|---|
| `class_weight` | `balanced_subsample` | `None` |
| `max_depth` | 6 | `None` (sin límite) |
| `max_features` | 0.3 | `sqrt` |
| `min_samples_leaf` | 2 | 1 |
| `min_samples_split` | 11 | 2 |
| `n_estimators` | 144 | 100 |

## Elección de la métrica: F2

StreamLoop pierde mucho más con un cliente que cancela sin ser detectado (falso negativo) que con una oferta de
retención innecesaria (falso positivo).

- **Accuracy** (la de sklearn por defecto) no sirve: decir "nadie cancela" ya acierta el 73,5 %. De hecho, nuestro modelo
  final tiene **menos accuracy** que la línea base y, aun así, detecta 117 cancelaciones más.
- **Recall solo** se maximiza marcando a todos los clientes como "cancela".
- **F2** combina precisión y recall dando **el doble de peso al recall**: premia detectar cancelaciones sin permitir
  la solución trivial. Se usó como `scoring` en `RandomizedSearchCV` y `GridSearchCV`.

## Proceso de búsqueda

Ambas búsquedas usaron solo la partición de entrenamiento, los mismos 5 folds estratificados, `n_jobs=-1`,
`scoring=F2` y `refit=True`.

1. **`RandomizedSearchCV`:** 60 combinaciones sobre `n_estimators`, `max_depth`, `min_samples_split`,
   `min_samples_leaf`, `max_features` y `class_weight`. Hallazgos:
   - `class_weight` es el factor decisivo: sin compensar el desequilibrio, el F2 mediano cae de ~0,70 a ~0,51.
   - Los árboles poco profundos (profundidad 4–6) generalizan mejor que los árboles sin límite.
   - Las 10 mejores combinaciones están separadas por 0,004 de F2, mucho menos que la variación entre folds (~0,02).
   - Mejor F2 en CV: **0,7226**.
2. **`GridSearchCV`:** 27 combinaciones alrededor del ganador (profundidad 4/6/8, hojas 1/2/5, 100/144/244 árboles).
   **Ninguna superó al ganador de la búsqueda aleatoria**, lo que confirma que estamos en una zona estable y no en un
   pico aislado.

## Trade-off de estabilidad considerado

Se revisó `cv_results_` para los mejores candidatos, con su media, su desviación y su peor y mejor fold:

| Candidato | F2 medio | Desviación | Peor fold | train − CV |
|---|---|---|---|---|
| **#1 (elegido)** · depth 6, 144 árboles | **0,7226** | 0,0166 | **0,6950** | 0,034 |
| #3 · depth 6, 100 árboles | 0,7201 | **0,0130** | 0,6951 | 0,036 |
| #4 · depth 4, 244 árboles | 0,7200 | 0,0263 | 0,6762 | **0,014** |

**Elegimos el #1, el de mayor promedio.**

- El #3 varía menos entre folds, pero su **peor fold es el mismo** (0,6951 frente a 0,6950): su estabilidad extra
  viene de tener folds buenos algo peores, no de proteger mejor en el peor caso. Costaría 0,0025 de F2 medio sin ninguna ganancia real.
- El #4 sobreajusta menos, pero es el más inestable: su peor fold baja a 0,676.
- **Confirmación en test:** el F2 en test del #1 (0,7174) es casi igual a su F2 en CV (0,7226), así que la estimación era
  fiable y el modelo no está sobreajustado a los folds.

Como el elegido es el mejor de la búsqueda, se usó directamente `grid_search.best_estimator_`, reentrenado por
`refit=True` sobre todo train, sin reentrenarlo a mano.

## Implicación para el negocio

- **Se detectan 117 cancelaciones más** de cada 374, pero el equipo de retención recibirá unas **560 alertas** en lugar de
  289, y algo más de la mitad (52,7 %) serán clientes que realmente iban a cancelar.
- El **ROC-AUC sube de 0,818 a 0,844**: el modelo también ordena mejor a los clientes por riesgo, no solo marca más.

## Siguientes pasos recomendados

- **Ajustar el umbral de decisión** (ahora 0,5) con el coste real de una retención frente al de un cliente perdido.
  Con ese dato, el umbral óptimo puede calcularse directamente.
- Probar otros clasificadores (gradient boosting, regresión logística) con el mismo proceso y la misma métrica.
