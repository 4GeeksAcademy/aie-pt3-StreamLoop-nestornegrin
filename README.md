# StreamLoop · Ajuste del clasificador de cancelación (churn)

Ajuste sistemático de hiperparámetros de un clasificador de churn para StreamLoop: línea base con valores por defecto,
búsqueda amplia con `RandomizedSearchCV`, búsqueda focalizada con `GridSearchCV` y elección justificada del modelo final.

## Resultado

| | Línea base | Modelo ajustado |
|---|---|---|
| F2 (test) | 0,499 | **0,717** |
| Cancelaciones detectadas | 178 de 374 | **295 de 374** |

## Archivos

- [`churn_tuning.ipynb`](churn_tuning.ipynb): proceso completo, ejecutado y con salidas.
- [`tuning_report.md`](tuning_report.md): informe para el tech lead (métrica, hiperparámetros, trade-off de estabilidad).
- `requirements.txt`: dependencias con versiones fijadas.

## Cómo reproducirlo

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace churn_tuning.ipynb
```

El dataset se descarga automáticamente desde la URL pública de IBM. La ejecución completa tarda unos 7 minutos con 2 CPU.
