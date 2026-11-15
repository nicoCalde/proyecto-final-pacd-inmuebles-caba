# Registro de decisiones

## Proyecto y datos

| Decisión | Motivo |
| --- | --- |
| Opción A: Mercado Inmobiliario | Tema elegido para el TP. |
| Datos de BA Data "Departamentos en venta" en formato shapefile | Para 2017–2019 el portal solo publica shapefiles. La tabla de datos está en el `.dbf`. |
| Usar solo 2018 y 2019 | Son los años más recientes con todas las variables (antigüedad, baños, ubicación), unos 173 mil avisos. |
| Descartar 2017 y 2020 | A 2017 le faltan variables. 2020 tiene un esquema distinto. |
| No versionar `datasets/` | Los `.dbf` superan los 100 MB que acepta GitHub. Los datos limpios se entregan en la base de datos. |
| Borrar los datos SGR (Opción C) | Pertenecían a otra opción de la consigna. |
| Leer solo la tabla `.dbf` de cada shapefile, con pyshp y en UTF-8 | Contiene todos los atributos. Las geometrías no hacen falta porque ya están latitud y longitud. pyshp es liviano, comparado con alternativas geoespaciales pesadas. |
| Mapeo explícito año → archivo en `config` | Es más robusto que deducir el año del nombre del archivo, que no sigue un formato uniforme. |
| Unificar los nombres de columna entre 2018 y 2019 (`ANTIG_`, `BARRIO_1`/`BARRIOS_1`, `COMUNA_1`/`COMUNAS`) | Ambos años describen las mismas variables con nombres distintos. Hay que unificarlos para poder unir las tablas. |
| Descartar la columna `AGE` de 2019 | Solo existe en 2019 y no se puede completar para 2018. |

## Limpieza

| Decisión | Motivo |
| --- | --- |
| Conservar solo ventas (`VTA`) de departamentos (`DTO`) | Es el universo del problema. Había 1 alquiler y 25 edificios. |
| `antig`: "A ESTRENAR"/"PRE-VENTA" → 0, "SIN DATO" → nulo, coma decimal → punto | En 2018 la antigüedad venía como texto. Así queda numérica en ambos años. |
| Unificar los barrios en los 48 oficiales: subbarrios a su barrio, variantes mal codificadas corregidas, desconocidos → nulo | Las categorías tienen que ser consistentes para codificarlas. Había 59 nombres para 48 barrios. |
| Eliminar duplicados exactos después de normalizar | El mismo aviso no debe pesar doble ni caer a la vez en train y en test. Normalizar primero detecta duplicados escritos distinto. |
| Valores fuera de un rango plausible (m², ambientes, baños, antigüedad, ubicación) → nulo, sin borrar la fila | Son errores de carga, como 0 m² o 321 ambientes. Así se conservan las filas, y la imputación queda a cargo del Pipeline, entrenado solo con train. |
| Eliminar filas sin precio válido o con m² cubiertos > m² totales | Sin precio no hay target, y una superficie incoherente indica un aviso mal cargado. |
| Eliminar el precio por m² fuera de los percentiles 1–99, sobre todo el dataset | Quita errores de carga extremos en el target. Los umbrales casi no dependen del conjunto de test, así que el impacto metodológico es mínimo. |
| Conservar `preciousd` en la tabla limpia, nunca como feature | Sirve para análisis y gráficos. Usarlo como feature filtraría el target. |
| `id_propiedad` secuencial | Identificador único de cada aviso limpio, clave de la base de datos. |

## Metodología

| Decisión | Motivo |
| --- | --- |
| Target: segmento de valor por terciles de USD/m² (Accesible / Estándar / Alta) | Las clases quedan balanceadas. El USD/m² permite comparar departamentos de distinto tamaño. |
| Umbrales de los terciles calculados solo con train | Evita que el conjunto de test influya en la definición de las clases. |
| Un precio igual a un umbral va a la clase superior | La regla de borde queda explícita y es reproducible. |
| Guardar los umbrales aprendidos para registrarlos en la base de datos | La definición de cada clase queda auditable: se sabe exactamente qué USD/m² separa cada segmento. |
| El etiquetador del target queda fuera del `Pipeline` de scikit-learn | Transforma la variable objetivo, no las features, y el `Pipeline` solo procesa las features. |
| Excluir las columnas de precio como features | El target se deriva del precio. Usarlo como feature sería hacer trampa (*data leakage*). |
| LogisticRegression vs RandomForest | La consigna pide al menos dos modelos: uno lineal simple y uno no lineal. |
| `DummyClassifier` como línea base | Muestra cuánto mejora cada modelo respecto de predecir siempre la clase más frecuente. Sin esa referencia, una métrica aislada no dice si el modelo aprende algo. |
| Split 80/20 estratificado por una variable auxiliar de precio | El target se define después del split, así que no se puede estratificar por él. Estratificar por tramos de precio mantiene la misma proporción de segmentos en train y test. |
| GridSearchCV con F1 macro y validación cruzada estratificada de 5 folds | F1 macro pesa igual a todas las clases. La estratificación mantiene la proporción de clases en cada fold. |
| Elegir el modelo por la media de CV y reportar el desvío entre folds | El conjunto de test se usa una sola vez, al final, para confirmar. El desvío muestra qué tan estable es cada modelo. |
| Guardar el pipeline entrenado con joblib | Permite reutilizar el modelo sin reentrenarlo. |
| Semilla fija (42) | Resultados reproducibles. |

## Visualización y presentación

| Decisión | Motivo |
| --- | --- |
| Cada gráfico se elige por la pregunta que responde: barras para comparar, líneas para evolución, dispersión para relación, histograma para distribución | Es la guía de las "cuatro gráficas comodín" de la cátedra. Por eso se descarta el heatmap de correlaciones. |
| Títulos que enuncian la conclusión, no el tema | El deck se entrega en PDF y tiene que entenderse sin la exposición (guía SWD). |
| Eje en 0 y valores rotulados en las barras de métricas | Un eje truncado exagera diferencias chicas entre modelos. |
| Mismo color para cada segmento en todos los gráficos | La consistencia visual facilita la lectura. |
| Deck de 7 diapositivas: pregunta → datos → target → modelos → resultados → interpretación → conclusión | Una idea por diapositiva, de la pregunta al cierre, dentro de las 6 a 8 que pide la consigna. |

## Ingeniería

| Decisión | Motivo |
| --- | --- |
| SQLite | La consigna pide una BD accesible. Un archivo en el repo no necesita servidor. |
| Gráficos generados desde la BD | Demuestra la integración con la base de datos. |
| Python 3.12 en un entorno virtual con uv | Versión estable con soporte completo de todas las librerías del proyecto. El entorno virtual aísla las dependencias y permite reproducir la instalación en cualquier máquina. |
| Versiones exactas (`==`) en `requirements.txt` | Así cualquier instalación reproduce exactamente el entorno probado, como pide la cátedra. Por ejemplo, pandas 3 cambia comportamientos respecto de pandas 2. |
| Logging con consola y archivo, y al menos tres niveles (INFO, WARNING, ERROR) | Deja trazabilidad de cada etapa: filas leídas, filas descartadas, errores y métricas. Es una pauta de la cátedra. |
| Código en `src/inmuebles/` con un módulo por responsabilidad | Organización modular, evaluada en la rúbrica. |
| TDD con pytest, ciclo Red → Implementation → Green | **Red:** se escribe primero un test que falla, y así se comprueba que detecta la falta del comportamiento. **Implementation:** se escribe el código mínimo para cumplirlo. **Green:** se confirma que pasa sin romper lo anterior. Así cada requisito queda expresado como algo verificable, los tests prueban lo que el código debe hacer y no solo lo que ya hace, y forman una red de seguridad contra regresiones en las etapas siguientes. |
| black y flake8 (con docstrings y naming) | Estándar definido por el equipo para cumplir PEP 8, que exige la rúbrica. black formatea el código y flake8 verifica automáticamente el estilo, los docstrings y los nombres. |
| `pyproject.toml` y `.flake8` | Configuración de pytest y black, y de flake8, que no lee `pyproject.toml`. |
| Entregables en español | La consigna, la cursada y la defensa son en español. De no ser así, irían en inglés, que es la convención en el desarrollo de software. |
