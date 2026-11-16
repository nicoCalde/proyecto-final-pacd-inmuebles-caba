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
| Grilla de LogisticRegression: `C` ∈ {0,01; 0,1; 1; 10} | Recorre la regularización en escala logarítmica, de fuerte a débil. |
| Grilla de RandomForest: 100 o 200 árboles, profundidad 20 o sin límite, hoja mínima de 1 o 5 | Contrasta árboles acotados con árboles completos: 8 combinaciones, en unos 3 minutos de entrenamiento. |
| El RandomForest ganador es la configuración sin restricciones | El F1 de CV se mide en datos que cada fold no vio, así que no es sobreajuste a train. La comparación CV vs test de la Etapa 7 lo confirma. |
| Modelo guardado con compresión (`compress=3`) y fuera de git | El RandomForest pasa de 1,2 GB a unos 200 MB. Aun así supera el límite de GitHub, por eso se regenera con el script de entrenamiento. |
| Métricas de test con promedio macro y orden de etiquetas explícito | Es coherente con el F1 macro de la CV, así que ambas se pueden comparar, y las tres clases pesan igual. |
| Test usado una sola vez, para todos los modelos (línea base incluida) | Con test no se elige nada, solo se confirma. La línea base da el piso de referencia. |
| Comparación CV vs test: RandomForest 0,729 → 0,739, LogisticRegression 0,610 → 0,610, línea base 0,167 → 0,167 | Test confirma la CV: no hay sobreajuste. |
| Reporte por clase | Muestra que "Estándar" es la clase más difícil (F1 0,640 contra ~0,79 de las otras): al ser el tercil del medio, limita con las otras dos. |
| Tabla larga de métricas (modelo, métrica, origen, media, desvío) | Tiene la forma de la tabla `metricas` de la base de datos y permite comparar CV y test en una sola consulta. |
| Precisión macro del `DummyClassifier` = 0 en las clases que nunca predice | Siempre predice la misma clase, así que la precisión de las otras dos queda indefinida y scikit-learn la toma como 0 (con una advertencia esperable). |
| Semilla fija (42) | Resultados reproducibles. |

## Preprocesamiento

| Decisión | Motivo |
| --- | --- |
| Features: antigüedad, m² totales y cubiertos, ambientes, baños, latitud, longitud, barrio y comuna | Son los atributos del inmueble y su ubicación que pide la consigna, sin ninguna columna de precio. |
| Imputar las numéricas con la mediana | Es robusta a valores extremos. La media se sesga con las colas de m² y antigüedad. |
| Imputar las categóricas con la moda | Hay muy pocos nulos (2 en barrio y 4 en comuna), así que el sesgo es despreciable. |
| `StandardScaler` en las numéricas | La regresión logística con regularización es sensible a la escala de las variables. |
| One-hot encoding con `handle_unknown="ignore"` | Un barrio o una comuna que no aparezca en train se codifica como ceros en lugar de hacer fallar el modelo. |
| `comuna` como categórica y no numérica | Es un código de distrito sin orden. Además no es redundante con barrio: 8 barrios abarcan más de una comuna. |
| Preprocesador entregado sin ajustar | Se ajusta dentro del `Pipeline` del modelo y solo con train, así no se filtra información de test. |

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
| Tablas: `propiedades_limpias` (Tabla 1), `metricas` y `predicciones` (Tabla 2), `modelos_config` (Tabla 3) y `ejecuciones` | Cubre las tres tablas de la consigna. `ejecuciones` tiene una fila por entrenamiento y es la tabla padre de los resultados. |
| `run_id` (fecha y hora UTC) en todas las tablas de resultados | Cada entrenamiento queda identificado y se pueden comparar corridas. La tabla limpia no lo lleva porque es la fuente, cargada aparte del entrenamiento. |
| Tabla limpia sin la columna `segmento` | El segmento depende de los umbrales de cada corrida, que se guardan en `ejecuciones`. |
| Predicciones solo sobre test, para todos los modelos | Son los datos que ningún modelo vio. Guardar todos los modelos permite compararlos caso por caso. |
| Matriz de confusión como vista (`matriz_confusion`) y no como tabla | Se calcula a partir de `predicciones`. Guardarla aparte duplicaría datos que podrían quedar inconsistentes. |
| Claves primarias y foráneas, `CHECK` y `PRAGMA foreign_keys=ON` | Es la base la que rechaza datos inválidos, como un segmento inexistente, una predicción de un aviso que no existe o un desvío en una métrica de test. |
| Escrituras en una sola transacción, idempotentes | Si algo falla no queda una carga a medias. Volver a escribir la misma corrida la reemplaza, sin duplicar filas. |
| Escritura con `sqlite3` (`executemany`) en lugar de `pandas.to_sql` | `to_sql` confirma los cambios por su cuenta, y eso rompe la transacción única. Las lecturas sí usan `pandas.read_sql`. |
| Recargar los mismos datos limpios no cambia nada. Si cambiaron y hay corridas guardadas, la carga falla, salvo que se pida explícitamente borrarlas | Los ids se numeran al limpiar. Con datos nuevos, las predicciones viejas apuntarían a otros avisos sin que nadie lo note. |
| Validar columnas y cantidad de filas después de cada carga | Confirma que lo guardado es lo que se quiso guardar. Es una pauta de la cátedra. |
| Índice sobre `predicciones.id_propiedad` | Acelera el control de la clave foránea al borrar avisos y los cruces con la tabla limpia. |
| Dos scripts: `cargar_bd.py` (crudos → Tabla 1) y `entrenar.py` (Tabla 1 → Tablas 2 y 3) | Cargar datos y entrenar son pasos independientes: se puede reentrenar sin releer los shapefiles. Los revisores sin `datasets/` pueden entrenar desde la base entregada. |
| Scripts con `main()` y `argparse`, sin lógica propia | Toda la lógica está en el paquete, que tiene tests. Las rutas tienen valores por defecto en `config` y se pueden cambiar sin tocar el código. |
| Paquete instalable en modo editable (`-e .` en `requirements.txt`) | Los scripts importan `inmuebles` desde cualquier carpeta, sin manipular `sys.path`. |
| Opción `--rapido` con grillas reducidas | Permite una demo o una prueba de punta a punta en segundos. Las grillas completas siguen siendo las de la entrega. |
| Un archivo de log por ejecución en `logs/`, con el `run_id` en el nombre y horas en UTC | Cada corrida de la base tiene su log y se encuentran por el mismo identificador. `logs/` no se versiona porque se genera en cada ejecución. |
| Nivel WARNING para filas descartadas y valores enmascarados en la limpieza | Deja a la vista cuántos datos se pierden en cada paso. Es una pauta de la cátedra. |
| Avisos de `warnings` (p. ej. de scikit-learn) redirigidos al log | Quedan en el archivo de la corrida en lugar de perderse en la consola. |
| Errores esperables (archivos faltantes, datos inválidos, fallas de SQLite) registrados como ERROR, con código de salida 1 | El usuario recibe un mensaje claro y el log registra la causa. Un error inesperado se propaga con su traceback para no ocultar bugs. |
| `entrenar.py` no crea la base si no existe | Una base vacía creada por error escondería que falta correr `cargar_bd.py`. |
| El modelo se guarda recién después de escribir la corrida en la base | Así el archivo `.joblib` nunca corresponde a una corrida que no existe en la base. |
| `data/inmuebles.db` se versiona con una sola corrida, entrenada con las grillas completas | Los revisores acceden a la base sin descargar los datos crudos, como pide la consigna. Una sola corrida mantiene el archivo chico. |
| Gráficos generados desde la BD | Demuestra la integración con la base de datos. |
| Python 3.12 en un entorno virtual con uv | Versión estable con soporte completo de todas las librerías del proyecto. El entorno virtual aísla las dependencias y permite reproducir la instalación en cualquier máquina. |
| Versiones exactas (`==`) en `requirements.txt` | Así cualquier instalación reproduce exactamente el entorno probado, como pide la cátedra. Por ejemplo, pandas 3 cambia comportamientos respecto de pandas 2. |
| Logging con consola y archivo, y al menos tres niveles (INFO, WARNING, ERROR) | Deja trazabilidad de cada etapa: filas leídas, filas descartadas, errores y métricas. Es una pauta de la cátedra. |
| Código en `src/inmuebles/` con un módulo por responsabilidad | Organización modular, evaluada en la rúbrica. |
| TDD con pytest, ciclo Red → Implementation → Green | **Red:** se escribe primero un test que falla, y así se comprueba que detecta la falta del comportamiento. **Implementation:** se escribe el código mínimo para cumplirlo. **Green:** se confirma que pasa sin romper lo anterior. Así cada requisito queda expresado como algo verificable, los tests prueban lo que el código debe hacer y no solo lo que ya hace, y forman una red de seguridad contra regresiones en las etapas siguientes. |
| black y flake8 (con docstrings y naming) | Estándar definido por el equipo para cumplir PEP 8, que exige la rúbrica. black formatea el código y flake8 verifica automáticamente el estilo, los docstrings y los nombres. |
| `pyproject.toml` y `.flake8` | Configuración de pytest y black, y de flake8, que no lee `pyproject.toml`. |
| Entregables en español | La consigna, la cursada y la defensa son en español. De no ser así, irían en inglés, que es la convención en el desarrollo de software. |
