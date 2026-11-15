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

## Metodología

| Decisión | Motivo |
| --- | --- |
| Target: segmento de valor por terciles de USD/m² (Accesible / Estándar / Alta) | Las clases quedan balanceadas. El USD/m² permite comparar departamentos de distinto tamaño. |
| Umbrales de los terciles calculados solo con train | Evita que el conjunto de test influya en la definición de las clases. |
| Excluir las columnas de precio como features | El target se deriva del precio. Usarlo como feature sería hacer trampa (*data leakage*). |
| LogisticRegression vs RandomForest | La consigna pide al menos dos modelos: uno lineal simple y uno no lineal. |
| GridSearchCV con F1 macro y validación cruzada estratificada de 5 folds | F1 macro pesa igual a todas las clases. La estratificación mantiene la proporción de clases en cada fold. |
| Semilla fija (42) | Resultados reproducibles. |

## Ingeniería

| Decisión | Motivo |
| --- | --- |
| SQLite | La consigna pide una BD accesible. Un archivo en el repo no necesita servidor. |
| Gráficos generados desde la BD | Demuestra la integración con la base de datos. |
| Python 3.12 en un entorno virtual con uv | Versión estable con soporte completo de todas las librerías del proyecto. El entorno virtual aísla las dependencias y permite reproducir la instalación en cualquier máquina. |
| Mínimos de `requirements.txt` iguales a las versiones probadas | pandas 3 cambia comportamientos. Con estos mínimos, el revisor obtiene los mismos resultados. |
| Código en `src/inmuebles/` con un módulo por responsabilidad | Organización modular, evaluada en la rúbrica. |
| TDD con pytest (tests primero) | Se pidió en el plan. Garantiza que cada etapa funcione antes de avanzar. |
| black y flake8 (con docstrings y naming) | Estándar definido por el equipo para cumplir PEP 8, que exige la rúbrica. black formatea el código y flake8 verifica automáticamente el estilo, los docstrings y los nombres. |
| `pyproject.toml` y `.flake8` | Configuración de pytest y black, y de flake8, que no lee `pyproject.toml`. |
| Entregables en español | La consigna, la cursada y la defensa son en español. De no ser así, irían en inglés, que es la convención en el desarrollo de software. |
