"""Persistencia en SQLite: tabla limpia, corridas de entrenamiento y resultados.

El esquema es DDL explícito con claves primarias y foráneas, y cada conexión
activa ``PRAGMA foreign_keys``. Las escrituras usan ``executemany`` con
marcadores ``?`` dentro de una única transacción (``with conn:``), por lo que
son atómicas e idempotentes. Las lecturas usan ``pandas.read_sql`` con
columnas explícitas. Los nombres de tabla solo salen de una lista blanca.

Tablas: ``propiedades_limpias`` (Tabla 1), ``metricas`` y ``predicciones``
(Tabla 2), ``modelos_config`` (Tabla 3) y ``ejecuciones`` (una fila por
corrida, padre de las demás tablas de resultados). La matriz de confusión no
se guarda: se deriva de ``predicciones`` con ``GROUP BY``.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from inmuebles.config import (
    CLEAN_COLUMNS,
    CLEAN_TABLE,
    CONFIG_COLUMNS,
    CONFIG_TABLE,
    CONFUSION_AXES,
    CONFUSION_COUNT_COLUMN,
    CONFUSION_VIEW,
    CV_ORIGIN,
    DB_PATH,
    ID_COLUMN,
    METRICS_DB_COLUMNS,
    METRICS_TABLE,
    PREDICTION_COLUMNS,
    PREDICTIONS_TABLE,
    RANDOM_STATE,
    REFIT_METRIC,
    RUN_COLUMNS,
    RUN_ID_COLUMN,
    RUN_ID_FORMAT,
    RUNS_TABLE,
    SEGMENT_LABELS,
    TEST_ORIGIN,
    TEST_SIZE,
)
from inmuebles.evaluation import metrics_table

logger = logging.getLogger(__name__)

# Ruta especial de SQLite para una base en memoria (no crea carpetas)
_MEMORY_PATH = ":memory:"
# Prefijo de los hiperparámetros del paso "clf" en las grillas del Pipeline
_CLF_PREFIX = "clf__"
_CLF_STEP = "clf"

# Valores válidos para las restricciones CHECK
_SEGMENTS_SQL = ", ".join(f"'{label}'" for label in SEGMENT_LABELS)

_SCHEMA_DDL = f"""
CREATE TABLE IF NOT EXISTS {CLEAN_TABLE} (
    id_propiedad INTEGER PRIMARY KEY,
    anio INTEGER NOT NULL,
    barrio TEXT,
    comuna INTEGER,
    antig REAL,
    m2total REAL,
    m2cub REAL,
    ambientes REAL,
    banos REAL,
    latitud REAL,
    longitud REAL,
    preciousd REAL NOT NULL,
    preciousdm REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS {RUNS_TABLE} (
    run_id TEXT PRIMARY KEY,
    fecha TEXT NOT NULL,
    modelo_elegido TEXT NOT NULL,
    metrica_seleccion TEXT NOT NULL,
    umbral_bajo REAL NOT NULL,
    umbral_alto REAL NOT NULL,
    semilla INTEGER NOT NULL,
    proporcion_test REAL NOT NULL,
    filas_train INTEGER NOT NULL,
    filas_test INTEGER NOT NULL,
    CHECK (umbral_bajo < umbral_alto)
);
CREATE TABLE IF NOT EXISTS {CONFIG_TABLE} (
    run_id TEXT NOT NULL REFERENCES {RUNS_TABLE}(run_id) ON DELETE CASCADE,
    modelo TEXT NOT NULL,
    clase TEXT NOT NULL,
    hiperparametros TEXT NOT NULL,
    parametros TEXT NOT NULL,
    PRIMARY KEY (run_id, modelo)
);
CREATE TABLE IF NOT EXISTS {METRICS_TABLE} (
    run_id TEXT NOT NULL,
    modelo TEXT NOT NULL,
    metrica TEXT NOT NULL,
    origen TEXT NOT NULL CHECK (origen IN ('{CV_ORIGIN}', '{TEST_ORIGIN}')),
    media REAL NOT NULL,
    desvio REAL,
    PRIMARY KEY (run_id, modelo, metrica, origen),
    FOREIGN KEY (run_id, modelo)
        REFERENCES {CONFIG_TABLE}(run_id, modelo) ON DELETE CASCADE,
    CHECK (
        (origen = '{CV_ORIGIN}' AND desvio IS NOT NULL)
        OR (origen = '{TEST_ORIGIN}' AND desvio IS NULL)
    )
);
CREATE TABLE IF NOT EXISTS {PREDICTIONS_TABLE} (
    run_id TEXT NOT NULL,
    modelo TEXT NOT NULL,
    id_propiedad INTEGER NOT NULL REFERENCES {CLEAN_TABLE}(id_propiedad),
    real TEXT NOT NULL CHECK (real IN ({_SEGMENTS_SQL})),
    predicho TEXT NOT NULL CHECK (predicho IN ({_SEGMENTS_SQL})),
    PRIMARY KEY (run_id, modelo, id_propiedad),
    FOREIGN KEY (run_id, modelo)
        REFERENCES {CONFIG_TABLE}(run_id, modelo) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_{PREDICTIONS_TABLE}_{ID_COLUMN}
    ON {PREDICTIONS_TABLE}({ID_COLUMN});
CREATE VIEW IF NOT EXISTS {CONFUSION_VIEW} AS
    SELECT run_id, modelo, real, predicho, COUNT(*) AS {CONFUSION_COUNT_COLUMN}
    FROM {PREDICTIONS_TABLE}
    GROUP BY run_id, modelo, real, predicho;
"""

# Lista blanca: tabla -> (columnas en orden, clave primaria en orden)
_TABLES = {
    CLEAN_TABLE: (CLEAN_COLUMNS, (ID_COLUMN,)),
    RUNS_TABLE: (RUN_COLUMNS, (RUN_ID_COLUMN,)),
    CONFIG_TABLE: (CONFIG_COLUMNS, (RUN_ID_COLUMN, "modelo")),
    METRICS_TABLE: (
        METRICS_DB_COLUMNS,
        (RUN_ID_COLUMN, "modelo", "metrica", "origen"),
    ),
    PREDICTIONS_TABLE: (PREDICTION_COLUMNS, (RUN_ID_COLUMN, "modelo", ID_COLUMN)),
}
# Tablas de una corrida, de padres a hijos (orden de inserción)
_RUN_TABLES = (RUNS_TABLE, CONFIG_TABLE, METRICS_TABLE, PREDICTIONS_TABLE)


def connect(path=DB_PATH):
    """Abrir una conexión a SQLite con las claves foráneas activadas.

    Parameters
    ----------
    path : str or pathlib.Path, default DB_PATH
        Ruta del archivo ``.db`` (se crean las carpetas intermedias) o
        ``":memory:"`` para una base en memoria.

    Returns
    -------
    sqlite3.Connection
        Conexión con ``PRAGMA foreign_keys = ON``.
    """
    if str(path) != _MEMORY_PATH:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_schema(conn):
    """Crear las tablas, el índice y la vista si no existen (idempotente).

    ``executescript`` confirma antes cualquier transacción abierta, así que se
    llama al abrir la base, antes de escribir.

    Parameters
    ----------
    conn : sqlite3.Connection
        Conexión abierta con ``connect``.
    """
    conn.executescript(_SCHEMA_DDL)
    logger.info("Esquema verificado: %d tablas", len(_TABLES))


def _table_spec(table):
    """Devolver ``(columnas, clave primaria)`` de una tabla de la lista blanca.

    Raises
    ------
    ValueError
        Si ``table`` no es una tabla conocida del proyecto.
    """
    if table not in _TABLES:
        raise ValueError(f"Tabla desconocida: {table!r}. Válidas: {sorted(_TABLES)}")
    return _TABLES[table]


def _check_columns(table, actual):
    """Validar que ``actual`` sean las columnas de ``table``, en orden.

    Raises
    ------
    ValueError
        Si las columnas difieren en nombres o en orden.
    """
    expected = list(_table_spec(table)[0])
    if list(actual) != expected:
        raise ValueError(
            f"Columnas de {table} inesperadas: {list(actual)} != {expected}"
        )


def _to_python(value):
    """Convertir un escalar de numpy a su tipo de Python (los demás, igual)."""
    return value.item() if isinstance(value, np.generic) else value


def _records(df):
    """Convertir ``df`` en tuplas de tipos de Python, con NaN como ``None``."""
    frame = df.astype(object).where(df.notna(), None)
    return [
        tuple(_to_python(value) for value in row)
        for row in frame.itertuples(index=False, name=None)
    ]


def _insert_sql(table):
    """Armar el ``INSERT`` con columnas explícitas y marcadores ``?``."""
    columns = _table_spec(table)[0]
    marks = ", ".join("?" for _ in columns)
    return f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({marks})"


def _count_runs(conn):
    """Contar las corridas guardadas."""
    return conn.execute(f"SELECT COUNT(*) FROM {RUNS_TABLE}").fetchone()[0]


def _check_stale_runs(conn, replace_runs):
    """Impedir que cambien los datos limpios debajo de corridas guardadas.

    Los ids se numeran al limpiar, así que con datos nuevos las predicciones
    viejas apuntarían a otros avisos.

    Returns
    -------
    int
        Cantidad de corridas que hay que borrar.

    Raises
    ------
    ValueError
        Si hay corridas guardadas y ``replace_runs`` es falso.
    """
    runs = _count_runs(conn)
    if runs and not replace_runs:
        raise ValueError(
            f"Los datos limpios cambiaron y hay {runs} corridas guardadas: "
            "usar replace_runs=True para borrarlas y recargar."
        )
    return runs


def write_clean(conn, df, replace_runs=False):
    """Cargar la tabla limpia, reemplazándola solo si los datos cambiaron.

    Si ``df`` coincide con lo guardado no se escribe nada (idempotente). Si
    cambió, en una sola transacción se borran las corridas (cuando
    ``replace_runs`` lo permite) y se reemplaza la tabla entera.

    Parameters
    ----------
    conn : sqlite3.Connection
        Conexión con el esquema creado.
    df : pandas.DataFrame
        Tabla limpia no vacía con exactamente ``CLEAN_COLUMNS``, en orden.
    replace_runs : bool, default False
        Si es verdadero y los datos cambiaron, borra las corridas guardadas,
        cuyos resultados ya no corresponden a la tabla nueva.

    Returns
    -------
    int
        Cantidad de filas de la tabla limpia.

    Raises
    ------
    ValueError
        Si las columnas no son ``CLEAN_COLUMNS`` en orden, ``df`` está vacía,
        o los datos cambiaron, hay corridas y ``replace_runs`` es falso.
    sqlite3.IntegrityError
        Si alguna fila viola una restricción; la base queda sin cambios.
    """
    _check_columns(CLEAN_TABLE, df.columns)
    if df.empty:
        raise ValueError("La tabla limpia está vacía: no se carga nada.")
    records = _records(df.sort_values(ID_COLUMN))
    if _records(read_table(conn, CLEAN_TABLE)) == records:
        logger.info("%s sin cambios: %d filas", CLEAN_TABLE, len(records))
        return len(records)
    runs = _check_stale_runs(conn, replace_runs)
    with conn:
        conn.execute(f"DELETE FROM {RUNS_TABLE}")
        conn.execute(f"DELETE FROM {CLEAN_TABLE}")
        conn.executemany(_insert_sql(CLEAN_TABLE), records)
    if runs:
        logger.warning("Se borraron %d corridas de datos anteriores", runs)
    logger.info("%s: %d filas escritas", CLEAN_TABLE, len(records))
    return len(records)


def new_run_id():
    """Generar un ``run_id`` con la marca de tiempo UTC actual.

    Returns
    -------
    str
        Marca con formato ``RUN_ID_FORMAT`` (p. ej. ``20261116T153012Z``).
    """
    return datetime.now(timezone.utc).strftime(RUN_ID_FORMAT)


@dataclass(frozen=True)
class RunArtifacts:
    """Artefactos de una corrida de entrenamiento y evaluación.

    Attributes
    ----------
    run_id : str
        Identificador de la corrida (ver ``new_run_id``).
    results : dict
        Salida de ``train_models``.
    evaluations : dict
        Salida de ``evaluate_models``.
    labeler : inmuebles.target.ValueSegmentLabeler
        Etiquetador ajustado sobre train.
    best : str
        Nombre del modelo elegido.
    test_ids : pandas.Series
        ``id_propiedad`` de las filas de test, con el índice de las
        predicciones.
    y_test : pandas.Series
        Segmentos reales de test, con el mismo índice.
    n_train : int
        Cantidad de filas de train.
    """

    run_id: str
    results: dict
    evaluations: dict
    labeler: object
    best: str
    test_ids: pd.Series
    y_test: pd.Series
    n_train: int


def _check_artifacts(artifacts):
    """Validar los modelos, el modelo elegido y el etiquetador.

    Raises
    ------
    ValueError
        Si ``results`` y ``evaluations`` no tienen los mismos modelos, ``best``
        no está en ``results`` o el etiquetador no está ajustado.
    """
    if set(artifacts.results) != set(artifacts.evaluations):
        raise ValueError(
            f"Resultados y evaluaciones con distintos modelos: "
            f"{sorted(artifacts.results)} != {sorted(artifacts.evaluations)}"
        )
    if artifacts.best not in artifacts.results:
        raise ValueError(f"Modelo elegido desconocido: {artifacts.best!r}")
    if artifacts.labeler.thresholds_ is None:
        raise ValueError("El etiquetador no está ajustado (thresholds_ es None).")


def _check_alignment(artifacts):
    """Validar que las predicciones tengan el índice de ``test_ids`` e ``y_test``.

    Raises
    ------
    ValueError
        Si algún índice de predicciones difiere de ``test_ids`` o ``y_test``.
    """
    for name, evaluation in artifacts.evaluations.items():
        index = evaluation["predictions"].index
        test_ids, y_test = artifacts.test_ids, artifacts.y_test
        if not (index.equals(test_ids.index) and index.equals(y_test.index)):
            raise ValueError(f"Índice de predicciones de {name} desalineado.")


def _run_row(artifacts):
    """Armar la tabla de la corrida (una fila)."""
    low, high = artifacts.labeler.thresholds_
    fecha = datetime.now(timezone.utc).isoformat(timespec="seconds")
    values = (
        artifacts.run_id,
        fecha,
        artifacts.best,
        REFIT_METRIC,
        low,
        high,
        RANDOM_STATE,
        TEST_SIZE,
        artifacts.n_train,
        len(artifacts.test_ids),
    )
    return pd.DataFrame([values], columns=list(RUN_COLUMNS))


def _json_default(value):
    """Convertir para JSON: escalares de numpy a Python y lo demás a texto."""
    return value.item() if isinstance(value, np.generic) else str(value)


def _to_json(mapping):
    """Serializar a JSON con claves ordenadas (ver ``_json_default``)."""
    return json.dumps(mapping, sort_keys=True, default=_json_default)


def _config_rows(artifacts):
    """Armar la configuración de cada modelo, ordenada por nombre."""
    rows = []
    for name in sorted(artifacts.results):
        result = artifacts.results[name]
        clf = result["estimator"].named_steps[_CLF_STEP]
        tuned = {k.removeprefix(_CLF_PREFIX): v for k, v in result["params"].items()}
        rows.append(
            (
                artifacts.run_id,
                name,
                type(clf).__name__,
                _to_json(tuned),
                _to_json(clf.get_params()),
            )
        )
    return pd.DataFrame(rows, columns=list(CONFIG_COLUMNS))


def _metrics_rows(artifacts):
    """Armar la tabla larga de métricas con el ``run_id`` adelante."""
    metrics = metrics_table(artifacts.results, artifacts.evaluations)
    metrics.insert(0, RUN_ID_COLUMN, artifacts.run_id)
    return metrics


def _prediction_rows(artifacts):
    """Armar una fila por modelo y caso de test, con el real y el predicho."""
    frames = [
        pd.DataFrame(
            {
                RUN_ID_COLUMN: artifacts.run_id,
                "modelo": name,
                ID_COLUMN: artifacts.test_ids.to_numpy(),
                "real": artifacts.y_test.to_numpy(),
                "predicho": artifacts.evaluations[name]["predictions"].to_numpy(),
            },
            columns=list(PREDICTION_COLUMNS),
        )
        for name in sorted(artifacts.evaluations)
    ]
    return pd.concat(frames, ignore_index=True)


def build_run_tables(artifacts):
    """Armar las cuatro tablas de resultados de una corrida.

    Parameters
    ----------
    artifacts : RunArtifacts
        Artefactos de la corrida.

    Returns
    -------
    dict of str to pandas.DataFrame
        ``{RUNS_TABLE, CONFIG_TABLE, METRICS_TABLE, PREDICTIONS_TABLE}``, cada
        una con sus columnas en orden.

    Raises
    ------
    ValueError
        Si ``results`` y ``evaluations`` no tienen los mismos modelos, ``best``
        no está en ``results``, el etiquetador no está ajustado o los índices
        de predicciones, ``test_ids`` e ``y_test`` difieren.
    """
    _check_artifacts(artifacts)
    _check_alignment(artifacts)
    return {
        RUNS_TABLE: _run_row(artifacts),
        CONFIG_TABLE: _config_rows(artifacts),
        METRICS_TABLE: _metrics_rows(artifacts),
        PREDICTIONS_TABLE: _prediction_rows(artifacts),
    }


def write_run(conn, tables):
    """Escribir una corrida reemplazando la versión previa de ese ``run_id``.

    En una sola transacción se borra la corrida (las tablas hijas caen por
    ``ON DELETE CASCADE``) y se insertan las cuatro tablas, de padres a
    hijos. Si algo falla, no queda ninguna fila parcial.

    Parameters
    ----------
    conn : sqlite3.Connection
        Conexión con el esquema creado.
    tables : dict of str to pandas.DataFrame
        Salida de ``build_run_tables``.

    Raises
    ------
    ValueError
        Si alguna tabla no tiene sus columnas en orden.
    sqlite3.IntegrityError
        Si alguna fila viola una restricción; la base queda sin cambios.
    """
    for table in _RUN_TABLES:
        _check_columns(table, tables[table].columns)
    run_id = tables[RUNS_TABLE][RUN_ID_COLUMN].iloc[0]
    with conn:
        conn.execute(f"DELETE FROM {RUNS_TABLE} WHERE {RUN_ID_COLUMN} = ?", (run_id,))
        for table in _RUN_TABLES:
            conn.executemany(_insert_sql(table), _records(tables[table]))
    logger.info("Corrida %s escrita en %d tablas", run_id, len(_RUN_TABLES))


def _run_filter(table, run_id):
    """Armar la cláusula ``WHERE`` y sus parámetros para filtrar por corrida.

    Raises
    ------
    ValueError
        Si se pide filtrar una tabla sin columna ``run_id``.
    """
    if run_id is None:
        return "", ()
    if RUN_ID_COLUMN not in _table_spec(table)[0]:
        raise ValueError(f"La tabla {table} no tiene columna {RUN_ID_COLUMN}.")
    return f" WHERE {RUN_ID_COLUMN} = ?", (run_id,)


def validate_load(conn, table, expected, run_id=None):
    """Validar las columnas y la cantidad de filas cargadas en una tabla.

    Parameters
    ----------
    conn : sqlite3.Connection
        Conexión con los datos cargados.
    table : str
        Tabla de la lista blanca del proyecto.
    expected : pandas.DataFrame
        DataFrame de origen de la carga.
    run_id : str, optional
        Si se indica, se cuentan solo las filas de esa corrida.

    Raises
    ------
    ValueError
        Si la tabla es desconocida, sus columnas difieren de las de
        ``expected`` (nombres u orden) o la cantidad de filas no coincide.
    """
    _table_spec(table)
    columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    if columns != list(expected.columns):
        raise ValueError(f"Columnas de {table}: {columns} != {list(expected.columns)}")
    where, params = _run_filter(table, run_id)
    count = conn.execute(f"SELECT COUNT(*) FROM {table}{where}", params).fetchone()[0]
    if count != len(expected):
        raise ValueError(f"Filas de {table}: {count} en la base != {len(expected)}")
    logger.info("Carga validada: %s con %d filas", table, count)


def read_table(conn, table, run_id=None):
    """Leer una tabla con columnas explícitas, ordenada por su clave primaria.

    Parameters
    ----------
    conn : sqlite3.Connection
        Conexión abierta.
    table : str
        Tabla de la lista blanca del proyecto.
    run_id : str, optional
        Si se indica, se leen solo las filas de esa corrida.

    Returns
    -------
    pandas.DataFrame
        Filas de la tabla, con sus columnas en orden.

    Raises
    ------
    ValueError
        Si la tabla es desconocida o no admite el filtro por ``run_id``.
    """
    columns, key = _table_spec(table)
    where, params = _run_filter(table, run_id)
    sql = (
        f"SELECT {', '.join(columns)} FROM {table}{where} " f"ORDER BY {', '.join(key)}"
    )
    return pd.read_sql(sql, conn, params=params)


def read_confusion(conn, run_id, model):
    """Leer la matriz de confusión de un modelo desde la vista de la base.

    Parameters
    ----------
    conn : sqlite3.Connection
        Conexión abierta.
    run_id : str
        Corrida a leer.
    model : str
        Nombre del modelo.

    Returns
    -------
    pandas.DataFrame
        Conteos con filas = real y columnas = predicho, en el orden de
        ``SEGMENT_LABELS`` (las celdas sin casos quedan en 0).

    Raises
    ------
    ValueError
        Si no hay predicciones de ese modelo en esa corrida.
    """
    real, predicted = CONFUSION_AXES
    sql = (
        f"SELECT {real}, {predicted}, {CONFUSION_COUNT_COLUMN} FROM {CONFUSION_VIEW} "
        f"WHERE {RUN_ID_COLUMN} = ? AND modelo = ?"
    )
    cells = pd.read_sql(sql, conn, params=(run_id, model))
    if cells.empty:
        raise ValueError(f"No hay predicciones de {model!r} en la corrida {run_id!r}.")
    labels = list(SEGMENT_LABELS)
    matrix = cells.pivot(index=real, columns=predicted, values=CONFUSION_COUNT_COLUMN)
    return matrix.reindex(index=labels, columns=labels).fillna(0).astype(int)


def latest_run_id(conn):
    """Devolver el ``run_id`` de la corrida más reciente.

    Parameters
    ----------
    conn : sqlite3.Connection
        Conexión abierta.

    Returns
    -------
    str or None
        ``run_id`` con la mayor ``fecha`` (a igual fecha, el mayor ``run_id``),
        o ``None`` si no hay corridas.
    """
    sql = (
        f"SELECT {RUN_ID_COLUMN} FROM {RUNS_TABLE} "
        f"ORDER BY fecha DESC, {RUN_ID_COLUMN} DESC LIMIT 1"
    )
    row = conn.execute(sql).fetchone()
    return None if row is None else row[0]
