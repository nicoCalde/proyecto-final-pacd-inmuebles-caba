"""Pruebas de la persistencia en SQLite: esquema, tabla limpia y corridas.

Todas las bases son temporales (``:memory:`` o ``tmp_path``); nunca se toca
``data/`` ni ``datasets/``. Las corridas se arman con un flujo real diminuto
(grillas mínimas, ``n_jobs=1``), igual que en ``test_evaluation.py``.
"""

import dataclasses
import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from inmuebles.config import (
    BASELINE_NAME,
    CLEAN_COLUMNS,
    CLEAN_TABLE,
    CONFIG_COLUMNS,
    CONFIG_TABLE,
    ID_COLUMN,
    METRICS_DB_COLUMNS,
    METRICS_TABLE,
    PREDICTION_COLUMNS,
    PREDICTIONS_TABLE,
    RANDOM_STATE,
    REFIT_METRIC,
    RUN_COLUMNS,
    RUN_ID_COLUMN,
    RUNS_TABLE,
    SEGMENT_LABELS,
    TARGET_SOURCE,
    TEST_ORIGIN,
    TEST_SIZE,
)
from inmuebles.database import (
    RunArtifacts,
    build_run_tables,
    connect,
    create_schema,
    latest_run_id,
    new_run_id,
    read_confusion,
    read_table,
    validate_load,
    write_clean,
    write_run,
)
from inmuebles.evaluation import evaluate_models, metrics_table
from inmuebles.modeling import make_xy, select_best, split_data, train_models
from inmuebles.target import ValueSegmentLabeler

ACC, STD, HIGH = SEGMENT_LABELS

# Grillas diminutas para que el flujo completo sea rápido.
TINY_GRIDS = {
    "logistic_regression": {"clf__C": [1.0]},
    "random_forest": {"clf__n_estimators": [10], "clf__max_depth": [5]},
}

RUN_ID = "20260101T000000Z"
OTHER_RUN_ID = "20260102T000000Z"
# Tablas de una corrida, de padres a hijos.
RUN_TABLES = (RUNS_TABLE, CONFIG_TABLE, METRICS_TABLE, PREDICTIONS_TABLE)
SCHEMA = {
    CLEAN_TABLE: CLEAN_COLUMNS,
    RUNS_TABLE: RUN_COLUMNS,
    CONFIG_TABLE: CONFIG_COLUMNS,
    METRICS_TABLE: METRICS_DB_COLUMNS,
    PREDICTIONS_TABLE: PREDICTION_COLUMNS,
}
UNKNOWN_TABLES = ("sqlite_master", "otra_tabla", "predicciones; DROP TABLE x")
ID_REGEX = re.compile(r"^\d{8}T\d{6}Z$")

# Filas mínimas válidas para insertar a mano con SQL crudo.
RAW_MODEL = "random_forest"
RAW_RUN = ("r1", "2026-01-01T00:00:00+00:00", RAW_MODEL, "f1", 1.0, 2.0, 42, 0.2)
RAW_SIZES = (8, 2)
RAW_CONFIG = ("r1", RAW_MODEL, "RandomForestClassifier", "{}", "{}")


# --- Ayudantes -------------------------------------------------------------


def _insert(conn, table, columns, values):
    """Insertar una fila con SQL crudo sobre columnas explícitas y confirmar."""
    names = ", ".join(columns)
    marks = ", ".join("?" for _ in columns)
    with conn:
        conn.execute(f"INSERT INTO {table} ({names}) VALUES ({marks})", values)


def _columns(conn, table):
    """Devolver los nombres de columna de ``table``, en orden."""
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def _count(conn, table, run_id=None):
    """Contar las filas de ``table``, filtrando por ``run_id`` si se indica."""
    if run_id is None:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    sql = f"SELECT COUNT(*) FROM {table} WHERE {RUN_ID_COLUMN} = ?"
    return conn.execute(sql, (run_id,)).fetchone()[0]


def _copy_tables(tables):
    """Copiar en profundidad el diccionario de tablas de una corrida."""
    return {name: df.copy() for name, df in tables.items()}


def _with_run_id(tables, run_id, fecha=None):
    """Devolver una copia de ``tables`` con otro ``run_id`` (y otra fecha)."""
    copy = _copy_tables(tables)
    for df in copy.values():
        df[RUN_ID_COLUMN] = run_id
    if fecha is not None:
        copy[RUNS_TABLE]["fecha"] = fecha
    return copy


def _insert_parents(conn):
    """Insertar una fila limpia (id 0), una corrida y su configuración."""
    columns = (ID_COLUMN, "anio", "preciousd", "preciousdm")
    _insert(conn, CLEAN_TABLE, columns, (0, 2018, 1e5, 2e3))
    _insert(conn, RUNS_TABLE, RUN_COLUMNS, RAW_RUN + RAW_SIZES)
    _insert(conn, CONFIG_TABLE, CONFIG_COLUMNS, RAW_CONFIG)


# --- Fixtures --------------------------------------------------------------


@pytest.fixture
def conn():
    """Abrir una base en memoria con el esquema creado y cerrarla al final."""
    connection = connect(":memory:")
    create_schema(connection)
    yield connection
    connection.close()


@pytest.fixture
def small_clean(make_clean):
    """Devolver una tabla limpia de 3 filas; la primera con barrio y antig NaN."""
    return make_clean(
        (
            {"barrio": np.nan, "antig": np.nan},
            {"barrio": "BOCA", "comuna": 4.0},
            {"m2total": 120.0, "preciousdm": 3100.0},
        )
    )


@pytest.fixture(scope="module")
def flow(learnable_clean):
    """Entrenar una sola vez con grillas diminutas y armar los artefactos."""
    train, test = split_data(learnable_clean)
    labeler = ValueSegmentLabeler().fit(train[TARGET_SOURCE])
    x_train, y_train = make_xy(train, labeler)
    x_test, y_test = make_xy(test, labeler)
    results = train_models(x_train, y_train, grids=TINY_GRIDS, n_jobs=1)
    artifacts = RunArtifacts(
        run_id=RUN_ID,
        results=results,
        evaluations=evaluate_models(results, x_test, y_test),
        labeler=labeler,
        best=select_best(results),
        test_ids=test[ID_COLUMN],
        y_test=y_test,
        n_train=len(train),
    )
    return artifacts, build_run_tables(artifacts)


@pytest.fixture
def run_conn(conn, learnable_clean):
    """Abrir una base con la tabla limpia aprendible ya cargada."""
    write_clean(conn, learnable_clean)
    return conn


# --- Esquema y conexión ----------------------------------------------------


@pytest.mark.parametrize("table", list(SCHEMA))
def test_schema_tables_have_expected_columns(conn, table):
    """Verificar que cada tabla existe con sus columnas en el orden pactado."""
    assert _columns(conn, table) == list(SCHEMA[table])


def test_schema_has_exactly_the_five_tables(conn):
    """Verificar que el esquema crea las cinco tablas del proyecto."""
    sql = "SELECT name FROM sqlite_master WHERE type = 'table'"
    names = {row[0] for row in conn.execute(sql)}
    assert names == set(SCHEMA)


def test_schema_indexes_prediction_ids(conn):
    """Verificar el índice sobre ``id_propiedad`` (la FK no lidera la PK)."""
    indexes = conn.execute(f"PRAGMA index_list({PREDICTIONS_TABLE})").fetchall()
    indexed = {
        tuple(row[2] for row in conn.execute(f"PRAGMA index_info({index[1]})"))
        for index in indexes
    }
    assert (ID_COLUMN,) in indexed


@pytest.mark.parametrize("target", [":memory:", "file"])
def test_connect_enables_foreign_keys(tmp_path, target):
    """Verificar que ``connect`` activa ``PRAGMA foreign_keys``."""
    path = ":memory:" if target == ":memory:" else tmp_path / "x.db"
    connection = connect(path)
    try:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        connection.close()


def test_connect_creates_parent_folders(tmp_path):
    """Verificar que ``connect`` crea las carpetas intermedias de la ruta."""
    path = tmp_path / "a" / "b" / "inmuebles.db"
    connection = connect(path)
    create_schema(connection)
    connection.close()
    assert path.is_file()


def test_create_schema_twice_keeps_data(conn, small_clean):
    """Verificar que crear el esquema de nuevo no borra los datos cargados."""
    write_clean(conn, small_clean)
    create_schema(conn)
    assert _count(conn, CLEAN_TABLE) == len(small_clean)


# --- Tabla limpia ----------------------------------------------------------


def test_write_clean_roundtrip_with_nulls(conn, small_clean):
    """Verificar que los valores vuelven iguales y que NaN se guarda como NULL."""
    assert write_clean(conn, small_clean) == len(small_clean)
    sql = f"SELECT COUNT(*) FROM {CLEAN_TABLE} WHERE barrio IS NULL AND antig IS NULL"
    assert conn.execute(sql).fetchone()[0] == 1
    back = read_table(conn, CLEAN_TABLE)
    assert back.loc[0, ["barrio", "antig"]].isna().all()
    pd.testing.assert_frame_equal(
        back.iloc[1:].reset_index(drop=True),
        small_clean.iloc[1:].reset_index(drop=True),
        check_dtype=False,
    )


def test_write_clean_is_idempotent(conn, small_clean):
    """Verificar que cargar dos veces la misma tabla deja el mismo contenido."""
    write_clean(conn, small_clean)
    first = read_table(conn, CLEAN_TABLE)
    assert write_clean(conn, small_clean) == len(small_clean)
    pd.testing.assert_frame_equal(read_table(conn, CLEAN_TABLE), first)


def test_write_clean_updates_changed_values(conn, small_clean):
    """Verificar que una segunda carga actualiza los valores de un id existente."""
    write_clean(conn, small_clean)
    changed = small_clean.copy()
    changed.loc[1, "m2total"] = 99.0
    write_clean(conn, changed)
    back = read_table(conn, CLEAN_TABLE).set_index(ID_COLUMN)
    assert back.loc[changed.loc[1, ID_COLUMN], "m2total"] == pytest.approx(99.0)


def test_write_clean_shrink_removes_missing_ids(conn, small_clean):
    """Verificar que una carga con menos filas borra los ids que ya no están."""
    write_clean(conn, small_clean)
    assert write_clean(conn, small_clean.iloc[:2]) == 2
    ids = read_table(conn, CLEAN_TABLE)[ID_COLUMN].tolist()
    assert ids == small_clean[ID_COLUMN].iloc[:2].tolist()


@pytest.mark.parametrize("change", ["drop", "extra", "reorder"])
def test_write_clean_rejects_wrong_columns(conn, small_clean, change):
    """Verificar que columnas faltantes, sobrantes o desordenadas dan ``ValueError``."""
    if change == "drop":
        df = small_clean.drop(columns="banos")
    elif change == "extra":
        df = small_clean.assign(precioars=1.0)
    else:
        df = small_clean[list(reversed(CLEAN_COLUMNS))]
    with pytest.raises(ValueError):
        write_clean(conn, df)
    assert _count(conn, CLEAN_TABLE) == 0


def test_write_clean_rejects_empty_table(conn, small_clean):
    """Verificar que una tabla vacía da ``ValueError`` y no borra lo cargado."""
    write_clean(conn, small_clean)
    with pytest.raises(ValueError, match="vacía"):
        write_clean(conn, small_clean.iloc[0:0])
    assert _count(conn, CLEAN_TABLE) == len(small_clean)


def test_write_clean_is_atomic_on_failure(conn, small_clean):
    """Verificar que una carga fallida (id duplicado) deja la versión anterior."""
    write_clean(conn, small_clean)
    before = read_table(conn, CLEAN_TABLE)
    broken = small_clean.copy()
    broken.loc[0, "m2total"] = 77.0
    broken.loc[2, ID_COLUMN] = broken.loc[1, ID_COLUMN]
    with pytest.raises(sqlite3.IntegrityError):
        write_clean(conn, broken)
    pd.testing.assert_frame_equal(read_table(conn, CLEAN_TABLE), before)


def _shifted(df):
    """Simular una limpieza nueva que pierde la primera fila y renumera los ids."""
    shifted = df.iloc[1:].reset_index(drop=True)
    shifted[ID_COLUMN] = df[ID_COLUMN].iloc[:-1].to_numpy()
    return shifted


def test_write_clean_same_data_keeps_runs(run_conn, flow, learnable_clean):
    """Verificar que recargar los mismos datos conserva las corridas guardadas."""
    _, tables = flow
    write_run(run_conn, tables)
    assert write_clean(run_conn, learnable_clean) == len(learnable_clean)
    for table in RUN_TABLES:
        assert _count(run_conn, table, RUN_ID) == len(tables[table])


def test_write_clean_changed_data_with_runs_raises(run_conn, flow, learnable_clean):
    """Verificar que datos distintos con corridas guardadas fallan sin cambiar nada.

    Si se aceptaran, las predicciones viejas quedarían apuntando a otros avisos
    (los ids se renumeran al limpiar).
    """
    _, tables = flow
    write_run(run_conn, tables)
    before = read_table(run_conn, CLEAN_TABLE)
    with pytest.raises(ValueError, match="corridas"):
        write_clean(run_conn, _shifted(learnable_clean))
    pd.testing.assert_frame_equal(read_table(run_conn, CLEAN_TABLE), before)
    for table in RUN_TABLES:
        assert _count(run_conn, table, RUN_ID) == len(tables[table])


def test_write_clean_replace_runs_drops_stale_results(
    run_conn, flow, learnable_clean, caplog
):
    """Verificar que ``replace_runs`` recarga los datos y borra las corridas viejas."""
    _, tables = flow
    write_run(run_conn, tables)
    shifted = _shifted(learnable_clean)
    with caplog.at_level("WARNING"):
        assert write_clean(run_conn, shifted, replace_runs=True) == len(shifted)
    for table in RUN_TABLES:
        assert _count(run_conn, table) == 0
    pd.testing.assert_frame_equal(
        read_table(run_conn, CLEAN_TABLE), shifted, check_dtype=False
    )
    assert any(record.levelname == "WARNING" for record in caplog.records)


# --- Restricciones del esquema ---------------------------------------------


def _prediction(model=RAW_MODEL, id_=0, real=ACC, predicted=ACC):
    """Armar una fila de ``predicciones`` para insertar con SQL crudo."""
    return PREDICTIONS_TABLE, PREDICTION_COLUMNS, ("r1", model, id_, real, predicted)


def _metric(origin, desvio):
    """Armar una fila de ``metricas`` para insertar con SQL crudo."""
    return (
        METRICS_TABLE,
        METRICS_DB_COLUMNS,
        ("r1", RAW_MODEL, "f1", origin, 0.5, desvio),
    )


INVALID_CHILD_ROWS = {
    "id_inexistente": _prediction(id_=999),
    "real_fuera_de_segmentos": _prediction(real="Otro"),
    "predicho_fuera_de_segmentos": _prediction(predicted="Otro"),
    "modelo_sin_config": _prediction(model="otro"),
    "origen_invalido": _metric("x", 0.1),
    "test_con_desvio": _metric("test", 0.1),
    "cv_sin_desvio": _metric("cv", None),
    "config_sin_corrida": (
        CONFIG_TABLE,
        CONFIG_COLUMNS,
        ("r9", RAW_MODEL, "X", "{}", "{}"),
    ),
}

VALID_CHILD_ROWS = {
    "prediccion": _prediction(predicted=HIGH),
    "metrica_cv": _metric("cv", 0.1),
    "metrica_test": _metric("test", None),
}


@pytest.mark.parametrize("case", list(VALID_CHILD_ROWS))
def test_db_accepts_valid_child_rows(conn, case):
    """Verificar que las filas hijas válidas se aceptan (control de las inválidas)."""
    _insert_parents(conn)
    table, columns, values = VALID_CHILD_ROWS[case]
    _insert(conn, table, columns, values)
    assert _count(conn, table) == 1


@pytest.mark.parametrize("case", list(INVALID_CHILD_ROWS))
def test_db_rejects_invalid_child_rows(conn, case):
    """Verificar que la base rechaza FKs rotas, orígenes y etiquetas inválidas."""
    _insert_parents(conn)
    table, columns, values = INVALID_CHILD_ROWS[case]
    with pytest.raises(sqlite3.IntegrityError):
        _insert(conn, table, columns, values)


@pytest.mark.parametrize("thresholds", [(2.0, 1.0), (1.5, 1.5)])
def test_db_rejects_unordered_thresholds(conn, thresholds):
    """Verificar que la base exige ``umbral_bajo < umbral_alto``."""
    values = RAW_RUN[:4] + thresholds + RAW_RUN[6:] + RAW_SIZES
    with pytest.raises(sqlite3.IntegrityError):
        _insert(conn, RUNS_TABLE, RUN_COLUMNS, values)


# --- Armado de las tablas de una corrida -----------------------------------


@pytest.mark.parametrize("table", RUN_TABLES)
def test_build_run_tables_columns_in_order(flow, table):
    """Verificar que cada tabla de la corrida tiene sus columnas en orden."""
    _, tables = flow
    assert set(tables) == set(RUN_TABLES)
    assert list(tables[table].columns) == list(SCHEMA[table])
    assert (tables[table][RUN_ID_COLUMN] == RUN_ID).all()


def test_build_run_tables_run_row(flow):
    """Verificar umbrales, modelo elegido, tamaños y parámetros de la corrida."""
    artifacts, tables = flow
    runs = tables[RUNS_TABLE]
    assert len(runs) == 1
    row = runs.iloc[0]
    assert (row["umbral_bajo"], row["umbral_alto"]) == pytest.approx(
        artifacts.labeler.thresholds_
    )
    assert row["modelo_elegido"] == artifacts.best
    assert row["filas_train"] == artifacts.n_train
    assert row["filas_test"] == len(artifacts.test_ids)
    assert row["semilla"] == RANDOM_STATE
    assert row["proporcion_test"] == pytest.approx(TEST_SIZE)
    assert row["metrica_seleccion"] == REFIT_METRIC


def test_build_run_tables_fecha_is_recent_utc_iso(flow):
    """Verificar que ``fecha`` es ISO 8601 en UTC y corresponde a esta ejecución."""
    _, tables = flow
    fecha = datetime.fromisoformat(tables[RUNS_TABLE].iloc[0]["fecha"])
    assert fecha.utcoffset() == timedelta(0)
    assert abs(datetime.now(timezone.utc) - fecha) < timedelta(hours=1)


def test_build_run_tables_config_rows(flow):
    """Verificar una fila por modelo, clase del clasificador y JSON sin ``clf__``."""
    artifacts, tables = flow
    config = tables[CONFIG_TABLE]
    assert config["modelo"].tolist() == sorted(artifacts.results)
    for row in config.itertuples(index=False):
        clf = artifacts.results[row.modelo]["estimator"].named_steps["clf"]
        assert row.clase == type(clf).__name__
        hyper = json.loads(row.hiperparametros)
        assert not any(key.startswith("clf__") for key in hyper)
        tuned = artifacts.results[row.modelo]["params"]
        assert set(hyper) == {key.removeprefix("clf__") for key in tuned}
        assert set(json.loads(row.parametros)) == set(clf.get_params())


def test_build_run_tables_baseline_has_no_hyperparameters(flow):
    """Verificar que el baseline se guarda con hiperparámetros ``{}``."""
    _, tables = flow
    config = tables[CONFIG_TABLE].set_index("modelo")
    assert json.loads(config.loc[BASELINE_NAME, "hiperparametros"]) == {}


def test_build_run_tables_metrics_match_metrics_table(flow):
    """Verificar que las métricas son ``metrics_table`` con ``run_id`` adelante."""
    artifacts, tables = flow
    expected = metrics_table(artifacts.results, artifacts.evaluations)
    got = tables[METRICS_TABLE].drop(columns=RUN_ID_COLUMN)
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True), expected.reset_index(drop=True)
    )


def test_build_run_tables_predictions_rows(flow):
    """Verificar una fila por modelo y caso de test, con ids, real y predicho."""
    artifacts, tables = flow
    preds = tables[PREDICTIONS_TABLE]
    assert len(preds) == len(artifacts.results) * len(artifacts.test_ids)
    for name, group in preds.groupby("modelo"):
        expected = artifacts.evaluations[name]["predictions"]
        assert group[ID_COLUMN].tolist() == artifacts.test_ids.tolist()
        assert group["real"].tolist() == artifacts.y_test.tolist()
        assert group["predicho"].tolist() == expected.tolist()


@pytest.mark.parametrize("field", ["test_ids", "y_test"])
def test_build_run_tables_rejects_misaligned_index(flow, field):
    """Verificar que un índice desalineado con las predicciones da ``ValueError``."""
    artifacts, _ = flow
    shifted = getattr(artifacts, field).reset_index(drop=True)
    with pytest.raises(ValueError):
        build_run_tables(dataclasses.replace(artifacts, **{field: shifted}))


@pytest.mark.parametrize("change", ["falta", "sobra"])
def test_build_run_tables_rejects_mismatched_evaluations(flow, change):
    """Verificar ``ValueError`` si evaluaciones y resultados difieren en modelos."""
    artifacts, _ = flow
    evaluations = dict(artifacts.evaluations)
    if change == "falta":
        evaluations.pop(BASELINE_NAME)
    else:
        evaluations["otro"] = evaluations[BASELINE_NAME]
    with pytest.raises(ValueError, match="modelos"):
        build_run_tables(dataclasses.replace(artifacts, evaluations=evaluations))


def test_build_run_tables_json_keeps_numpy_numbers(flow):
    """Verificar que un hiperparámetro de numpy se guarda como número, no texto."""
    artifacts, _ = flow
    results = dict(artifacts.results)
    forest = dict(results["random_forest"])
    forest["params"] = {"clf__n_estimators": np.int64(10)}
    results["random_forest"] = forest
    tables = build_run_tables(dataclasses.replace(artifacts, results=results))
    config = tables[CONFIG_TABLE].set_index("modelo")
    hyper = json.loads(config.loc["random_forest", "hiperparametros"])
    assert hyper == {"n_estimators": 10}


@pytest.mark.parametrize(
    "change",
    [{"best": "modelo_inexistente"}, {"labeler": ValueSegmentLabeler()}],
    ids=["best_desconocido", "labeler_sin_ajustar"],
)
def test_build_run_tables_rejects_invalid_artifacts(flow, change):
    """Verificar ``ValueError`` si falta el mejor modelo o el etiquetador no ajustó."""
    artifacts, _ = flow
    with pytest.raises(ValueError):
        build_run_tables(dataclasses.replace(artifacts, **change))


# --- Escritura de corridas -------------------------------------------------


def test_write_run_counts_per_table(run_conn, flow):
    """Verificar que cada tabla de la corrida queda con todas sus filas."""
    _, tables = flow
    write_run(run_conn, tables)
    for table in RUN_TABLES:
        assert _count(run_conn, table, RUN_ID) == len(tables[table])


def test_write_run_is_idempotent(run_conn, flow):
    """Verificar que reescribir la misma corrida no duplica filas."""
    _, tables = flow
    write_run(run_conn, tables)
    write_run(run_conn, tables)
    for table in RUN_TABLES:
        assert _count(run_conn, table) == len(tables[table])


def test_write_run_second_run_leaves_first_untouched(run_conn, flow):
    """Verificar que otra corrida convive y no altera la primera."""
    _, tables = flow
    write_run(run_conn, tables)
    before = {t: read_table(run_conn, t, run_id=RUN_ID) for t in RUN_TABLES}
    write_run(run_conn, _with_run_id(tables, OTHER_RUN_ID))
    for table in RUN_TABLES:
        assert _count(run_conn, table, OTHER_RUN_ID) == len(tables[table])
        after = read_table(run_conn, table, run_id=RUN_ID)
        pd.testing.assert_frame_equal(after, before[table])


def _corrupted(tables):
    """Devolver una copia cuya primera predicción apunta a un id inexistente."""
    copy = _copy_tables(tables)
    preds = copy[PREDICTIONS_TABLE]
    preds.loc[preds.index[0], ID_COLUMN] = 10**9
    return copy


def test_write_run_is_atomic_on_failure(run_conn, flow):
    """Verificar que un insert fallido no deja ninguna fila de esa corrida."""
    _, tables = flow
    with pytest.raises(sqlite3.IntegrityError):
        write_run(run_conn, _corrupted(tables))
    for table in RUN_TABLES:
        assert _count(run_conn, table, RUN_ID) == 0


def test_write_run_failed_rewrite_keeps_previous_version(run_conn, flow):
    """Verificar que una reescritura fallida conserva la corrida anterior entera."""
    _, tables = flow
    write_run(run_conn, tables)
    with pytest.raises(sqlite3.IntegrityError):
        write_run(run_conn, _corrupted(tables))
    for table in RUN_TABLES:
        assert _count(run_conn, table, RUN_ID) == len(tables[table])


def test_write_run_test_metrics_have_null_desvio(run_conn, flow):
    """Verificar que el desvío de las métricas de test vuelve como NULL."""
    _, tables = flow
    write_run(run_conn, tables)
    metrics = read_table(run_conn, METRICS_TABLE, run_id=RUN_ID)
    is_test = metrics["origen"] == TEST_ORIGIN
    assert is_test.any()
    assert metrics.loc[is_test, "desvio"].isna().all()
    assert metrics.loc[~is_test, "desvio"].notna().all()


# --- Validación de la carga ------------------------------------------------


def test_validate_load_passes_after_correct_load(
    run_conn, flow, learnable_clean, caplog
):
    """Verificar que una carga correcta valida sin errores y lo registra en INFO."""
    _, tables = flow
    write_run(run_conn, tables)
    with caplog.at_level("INFO"):
        validate_load(run_conn, CLEAN_TABLE, learnable_clean)
        for table in RUN_TABLES:
            validate_load(run_conn, table, tables[table], run_id=RUN_ID)
    assert any(record.levelname == "INFO" for record in caplog.records)


def test_validate_load_wrong_count_raises(conn, small_clean):
    """Verificar que una cantidad de filas distinta da ``ValueError``."""
    write_clean(conn, small_clean)
    with pytest.raises(ValueError):
        validate_load(conn, CLEAN_TABLE, small_clean.iloc[:-1])


@pytest.mark.parametrize("change", ["rename", "reorder"])
def test_validate_load_wrong_columns_raises(conn, small_clean, change):
    """Verificar que columnas distintas o en otro orden dan ``ValueError``."""
    write_clean(conn, small_clean)
    if change == "rename":
        expected = small_clean.rename(columns={"banos": "baños"})
    else:
        expected = small_clean[list(reversed(CLEAN_COLUMNS))]
    with pytest.raises(ValueError):
        validate_load(conn, CLEAN_TABLE, expected)


@pytest.mark.parametrize("table", UNKNOWN_TABLES)
def test_validate_load_unknown_table_raises(conn, small_clean, table):
    """Verificar que un nombre de tabla fuera de la lista blanca da ``ValueError``."""
    with pytest.raises(ValueError):
        validate_load(conn, table, small_clean)


def test_validate_load_filters_by_run_id(run_conn, flow):
    """Verificar que el conteo se filtra por ``run_id`` cuando se indica."""
    _, tables = flow
    write_run(run_conn, tables)
    write_run(run_conn, _with_run_id(tables, OTHER_RUN_ID))
    expected = tables[PREDICTIONS_TABLE]
    validate_load(run_conn, PREDICTIONS_TABLE, expected, run_id=RUN_ID)
    with pytest.raises(ValueError):
        validate_load(run_conn, PREDICTIONS_TABLE, expected)


# --- Lectura y utilidades --------------------------------------------------


def test_read_table_orders_by_primary_key(conn, small_clean):
    """Verificar columnas explícitas en orden y filas ordenadas por la PK."""
    write_clean(conn, small_clean.iloc[::-1])
    back = read_table(conn, CLEAN_TABLE)
    assert list(back.columns) == list(CLEAN_COLUMNS)
    assert back[ID_COLUMN].tolist() == sorted(small_clean[ID_COLUMN].tolist())


def test_read_table_filters_predictions_by_run_id(run_conn, flow):
    """Verificar el filtro por ``run_id`` y el orden por modelo e id."""
    _, tables = flow
    write_run(run_conn, tables)
    write_run(run_conn, _with_run_id(tables, OTHER_RUN_ID))
    preds = read_table(run_conn, PREDICTIONS_TABLE, run_id=OTHER_RUN_ID)
    assert len(preds) == len(tables[PREDICTIONS_TABLE])
    assert (preds[RUN_ID_COLUMN] == OTHER_RUN_ID).all()
    keys = list(zip(preds["modelo"], preds[ID_COLUMN]))
    assert keys == sorted(keys)


RUN_KEYS = {
    CONFIG_TABLE: [RUN_ID_COLUMN, "modelo"],
    METRICS_TABLE: [RUN_ID_COLUMN, "modelo", "metrica", "origen"],
    PREDICTIONS_TABLE: [RUN_ID_COLUMN, "modelo", ID_COLUMN],
}


@pytest.mark.parametrize("table", list(RUN_KEYS))
def test_read_table_orders_runs_by_primary_key(run_conn, flow, table):
    """Verificar el orden por PK aunque las corridas se escriban desordenadas."""
    _, tables = flow
    write_run(run_conn, _with_run_id(tables, OTHER_RUN_ID))
    write_run(run_conn, tables)
    back = read_table(run_conn, table)
    keys = list(back[RUN_KEYS[table]].itertuples(index=False, name=None))
    assert keys == sorted(keys)


def test_read_confusion_matches_evaluation(run_conn, flow):
    """Verificar que la vista de la matriz reproduce la matriz de la evaluación."""
    artifacts, tables = flow
    write_run(run_conn, tables)
    write_run(run_conn, _with_run_id(tables, OTHER_RUN_ID))
    for name, evaluation in artifacts.evaluations.items():
        matrix = read_confusion(run_conn, RUN_ID, name)
        pd.testing.assert_frame_equal(
            matrix, evaluation["confusion"], check_dtype=False
        )


def test_read_confusion_unknown_run_raises(run_conn):
    """Verificar que pedir la matriz de una corrida inexistente da ``ValueError``."""
    with pytest.raises(ValueError):
        read_confusion(run_conn, "no_existe", BASELINE_NAME)


@pytest.mark.parametrize("table", UNKNOWN_TABLES)
def test_read_table_unknown_table_raises(conn, table):
    """Verificar que leer una tabla fuera de la lista blanca da ``ValueError``."""
    with pytest.raises(ValueError):
        read_table(conn, table)


def test_latest_run_id_none_when_empty(conn):
    """Verificar que sin corridas no hay ``run_id`` más reciente."""
    assert latest_run_id(conn) is None


def test_latest_run_id_returns_newest_fecha(run_conn, flow):
    """Verificar que se elige la corrida con la fecha más reciente."""
    _, tables = flow
    newer = _with_run_id(tables, "b_nueva", fecha="2026-03-01T00:00:00+00:00")
    older = _with_run_id(tables, "z_vieja", fecha="2026-01-01T00:00:00+00:00")
    write_run(run_conn, newer)
    write_run(run_conn, older)
    assert latest_run_id(run_conn) == "b_nueva"


def test_latest_run_id_breaks_ties_by_run_id(run_conn, flow):
    """Verificar que, a igual fecha, gana el ``run_id`` mayor."""
    _, tables = flow
    fecha = "2026-01-01T00:00:00+00:00"
    write_run(run_conn, _with_run_id(tables, OTHER_RUN_ID, fecha=fecha))
    write_run(run_conn, _with_run_id(tables, RUN_ID, fecha=fecha))
    assert latest_run_id(run_conn) == OTHER_RUN_ID


def test_new_run_id_is_utc_timestamp():
    """Verificar el formato ``AAAAMMDDTHHMMSSZ`` y que corresponde a ahora en UTC."""
    run_id = new_run_id()
    assert ID_REGEX.match(run_id)
    stamp = datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    assert abs(datetime.now(timezone.utc) - stamp) < timedelta(minutes=1)
