"""Pruebas de punta a punta de los scripts de carga, entrenamiento y gráficos.

Los scripts se importan desde ``scripts/`` y se llama a su ``main`` con rutas
explícitas bajo el directorio temporal: nunca se tocan ``datasets/``, ``data/``,
``logs/``, ``models/`` ni ``reports/``. Los ``.dbf`` crudos se escriben con la
fábrica ``make_raw_dir`` y el entrenamiento usa ``--rapido --n-jobs 1``.
"""

import importlib.util
import re
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd
import pytest

from inmuebles.cleaning import clean
from inmuebles.config import (
    BASELINE_NAME,
    CHART_LIMITATIONS,
    CLEAN_TABLE,
    CONFIG_TABLE,
    CV_ORIGIN,
    EXIT_ERROR,
    EXIT_OK,
    FEATURE_COLUMNS,
    FIGURE_FORMAT,
    ID_COLUMN,
    IMPORTANCE_TABLE,
    METRICS_TABLE,
    MODEL_GRIDS,
    N_JOBS,
    PREDICTIONS_TABLE,
    QUICK_GRIDS,
    RAW_FILE_NAMES,
    RUN_ID_COLUMN,
    RUNS_TABLE,
    SEGMENT_LABELS,
    TEST_ORIGIN,
)
from inmuebles.data_loading import load_raw
from inmuebles.database import (
    connect,
    create_schema,
    latest_run_id,
    read_table,
)
from inmuebles.modeling import split_data

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
# Línea de log: fecha, hora, logger, nivel y mensaje
LOG_LINE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} (?P<name>\S+) (?P<level>[A-Z]+) "
)
# Todos los modelos de una corrida: el baseline y los modelos reales
ALL_MODELS = {BASELINE_NAME, *MODEL_GRIDS}
# Gráficos esperados: uno por limitación y una matriz por cada modelo real
CONFUSION_CHART = "matriz_confusion"
EXPECTED_FIGURES = {
    *(f"{key}.{FIGURE_FORMAT}" for key in CHART_LIMITATIONS if key != CONFUSION_CHART),
    *(f"{CONFUSION_CHART}_{model}.{FIGURE_FORMAT}" for model in MODEL_GRIDS),
}


def _load_script(name):
    """Importar ``scripts/<name>.py`` como módulo, sin agregarlo a ``sys.modules``."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw_files(raw_dir):
    """Armar el mapeo año -> ``.dbf`` de una carpeta de datos crudos."""
    return {year: raw_dir / name for year, name in RAW_FILE_NAMES.items()}


def _load(script, raw_dir, db, logs, *extra):
    """Ejecutar ``cargar_bd.main`` y devolver su código de salida."""
    argv = ["--crudos", str(raw_dir), "--db", str(db), "--logs", str(logs), *extra]
    return script.main(argv)


def _train(script, db, model, logs):
    """Ejecutar ``entrenar.main`` en modo rápido y devolver su código de salida."""
    argv = ["--db", str(db), "--modelo", str(model), "--logs", str(logs)]
    return script.main([*argv, "--rapido", "--n-jobs", "1"])


def _log_lines(log_dir):
    """Leer todas las líneas de los archivos ``.log`` de una carpeta."""
    return [
        line
        for path in sorted(Path(log_dir).glob("*.log"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _levels(log_dir):
    """Devolver los niveles presentes en las líneas con el formato del proyecto."""
    matches = (LOG_LINE.match(line) for line in _log_lines(log_dir))
    return {match["level"] for match in matches if match}


def _errors(log_dir):
    """Devolver las líneas de nivel ERROR de una carpeta de logs."""
    return [line for line in _log_lines(log_dir) if " ERROR " in line]


def _read(db, table):
    """Leer una tabla de una base existente y cerrar la conexión."""
    with closing(connect(db)) as conn:
        return read_table(conn, table)


# --- Fixtures --------------------------------------------------------------


@pytest.fixture(scope="module")
def cargar_bd():
    """Importar el script ``scripts/cargar_bd.py``."""
    return _load_script("cargar_bd")


@pytest.fixture(scope="module")
def entrenar():
    """Importar el script ``scripts/entrenar.py``."""
    return _load_script("entrenar")


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory, make_raw_dir):
    """Escribir una carpeta de datos crudos con la semilla 0."""
    return make_raw_dir(tmp_path_factory.mktemp("crudos"), seed=0)


@pytest.fixture(scope="module")
def other_raw_dir(tmp_path_factory, make_raw_dir):
    """Escribir una carpeta de datos crudos distinta (semilla 1)."""
    return make_raw_dir(tmp_path_factory.mktemp("crudos_otros"), seed=1)


@pytest.fixture(scope="module")
def e2e(tmp_path_factory, raw_dir, cargar_bd, entrenar):
    """Cargar la base y entrenar una sola vez; devolver rutas y códigos de salida."""
    root = tmp_path_factory.mktemp("e2e")
    run = SimpleNamespace(
        raw=raw_dir,
        db=root / "data" / "inmuebles.db",
        model=root / "models" / "modelo.joblib",
        load_logs=root / "logs_carga",
        train_logs=root / "logs_entrenar",
    )
    run.load_code = _load(cargar_bd, run.raw, run.db, run.load_logs)
    run.train_code = _train(entrenar, run.db, run.model, run.train_logs)
    return run


@pytest.fixture
def reload(e2e, cargar_bd, tmp_path):
    """Copiar la base ya entrenada y devolver una función que la recarga.

    La función recibe la carpeta de datos crudos y opciones extra, ejecuta
    ``cargar_bd`` sobre la copia y devuelve su código de salida; la copia
    queda en el atributo ``db`` y sus logs en ``logs``.
    """
    db = tmp_path / "copia.db"
    shutil.copy(e2e.db, db)

    logs = tmp_path / "logs"

    def _run(raw, *extra):
        return _load(cargar_bd, raw, db, logs, *extra)

    return SimpleNamespace(db=db, logs=logs, run=_run)


# --- cargar_bd -------------------------------------------------------------


def test_cargar_bd_loads_clean_table(e2e):
    """Verificar código 0 y que la Tabla 1 tiene las filas de la limpieza."""
    assert e2e.load_code == EXIT_OK
    expected = clean(load_raw(_raw_files(e2e.raw)))
    pd.testing.assert_frame_equal(
        _read(e2e.db, CLEAN_TABLE), expected, check_dtype=False
    )


def test_cargar_bd_writes_log_with_info_and_warning(e2e):
    """Verificar un archivo de log de la carga con líneas INFO y WARNING."""
    logs = list(e2e.load_logs.glob("*.log"))
    assert len(logs) == 1
    assert logs[0].name.startswith("cargar_bd")
    assert {"INFO", "WARNING"} <= _levels(e2e.load_logs)


def test_cargar_bd_second_run_keeps_content(cargar_bd, raw_dir, tmp_path):
    """Verificar que cargar dos veces los mismos datos deja la misma Tabla 1."""
    db, logs = tmp_path / "bd.db", tmp_path / "logs"
    assert _load(cargar_bd, raw_dir, db, logs) == EXIT_OK
    first = _read(db, CLEAN_TABLE)
    assert _load(cargar_bd, raw_dir, db, logs) == EXIT_OK
    assert _read(db, CLEAN_TABLE).equals(first)


def test_cargar_bd_missing_raw_dir_fails_before_db(cargar_bd, tmp_path):
    """Verificar código 1, ERROR que remite al README y ninguna base creada."""
    db, logs = tmp_path / "bd.db", tmp_path / "logs"
    assert _load(cargar_bd, tmp_path / "no_existe", db, logs) == EXIT_ERROR
    errors = _errors(logs)
    assert errors
    assert any("README" in line for line in errors)
    assert not db.exists()


def test_cargar_bd_validates_clean_table(cargar_bd, raw_dir, tmp_path, monkeypatch):
    """Verificar que la carga valida la Tabla 1 contra la tabla limpia."""
    calls = []
    monkeypatch.setattr(
        cargar_bd, "validate_load", lambda conn, table, df: calls.append(table)
    )
    assert _load(cargar_bd, raw_dir, tmp_path / "bd.db", tmp_path / "logs") == 0
    assert calls == [CLEAN_TABLE]


def test_cargar_bd_database_error_is_logged(cargar_bd, raw_dir, tmp_path, monkeypatch):
    """Verificar que una falla de SQLite da código 1 y una línea ERROR."""

    def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(cargar_bd, "write_clean", locked)
    logs = tmp_path / "logs"
    assert _load(cargar_bd, raw_dir, tmp_path / "bd.db", logs) == EXIT_ERROR
    assert any("database is locked" in line for line in _errors(logs))


def test_cargar_bd_unexpected_error_propagates(cargar_bd, tmp_path, monkeypatch):
    """Verificar que un error inesperado no se convierte en código de salida."""

    def boom(files):
        raise RuntimeError("inesperado")

    monkeypatch.setattr(cargar_bd, "load_raw", boom)
    with pytest.raises(RuntimeError):
        _load(cargar_bd, tmp_path, tmp_path / "bd.db", tmp_path / "logs")


# --- entrenar --------------------------------------------------------------


def test_entrenar_exits_ok(e2e):
    """Verificar que el entrenamiento termina con código 0."""
    assert e2e.train_code == EXIT_OK


def test_entrenar_writes_one_run_equal_to_latest(e2e):
    """Verificar una única corrida en ``ejecuciones`` y que es la más reciente."""
    runs = _read(e2e.db, RUNS_TABLE)
    with closing(connect(e2e.db)) as conn:
        latest = latest_run_id(conn)
    assert runs[RUN_ID_COLUMN].tolist() == [latest]


def test_entrenar_config_has_baseline_and_models(e2e):
    """Verificar que la Tabla 3 tiene el baseline y ambos modelos reales."""
    assert set(_read(e2e.db, CONFIG_TABLE)["modelo"]) == ALL_MODELS


def test_entrenar_metrics_have_cv_and_test_per_model(e2e):
    """Verificar métricas de CV y de test para cada modelo."""
    metrics = _read(e2e.db, METRICS_TABLE)
    origins = metrics.groupby("modelo")["origen"].agg(set).to_dict()
    assert origins == {model: {CV_ORIGIN, TEST_ORIGIN} for model in ALL_MODELS}


def test_entrenar_predictions_cover_test_for_each_model(e2e):
    """Verificar una predicción por modelo y caso de test, con ids de la Tabla 1."""
    clean_table = _read(e2e.db, CLEAN_TABLE)
    n_test = len(split_data(clean_table)[1])
    predictions = _read(e2e.db, PREDICTIONS_TABLE)
    assert len(predictions) == len(ALL_MODELS) * n_test
    counts = predictions.groupby("modelo").size().to_dict()
    assert counts == {model: n_test for model in ALL_MODELS}
    assert set(predictions[ID_COLUMN]) <= set(clean_table[ID_COLUMN])


def test_entrenar_saves_model_that_predicts_segments(e2e):
    """Verificar que el modelo guardado se carga y predice segmentos válidos."""
    assert e2e.model.is_file()
    features = _read(e2e.db, CLEAN_TABLE).loc[:, list(FEATURE_COLUMNS)]
    predicted = joblib.load(e2e.model).predict(features)
    assert set(predicted) <= set(SEGMENT_LABELS)


def test_entrenar_saves_the_chosen_model(e2e):
    """Verificar que el modelo guardado es el elegido en la corrida."""
    chosen = _read(e2e.db, RUNS_TABLE)["modelo_elegido"].item()
    config = _read(e2e.db, CONFIG_TABLE)
    expected = config.loc[config["modelo"] == chosen, "clase"].item()
    assert type(joblib.load(e2e.model).named_steps["clf"]).__name__ == expected


def test_entrenar_log_name_contains_run_id(e2e):
    """Verificar que el log del entrenamiento lleva el ``run_id`` en el nombre."""
    run_id = _read(e2e.db, RUNS_TABLE)[RUN_ID_COLUMN].iloc[0]
    assert (e2e.train_logs / f"entrenar_{run_id}.log").is_file()
    assert "INFO" in _levels(e2e.train_logs)


def test_entrenar_without_db_fails_without_creating_it(entrenar, tmp_path):
    """Verificar código 1, ERROR que remite a ``cargar_bd`` y ninguna base creada."""
    db, logs = tmp_path / "no_existe.db", tmp_path / "logs"
    assert _train(entrenar, db, tmp_path / "modelo.joblib", logs) == EXIT_ERROR
    assert any("cargar_bd" in line for line in _errors(logs))
    assert not db.exists()


def test_entrenar_empty_clean_table_fails(entrenar, tmp_path):
    """Verificar código 1 y ERROR si la base existe pero la Tabla 1 está vacía."""
    db, logs = tmp_path / "vacia.db", tmp_path / "logs"
    with closing(connect(db)) as conn:
        create_schema(conn)
    model = tmp_path / "modelo.joblib"
    assert _train(entrenar, db, model, logs) == EXIT_ERROR
    assert any("cargar_bd" in line for line in _errors(logs))
    assert not model.exists()


@pytest.fixture
def trained_copy(e2e, tmp_path):
    """Copiar la base ya entrenada; devolver rutas de base, modelo y logs."""
    db = tmp_path / "copia.db"
    shutil.copy(e2e.db, db)
    return SimpleNamespace(
        db=db, model=tmp_path / "modelo.joblib", logs=tmp_path / "logs"
    )


def test_entrenar_log_stamp_is_run_id(entrenar, tmp_path, monkeypatch):
    """Verificar que el nombre del log usa el ``run_id`` de la corrida."""
    monkeypatch.setattr(entrenar, "new_run_id", lambda: "20000101T000000Z")
    logs = tmp_path / "logs"
    _train(entrenar, tmp_path / "no_existe.db", tmp_path / "m.joblib", logs)
    assert (logs / "entrenar_20000101T000000Z.log").is_file()


@pytest.mark.parametrize(
    ("extra", "grids", "n_jobs"),
    [((), MODEL_GRIDS, N_JOBS), (("--rapido", "--n-jobs", "2"), QUICK_GRIDS, 2)],
    ids=["completo", "rapido"],
)
def test_entrenar_passes_grids_and_n_jobs(
    entrenar, trained_copy, monkeypatch, extra, grids, n_jobs
):
    """Verificar que ``--rapido`` y ``--n-jobs`` llegan a ``train_models``."""
    seen = {}

    def spy(x, y, grids=None, n_jobs=None):
        seen.update(grids=grids, n_jobs=n_jobs)
        raise ValueError("corte de la prueba")

    monkeypatch.setattr(entrenar, "train_models", spy)
    paths = trained_copy
    argv = ["--db", str(paths.db), "--modelo", str(paths.model)]
    assert entrenar.main([*argv, "--logs", str(paths.logs), *extra]) == EXIT_ERROR
    assert seen == {"grids": grids, "n_jobs": n_jobs}


def test_entrenar_validates_each_run_table(entrenar, trained_copy, monkeypatch):
    """Verificar que se valida cada tabla de la corrida, filtrando por su run_id."""
    calls = []

    def spy(conn, table, df, run_id=None):
        calls.append((table, run_id))

    monkeypatch.setattr(entrenar, "validate_load", spy)
    paths = trained_copy
    assert _train(entrenar, paths.db, paths.model, paths.logs) == EXIT_OK
    tables = {
        RUNS_TABLE,
        CONFIG_TABLE,
        METRICS_TABLE,
        PREDICTIONS_TABLE,
        IMPORTANCE_TABLE,
    }
    assert {table for table, _ in calls} == tables
    assert len({run_id for _, run_id in calls}) == 1
    assert None not in {run_id for _, run_id in calls}


def test_entrenar_persist_failure_saves_no_model(entrenar, trained_copy, monkeypatch):
    """Verificar que si falla la escritura de la corrida no se guarda el modelo."""

    def locked(conn, tables):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(entrenar, "write_run", locked)
    paths = trained_copy
    assert _train(entrenar, paths.db, paths.model, paths.logs) == EXIT_ERROR
    assert any("database is locked" in line for line in _errors(paths.logs))
    assert not paths.model.exists()


def test_entrenar_unexpected_error_propagates(entrenar, trained_copy, monkeypatch):
    """Verificar que un error inesperado del entrenamiento se propaga."""

    def boom(df):
        raise RuntimeError("inesperado")

    monkeypatch.setattr(entrenar, "split_data", boom)
    paths = trained_copy
    with pytest.raises(RuntimeError):
        _train(entrenar, paths.db, paths.model, paths.logs)


# --- Recarga con corridas guardadas ----------------------------------------


@pytest.mark.parametrize(
    ("raw_fixture", "extra", "code", "runs"),
    [
        ("raw_dir", (), EXIT_OK, 1),
        ("other_raw_dir", (), EXIT_ERROR, 1),
        ("other_raw_dir", ("--reemplazar-corridas",), EXIT_OK, 0),
    ],
    ids=["mismos_datos", "datos_cambiados", "datos_cambiados_reemplazar"],
)
def test_cargar_bd_reload_after_run(request, reload, raw_fixture, extra, code, runs):
    """Verificar que recargar conserva la corrida salvo con ``--reemplazar-corridas``.

    Con datos distintos y sin la opción la carga falla y no cambia nada.
    """
    raw = request.getfixturevalue(raw_fixture)
    assert reload.run(raw, *extra) == code
    assert len(_read(reload.db, RUNS_TABLE)) == runs


def test_cargar_bd_changed_data_keeps_table_and_names_flag(reload, other_raw_dir):
    """Verificar que con datos distintos la Tabla 1 no cambia y se nombra la opción."""
    before = _read(reload.db, CLEAN_TABLE)
    assert reload.run(other_raw_dir) == EXIT_ERROR
    pd.testing.assert_frame_equal(_read(reload.db, CLEAN_TABLE), before)
    assert any("--reemplazar-corridas" in line for line in _errors(reload.logs))


# --- generar_graficos ------------------------------------------------------


def _charts(script, db, out, logs, *extra):
    """Ejecutar ``generar_graficos.main`` y devolver su código de salida."""
    argv = ["--db", str(db), "--salida", str(out), "--logs", str(logs), *extra]
    return script.main(argv)


@pytest.fixture(scope="module")
def generar_graficos():
    """Importar el script ``scripts/generar_graficos.py``."""
    return _load_script("generar_graficos")


@pytest.fixture(scope="module")
def charts(e2e, generar_graficos, tmp_path_factory):
    """Generar una sola vez los gráficos de una copia de la base entrenada."""
    root = tmp_path_factory.mktemp("graficos")
    db = root / "copia.db"
    shutil.copy(e2e.db, db)
    run = SimpleNamespace(db=db, out=root / "figuras", logs=root / "logs")
    run.code = _charts(generar_graficos, run.db, run.out, run.logs)
    return run


def test_generar_graficos_writes_every_figure(charts):
    """Verificar código 0 y un archivo no vacío por cada gráfico esperado."""
    assert charts.code == EXIT_OK
    written = {path.name: path for path in charts.out.iterdir()}
    assert set(written) == EXPECTED_FIGURES
    assert all(path.stat().st_size > 0 for path in written.values())


def test_generar_graficos_writes_log_with_info(charts):
    """Verificar un archivo de log de los gráficos con líneas INFO."""
    logs = list(charts.logs.glob("*.log"))
    assert len(logs) == 1
    assert logs[0].name.startswith("generar_graficos")
    assert "INFO" in _levels(charts.logs)


def test_generar_graficos_accepts_explicit_run_id(charts, generar_graficos, tmp_path):
    """Verificar que ``--run-id`` con una corrida existente genera los gráficos."""
    run_id = _read(charts.db, RUNS_TABLE)[RUN_ID_COLUMN].item()
    out = tmp_path / "figuras"
    code = _charts(
        generar_graficos, charts.db, out, tmp_path / "logs", "--run-id", run_id
    )
    assert code == EXIT_OK
    assert {path.name for path in out.iterdir()} == EXPECTED_FIGURES


def test_generar_graficos_without_db_fails_without_creating_it(
    generar_graficos, tmp_path
):
    """Verificar código 1, ERROR que remite a los scripts previos y ninguna base."""
    db, logs = tmp_path / "no_existe.db", tmp_path / "logs"
    assert _charts(generar_graficos, db, tmp_path / "figuras", logs) == EXIT_ERROR
    errors = _errors(logs)
    assert any("cargar_bd" in line for line in errors)
    assert any("entrenar" in line for line in errors)
    assert not db.exists()


def test_generar_graficos_old_db_without_weights_fails(
    charts, generar_graficos, tmp_path
):
    """Verificar código 1 y ERROR que remite a entrenar.py en una base vieja."""
    db, logs = tmp_path / "vieja.db", tmp_path / "logs"
    shutil.copy(charts.db, db)
    with closing(sqlite3.connect(db)) as conn:
        conn.execute(f"DROP TABLE {IMPORTANCE_TABLE}")
        conn.commit()
    assert _charts(generar_graficos, db, tmp_path / "figuras", logs) == EXIT_ERROR
    assert any("entrenar" in line for line in _errors(logs))
    with closing(sqlite3.connect(db)) as conn:
        sql = "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?"
        assert conn.execute(sql, (IMPORTANCE_TABLE,)).fetchone() is None


def test_generar_graficos_unknown_run_id_fails(charts, generar_graficos, tmp_path):
    """Verificar código 1 y una línea ERROR si el ``--run-id`` no existe."""
    logs = tmp_path / "logs"
    extra = ("--run-id", "no_existe")
    code = _charts(generar_graficos, charts.db, tmp_path / "figuras", logs, *extra)
    assert code == EXIT_ERROR
    assert _errors(logs)
