"""Entrenar y evaluar los modelos y guardar los resultados en SQLite.

Lee la Tabla 1, la parte en train y test, define el segmento con los
terciles de train, ajusta los modelos con ``GridSearchCV``, elige el mejor
por la CV y lo evalúa una sola vez en test. La corrida (``run_id``) se
escribe en las Tablas 2 y 3, y el log lleva el ``run_id`` en el nombre. El
modelo se guarda con joblib solo después de escribir la corrida. Requiere
haber ejecutado antes ``cargar_bd.py``.

Uso::

    python scripts/entrenar.py
    python scripts/entrenar.py --rapido --n-jobs 1
    python scripts/entrenar.py --db data/inmuebles.db --modelo models/m.joblib
"""

import argparse
import logging
import sys
from contextlib import closing
from pathlib import Path

from inmuebles.config import (
    CLEAN_TABLE,
    DB_PATH,
    EXIT_ERROR,
    EXIT_OK,
    ID_COLUMN,
    LOGS_DIR,
    MODEL_GRIDS,
    MODEL_PATH,
    N_JOBS,
    QUICK_GRIDS,
    REFIT_METRIC,
    SCRIPT_ERRORS,
    TARGET_SOURCE,
)
from inmuebles.database import (
    RunArtifacts,
    build_run_tables,
    connect,
    create_schema,
    new_run_id,
    read_table,
    validate_load,
    write_run,
)
from inmuebles.evaluation import evaluate_models
from inmuebles.logging_setup import log_session
from inmuebles.modeling import (
    make_xy,
    save_model,
    select_best,
    split_data,
    train_models,
)
from inmuebles.target import ValueSegmentLabeler

SCRIPT_NAME = "entrenar"
logger = logging.getLogger(SCRIPT_NAME)
# Indicación para cuando falta la Tabla 1
LOAD_HINT = "Ejecutar antes scripts/cargar_bd.py."


def parse_args(argv=None):
    """Leer los argumentos de la línea de comandos.

    Parameters
    ----------
    argv : list of str, optional
        Argumentos; por defecto, los de ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Con ``db``, ``modelo``, ``logs``, ``rapido`` y ``n_jobs``.
    """
    parser = argparse.ArgumentParser(
        description="Entrenar los modelos y guardar la corrida en la base."
    )
    parser.add_argument(
        "--db", type=Path, default=DB_PATH, help="ruta de la base SQLite"
    )
    parser.add_argument(
        "--modelo", type=Path, default=MODEL_PATH, help="ruta del modelo .joblib"
    )
    parser.add_argument(
        "--logs", type=Path, default=LOGS_DIR, help="carpeta de los archivos de log"
    )
    parser.add_argument(
        "--rapido", action="store_true", help="usar las grillas reducidas (demo)"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=N_JOBS, help="procesos en paralelo (-1: todos)"
    )
    return parser.parse_args(argv)


def _read_clean(conn):
    """Leer la Tabla 1, que no puede estar vacía.

    Raises
    ------
    ValueError
        Si la Tabla 1 está vacía; el mensaje remite a ``cargar_bd.py``.
    """
    create_schema(conn)
    df = read_table(conn, CLEAN_TABLE)
    if df.empty:
        raise ValueError(f"La tabla {CLEAN_TABLE} está vacía. {LOAD_HINT}")
    return df


def _fit(df, args, run_id):
    """Partir, etiquetar, entrenar, elegir y evaluar; devolver los artefactos.

    El modelo todavía no se guarda: eso ocurre recién cuando la corrida quedó
    escrita en la base, para que el archivo nunca corresponda a una corrida
    que no existe.
    """
    train, test = split_data(df)
    labeler = ValueSegmentLabeler().fit(train[TARGET_SOURCE])
    x_train, y_train = make_xy(train, labeler)
    x_test, y_test = make_xy(test, labeler)
    grids = QUICK_GRIDS if args.rapido else MODEL_GRIDS
    results = train_models(x_train, y_train, grids=grids, n_jobs=args.n_jobs)
    best = select_best(results)
    evaluations = evaluate_models(results, x_test, y_test)
    return RunArtifacts(
        run_id,
        results,
        evaluations,
        labeler,
        best,
        test_ids=test[ID_COLUMN],
        y_test=y_test,
        n_train=len(train),
    )


def _persist(conn, artifacts):
    """Escribir la corrida en las Tablas 2 y 3 y validar cada tabla."""
    tables = build_run_tables(artifacts)
    write_run(conn, tables)
    for table, df in tables.items():
        validate_load(conn, table, df, run_id=artifacts.run_id)


def _run(args, run_id):
    """Entrenar sobre la Tabla 1 y guardar la corrida ``run_id``."""
    if not args.db.is_file():
        raise FileNotFoundError(f"No existe la base {args.db}. {LOAD_HINT}")
    grids = "reducidas" if args.rapido else "completas"
    logger.info("Corrida %s: base=%s, grillas %s", run_id, args.db, grids)
    with closing(connect(args.db)) as conn:
        artifacts = _fit(_read_clean(conn), args, run_id)
        _persist(conn, artifacts)
    save_model(artifacts.results[artifacts.best]["estimator"], args.modelo)
    f1 = artifacts.evaluations[artifacts.best]["metrics"][REFIT_METRIC]
    logger.info(
        "Corrida %s: mejor modelo %s (%s test=%.3f)",
        run_id,
        artifacts.best,
        REFIT_METRIC,
        f1,
    )


def main(argv=None):
    """Ejecutar el entrenamiento y devolver el código de salida.

    Parameters
    ----------
    argv : list of str, optional
        Argumentos de la línea de comandos.

    Returns
    -------
    int
        ``EXIT_OK`` si la corrida se guardó; ``EXIT_ERROR`` si falta la base,
        la Tabla 1 está vacía, algún dato no es válido o falla la base
        (``SCRIPT_ERRORS``).
    """
    args = parse_args(argv)
    run_id = new_run_id()
    with log_session(SCRIPT_NAME, args.logs, stamp=run_id):
        try:
            _run(args, run_id)
        except SCRIPT_ERRORS as error:
            logger.error("El entrenamiento falló: %s", error)
            return EXIT_ERROR
    return EXIT_OK


if __name__ == "__main__":
    # Acentos legibles en cualquier consola de Windows (p. ej. Git Bash)
    sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
