"""Cargar los datos crudos, limpiarlos y guardarlos en la Tabla 1 de SQLite.

Lee los ``.dbf`` de 2018 y 2019 de la carpeta de datos crudos, aplica la
limpieza y escribe ``propiedades_limpias`` en la base, validando columnas y
cantidad de filas. Recargar los mismos datos no cambia nada.

Uso::

    python scripts/cargar_bd.py
    python scripts/cargar_bd.py --crudos datasets --db data/inmuebles.db
    python scripts/cargar_bd.py --reemplazar-corridas
"""

import argparse
import logging
import sys
from contextlib import closing
from pathlib import Path

from inmuebles.cleaning import clean
from inmuebles.config import (
    CLEAN_TABLE,
    DB_PATH,
    EXIT_ERROR,
    EXIT_OK,
    LOGS_DIR,
    RAW_DIR,
    RAW_FILE_NAMES,
    SCRIPT_ERRORS,
)
from inmuebles.data_loading import load_raw
from inmuebles.database import connect, create_schema, validate_load, write_clean
from inmuebles.logging_setup import log_session

SCRIPT_NAME = "cargar_bd"
logger = logging.getLogger(SCRIPT_NAME)


def parse_args(argv=None):
    """Leer los argumentos de la línea de comandos.

    Parameters
    ----------
    argv : list of str, optional
        Argumentos; por defecto, los de ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Con ``db``, ``crudos``, ``logs`` y ``reemplazar_corridas``.
    """
    parser = argparse.ArgumentParser(
        description="Cargar los datos crudos limpios en la Tabla 1 de la base."
    )
    parser.add_argument(
        "--db", type=Path, default=DB_PATH, help="ruta de la base SQLite"
    )
    parser.add_argument(
        "--crudos",
        type=Path,
        default=RAW_DIR,
        help="carpeta con los datos crudos de 2018 y 2019",
    )
    parser.add_argument(
        "--logs", type=Path, default=LOGS_DIR, help="carpeta de los archivos de log"
    )
    parser.add_argument(
        "--reemplazar-corridas",
        action="store_true",
        help="si los datos cambiaron, borrar las corridas guardadas",
    )
    return parser.parse_args(argv)


def _load_clean(raw_dir):
    """Leer los ``.dbf`` de ``raw_dir`` y devolver la tabla limpia.

    Raises
    ------
    FileNotFoundError
        Si falta algún archivo crudo; el mensaje remite al README.
    """
    files = {year: raw_dir / name for year, name in RAW_FILE_NAMES.items()}
    try:
        raw = load_raw(files)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{error}. Descargar los datos crudos como indica el README."
        ) from error
    return clean(raw)


def _run(args):
    """Limpiar los datos crudos y cargarlos en la Tabla 1, validando la carga."""
    logger.info("Carga: crudos=%s, base=%s", args.crudos, args.db)
    df = _load_clean(args.crudos)
    with closing(connect(args.db)) as conn:
        create_schema(conn)
        write_clean(conn, df, replace_runs=args.reemplazar_corridas)
        validate_load(conn, CLEAN_TABLE, df)
    logger.info("Tabla 1 cargada: %d filas en %s", len(df), args.db)


def main(argv=None):
    """Ejecutar la carga y devolver el código de salida.

    Parameters
    ----------
    argv : list of str, optional
        Argumentos de la línea de comandos.

    Returns
    -------
    int
        ``EXIT_OK`` si la carga terminó bien; ``EXIT_ERROR`` si faltan datos
        crudos, la carga no es válida o falla la base (``SCRIPT_ERRORS``).
    """
    args = parse_args(argv)
    with log_session(SCRIPT_NAME, args.logs):
        try:
            _run(args)
        except SCRIPT_ERRORS as error:
            logger.error("La carga falló: %s", error)
            return EXIT_ERROR
    return EXIT_OK


if __name__ == "__main__":
    # Acentos legibles en cualquier consola de Windows (p. ej. Git Bash)
    sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
