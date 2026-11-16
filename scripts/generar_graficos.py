"""Generar los gráficos de una corrida a partir de la base SQLite.

Lee de la base la tabla limpia y los resultados de una corrida (por defecto,
la más reciente), arma cada gráfico y lo guarda como PNG en la carpeta de
salida. Solo necesita ``data/inmuebles.db``: requiere haber ejecutado antes
``cargar_bd.py`` y ``entrenar.py``. La base no se modifica.

Uso::

    python scripts/generar_graficos.py
    python scripts/generar_graficos.py --run-id 20261116T153012Z
    python scripts/generar_graficos.py --db data/inmuebles.db --salida reports/figures
"""

import argparse
import logging
import sys
from contextlib import closing
from pathlib import Path

from inmuebles.config import (
    DB_PATH,
    EXIT_ERROR,
    EXIT_OK,
    FIGURES_DIR,
    LOGS_DIR,
    RUN_ID_COLUMN,
    SCRIPT_ERRORS,
)
from inmuebles.database import connect
from inmuebles.logging_setup import log_session
from inmuebles.visualization import build_figures, load_chart_data, save_figures

SCRIPT_NAME = "generar_graficos"
logger = logging.getLogger(SCRIPT_NAME)
# Indicación para cuando falta la base
PIPELINE_HINT = "Ejecutar antes scripts/cargar_bd.py y scripts/entrenar.py."


def parse_args(argv=None):
    """Leer los argumentos de la línea de comandos.

    Parameters
    ----------
    argv : list of str, optional
        Argumentos; por defecto, los de ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Con ``db``, ``salida``, ``run_id`` y ``logs``.
    """
    parser = argparse.ArgumentParser(
        description="Generar los gráficos de una corrida desde la base."
    )
    parser.add_argument(
        "--db", type=Path, default=DB_PATH, help="ruta de la base SQLite"
    )
    parser.add_argument(
        "--salida", type=Path, default=FIGURES_DIR, help="carpeta de los gráficos"
    )
    parser.add_argument(
        "--run-id", default=None, help="corrida a graficar (por defecto, la última)"
    )
    parser.add_argument(
        "--logs", type=Path, default=LOGS_DIR, help="carpeta de los archivos de log"
    )
    return parser.parse_args(argv)


def _run(args):
    """Leer la corrida de la base y guardar sus gráficos en ``args.salida``."""
    if not args.db.is_file():
        raise FileNotFoundError(f"No existe la base {args.db}. {PIPELINE_HINT}")
    logger.info("Gráficos desde la base %s", args.db)
    with closing(connect(args.db)) as conn:
        data = load_chart_data(conn, run_id=args.run_id)
    paths = save_figures(build_figures(data), args.salida)
    logger.info(
        "Corrida %s: %d gráficos en %s",
        data.run[RUN_ID_COLUMN],
        len(paths),
        args.salida,
    )


def main(argv=None):
    """Generar los gráficos y devolver el código de salida.

    Parameters
    ----------
    argv : list of str, optional
        Argumentos de la línea de comandos.

    Returns
    -------
    int
        ``EXIT_OK`` si se guardaron los gráficos; ``EXIT_ERROR`` si falta la
        base, no hay corridas, no existe el ``--run-id`` o falla la base
        (``SCRIPT_ERRORS``).
    """
    args = parse_args(argv)
    with log_session(SCRIPT_NAME, args.logs):
        try:
            _run(args)
        except SCRIPT_ERRORS as error:
            logger.error("La generación de gráficos falló: %s", error)
            return EXIT_ERROR
    return EXIT_OK


if __name__ == "__main__":
    # Acentos legibles en cualquier consola de Windows (p. ej. Git Bash)
    sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
