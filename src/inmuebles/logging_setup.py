"""Sesión de logging de los scripts: consola más un archivo por ejecución.

``log_session`` agrega al logger raíz un handler de consola (stderr) y uno de
archivo en ``LOGS_DIR``, ambos con el formato ``LOG_FORMAT`` y la hora en UTC
(como el nombre del archivo y el ``run_id``), y al salir los quita y cierra,
aun si hubo un error. Los demás handlers no se tocan. Mientras dura la sesión,
los avisos de ``warnings`` (p. ej. de scikit-learn) también se registran.
"""

import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from inmuebles.config import LOG_FORMAT, LOG_LEVEL, LOGS_DIR, RUN_ID_FORMAT


def _log_path(name, log_dir, stamp):
    """Armar la ruta ``log_dir/name_stamp.log``, creando la carpeta."""
    if stamp is None:
        stamp = datetime.now(timezone.utc).strftime(RUN_ID_FORMAT)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{name}_{stamp}.log"


def _make_handlers(path):
    """Crear los handlers de consola y de archivo con el formato del proyecto."""
    handlers = (logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8"))
    formatter = logging.Formatter(LOG_FORMAT)
    formatter.converter = time.gmtime
    for handler in handlers:
        handler.setFormatter(formatter)
    return handlers


@contextmanager
def log_session(name, log_dir=LOGS_DIR, stamp=None):
    """Registrar en consola y en un archivo mientras dura el bloque ``with``.

    Parameters
    ----------
    name : str
        Prefijo del archivo de log (p. ej. el nombre del script).
    log_dir : str or pathlib.Path, default LOGS_DIR
        Carpeta del archivo; se crea si no existe.
    stamp : str, optional
        Marca del nombre del archivo; por defecto, la hora UTC actual con
        ``RUN_ID_FORMAT``.

    Yields
    ------
    pathlib.Path
        Ruta del archivo de log ``log_dir/name_stamp.log``.
    """
    path = _log_path(name, log_dir, stamp)
    root = logging.getLogger()
    previous_level = root.level
    handlers = _make_handlers(path)
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(LOG_LEVEL)
    logging.captureWarnings(True)
    try:
        yield path
    finally:
        logging.captureWarnings(False)
        for handler in handlers:
            root.removeHandler(handler)
            handler.close()
        root.setLevel(previous_level)
