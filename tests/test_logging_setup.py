"""Pruebas de la sesión de logging de los scripts: consola más archivo.

Cada prueba escribe sus logs en ``tmp_path``; nunca se toca ``logs/``.
"""

import logging
import re
import warnings
from datetime import datetime, timedelta, timezone

import pytest

from inmuebles.logging_setup import log_session

NAME = "prueba"
STAMP = "20260101T000000Z"
LOGGER_NAME = "inmuebles.prueba"
# Marca ``asctime`` del formato ``asctime name levelname message``
TIMESTAMP = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}"


def _lines(path):
    """Leer las líneas del archivo de log."""
    return path.read_text(encoding="utf-8").splitlines()


@pytest.fixture
def quiet_root():
    """Dejar el logger raíz en ERROR durante la prueba y restaurarlo al final."""
    root = logging.getLogger()
    previous = root.level
    root.setLevel(logging.ERROR)
    yield root
    root.setLevel(previous)


def test_log_session_creates_file_and_folders(tmp_path):
    """Verificar que el archivo es ``log_dir/name_stamp.log`` y se crea la carpeta."""
    log_dir = tmp_path / "sub" / "logs"
    with log_session(NAME, log_dir, stamp=STAMP) as path:
        logging.getLogger(LOGGER_NAME).info("hola")
    assert path == log_dir / f"{NAME}_{STAMP}.log"
    assert path.is_file()


@pytest.mark.parametrize("level", ["INFO", "WARNING", "ERROR"])
def test_log_session_writes_formatted_lines(tmp_path, quiet_root, level):
    """Verificar que cada nivel se escribe con el formato del proyecto."""
    with log_session(NAME, tmp_path, stamp=STAMP) as path:
        logging.getLogger(LOGGER_NAME).log(getattr(logging, level), "mensaje %d", 7)
    pattern = f"{TIMESTAMP} {re.escape(LOGGER_NAME)} {level} mensaje 7"
    assert any(re.fullmatch(pattern, line) for line in _lines(path))


def test_log_session_skips_debug(tmp_path):
    """Verificar que los mensajes DEBUG no se escriben."""
    with log_session(NAME, tmp_path, stamp=STAMP) as path:
        logging.getLogger(LOGGER_NAME).debug("detalle oculto")
        logging.getLogger(LOGGER_NAME).info("visible")
    text = path.read_text(encoding="utf-8")
    assert "detalle oculto" not in text
    assert "visible" in text


def test_log_session_also_logs_to_stderr(tmp_path, capsys):
    """Verificar que los mensajes también salen por la consola (stderr)."""
    with log_session(NAME, tmp_path, stamp=STAMP):
        logging.getLogger(LOGGER_NAME).warning("aviso en consola")
    assert "aviso en consola" in capsys.readouterr().err


def _session(log_dir, fail):
    """Abrir y cerrar una sesión que registra un mensaje y opcionalmente falla."""
    with log_session(NAME, log_dir, stamp=STAMP):
        logging.getLogger(LOGGER_NAME).info("dentro")
        if fail:
            raise RuntimeError("falla dentro de la sesión")


@pytest.mark.parametrize("fail", [False, True])
def test_log_session_restores_root_and_closes_file(tmp_path, quiet_root, fail):
    """Verificar que al salir, aun con error, se restaura el raíz y se cierra el log."""
    handlers = list(quiet_root.handlers)
    if fail:
        with pytest.raises(RuntimeError):
            _session(tmp_path, fail)
    else:
        _session(tmp_path, fail)
    assert quiet_root.handlers == handlers
    assert quiet_root.level == logging.ERROR
    path = tmp_path / f"{NAME}_{STAMP}.log"
    path.unlink()
    assert not path.exists()


def test_log_session_twice_does_not_duplicate_lines(tmp_path):
    """Verificar que dos sesiones seguidas no duplican los mensajes."""
    for message in ("primera", "segunda"):
        with log_session(NAME, tmp_path, stamp=STAMP) as path:
            logging.getLogger(LOGGER_NAME).info(message)
    text = path.read_text(encoding="utf-8")
    assert text.count("primera") == 1
    assert text.count("segunda") == 1


def test_log_session_default_stamp_is_utc_run_id(tmp_path):
    """Verificar que sin ``stamp`` el nombre lleva la marca UTC del ``run_id``."""
    with log_session(NAME, tmp_path) as path:
        pass
    assert re.fullmatch(rf"{NAME}_\d{{8}}T\d{{6}}Z\.log", path.name)


def test_log_session_closes_file_handler(tmp_path, monkeypatch):
    """Verificar que al salir se cierra el handler de archivo."""
    root = logging.getLogger()
    added, original = [], root.addHandler
    monkeypatch.setattr(root, "addHandler", lambda h: (added.append(h), original(h)))
    with log_session(NAME, tmp_path, stamp=STAMP):
        pass
    files = [h for h in added if isinstance(h, logging.FileHandler)]
    assert len(files) == 1
    assert files[0].stream is None


def test_log_session_records_python_warnings(tmp_path):
    """Verificar que un aviso de ``warnings`` queda en el archivo de log."""
    showwarning = warnings.showwarning
    with log_session(NAME, tmp_path, stamp=STAMP) as path:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn("aviso de prueba", UserWarning)
    lines = _lines(path)
    assert any("WARNING" in line and "aviso de prueba" in line for line in lines)
    assert warnings.showwarning is showwarning


def test_log_session_times_are_utc(tmp_path):
    """Verificar que la hora de cada línea es UTC, como el nombre del archivo."""
    with log_session(NAME, tmp_path, stamp=STAMP) as path:
        logging.getLogger(LOGGER_NAME).info("hora")
    stamp = datetime.strptime(_lines(path)[-1][:19], "%Y-%m-%d %H:%M:%S")
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    assert abs(now - stamp) < timedelta(minutes=1)
