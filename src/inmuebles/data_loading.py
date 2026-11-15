"""Carga de los datos crudos de departamentos en venta desde archivos ``.dbf``.

Se lee solo la tabla de atributos de cada shapefile, se unifican los nombres
de columna entre años y se concatenan los años en una única tabla, sin
convertir tipos.
"""

import logging
import unicodedata
from pathlib import Path

import pandas as pd
import shapefile

from inmuebles.config import COLUMN_ALIASES, DBF_ENCODING, RAW_COLUMNS, YEAR_COLUMN

logger = logging.getLogger(__name__)


def read_dbf(path):
    """Leer la tabla de atributos de un archivo ``.dbf``.

    Parameters
    ----------
    path : str or pathlib.Path
        Ruta al archivo ``.dbf``. No se requieren ``.shp`` ni ``.shx``.

    Returns
    -------
    pandas.DataFrame
        Una fila por registro y una columna por campo, con los nombres
        originales del archivo.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo .dbf: {path}")
    with (
        open(path, "rb") as dbf,
        shapefile.Reader(dbf=dbf, encoding=DBF_ENCODING) as reader,
    ):
        # El primer campo es la marca de borrado interna del formato dBase
        names = [field[0] for field in reader.fields[1:]]
        records = [list(record) for record in reader.iterRecords()]
    logger.info("Leídas %d filas de %s", len(records), path.name)
    return pd.DataFrame(records, columns=names)


def _clean_name(name):
    """Recortar, pasar a minúsculas, quitar acentos y aplicar ``COLUMN_ALIASES``."""
    decomposed = unicodedata.normalize("NFKD", name.strip().lower())
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return COLUMN_ALIASES.get(plain, plain)


def normalize_columns(df):
    """Normalizar los nombres de columna y aplicar los alias del proyecto.

    Parameters
    ----------
    df : pandas.DataFrame
        Tabla con los nombres de columna originales.

    Returns
    -------
    pandas.DataFrame
        Copia de ``df`` con los nombres sin espacios en los extremos, en
        minúsculas, sin acentos y unificados según ``COLUMN_ALIASES``.
    """
    return df.rename(columns=_clean_name)


def validate_columns(df, required):
    """Verificar que ``df`` contenga todas las columnas requeridas.

    Parameters
    ----------
    df : pandas.DataFrame
        Tabla a verificar.
    required : Iterable of str
        Nombres de columna que deben estar presentes.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Si falta alguna columna; el mensaje enumera las faltantes.
    """
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(missing)}")


def load_raw(files):
    """Cargar y unificar los ``.dbf`` crudos de varios años.

    Parameters
    ----------
    files : Mapping of int to pathlib.Path
        Año de publicación asociado a la ruta de su archivo ``.dbf``.

    Returns
    -------
    pandas.DataFrame
        Columnas ``RAW_COLUMNS`` más ``YEAR_COLUMN``, con índice continuo y
        los tipos tal como se leyeron.

    Raises
    ------
    ValueError
        Si ``files`` está vacío o algún archivo no tiene las columnas
        requeridas.
    FileNotFoundError
        Si algún archivo no existe.
    """
    if not files:
        raise ValueError("No se indicaron archivos para cargar.")
    frames = [_load_year(year, path) for year, path in files.items()]
    return pd.concat(frames, ignore_index=True)


def _load_year(year, path):
    """Leer, normalizar y validar el ``.dbf`` de un año."""
    df = normalize_columns(read_dbf(path))
    validate_columns(df, RAW_COLUMNS)
    return df[list(RAW_COLUMNS)].assign(**{YEAR_COLUMN: year})
