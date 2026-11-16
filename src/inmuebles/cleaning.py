"""Limpieza de los avisos crudos de departamentos en venta.

Cada función recibe la tabla con la forma de ``load_raw`` y devuelve un
DataFrame nuevo, sin modificar la entrada. ``clean`` encadena todos los pasos
y devuelve la tabla limpia con las columnas ``CLEAN_COLUMNS``.
"""

import logging
import re

import pandas as pd

from inmuebles.config import (
    ANTIG_NEW_LABELS,
    APARTMENT_TYPE,
    BARRIO_ALIASES,
    CABA_BARRIOS,
    CLEAN_COLUMNS,
    ID_COLUMN,
    PRICE_QUANTILES,
    SALE_OPERATION,
    TARGET_SOURCE,
    VALID_RANGES,
)

logger = logging.getLogger(__name__)

# Todo lo que no sea letra ASCII mayúscula o espacio; al quitarlo, las
# variantes mal codificadas de un nombre (p. ej. de NUÑEZ) coinciden.
_NON_SKELETON = re.compile(r"[^A-Z ]")


def _skeleton(name):
    """Reducir un nombre en mayúsculas a sus letras ASCII y espacios."""
    return _NON_SKELETON.sub("", name)


# Esqueleto de cada barrio oficial o alias -> nombre oficial
_BARRIO_LOOKUP = {
    **{_skeleton(barrio): barrio for barrio in CABA_BARRIOS},
    **{_skeleton(alias): barrio for alias, barrio in BARRIO_ALIASES.items()},
}


def filter_listings(df):
    """Conservar solo los avisos de venta de departamentos.

    Parameters
    ----------
    df : pandas.DataFrame
        Tabla con las columnas ``operacion`` y ``tipo``.

    Returns
    -------
    pandas.DataFrame
        Filas con ``operacion == SALE_OPERATION`` y ``tipo == APARTMENT_TYPE``.
    """
    keep = (df["operacion"] == SALE_OPERATION) & (df["tipo"] == APARTMENT_TYPE)
    return df.loc[keep].copy()


def parse_antig(df):
    """Convertir la antigüedad a ``float64``.

    Las etiquetas de ``ANTIG_NEW_LABELS`` valen 0, la coma decimal se toma
    como punto y cualquier otro texto no numérico pasa a NaN.

    Parameters
    ----------
    df : pandas.DataFrame
        Tabla con la columna ``antig`` como texto, número o mezcla de ambos.

    Returns
    -------
    pandas.DataFrame
        Copia de ``df`` con ``antig`` de tipo ``float64``.
    """
    text = df["antig"].astype("string").str.strip().str.upper()
    text = text.mask(text.isin(ANTIG_NEW_LABELS), "0")
    text = text.str.replace(",", ".", regex=False)
    antig = pd.to_numeric(text, errors="coerce").astype("float64")
    return df.assign(antig=antig)


def normalize_barrio(df):
    """Unificar los barrios con los 48 nombres oficiales de ``CABA_BARRIOS``.

    Se recortan espacios, se pasa a mayúsculas, se aplican los alias de
    ``BARRIO_ALIASES`` y se compara el esqueleto del nombre: solo sus letras
    A-Z y espacios. Así coinciden las variantes mal codificadas, como las de
    NUÑEZ. Las letras acentuadas se eliminan del esqueleto en lugar de
    convertirse a su versión sin acento, lo que alcanza para los nombres de
    los datos. Lo desconocido pasa a NaN.

    Parameters
    ----------
    df : pandas.DataFrame
        Tabla con la columna ``barrio``.

    Returns
    -------
    pandas.DataFrame
        Copia de ``df`` con ``barrio`` normalizado.
    """
    barrio = df["barrio"].astype("string").str.strip().str.upper()
    skeleton = barrio.str.replace(_NON_SKELETON, "", regex=True)
    return df.assign(barrio=skeleton.map(_BARRIO_LOOKUP))


def mask_out_of_range(df):
    """Reemplazar por NaN los valores fuera de los rangos de ``VALID_RANGES``.

    Parameters
    ----------
    df : pandas.DataFrame
        Tabla con las columnas de ``VALID_RANGES``.

    Returns
    -------
    pandas.DataFrame
        Copia de ``df`` donde cada valor fuera de su rango inclusivo es NaN.

    Notes
    -----
    Por cada columna con algún valor enmascarado (no nulo antes, NaN después)
    se registra un WARNING con la cantidad.
    """
    masked = {
        column: df[column].where(df[column].between(low, high))
        for column, (low, high) in VALID_RANGES.items()
    }
    for column, values in masked.items():
        _warn_masked(column, df[column].notna() & values.isna())
    return df.assign(**masked)


def _warn_masked(column, newly_masked):
    """Registrar un WARNING si ``column`` tiene valores recién enmascarados."""
    count = int(newly_masked.sum())
    if count:
        logger.warning("%s: %d valores fuera de rango pasan a NaN", column, count)


def drop_invalid_rows(df):
    """Descartar filas sin precio válido o con superficies inconsistentes.

    Parameters
    ----------
    df : pandas.DataFrame
        Tabla con ``TARGET_SOURCE``, ``m2cub`` y ``m2total``.

    Returns
    -------
    pandas.DataFrame
        Filas con ``TARGET_SOURCE`` positivo y sin ``m2cub > m2total``; si
        falta alguna superficie, la fila se conserva.
    """
    valid_price = df[TARGET_SOURCE] > 0
    inconsistent_m2 = df["m2cub"] > df["m2total"]
    return df.loc[valid_price & ~inconsistent_m2].copy()


def drop_price_outliers(df):
    """Descartar los precios por m² fuera de los cuantiles ``PRICE_QUANTILES``.

    Parameters
    ----------
    df : pandas.DataFrame
        Tabla con la columna ``TARGET_SOURCE``.

    Returns
    -------
    pandas.DataFrame
        Filas cuyo ``TARGET_SOURCE`` está entre ambos cuantiles, inclusive.
    """
    price = df[TARGET_SOURCE]
    low, high = price.quantile(list(PRICE_QUANTILES))
    return df.loc[price.between(low, high)].copy()


def _drop_duplicates(df):
    """Descartar los avisos repetidos exactamente."""
    return df.drop_duplicates()


# Pasos de ``clean`` en orden de aplicación. Los duplicados se buscan después
# de normalizar, para que avisos iguales escritos distinto cuenten una vez, y
# los precios inválidos se descartan antes de calcular los cuantiles.
_STEPS = (
    filter_listings,
    parse_antig,
    normalize_barrio,
    _drop_duplicates,
    mask_out_of_range,
    drop_invalid_rows,
    drop_price_outliers,
)


def _log_dropped(step_name, dropped):
    """Registrar las filas eliminadas por un paso: WARNING si hubo, si no INFO."""
    level = logging.WARNING if dropped else logging.INFO
    logger.log(level, "%s: %d filas eliminadas", step_name, dropped)


def clean(df):
    """Aplicar toda la limpieza y devolver la tabla con ``CLEAN_COLUMNS``.

    Parameters
    ----------
    df : pandas.DataFrame
        Tabla cruda con la forma de la salida de ``load_raw``.

    Returns
    -------
    pandas.DataFrame
        Tabla limpia con las columnas ``CLEAN_COLUMNS``, ``ID_COLUMN``
        numerado desde 1 e índice continuo desde 0.
    """
    result = df
    for step in _STEPS:
        before = len(result)
        result = step(result)
        _log_dropped(step.__name__, before - len(result))
    columns = [column for column in CLEAN_COLUMNS if column != ID_COLUMN]
    result = result.loc[:, columns].reset_index(drop=True)
    result.insert(0, ID_COLUMN, range(1, len(result) + 1))
    logger.info("Limpieza terminada: %d de %d filas", len(result), len(df))
    return result
