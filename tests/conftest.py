"""Fixtures compartidas: shapefiles diminutos y tablas crudas sintéticas.

Las pruebas nunca leen ``datasets/``; en su lugar se escriben con pyshp
shapefiles de puntos con pocas filas en el directorio temporal de cada prueba.
Los esquemas copian el orden, los tipos y los tamaños de los ``.dbf`` reales.
Para la limpieza se arman en memoria tablas con la forma de ``load_raw`` y,
para el preprocesamiento y el modelado, tablas con la forma de la salida de
``clean``.
"""

import numpy as np
import pandas as pd
import pytest
import shapefile

from inmuebles.config import CLEAN_COLUMNS, ID_COLUMN, RAW_COLUMNS, YEAR_COLUMN

# Esquema real 2018: ANTIG_ y CAMBIO como texto, BARRIO_1 y COMUNA_1 al final.
SCHEMA_2018 = (
    ("OPERACION", "C", 254, 0),
    ("TIPO", "C", 254, 0),
    ("ANTIG_", "C", 254, 0),
    ("M2TOTAL", "N", 16, 6),
    ("M2CUB", "N", 16, 6),
    ("AMBIENTES", "N", 16, 6),
    ("BANOS", "N", 16, 6),
    ("DIRECCION", "C", 254, 0),
    ("CAMBIO", "C", 254, 0),
    ("PRECIOUSD", "N", 16, 6),
    ("PRECIOARS", "N", 16, 6),
    ("PRECIOUSDM", "N", 16, 6),
    ("PRECIOARSM", "N", 16, 6),
    ("LATITUD", "N", 16, 6),
    ("LONGITUD", "N", 16, 6),
    ("MES", "C", 254, 0),
    ("BARRIO_1", "C", 254, 0),
    ("COMUNA_1", "N", 16, 6),
)

ROWS_2018 = (
    (
        "VTA", "DTO", "10", 50, 45, 2, 1, "Av. Cabildo 1500",
        "28.5", 100000, 2850000, 2000.0, 57000.0,
        -34.56, -58.45, "ENERO", "NUÑEZ", 13,
    ),
    (
        "VTA", "DTO", "a estrenar", 80, 70, 3, 2, "Gral. Güemes 3200",
        "28.5", 200000, 5700000, 2500.0, 71250.0,
        -34.59, -58.41, "FEBRERO", "PALERMO", 14,
    ),
)  # fmt: skip

# Esquema real 2019: otro orden, ANTIG numérico, AGE extra, BARRIOS_1 y
# COMUNAS antes de MES, LATITUD y LONGITUD.
SCHEMA_2019 = (
    ("OPERACION", "C", 254, 0),
    ("TIPO", "C", 254, 0),
    ("ANTIG", "F", 19, 11),
    ("AGE", "C", 254, 0),
    ("M2TOTAL", "F", 19, 11),
    ("M2CUB", "F", 19, 11),
    ("AMBIENTES", "F", 19, 11),
    ("BANOS", "F", 19, 11),
    ("DIRECCION", "C", 254, 0),
    ("CAMBIO", "C", 254, 0),
    ("PRECIOUSD", "F", 19, 11),
    ("PRECIOARS", "F", 19, 11),
    ("PRECIOUSDM", "N", 5, 0),
    ("PRECIOARSM", "F", 19, 11),
    ("BARRIOS_1", "C", 50, 0),
    ("COMUNAS", "N", 5, 0),
    ("MES", "C", 50, 0),
    ("LATITUD", "F", 19, 11),
    ("LONGITUD", "F", 19, 11),
)

ROWS_2019 = (
    (
        "VTA", "DTO", 5.0, "EST", 60, 55, 2, 1, "Av. Córdoba 2100",
        "39.0", 120000, 4680000, 2000, 78000.0,
        "BALVANERA", 3, "MARZO", -34.60, -58.40,
    ),
    (
        "VTA", "DTO", 30.0, "EST", 40, 38, 1, 1, "Peña 2800",
        "39.0", 90000, 3510000, 2250, 87750.0,
        "RECOLETA", 2, "ABRIL", -34.59, -58.40,
    ),
    (
        "VTA", "DTO", 0.0, "EST", 100, 90, 4, 2, "Av. del Libertador 5000",
        "39.0", 400000, 15600000, 4000, 156000.0,
        "BELGRANO", 13, "MAYO", -34.56, -58.44,
    ),
)  # fmt: skip


def _write_dbf(path, schema, rows):
    """Escribir un shapefile de puntos con ``schema`` y ``rows``."""
    with shapefile.Writer(str(path), shapeType=shapefile.POINT, encoding="utf-8") as w:
        for name, kind, size, decimal in schema:
            w.field(name, kind, size=size, decimal=decimal)
        for row in rows:
            w.point(0.0, 0.0)
            w.record(*row)
    return path.with_suffix(".dbf")


@pytest.fixture
def make_dbf(tmp_path):
    """Devolver una fábrica que escribe un ``.dbf`` de 2018 o 2019.

    La fábrica acepta el año (2018 o 2019) y, opcionalmente, un campo a
    omitir para simular un archivo incompleto. Devuelve la ruta al ``.dbf``.
    """
    specs = {2018: (SCHEMA_2018, ROWS_2018), 2019: (SCHEMA_2019, ROWS_2019)}

    def _make(year, drop=None):
        schema, rows = specs[year]
        keep = [i for i, field in enumerate(schema) if field[0] != drop]
        return _write_dbf(
            tmp_path / f"deptos_{year}_{drop or 'ok'}",
            [schema[i] for i in keep],
            [[row[i] for i in keep] for row in rows],
        )

    return _make


# Fila cruda válida por defecto, con la forma de la salida de ``load_raw``.
RAW_DEFAULTS = {
    "operacion": "VTA",
    "tipo": "DTO",
    "antig": 10.0,
    "m2total": 60.0,
    "m2cub": 55.0,
    "ambientes": 2.0,
    "banos": 1.0,
    "direccion": None,
    "cambio": "28.5",
    "preciousd": 150000.0,
    "precioars": 4275000.0,
    "preciousdm": 2500.0,
    "precioarsm": 71250.0,
    "latitud": -34.58,
    "longitud": -58.43,
    "mes": "ENERO",
    "barrio": "PALERMO",
    "comuna": 14.0,
    YEAR_COLUMN: 2018,
}


@pytest.fixture
def make_raw():
    """Devolver una fábrica de tablas crudas con una fila por cada ``override``.

    Cada fila parte de ``RAW_DEFAULTS`` (válida) y recibe una dirección única,
    salvo que el ``override`` la indique; así no hay duplicados accidentales.
    """

    def _make(overrides=({},)):
        rows = [
            {**RAW_DEFAULTS, "direccion": f"Calle {i}", **override}
            for i, override in enumerate(overrides)
        ]
        return pd.DataFrame(rows, columns=[*RAW_COLUMNS, YEAR_COLUMN])

    return _make


# Fila limpia válida por defecto, con la forma de la salida de ``clean``.
CLEAN_DEFAULTS = {
    ID_COLUMN: 0,
    YEAR_COLUMN: 2018,
    "barrio": "PALERMO",
    "comuna": 14.0,
    "antig": 10.0,
    "m2total": 60.0,
    "m2cub": 55.0,
    "ambientes": 2.0,
    "banos": 1.0,
    "latitud": -34.58,
    "longitud": -58.43,
    "preciousd": 150000.0,
    "preciousdm": 2500.0,
}


@pytest.fixture(scope="session")
def make_clean():
    """Devolver una fábrica de tablas limpias con una fila por cada ``override``.

    Cada fila parte de ``CLEAN_DEFAULTS`` y recibe un identificador secuencial,
    salvo que el ``override`` lo indique.
    """

    def _make(overrides=({},)):
        rows = [
            {**CLEAN_DEFAULTS, ID_COLUMN: i, **override}
            for i, override in enumerate(overrides)
        ]
        return pd.DataFrame(rows, columns=list(CLEAN_COLUMNS))

    return _make


# Barrios del dataset aprendible: (comuna, precio base en USD/m²).
LEARNABLE_BARRIOS = {
    "VILLA LUGANO": (8.0, 1200.0),
    "BOCA": (4.0, 1800.0),
    "CABALLITO": (6.0, 2600.0),
    "PALERMO": (14.0, 3400.0),
}


@pytest.fixture(scope="module")
def learnable_clean(make_clean):
    """Devolver una tabla limpia de 200 filas cuyo precio por m² es aprendible.

    El precio depende del barrio, la latitud y la superficie, con poco ruido,
    de modo que un modelo real supera al baseline. Algunas features tienen NaN
    para ejercitar la imputación del ``Pipeline``.
    """
    rng = np.random.default_rng(0)
    n_rows = 200
    barrios = rng.choice(list(LEARNABLE_BARRIOS), size=n_rows)
    latitud = rng.uniform(-34.70, -34.55, size=n_rows)
    m2total = rng.uniform(30.0, 150.0, size=n_rows)
    noise = rng.normal(0.0, 50.0, size=n_rows)
    overrides = []
    for i, barrio in enumerate(barrios):
        comuna, base = LEARNABLE_BARRIOS[barrio]
        price = base + 4000.0 * (latitud[i] + 34.70) - 3.0 * m2total[i] + noise[i]
        overrides.append(
            {
                "barrio": barrio,
                "comuna": comuna,
                "latitud": latitud[i],
                "m2total": m2total[i],
                "m2cub": 0.9 * m2total[i],
                "antig": float(rng.integers(0, 60)),
                "preciousdm": price,
                "preciousd": price * m2total[i],
            }
        )
    df = make_clean(overrides)
    df.loc[::25, "antig"] = np.nan
    df.loc[3::40, "m2cub"] = np.nan
    df.loc[7::50, "comuna"] = np.nan
    return df


# --- Carpeta de datos crudos para las pruebas de punta a punta --------------

# Filas aprendibles por año y tipo de cambio de cada año (texto, como en los
# .dbf reales)
RAW_DIR_ROWS = 150
RAW_DIR_CAMBIO = {2018: "28.5", 2019: "39.0"}
RAW_DIR_SCHEMAS = {2018: SCHEMA_2018, 2019: SCHEMA_2019}
# Filas que la limpieza descarta (otra operación) o enmascara (fuera de rango)
RAW_DIR_DISCARDED = ({"operacion": "ALQ"}, {"m2total": 5000.0})


def _schema_key(name):
    """Normalizar un campo del esquema como lo hace ``load_raw``."""
    from inmuebles.config import COLUMN_ALIASES

    return COLUMN_ALIASES.get(name.lower(), name.lower())


def _learnable_raw_row(rng, year, index, seed):
    """Armar una fila cruda válida cuyo precio por m² es aprendible."""
    barrio = str(rng.choice(list(LEARNABLE_BARRIOS)))
    comuna, base = LEARNABLE_BARRIOS[barrio]
    latitud = rng.uniform(-34.70, -34.55)
    m2total = rng.uniform(30.0, 150.0)
    price = base + 4000.0 * (latitud + 34.70) - 3.0 * m2total + rng.normal(0, 50)
    cambio = float(RAW_DIR_CAMBIO[year])
    return {
        "operacion": "VTA",
        "tipo": "DTO",
        "antig": float(rng.integers(0, 60)),
        "age": "EST",
        "m2total": m2total,
        "m2cub": 0.9 * m2total,
        "ambientes": float(rng.integers(1, 5)),
        "banos": float(rng.integers(1, 3)),
        "direccion": f"Calle {seed}-{year}-{index}",
        "cambio": RAW_DIR_CAMBIO[year],
        "preciousd": price * m2total,
        "precioars": price * m2total * cambio,
        "preciousdm": price,
        "precioarsm": price * cambio,
        "latitud": latitud,
        "longitud": rng.uniform(-58.50, -58.36),
        "mes": "ENERO",
        "barrio": barrio,
        "comuna": comuna,
    }


def _year_records(year, seed):
    """Armar los registros de un año en el orden y con los tipos del esquema."""
    rng = np.random.default_rng([seed, year])
    rows = [_learnable_raw_row(rng, year, i, seed) for i in range(RAW_DIR_ROWS)]
    for i, override in enumerate(RAW_DIR_DISCARDED):
        rows[i] = {**rows[i], **override}
    for row in rows:
        if year == 2018:
            row["antig"] = str(int(row["antig"]))
        else:
            row["preciousdm"] = int(round(row["preciousdm"]))
            row["comuna"] = int(row["comuna"])
    keys = [_schema_key(field[0]) for field in RAW_DIR_SCHEMAS[year]]
    return [[row[key] for key in keys] for row in rows]


@pytest.fixture(scope="session")
def make_raw_dir():
    """Devolver una fábrica que escribe los ``.dbf`` de 2018 y 2019 en una carpeta.

    La fábrica recibe la carpeta raíz y una semilla, escribe cada año en
    ``root / RAW_FILE_NAMES[year]`` con su esquema real y devuelve ``root``.
    Cada año tiene ``RAW_DIR_ROWS`` filas aprendibles con direcciones únicas;
    la primera es un alquiler (se descarta) y la segunda tiene ``m2total``
    fuera de rango (se enmascara). Semillas distintas dan datos distintos.
    """
    from inmuebles.config import RAW_FILE_NAMES

    def _make(root, seed=0):
        for year, name in RAW_FILE_NAMES.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            records = _year_records(year, seed)
            _write_dbf(path.with_suffix(""), RAW_DIR_SCHEMAS[year], records)
        return root

    return _make
