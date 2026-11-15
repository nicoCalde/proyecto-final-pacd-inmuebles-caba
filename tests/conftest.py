"""Fixtures compartidas: shapefiles diminutos que imitan los esquemas reales.

Las pruebas nunca leen ``datasets/``; en su lugar se escriben con pyshp
shapefiles de puntos con pocas filas en el directorio temporal de cada prueba.
Los esquemas copian el orden, los tipos y los tamaños de los ``.dbf`` reales.
"""

import pytest
import shapefile

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
