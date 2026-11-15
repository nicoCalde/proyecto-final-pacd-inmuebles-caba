"""Configuración central del proyecto: rutas y constantes de modelado.

Las rutas se anclan en la raíz del repositorio, por lo que no dependen del
directorio de trabajo actual. Importar este módulo no crea directorios.
"""

from pathlib import Path

# Raíz del repositorio: src/inmuebles/config.py -> parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Rutas de datos y salidas
RAW_DIR = PROJECT_ROOT / "datasets"
DB_PATH = PROJECT_ROOT / "data" / "inmuebles.db"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Reproducibilidad y partición train/test
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Archivos .dbf crudos por año (la capitalización de las carpetas es la real)
RAW_FILES = {
    2018: RAW_DIR / "Departamentos en Venta 2018" / "Departamentos_venta_2018.dbf",
    2019: RAW_DIR / "Departamentos en venta 2019" / "Departamentos_venta_2019.dbf",
}
DBF_ENCODING = "utf-8"

# Alias para unificar los nombres de columna de 2018 y 2019
COLUMN_ALIASES = {
    "antig_": "antig",
    "barrio_1": "barrio",
    "barrios_1": "barrio",
    "comuna_1": "comuna",
    "comunas": "comuna",
}

# Columnas comunes que se conservan de los datos crudos
RAW_COLUMNS = (
    "operacion",
    "tipo",
    "antig",
    "m2total",
    "m2cub",
    "ambientes",
    "banos",
    "direccion",
    "cambio",
    "preciousd",
    "precioars",
    "preciousdm",
    "precioarsm",
    "latitud",
    "longitud",
    "mes",
    "barrio",
    "comuna",
)

# Columna con el año de publicación de cada aviso
YEAR_COLUMN = "anio"
