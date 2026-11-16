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

# Filtro de avisos: solo ventas ("VTA") de departamentos ("DTO")
SALE_OPERATION = "VTA"
APARTMENT_TYPE = "DTO"

# Etiquetas de antigüedad que indican un inmueble nuevo (antigüedad 0)
ANTIG_NEW_LABELS = ("A ESTRENAR", "PRE-VENTA")

# Los 48 barrios oficiales de CABA, con la grafía del dataset
CABA_BARRIOS = (
    "AGRONOMIA",
    "ALMAGRO",
    "BALVANERA",
    "BARRACAS",
    "BELGRANO",
    "BOCA",
    "BOEDO",
    "CABALLITO",
    "CHACARITA",
    "COGHLAN",
    "COLEGIALES",
    "CONSTITUCION",
    "FLORES",
    "FLORESTA",
    "LINIERS",
    "MATADEROS",
    "MONTE CASTRO",
    "MONTSERRAT",
    "NUEVA POMPEYA",
    "NUÑEZ",
    "PALERMO",
    "PARQUE AVELLANEDA",
    "PARQUE CHACABUCO",
    "PARQUE CHAS",
    "PARQUE PATRICIOS",
    "PATERNAL",
    "PUERTO MADERO",
    "RECOLETA",
    "RETIRO",
    "SAAVEDRA",
    "SAN CRISTOBAL",
    "SAN NICOLAS",
    "SAN TELMO",
    "VELEZ SARSFIELD",
    "VERSALLES",
    "VILLA CRESPO",
    "VILLA DEL PARQUE",
    "VILLA DEVOTO",
    "VILLA GRAL. MITRE",
    "VILLA LUGANO",
    "VILLA LURO",
    "VILLA ORTUZAR",
    "VILLA PUEYRREDON",
    "VILLA REAL",
    "VILLA RIACHUELO",
    "VILLA SANTA RITA",
    "VILLA SOLDATI",
    "VILLA URQUIZA",
)

# Subdivisiones no oficiales que se unifican con su barrio oficial
BARRIO_ALIASES = {
    "BARRACAS ESTE": "BARRACAS",
    "BARRACAS OESTE": "BARRACAS",
    "FLORES NORTE": "FLORES",
    "FLORES SUR": "FLORES",
    "VILLA DEVOTO NORTE": "VILLA DEVOTO",
    "VILLA DEVOTO SUR": "VILLA DEVOTO",
}

# Rangos plausibles (inclusivos); los valores fuera de rango pasan a NaN
VALID_RANGES = {
    "antig": (0, 150),
    "m2total": (10, 2000),
    "m2cub": (10, 2000),
    "ambientes": (1, 15),
    "banos": (1, 10),
    "latitud": (-34.71, -34.52),
    "longitud": (-58.54, -58.33),
    "comuna": (1, 15),
}

# Cuantiles (inclusivos) del precio por m² fuera de los cuales se descarta
PRICE_QUANTILES = (0.01, 0.99)

# Columna de la que se derivará la variable objetivo: precio en USD por m²
TARGET_SOURCE = "preciousdm"

# Identificador secuencial de cada aviso limpio
ID_COLUMN = "id_propiedad"

# Columnas de la tabla limpia, en orden
CLEAN_COLUMNS = (
    ID_COLUMN,
    YEAR_COLUMN,
    "barrio",
    "comuna",
    "antig",
    "m2total",
    "m2cub",
    "ambientes",
    "banos",
    "latitud",
    "longitud",
    "preciousd",
    TARGET_SOURCE,
)

# Variable objetivo: segmento de valor por terciles del precio en USD por m²
TARGET_COLUMN = "segmento"
# Etiquetas ordenadas de menor a mayor valor
SEGMENT_LABELS = ("Accesible", "Estándar", "Alta")
# Cuantiles que separan los segmentos (terciles), ajustados solo sobre train
SEGMENT_QUANTILES = (1 / 3, 2 / 3)

# Features del modelo: numéricas (se imputan y escalan) y categóricas
NUMERIC_FEATURES = (
    "antig",
    "m2total",
    "m2cub",
    "ambientes",
    "banos",
    "latitud",
    "longitud",
)
# comuna es un código de distrito: categórica, no ordinal
CATEGORICAL_FEATURES = ("barrio", "comuna")
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columnas derivadas del precio (el target): nunca se usan como features
LEAKAGE_COLUMNS = ("preciousd", "precioars", TARGET_SOURCE, "precioarsm", "cambio")
