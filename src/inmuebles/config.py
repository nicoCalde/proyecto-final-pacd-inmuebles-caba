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
