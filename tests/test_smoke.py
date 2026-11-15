"""Pruebas de humo del paquete ``inmuebles`` y de su configuración."""

from pathlib import Path

import pytest

from inmuebles import config


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("RAW_DIR", Path("datasets")),
        ("DB_PATH", Path("data", "inmuebles.db")),
        ("FIGURES_DIR", Path("reports", "figures")),
    ],
)
def test_paths_are_absolute_and_anchored_at_root(name, expected):
    """Verificar que cada ruta es absoluta y cuelga de ``PROJECT_ROOT``."""
    path = getattr(config, name)
    assert isinstance(path, Path)
    assert path.is_absolute()
    assert path.relative_to(config.PROJECT_ROOT) == expected


def test_project_root_is_repo_root():
    """Verificar que ``PROJECT_ROOT`` apunta a la raíz real del repositorio."""
    root = config.PROJECT_ROOT
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "inmuebles").is_dir()


def test_modeling_constants():
    """Verificar la semilla aprobada y la proporción del conjunto de prueba."""
    assert config.RANDOM_STATE == 42
    assert 0 < config.TEST_SIZE < 1
