"""Pruebas de la carga de datos crudos desde los ``.dbf`` de 2018 y 2019."""

import pandas as pd
import pytest

from inmuebles.config import RAW_COLUMNS, YEAR_COLUMN
from inmuebles.data_loading import (
    load_raw,
    normalize_columns,
    read_dbf,
    validate_columns,
)


def test_read_dbf_returns_rows_with_non_ascii_text_intact(make_dbf):
    """Verificar que ``read_dbf`` devuelve las filas y respeta UTF-8."""
    df = read_dbf(make_dbf(2018))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df["BARRIO_1"].tolist() == ["NUÑEZ", "PALERMO"]
    assert df["DIRECCION"].iloc[1] == "Gral. Güemes 3200"
    assert df["AMBIENTES"].tolist() == [2, 3]


def test_read_dbf_missing_file_raises(tmp_path):
    """Verificar que un archivo inexistente produce ``FileNotFoundError``."""
    missing = tmp_path / "no_existe.dbf"
    with pytest.raises(FileNotFoundError, match="no_existe"):
        read_dbf(missing)


def test_normalize_columns_cleans_names_and_applies_aliases():
    """Verificar que se limpian los nombres y se aplican los alias."""
    df = pd.DataFrame(columns=[" Año ", "BARRIO_1", "COMUNAS", "ANTIG_", "MES"])
    result = normalize_columns(df)
    assert list(result.columns) == ["ano", "barrio", "comuna", "antig", "mes"]


def test_normalize_columns_does_not_mutate_input():
    """Verificar que ``normalize_columns`` no modifica el DataFrame original."""
    df = pd.DataFrame({"BARRIOS_1": ["NUÑEZ"]})
    result = normalize_columns(df)
    assert result is not df
    assert list(df.columns) == ["BARRIOS_1"]


def test_validate_columns_reports_missing_names():
    """Verificar que el error enumera las columnas faltantes."""
    df = pd.DataFrame(columns=["barrio"])
    with pytest.raises(ValueError, match="comuna") as excinfo:
        validate_columns(df, ("barrio", "comuna", "banos"))
    assert "banos" in str(excinfo.value)


def test_validate_columns_passes_when_complete():
    """Verificar que no hay error cuando están todas las columnas."""
    df = pd.DataFrame(columns=["barrio", "comuna"])
    assert validate_columns(df, ("barrio", "comuna")) is None


def test_load_raw_harmonizes_and_concatenates_years(make_dbf):
    """Verificar que se unifican ambos esquemas en una sola tabla."""
    df = load_raw({2018: make_dbf(2018), 2019: make_dbf(2019)})
    assert list(df.columns) == list(RAW_COLUMNS) + [YEAR_COLUMN]
    assert len(df) == 5
    assert df.index.tolist() == list(range(5))
    assert df[YEAR_COLUMN].tolist() == [2018] * 2 + [2019] * 3
    assert "age" not in df.columns
    for column in ("barrio", "comuna", "antig"):
        assert df[column].notna().all(), column
    assert df["barrio"].tolist()[:3] == ["NUÑEZ", "PALERMO", "BALVANERA"]


def test_load_raw_keeps_raw_types(make_dbf):
    """Verificar que no se convierten tipos: ``antig`` de 2018 sigue siendo texto."""
    df = load_raw({2018: make_dbf(2018)})
    assert df["antig"].tolist() == ["10", "a estrenar"]


def test_load_raw_missing_required_column_raises(make_dbf):
    """Verificar que un archivo sin una columna requerida produce ``ValueError``."""
    with pytest.raises(ValueError, match="banos"):
        load_raw({2018: make_dbf(2018, drop="BANOS")})


def test_load_raw_empty_mapping_raises():
    """Verificar que un mapeo vacío produce ``ValueError``."""
    with pytest.raises(ValueError, match="archivos"):
        load_raw({})
