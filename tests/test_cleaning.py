"""Pruebas de la limpieza de los avisos crudos de departamentos en venta."""

import logging

import numpy as np
import pandas as pd
import pytest

from inmuebles.cleaning import (
    clean,
    drop_invalid_rows,
    drop_price_outliers,
    filter_listings,
    mask_out_of_range,
    normalize_barrio,
    parse_antig,
)
from inmuebles.config import (
    BARRIO_ALIASES,
    CABA_BARRIOS,
    CLEAN_COLUMNS,
    ID_COLUMN,
    VALID_RANGES,
)


def test_caba_barrios_and_aliases_are_consistent():
    """Verificar que hay 48 barrios únicos y que los alias apuntan a ellos."""
    assert len(set(CABA_BARRIOS)) == len(CABA_BARRIOS) == 48
    assert {"NUÑEZ", "BOCA", "PATERNAL", "VILLA GRAL. MITRE"} <= set(CABA_BARRIOS)
    assert set(BARRIO_ALIASES.values()) <= set(CABA_BARRIOS)


@pytest.mark.parametrize(
    "func",
    [
        filter_listings,
        parse_antig,
        normalize_barrio,
        mask_out_of_range,
        drop_invalid_rows,
        drop_price_outliers,
        clean,
    ],
)
def test_functions_do_not_mutate_input(make_raw, func):
    """Verificar que cada función devuelve un DataFrame nuevo sin tocar la entrada."""
    df = make_raw([{"barrio": "NUÃ‘EZ"}, {"m2total": 0.0}, {"preciousdm": 0.0}])
    original = df.copy()
    result = func(df)
    assert result is not df
    pd.testing.assert_frame_equal(df, original)


def test_filter_listings_keeps_only_apartment_sales(make_raw):
    """Verificar que solo quedan los avisos de venta de departamentos."""
    df = make_raw([{}, {"operacion": "ALQ"}, {"tipo": "EDIFICIO"}])
    result = filter_listings(df)
    assert result["direccion"].tolist() == ["Calle 0"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("40,000000", 40.0),
        ("-2,000000", -2.0),
        ("10", 10.0),
        ("A ESTRENAR", 0.0),
        ("PRE-VENTA", 0.0),
        ("SIN DATO", np.nan),
        (5.0, 5.0),
        (np.nan, np.nan),
    ],
)
def test_parse_antig_converts_to_float(make_raw, raw, expected):
    """Verificar la conversión de ``antig`` con coma decimal y etiquetas."""
    result = parse_antig(make_raw([{"antig": raw}]))
    assert result["antig"].dtype == float
    np.testing.assert_equal(result["antig"].iloc[0], expected)


def test_parse_antig_handles_mixed_years(make_raw):
    """Verificar que una columna con texto de 2018 y floats de 2019 queda float."""
    df = make_raw([{"antig": "40,000000"}, {"antig": 3.0}, {"antig": "SIN DATO"}])
    result = parse_antig(df)
    np.testing.assert_array_equal(result["antig"].to_numpy(), [40.0, 3.0, np.nan])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("PALERMO", "PALERMO"),
        ("  palermo ", "PALERMO"),
        ("NUÑEZ", "NUÑEZ"),
        ("NUÃ‘EZ", "NUÑEZ"),
        ("NU├æEZ", "NUÑEZ"),
        ("FLORES NORTE", "FLORES"),
        ("FLORES SUR", "FLORES"),
        ("BARRACAS ESTE", "BARRACAS"),
        ("BARRACAS OESTE", "BARRACAS"),
        ("VILLA DEVOTO NORTE", "VILLA DEVOTO"),
        ("VILLA DEVOTO SUR", "VILLA DEVOTO"),
        ("VILLA GRAL. MITRE", "VILLA GRAL. MITRE"),
        ("", np.nan),
        ("GOTHAM", np.nan),
    ],
)
def test_normalize_barrio_maps_to_official_names(make_raw, raw, expected):
    """Verificar la unificación de barrios: alias, codificación y desconocidos."""
    result = normalize_barrio(make_raw([{"barrio": raw}]))
    value = result["barrio"].iloc[0]
    if pd.isna(expected):
        assert pd.isna(value)
    else:
        assert value == expected


@pytest.mark.parametrize("column", sorted(VALID_RANGES))
def test_mask_out_of_range_sets_nan_outside_inclusive_bounds(make_raw, column):
    """Verificar que fuera de rango pasa a NaN y los bordes se conservan."""
    low, high = VALID_RANGES[column]
    values = [low, high, low - 1, high + 1, 0.0, np.nan]
    result = mask_out_of_range(make_raw([{column: v} for v in values]))
    zero = 0.0 if low <= 0 <= high else np.nan
    expected = [low, high, np.nan, np.nan, zero, np.nan]
    np.testing.assert_array_equal(result[column].to_numpy(dtype=float), expected)


def test_mask_out_of_range_leaves_other_columns_untouched(make_raw):
    """Verificar que solo se enmascaran los valores fuera de rango."""
    df = make_raw([{"m2total": 0.0}])
    result = mask_out_of_range(df)
    assert pd.isna(result["m2total"].iloc[0])
    other = [c for c in df.columns if c != "m2total"]
    pd.testing.assert_frame_equal(result[other], df[other])


def test_drop_invalid_rows_removes_bad_price_and_inconsistent_m2(make_raw):
    """Verificar que se descartan precios inválidos y ``m2cub`` > ``m2total``."""
    df = make_raw(
        [
            {},
            {"preciousdm": np.nan},
            {"preciousdm": 0.0},
            {"preciousdm": -5.0},
            {"m2total": 50.0, "m2cub": 60.0},
            {"m2total": 50.0, "m2cub": 50.0},
            {"m2total": np.nan, "m2cub": 60.0},
            {"m2total": 50.0, "m2cub": np.nan},
        ]
    )
    result = drop_invalid_rows(df)
    kept = ["Calle 0", "Calle 5", "Calle 6", "Calle 7"]
    assert result["direccion"].tolist() == kept


def test_drop_price_outliers_trims_extreme_quantiles(make_raw):
    """Verificar que se descartan los extremos y se conserva el centro."""
    df = make_raw([{"preciousdm": float(p)} for p in range(1, 101)])
    result = drop_price_outliers(df)
    assert result["preciousdm"].tolist() == [float(p) for p in range(2, 100)]


def test_drop_price_outliers_keeps_constant_prices(make_raw):
    """Verificar que con precios iguales los bordes inclusivos conservan todo."""
    result = drop_price_outliers(make_raw([{}, {}, {}]))
    assert len(result) == 3


def test_clean_returns_contract_columns_ids_and_index(make_raw):
    """Verificar columnas, identificadores consecutivos e índice nuevo."""
    df = make_raw([{}, {"operacion": "ALQ"}, {}, {"barrio": "NU├æEZ"}])
    result = clean(df)
    assert list(result.columns) == list(CLEAN_COLUMNS)
    assert result[ID_COLUMN].tolist() == [1, 2, 3]
    assert result.index.tolist() == [0, 1, 2]
    assert result["barrio"].tolist() == ["PALERMO", "PALERMO", "NUÑEZ"]


def test_clean_applies_full_chain(make_raw):
    """Verificar que ``clean`` encadena todas las reglas de limpieza."""
    df = make_raw(
        [
            {"antig": "A ESTRENAR", "barrio": "FLORES SUR"},
            {"antig": "-2,000000", "m2total": 0.0},
            {"tipo": "EDIFICIO"},
            {"preciousdm": 0.0},
            {"m2total": 40.0, "m2cub": 80.0},
        ]
    )
    result = clean(df)
    assert len(result) == 2
    assert result["antig"].iloc[0] == 0.0
    assert result["barrio"].iloc[0] == "FLORES"
    assert pd.isna(result["antig"].iloc[1])
    assert pd.isna(result["m2total"].iloc[1])


def test_clean_drops_exact_duplicates(make_raw):
    """Verificar que los avisos repetidos exactamente se cuentan una vez."""
    row = make_raw()
    result = clean(pd.concat([row, row, make_raw([{"direccion": "Otra"}])]))
    assert len(result) == 2


def test_clean_drops_duplicates_after_normalization(make_raw):
    """Verificar que avisos iguales salvo la codificación del barrio cuentan una vez."""
    df = make_raw(
        [
            {"direccion": "Cabildo 1", "barrio": "NUÑEZ", "antig": "40,000000"},
            {"direccion": "Cabildo 1", "barrio": "NUÃ‘EZ", "antig": 40.0},
        ]
    )
    assert len(clean(df)) == 1


def test_normalize_barrio_handles_all_missing_column(make_raw):
    """Verificar que una columna ``barrio`` sin ningún valor no rompe."""
    result = normalize_barrio(make_raw([{"barrio": np.nan}, {"barrio": np.nan}]))
    assert result["barrio"].isna().all()


def test_clean_drops_invalid_prices_before_trimming_outliers(make_raw):
    """Verificar que los precios inválidos no influyen en los cuantiles."""
    valid = [{"preciousdm": float(p)} for p in range(1, 101)]
    df = make_raw(valid + [{"preciousdm": 0.0}] * 20)
    result = clean(df)
    assert result["preciousdm"].tolist() == [float(p) for p in range(2, 100)]


def test_clean_empty_input_returns_empty_contract(make_raw):
    """Verificar que una entrada vacía devuelve una tabla vacía con el contrato."""
    result = clean(make_raw([]))
    assert result.empty
    assert list(result.columns) == list(CLEAN_COLUMNS)


def _messages(caplog, level):
    """Devolver los mensajes registrados por la limpieza con un nivel dado."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "inmuebles.cleaning" and record.levelno == level
    ]


def test_mask_out_of_range_warns_masked_count_per_column(make_raw, caplog):
    """Verificar un WARNING por columna enmascarada, con su cantidad de valores."""
    df = make_raw([{"m2total": 0.0}, {"m2total": 5000.0}, {"banos": 50.0}, {}])
    with caplog.at_level(logging.INFO, logger="inmuebles.cleaning"):
        mask_out_of_range(df)
    assert sorted(_messages(caplog, logging.WARNING)) == [
        "banos: 1 valores fuera de rango pasan a NaN",
        "m2total: 2 valores fuera de rango pasan a NaN",
    ]


def test_mask_out_of_range_no_warning_when_all_in_range(make_raw, caplog):
    """Verificar que sin valores fuera de rango (un NaN previo no cuenta) no avisa."""
    df = make_raw([{"antig": np.nan, "banos": np.nan}, {}])
    with caplog.at_level(logging.INFO, logger="inmuebles.cleaning"):
        mask_out_of_range(df)
    assert _messages(caplog, logging.WARNING) == []


def test_clean_warns_only_for_steps_that_drop_rows(make_raw, caplog):
    """Verificar WARNING en el paso que descarta filas e INFO en los demás."""
    df = make_raw([{}, {"operacion": "ALQ"}, {}])
    with caplog.at_level(logging.INFO, logger="inmuebles.cleaning"):
        clean(df)
    assert _messages(caplog, logging.WARNING) == ["filter_listings: 1 filas eliminadas"]
    infos = _messages(caplog, logging.INFO)
    assert "_drop_duplicates: 0 filas eliminadas" in infos
    assert any(message.startswith("Limpieza terminada") for message in infos)
