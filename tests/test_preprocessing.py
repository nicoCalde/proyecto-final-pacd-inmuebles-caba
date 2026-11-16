"""Pruebas del preprocesamiento: selección de features y ``ColumnTransformer``."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from inmuebles.config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    NUMERIC_FEATURES,
)
from inmuebles.preprocessing import build_preprocessor, select_features

# Filas de train con NaN en columnas numéricas y categóricas.
TRAIN_ROWS = (
    {"antig": 0.0, "m2total": 80.0, "barrio": "PALERMO", "comuna": 14.0},
    {"antig": 10.0, "m2total": np.nan, "barrio": "BOCA", "comuna": 4.0},
    {"antig": 40.0, "ambientes": 3.0, "m2cub": 70.0, "barrio": np.nan},
    {"antig": np.nan, "banos": 2.0, "longitud": -58.4, "comuna": np.nan},
    {"antig": 20.0, "latitud": -34.62, "barrio": "BOCA", "comuna": 4.0},
)
TRAIN_ANTIG_MEDIAN = 15.0


def _feature_names(fitted):
    """Devolver los nombres de salida sin el prefijo del transformador."""
    return [name.split("__")[-1] for name in fitted.get_feature_names_out()]


def _fit_transform(df):
    """Ajustar un preprocesador nuevo y devolverlo junto con su salida."""
    preprocessor = build_preprocessor()
    output = preprocessor.fit_transform(select_features(df))
    return preprocessor, output


def test_feature_constants_exclude_leakage():
    """Verificar que ninguna feature es una columna derivada del precio."""
    assert not set(FEATURE_COLUMNS) & set(LEAKAGE_COLUMNS)


def test_select_features_returns_exact_columns_in_order(make_clean):
    """Verificar que solo quedan ``FEATURE_COLUMNS``, en orden y sin precios."""
    result = select_features(make_clean([{}, {}]))
    assert tuple(result.columns) == FEATURE_COLUMNS
    assert not set(result.columns) & set(LEAKAGE_COLUMNS)
    assert len(result) == 2


def test_select_features_does_not_mutate_input(make_clean):
    """Verificar que se devuelve un DataFrame nuevo sin tocar la entrada."""
    df = make_clean([{}, {"barrio": "BOCA"}])
    original = df.copy()
    result = select_features(df)
    assert result is not df
    pd.testing.assert_frame_equal(df, original)


def test_select_features_missing_column_raises(make_clean):
    """Verificar que falta una feature lanza ``ValueError`` que la nombra."""
    df = make_clean().drop(columns="latitud")
    with pytest.raises(ValueError, match="latitud"):
        select_features(df)


def test_build_preprocessor_returns_unfitted_column_transformer():
    """Verificar que se devuelve un ``ColumnTransformer`` todavía sin ajustar."""
    preprocessor = build_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)
    with pytest.raises(NotFittedError):
        check_is_fitted(preprocessor)


def test_fit_transform_imputes_and_returns_dense_matrix(make_clean):
    """Verificar que la salida es una matriz densa 2-D, sin NaN y fila por fila."""
    df = make_clean(TRAIN_ROWS)
    _, output = _fit_transform(df)
    assert isinstance(output, np.ndarray)
    assert output.ndim == 2
    assert output.shape[0] == len(df)
    assert not np.isnan(output).any()


def test_output_width_is_numeric_plus_categories(make_clean):
    """Verificar el ancho: features numéricas más categorías vistas en fit."""
    df = make_clean(TRAIN_ROWS)
    _, output = _fit_transform(df)
    categories = sum(df[col].dropna().nunique() for col in CATEGORICAL_FEATURES)
    assert output.shape[1] == len(NUMERIC_FEATURES) + categories


def test_numeric_columns_are_standardized_on_fit_data(make_clean):
    """Verificar que cada columna numérica tiene media 0 y desvío 1 en train."""
    preprocessor, output = _fit_transform(make_clean(TRAIN_ROWS))
    names = _feature_names(preprocessor)
    for feature in NUMERIC_FEATURES:
        column = output[:, names.index(feature)]
        assert column.mean() == pytest.approx(0.0, abs=1e-9)
        assert column.std() == pytest.approx(1.0, abs=1e-9)


def test_feature_names_have_no_leakage(make_clean):
    """Verificar que ningún nombre de salida deriva de una columna de precio."""
    preprocessor, _ = _fit_transform(make_clean(TRAIN_ROWS))
    for name in preprocessor.get_feature_names_out():
        assert not any(leak in name for leak in LEAKAGE_COLUMNS)


def test_unseen_barrio_is_encoded_as_zeros(make_clean):
    """Verificar que un barrio no visto en fit no falla y queda todo en cero."""
    preprocessor, _ = _fit_transform(make_clean(TRAIN_ROWS))
    names = _feature_names(preprocessor)
    barrio_block = [i for i, name in enumerate(names) if name.startswith("barrio_")]
    test = select_features(make_clean([{"barrio": "VILLA LUGANO"}]))
    output = preprocessor.transform(test)
    assert barrio_block
    assert output[0, barrio_block].tolist() == [0.0] * len(barrio_block)


def test_transform_imputes_with_train_median(make_clean):
    """Verificar que un NaN en test se imputa con la mediana de train."""
    preprocessor, _ = _fit_transform(make_clean(TRAIN_ROWS))
    antig = _feature_names(preprocessor).index("antig")
    test = make_clean(
        [{"antig": np.nan}, {"antig": TRAIN_ANTIG_MEDIAN}, {"antig": 100.0}]
    )
    output = preprocessor.transform(select_features(test))
    assert output[0, antig] == pytest.approx(output[1, antig])
    assert output[0, antig] != pytest.approx(output[2, antig])
