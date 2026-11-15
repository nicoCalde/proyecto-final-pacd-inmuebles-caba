"""Pruebas de la variable objetivo: segmento de valor por terciles de USD/m²."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from inmuebles.config import SEGMENT_LABELS, SEGMENT_QUANTILES, TARGET_COLUMN
from inmuebles.target import ValueSegmentLabeler

LOW, MID, HIGH = SEGMENT_LABELS

# Terciles (interpolación lineal) de 1..9: posiciones 8/3 y 16/3.
TRAIN = np.arange(1.0, 10.0)
T1, T2 = 11 / 3, 19 / 3


@pytest.fixture
def labeler():
    """Devolver un etiquetador ajustado sobre los precios 1..9."""
    return ValueSegmentLabeler().fit(TRAIN)


def test_segment_constants_are_ordered_terciles():
    """Verificar que hay tres etiquetas ordenadas de menor a mayor valor."""
    assert SEGMENT_LABELS == ("Accesible", "Estándar", "Alta")
    assert SEGMENT_QUANTILES == pytest.approx((1 / 3, 2 / 3))
    assert TARGET_COLUMN == "segmento"


def test_fit_returns_self_and_learns_terciles():
    """Verificar que ``fit`` devuelve el objeto y guarda los terciles de train."""
    labeler = ValueSegmentLabeler()
    assert labeler.thresholds_ is None
    assert labeler.fit(TRAIN) is labeler
    assert labeler.thresholds_ == pytest.approx((T1, T2))


def test_transform_uses_train_thresholds_only(labeler):
    """Verificar que test se etiqueta con los umbrales de train, sin cambiarlos."""
    test = np.array([1000.0, 2000.0, 3000.0, 5.0])
    result = labeler.transform(test)
    assert labeler.thresholds_ == pytest.approx((T1, T2))
    assert result.tolist() == [HIGH, HIGH, HIGH, MID]


@pytest.mark.parametrize(
    ("price", "expected"),
    [(2.0, LOW), (5.0, MID), (8.0, HIGH)],
)
def test_transform_labels_by_range(labeler, price, expected):
    """Verificar la etiqueta de un valor dentro de cada tramo."""
    assert labeler.transform([price]).tolist() == [expected]


@pytest.mark.parametrize(("position", "expected"), [(0, MID), (1, HIGH)])
def test_transform_boundary_goes_to_upper_class(labeler, position, expected):
    """Verificar que un valor exactamente en un umbral va a la clase superior."""
    boundary = labeler.thresholds_[position]
    assert labeler.transform([boundary]).tolist() == [expected]


def test_transform_classifies_out_of_range_values(labeler):
    """Verificar que los valores fuera del rango de train igual se clasifican."""
    assert labeler.transform([-50.0, 0.0, 1e6]).tolist() == [LOW, LOW, HIGH]


def test_terciles_balance_a_uniform_sample():
    """Verificar que 300 valores distintos quedan en tres clases de 100."""
    prices = np.arange(1.0, 301.0)
    result = ValueSegmentLabeler().fit(prices).transform(prices)
    assert result.value_counts().to_dict() == {LOW: 100, MID: 100, HIGH: 100}


def test_transform_preserves_series_index_and_name(labeler):
    """Verificar que se conserva el índice de la Series y se usa ``TARGET_COLUMN``."""
    prices = pd.Series([2.0, 5.0, 8.0], index=[10, 20, 30], name="preciousdm")
    result = labeler.transform(prices)
    assert isinstance(result, pd.Series)
    assert result.name == TARGET_COLUMN
    assert result.index.tolist() == [10, 20, 30]
    assert result.tolist() == [LOW, MID, HIGH]


@pytest.mark.parametrize("convert", [list, np.asarray, pd.Series])
def test_fit_and_transform_accept_array_likes(convert):
    """Verificar que se aceptan listas, arreglos de numpy y Series."""
    labeler = ValueSegmentLabeler().fit(convert(TRAIN.tolist()))
    assert labeler.thresholds_ == pytest.approx((T1, T2))
    result = labeler.transform(convert([2.0, 8.0]))
    assert result.tolist() == [LOW, HIGH]
    assert result.index.tolist() == [0, 1]


def test_transform_before_fit_raises():
    """Verificar que transformar sin ajustar lanza ``NotFittedError``."""
    with pytest.raises(NotFittedError):
        ValueSegmentLabeler().transform([1.0])


@pytest.mark.parametrize(
    ("prices", "message"),
    [
        ([], "No hay precios"),
        ([1.0, np.nan, 3.0], "finitos"),
        ([1.0, 2.0, np.inf], "finitos"),
        ([5.0, 5.0, 5.0, 5.0], "crecientes"),
        ([[1.0, 2.0], [3.0, 4.0]], "unidimensional"),
    ],
    ids=["vacio", "nan", "infinito", "constante", "2d"],
)
def test_fit_rejects_invalid_prices(prices, message):
    """Verificar que ``fit`` rechaza cada tipo de entrada inválida con su motivo."""
    with pytest.raises(ValueError, match=message):
        ValueSegmentLabeler().fit(prices)


@pytest.mark.parametrize(
    ("prices", "message"),
    [
        ([2.0, np.nan], "finitos"),
        ([2.0, -np.inf], "finitos"),
        ([[2.0], [8.0]], "unidimensional"),
    ],
    ids=["nan", "infinito", "2d"],
)
def test_transform_rejects_invalid_prices(labeler, prices, message):
    """Verificar que ``transform`` rechaza precios no finitos o no 1-D."""
    with pytest.raises(ValueError, match=message):
        labeler.transform(prices)


def test_transform_empty_input_returns_empty_series(labeler):
    """Verificar que sin precios se devuelve una Series vacía con su nombre."""
    result = labeler.transform([])
    assert result.empty
    assert result.name == TARGET_COLUMN
