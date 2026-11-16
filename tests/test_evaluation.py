"""Pruebas de la evaluación final sobre test: métricas, matriz y tabla larga."""

import math
import warnings

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from inmuebles.config import (
    ALL_SEGMENTS,
    BASELINE_NAME,
    CV_ORIGIN,
    NUMERIC_FEATURES,
    SCORING,
    SEGMENT_LABELS,
    TARGET_SOURCE,
    TEST_ORIGIN,
)
from inmuebles.evaluation import (
    class_report,
    compute_metrics,
    confusion,
    evaluate_models,
    metrics_table,
    model_weights,
)
from inmuebles.modeling import make_xy, split_data, train_models
from inmuebles.target import ValueSegmentLabeler

ACC, STD, HIGH = SEGMENT_LABELS

# Ejemplo a mano (6 casos, 3 clases). Matriz (filas = real, columnas = pred.):
#   Accesible: [1, 1, 0]   Estándar: [0, 2, 0]   Alta: [1, 0, 1]
# Accuracy = 4/6.
# Precisión: Accesible 1/2, Estándar 2/3, Alta 1/1 -> macro = 13/18.
# Recall:    Accesible 1/2, Estándar 2/2, Alta 1/2 -> macro = 2/3.
# F1:        Accesible 1/2, Estándar 4/5, Alta 2/3 -> macro = 59/90.
Y_TRUE = pd.Series([ACC, ACC, STD, STD, HIGH, HIGH])
Y_PRED = pd.Series([ACC, STD, STD, STD, HIGH, ACC])
EXPECTED = {"accuracy": 4 / 6, "precision": 13 / 18, "recall": 2 / 3, "f1": 59 / 90}
EXPECTED_MATRIX = [[1, 1, 0], [0, 2, 0], [1, 0, 1]]

# Grillas diminutas para que el flujo completo sea rápido.
TINY_GRIDS = {
    "logistic_regression": {"clf__C": [1.0]},
    "random_forest": {"clf__n_estimators": [10], "clf__max_depth": [5]},
}
TABLE_COLUMNS = ["modelo", "metrica", "origen", "media", "desvio"]


@pytest.fixture(scope="module")
def flow(learnable_clean):
    """Entrenar una sola vez con grillas diminutas y evaluar todo en test."""
    train, test = split_data(learnable_clean)
    labeler = ValueSegmentLabeler().fit(train[TARGET_SOURCE])
    x_train, y_train = make_xy(train, labeler)
    x_test, y_test = make_xy(test, labeler)
    results = train_models(x_train, y_train, grids=TINY_GRIDS, n_jobs=1)
    return results, x_test, y_test, evaluate_models(results, x_test, y_test)


def test_compute_metrics_matches_hand_example():
    """Verificar accuracy y precisión/recall/F1 macro calculados a mano."""
    metrics = compute_metrics(Y_TRUE, Y_PRED)
    assert set(metrics) == set(SCORING)
    for name, value in EXPECTED.items():
        assert isinstance(metrics[name], float)
        assert metrics[name] == pytest.approx(value)


def test_compute_metrics_perfect_prediction():
    """Verificar que una predicción perfecta da 1.0 en todas las métricas."""
    metrics = compute_metrics(Y_TRUE, Y_TRUE)
    assert all(value == pytest.approx(1.0) for value in metrics.values())


def test_compute_metrics_class_never_predicted_counts_as_zero():
    """Verificar que una clase nunca predicha aporta precisión 0 sin avisos."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metrics = compute_metrics([ACC, STD, HIGH], [ACC, ACC, ACC])
    # Precisión: Accesible 1/3, Estándar 0, Alta 0 -> macro = 1/9.
    assert metrics["precision"] == pytest.approx(1 / 9)


def test_compute_metrics_uses_macro_not_weighted_average():
    """Verificar que con clases desbalanceadas cada clase pesa igual (macro)."""
    metrics = compute_metrics([ACC, ACC, ACC, ACC, STD, HIGH], [ACC] * 6)
    # Recall: Accesible 1, Estándar 0, Alta 0 -> macro = 1/3 (weighted = 4/6).
    assert metrics["recall"] == pytest.approx(1 / 3)


def test_class_report_class_never_predicted_counts_as_zero():
    """Verificar que el reporte por clase no avisa y pone 0 a lo no predicho."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = class_report([ACC, STD, HIGH], [ACC, ACC, ACC])
    assert report.loc[STD, "precision"] == 0.0


@pytest.mark.parametrize(
    ("y_true", "y_pred", "message"),
    [([ACC, STD], [ACC], "difieren"), ([], [], "observaciones")],
    ids=["largo", "vacio"],
)
def test_compute_metrics_rejects_bad_inputs(y_true, y_pred, message):
    """Verificar que largos distintos o entradas vacías levantan ``ValueError``."""
    with pytest.raises(ValueError, match=message):
        compute_metrics(y_true, y_pred)


def test_confusion_matches_hand_example():
    """Verificar conteos, orden de etiquetas y nombres real/predicho."""
    matrix = confusion(Y_TRUE, Y_PRED)
    assert list(matrix.index) == list(SEGMENT_LABELS)
    assert list(matrix.columns) == list(SEGMENT_LABELS)
    assert (matrix.index.name, matrix.columns.name) == ("real", "predicho")
    assert matrix.to_numpy().tolist() == EXPECTED_MATRIX
    assert matrix.sum(axis=1).to_dict() == Y_TRUE.value_counts().to_dict()


def test_confusion_keeps_absent_class_with_zeros():
    """Verificar que una clase ausente aparece igual, con fila y columna en 0."""
    matrix = confusion([ACC, STD, ACC], [ACC, STD, STD])
    assert list(matrix.index) == list(SEGMENT_LABELS)
    assert list(matrix.columns) == list(SEGMENT_LABELS)
    assert matrix.loc[HIGH].sum() == 0
    assert matrix[HIGH].sum() == 0
    assert matrix.to_numpy().sum() == 3


def test_class_report_matches_hand_example():
    """Verificar filas por segmento, soporte total y valores de una clase."""
    report = class_report(Y_TRUE, Y_PRED)
    assert list(report.index) == list(SEGMENT_LABELS)
    assert list(report.columns) == ["precision", "recall", "f1", "support"]
    assert pd.api.types.is_integer_dtype(report["support"])
    assert report["support"].sum() == len(Y_TRUE)
    assert report.loc[STD, "precision"] == pytest.approx(2 / 3)
    assert report.loc[STD, "recall"] == pytest.approx(1.0)
    assert report.loc[STD, "f1"] == pytest.approx(0.8)
    assert report.loc[HIGH, "recall"] == pytest.approx(0.5)


def test_evaluate_models_covers_every_model(flow):
    """Verificar que se evalúa cada modelo, baseline incluido, sobre test."""
    results, x_test, y_test, evaluations = flow
    assert set(evaluations) == set(results)
    assert BASELINE_NAME in evaluations
    for name, evaluation in evaluations.items():
        assert set(evaluation) == {"metrics", "confusion", "report", "predictions"}
        predictions = evaluation["predictions"]
        assert isinstance(predictions, pd.Series)
        assert predictions.index.equals(x_test.index)
        assert evaluation["confusion"].to_numpy().sum() == len(y_test)
        expected = compute_metrics(y_test, results[name]["estimator"].predict(x_test))
        assert evaluation["metrics"] == pytest.approx(expected)


def test_real_models_beat_dummy_on_test(flow):
    """Verificar que los modelos reales superan al baseline en F1 de test."""
    _, _, _, evaluations = flow
    dummy_f1 = evaluations[BASELINE_NAME]["metrics"]["f1"]
    for name, evaluation in evaluations.items():
        if name != BASELINE_NAME:
            assert evaluation["metrics"]["f1"] > dummy_f1


def test_metrics_table_shape_and_values(flow):
    """Verificar columnas, filas CV y test, desvíos y orden determinista."""
    results, _, _, evaluations = flow
    table = metrics_table(results, evaluations)
    assert list(table.columns) == TABLE_COLUMNS
    assert len(table) == len(results) * len(SCORING) * 2
    assert set(table["origen"]) == {CV_ORIGIN, TEST_ORIGIN}
    is_test = table["origen"] == TEST_ORIGIN
    assert table.loc[is_test, "desvio"].isna().all()
    assert table.loc[~is_test, "desvio"].notna().all()
    for row in table.itertuples(index=False):
        if row.origen == CV_ORIGIN:
            cv = results[row.modelo]["cv"][row.metrica]
            assert row.media == pytest.approx(cv["mean"])
            assert row.desvio == pytest.approx(cv["std"])
        else:
            expected = evaluations[row.modelo]["metrics"][row.metrica]
            assert row.media == pytest.approx(expected)
            assert math.isnan(row.desvio)


def test_metrics_table_is_deterministic(flow):
    """Verificar el orden por modelo, origen y métrica, sin depender de la entrada."""
    results, _, _, evaluations = flow
    table = metrics_table(results, evaluations)
    reversed_results = dict(reversed(list(results.items())))
    again = metrics_table(reversed_results, evaluations)
    pd.testing.assert_frame_equal(
        table.reset_index(drop=True), again.reset_index(drop=True)
    )
    # Dentro de cada (modelo, origen), las métricas siguen el orden de SCORING.
    for _, group in table.groupby(["modelo", "origen"], sort=False):
        assert list(group["metrica"]) == list(SCORING)


# --- Pesos de los modelos (importancias y coeficientes) --------------------

FOREST, LOGISTIC = "random_forest", "logistic_regression"
WEIGHT_COLUMNS = ["modelo", "variable", "segmento", "valor"]
# Columnas de salida del preprocesador y su nombre limpio esperado
CLEAN_NAMES = {
    **{f"num__{feature}": feature for feature in NUMERIC_FEATURES},
    "cat__barrio_PALERMO": "barrio=PALERMO",
    "cat__comuna_14.0": "comuna=14",
}


def _step(results, name, step):
    """Devolver un paso (``prep`` o ``clf``) del Pipeline ajustado de un modelo."""
    return results[name]["estimator"].named_steps[step]


def _raw_names(results, name):
    """Devolver los nombres de salida del preprocesador, sin limpiar."""
    return list(_step(results, name, "prep").get_feature_names_out())


def _rows(weights, model):
    """Filtrar los pesos de un modelo."""
    return weights[weights["modelo"] == model]


def _value(weights, model, segment, variable):
    """Devolver el único peso de un modelo, segmento y variable."""
    rows = _rows(weights, model)
    mask = (rows["segmento"] == segment) & (rows["variable"] == variable)
    return rows.loc[mask, "valor"].item()


@pytest.fixture(scope="module")
def weights(flow):
    """Calcular una sola vez los pesos de los modelos del flujo."""
    results, _, _, _ = flow
    return model_weights(results)


def test_model_weights_columns_and_index(weights):
    """Verificar columnas en orden, ``RangeIndex`` y valores de punto flotante."""
    assert list(weights.columns) == WEIGHT_COLUMNS
    assert isinstance(weights.index, pd.RangeIndex)
    assert pd.api.types.is_float_dtype(weights["valor"])
    assert not weights.empty


def test_model_weights_skips_baseline(flow, weights):
    """Verificar que el baseline (sin pesos) no aparece y sí los modelos reales."""
    results, _, _, _ = flow
    assert set(weights["modelo"]) == set(results) - {BASELINE_NAME}


def test_model_weights_forest_has_one_importance_per_feature(flow, weights):
    """Verificar una importancia por feature, sin segmento, que suman 1."""
    results, _, _, _ = flow
    forest = _rows(weights, FOREST)
    assert set(forest["segmento"]) == {ALL_SEGMENTS}
    assert len(forest) == len(_raw_names(results, FOREST))
    assert forest["variable"].is_unique
    assert forest["valor"].sum() == pytest.approx(1.0)


def test_model_weights_logistic_has_one_row_per_class_and_feature(flow, weights):
    """Verificar un coeficiente por clase y feature, con los tres segmentos."""
    results, _, _, _ = flow
    clf = _step(results, LOGISTIC, "clf")
    logistic = _rows(weights, LOGISTIC)
    assert len(logistic) == len(clf.classes_) * len(_raw_names(results, LOGISTIC))
    assert set(logistic["segmento"]) == set(SEGMENT_LABELS)
    assert not logistic.duplicated(["segmento", "variable"]).any()


@pytest.mark.parametrize("segment", SEGMENT_LABELS)
def test_model_weights_logistic_values_follow_classes(flow, weights, segment):
    """Verificar que cada coeficiente sale de la fila de su clase en ``classes_``.

    ``classes_`` está en orden alfabético, distinto de ``SEGMENT_LABELS``.
    """
    results, _, _, _ = flow
    clf = _step(results, LOGISTIC, "clf")
    row = list(clf.classes_).index(segment)
    raw = _raw_names(results, LOGISTIC)
    for raw_name, variable in CLEAN_NAMES.items():
        expected = clf.coef_[row, raw.index(raw_name)]
        assert _value(weights, LOGISTIC, segment, variable) == pytest.approx(expected)


def test_model_weights_forest_values_match_importances(flow, weights):
    """Verificar que cada importancia es la de su columna en el bosque."""
    results, _, _, _ = flow
    importances = _step(results, FOREST, "clf").feature_importances_
    raw = _raw_names(results, FOREST)
    for raw_name, variable in CLEAN_NAMES.items():
        expected = importances[raw.index(raw_name)]
        got = _value(weights, FOREST, ALL_SEGMENTS, variable)
        assert got == pytest.approx(expected)


def test_model_weights_cleans_feature_names(weights):
    """Verificar nombres sin prefijos y one-hot como ``col=valor`` sin ``.0``."""
    names = set(weights["variable"])
    assert not any(name.startswith(("num__", "cat__")) for name in names)
    assert set(NUMERIC_FEATURES) <= names
    comunas = {name for name in names if name.startswith("comuna=")}
    assert comunas
    assert not any(name.endswith(".0") for name in comunas)
    assert any(name.startswith("barrio=") for name in names)


def test_model_weights_is_sorted_and_deterministic(flow, weights):
    """Verificar el orden por modelo, segmento y variable, sin importar la entrada."""
    results, _, _, _ = flow
    keys = ["modelo", "segmento", "variable"]
    expected = weights.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(weights, expected)
    reversed_results = dict(reversed(list(results.items())))
    pd.testing.assert_frame_equal(model_weights(reversed_results), weights)


def test_model_weights_rejects_binary_coefficients():
    """Verificar ``ValueError`` si ``coef_`` no tiene una fila por clase (binario)."""
    x = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 0.0, 1.0, 0.0]})
    y = pd.Series([ACC, ACC, HIGH, HIGH])
    steps = [("prep", StandardScaler()), ("clf", LogisticRegression())]
    pipeline = Pipeline(steps).fit(x, y)
    with pytest.raises(ValueError, match="una fila de coeficientes por clase"):
        model_weights({LOGISTIC: {"estimator": pipeline}})
