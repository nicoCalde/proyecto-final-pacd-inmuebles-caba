"""Pruebas del modelado: partición, pipelines, búsqueda de grilla y guardado."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from inmuebles.config import (
    BASELINE_NAME,
    CV_FOLDS,
    FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    MODEL_GRIDS,
    RANDOM_STATE,
    REFIT_METRIC,
    SCORING,
    SEGMENT_LABELS,
    STRATIFY_BINS,
    TARGET_SOURCE,
)
from inmuebles.modeling import (
    build_pipeline,
    cv_summary,
    evaluate_baseline,
    make_xy,
    save_model,
    select_best,
    split_data,
    train_models,
    tune_model,
)
from inmuebles.target import ValueSegmentLabeler

# Grillas diminutas para que las pruebas sean rápidas.
TINY_GRIDS = {
    "logistic_regression": {"clf__C": [0.1, 1.0]},
    "random_forest": {"clf__n_estimators": [10], "clf__max_depth": [5]},
}
REAL_MODELS = ("logistic_regression", "random_forest")


@pytest.fixture
def xy_train(learnable_clean):
    """Devolver ``X`` e ``y`` de train con el etiquetador ajustado solo en train."""
    train, _ = split_data(learnable_clean)
    labeler = ValueSegmentLabeler().fit(train[TARGET_SOURCE])
    return make_xy(train, labeler)


def _results(scores):
    """Armar resultados a mano con la media de ``REFIT_METRIC`` de cada modelo."""
    return {
        name: {"estimator": None, "params": {}, "cv": {REFIT_METRIC: {"mean": s}}}
        for name, s in scores.items()
    }


def test_modeling_constants_are_consistent():
    """Verificar métricas de la consigna, refit válido y grillas sobre ``clf``."""
    assert set(SCORING) == {"accuracy", "precision", "recall", "f1"}
    assert REFIT_METRIC in SCORING
    assert BASELINE_NAME not in MODEL_GRIDS
    assert set(MODEL_GRIDS) == set(REAL_MODELS)
    for grid in MODEL_GRIDS.values():
        assert all(param.startswith("clf__") for param in grid)


def test_split_data_sizes_and_disjoint_indices(learnable_clean):
    """Verificar la partición 80/20 con índices disjuntos que cubren todo."""
    train, test = split_data(learnable_clean)
    assert (len(train), len(test)) == (160, 40)
    assert not set(train.index) & set(test.index)
    assert set(train.index) | set(test.index) == set(learnable_clean.index)


def test_split_data_keeps_original_rows(learnable_clean):
    """Verificar que cada fila conserva su índice y sus valores originales."""
    train, test = split_data(learnable_clean)
    for part in (train, test):
        pd.testing.assert_frame_equal(part, learnable_clean.loc[part.index])


def test_split_data_is_deterministic(learnable_clean):
    """Verificar que dos particiones de la misma tabla son idénticas."""
    first_train, first_test = split_data(learnable_clean)
    second_train, second_test = split_data(learnable_clean)
    assert first_train.index.equals(second_train.index)
    assert first_test.index.equals(second_test.index)


def test_split_data_stratifies_by_price_decile(learnable_clean):
    """Verificar que train y test tienen la misma mezcla de deciles de precio."""
    bins = pd.qcut(learnable_clean[TARGET_SOURCE], STRATIFY_BINS, labels=False)
    train, test = split_data(learnable_clean)
    train_share = bins.loc[train.index].value_counts(normalize=True).sort_index()
    test_share = bins.loc[test.index].value_counts(normalize=True).sort_index()
    assert len(test_share) == STRATIFY_BINS
    assert np.abs(train_share - test_share).max() <= 0.03


def test_make_xy_returns_features_and_labels(learnable_clean):
    """Verificar que ``X`` son las features sin precio e ``y`` las etiquetas."""
    labeler = ValueSegmentLabeler().fit(learnable_clean[TARGET_SOURCE])
    x, y = make_xy(learnable_clean, labeler)
    assert tuple(x.columns) == FEATURE_COLUMNS
    assert not set(x.columns) & set(LEAKAGE_COLUMNS)
    assert set(y) <= set(SEGMENT_LABELS)
    assert len(x) == len(y) == len(learnable_clean)
    assert x.index.equals(y.index)


def test_make_xy_labels_test_with_train_thresholds(learnable_clean):
    """Verificar que test se etiqueta con los umbrales de train, sin cambiarlos."""
    train, test = split_data(learnable_clean)
    labeler = ValueSegmentLabeler().fit(train[TARGET_SOURCE])
    thresholds = labeler.thresholds_
    _, y_test = make_xy(test, labeler)
    assert labeler.thresholds_ == thresholds
    low, high = thresholds
    expected = [
        SEGMENT_LABELS[(price >= low) + (price >= high)]
        for price in test[TARGET_SOURCE]
    ]
    assert y_test.tolist() == expected


@pytest.mark.parametrize(
    ("name", "estimator_class"),
    [
        ("logistic_regression", LogisticRegression),
        ("random_forest", RandomForestClassifier),
        (BASELINE_NAME, DummyClassifier),
    ],
)
def test_build_pipeline_returns_unfitted_pipeline(name, estimator_class):
    """Verificar los pasos ``prep``/``clf`` y el estimador según el nombre."""
    pipeline = build_pipeline(name)
    assert isinstance(pipeline, Pipeline)
    assert [step for step, _ in pipeline.steps] == ["prep", "clf"]
    assert isinstance(pipeline.named_steps["prep"], ColumnTransformer)
    assert isinstance(pipeline.named_steps["clf"], estimator_class)
    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline)


def test_build_pipeline_fixes_random_state_and_baseline_strategy():
    """Verificar la semilla de los modelos reales y la estrategia del baseline."""
    for name in REAL_MODELS:
        clf = build_pipeline(name).named_steps["clf"]
        assert clf.random_state == RANDOM_STATE
    assert build_pipeline(BASELINE_NAME).named_steps["clf"].strategy == (
        "most_frequent"
    )


def test_build_pipeline_unknown_name_raises():
    """Verificar que un nombre de modelo desconocido lanza ``ValueError``."""
    with pytest.raises(ValueError):
        build_pipeline("svm")


def test_tune_model_fits_grid_search(xy_train):
    """Verificar la búsqueda: grilla, CV estratificada, métricas y refit."""
    x, y = xy_train
    grid = TINY_GRIDS["logistic_regression"]
    search = tune_model("logistic_regression", x, y, grid=grid, n_jobs=1)
    check_is_fitted(search.best_estimator_)
    assert search.best_params_["clf__C"] in grid["clf__C"]
    assert search.refit == REFIT_METRIC
    assert search.scoring == SCORING
    assert isinstance(search.cv, StratifiedKFold)
    assert search.cv.n_splits == CV_FOLDS
    assert search.cv.shuffle
    assert search.cv.random_state == RANDOM_STATE
    for metric in SCORING:
        assert f"mean_test_{metric}" in search.cv_results_


def test_cv_summary_reads_mean_and_std(xy_train):
    """Verificar que el resumen trae media y desvío de cada métrica."""
    x, y = xy_train
    search = tune_model(
        "logistic_regression", x, y, grid=TINY_GRIDS["logistic_regression"], n_jobs=1
    )
    summary = cv_summary(search.cv_results_, search.best_index_)
    assert set(summary) == set(SCORING)
    for metric, stats in summary.items():
        assert set(stats) == {"mean", "std"}
        assert 0.0 <= stats["mean"] <= 1.0
        assert stats["std"] >= 0.0
        expected = search.cv_results_[f"mean_test_{metric}"][search.best_index_]
        assert stats["mean"] == pytest.approx(expected)


def test_evaluate_baseline_matches_majority_share(learnable_clean):
    """Verificar que la exactitud del dummy es la proporción de la clase mayor."""
    x = learnable_clean.loc[:, list(FEATURE_COLUMNS)]
    low, mid, high = SEGMENT_LABELS
    y = pd.Series([high] * 120 + [mid] * 50 + [low] * 30, index=x.index)
    result = evaluate_baseline(x, y)
    assert set(result) == {"estimator", "params", "cv"}
    assert result["params"] == {}
    assert isinstance(result["estimator"].named_steps["clf"], DummyClassifier)
    check_is_fitted(result["estimator"])
    assert set(result["cv"]) == set(SCORING)
    assert result["cv"]["accuracy"]["mean"] == pytest.approx(0.6, abs=0.02)


def test_train_models_returns_every_model(xy_train):
    """Verificar que hay baseline y modelos, y que estos superan al dummy."""
    x, y = xy_train
    results = train_models(x, y, grids=TINY_GRIDS, n_jobs=1)
    assert set(results) == {BASELINE_NAME, *REAL_MODELS}
    for entry in results.values():
        assert set(entry) == {"estimator", "params", "cv"}
        check_is_fitted(entry["estimator"])
        assert set(entry["cv"]) == set(SCORING)
    dummy_f1 = results[BASELINE_NAME]["cv"][REFIT_METRIC]["mean"]
    for name in REAL_MODELS:
        assert results[name]["cv"][REFIT_METRIC]["mean"] > dummy_f1 + 0.2
        assert set(results[name]["params"]) == set(TINY_GRIDS[name])


def test_train_models_is_reproducible(xy_train):
    """Verificar que dos entrenamientos iguales dan los mismos resultados."""
    x, y = xy_train
    first = train_models(x, y, grids=TINY_GRIDS, n_jobs=1)
    second = train_models(x, y, grids=TINY_GRIDS, n_jobs=1)
    for name in first:
        assert first[name]["params"] == second[name]["params"]
        for metric in SCORING:
            assert first[name]["cv"][metric]["mean"] == pytest.approx(
                second[name]["cv"][metric]["mean"]
            )


def test_select_best_ignores_baseline():
    """Verificar que gana el modelo real con mayor F1 medio, nunca el dummy."""
    results = _results(
        {BASELINE_NAME: 0.9, "logistic_regression": 0.7, "random_forest": 0.8}
    )
    assert select_best(results) == "random_forest"


def test_select_best_without_real_models_raises():
    """Verificar que solo con el baseline se lanza ``ValueError``."""
    with pytest.raises(ValueError):
        select_best(_results({BASELINE_NAME: 0.5}))


def test_save_model_roundtrip(tmp_path, xy_train):
    """Verificar que se crean las carpetas y el modelo cargado predice igual."""
    x, y = xy_train
    pipeline = build_pipeline("logistic_regression").fit(x, y)
    path = tmp_path / "sub" / "m.joblib"
    assert save_model(pipeline, path) == path
    assert path.is_file()
    loaded = joblib.load(path)
    np.testing.assert_array_equal(loaded.predict(x), pipeline.predict(x))
