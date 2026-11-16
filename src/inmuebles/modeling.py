"""Modelado: partición, pipelines, búsqueda de hiperparámetros y guardado.

La tabla limpia se parte en train/test estratificando por deciles del precio
por m² (el target solo existe después de la partición). Sobre train se
compara un baseline (``DummyClassifier``) con los modelos de ``MODEL_GRIDS``,
ajustados con ``GridSearchCV`` y validación cruzada estratificada. El mejor
modelo se elige por la media de ``REFIT_METRIC`` en la validación cruzada;
el conjunto de test no se usa en este módulo.
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from inmuebles.config import (
    BASELINE_NAME,
    CV_FOLDS,
    LOGREG_MAX_ITER,
    MODEL_COMPRESSION,
    MODEL_GRIDS,
    MODEL_PATH,
    RANDOM_STATE,
    REFIT_METRIC,
    SCORING,
    STRATIFY_BINS,
    TARGET_SOURCE,
    TEST_SIZE,
)
from inmuebles.preprocessing import build_preprocessor, select_features

logger = logging.getLogger(__name__)

# Fábricas de clasificadores sin ajustar, por nombre de modelo
_ESTIMATORS = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=LOGREG_MAX_ITER, random_state=RANDOM_STATE
    ),
    "random_forest": lambda: RandomForestClassifier(random_state=RANDOM_STATE),
    BASELINE_NAME: lambda: DummyClassifier(strategy="most_frequent"),
}


def split_data(df):
    """Partir la tabla limpia en train y test, estratificando por precio.

    Como el segmento (target) se define recién después de la partición, se
    estratifica por los deciles (``STRATIFY_BINS``) del precio por m²: así
    train y test tienen la misma distribución de precios.

    Parameters
    ----------
    df : pandas.DataFrame
        Tabla limpia, con la columna ``TARGET_SOURCE``.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(train, test)`` con el índice y los valores originales.
    """
    bins = pd.qcut(df[TARGET_SOURCE], STRATIFY_BINS, labels=False, duplicates="drop")
    train, test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=bins
    )
    logger.info("Partición: %d filas de train y %d de test", len(train), len(test))
    return train, test


def make_xy(df, labeler):
    """Separar las features y etiquetar el target con un etiquetador ajustado.

    Parameters
    ----------
    df : pandas.DataFrame
        Tabla limpia (train o test).
    labeler : inmuebles.target.ValueSegmentLabeler
        Etiquetador ya ajustado sobre los precios de train.

    Returns
    -------
    tuple
        ``(x, y)``: DataFrame de ``FEATURE_COLUMNS`` y Series de segmentos,
        con el mismo índice que ``df``.

    Raises
    ------
    ValueError
        Si ``x`` e ``y`` no tienen la misma cantidad de filas.
    """
    x = select_features(df)
    y = labeler.transform(df[TARGET_SOURCE])
    if len(x) != len(y):
        raise ValueError(f"X e y difieren en filas: {len(x)} != {len(y)}")
    return x, y


def build_pipeline(name):
    """Construir el ``Pipeline`` sin ajustar: preprocesamiento y clasificador.

    Parameters
    ----------
    name : str
        Nombre del modelo: una clave de ``MODEL_GRIDS`` o ``BASELINE_NAME``.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pasos ``prep`` (``ColumnTransformer``) y ``clf`` (clasificador).

    Raises
    ------
    ValueError
        Si el nombre no corresponde a ningún modelo conocido.
    """
    if name not in _ESTIMATORS:
        raise ValueError(
            f"Modelo desconocido: {name!r}. Válidos: {sorted(_ESTIMATORS)}"
        )
    return Pipeline([("prep", build_preprocessor()), ("clf", _ESTIMATORS[name]())])


def _make_cv():
    """Crear la validación cruzada estratificada, con barajado reproducible."""
    return StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def _summary(means, stds):
    """Armar ``{métrica: {"mean", "std"}}`` a partir de medias y desvíos."""
    return {m: {"mean": float(means[m]), "std": float(stds[m])} for m in SCORING}


def tune_model(name, x, y, grid=None, n_jobs=-1):
    """Buscar los mejores hiperparámetros de un modelo con ``GridSearchCV``.

    Parameters
    ----------
    name : str
        Nombre del modelo (clave de ``MODEL_GRIDS``).
    x : pandas.DataFrame
        Features de train.
    y : pandas.Series
        Segmentos de train.
    grid : dict, optional
        Grilla de parámetros; por defecto ``MODEL_GRIDS[name]``.
    n_jobs : int, default -1
        Procesos en paralelo (-1 usa todos los núcleos).

    Returns
    -------
    sklearn.model_selection.GridSearchCV
        Búsqueda ajustada, reajustada con ``REFIT_METRIC`` sobre todo train.
    """
    search = GridSearchCV(
        build_pipeline(name),
        MODEL_GRIDS[name] if grid is None else grid,
        scoring=SCORING,
        refit=REFIT_METRIC,
        cv=_make_cv(),
        n_jobs=n_jobs,
    )
    return search.fit(x, y)


def cv_summary(cv_results, index):
    """Resumir media y desvío entre folds de cada métrica para un candidato.

    Parameters
    ----------
    cv_results : dict
        Atributo ``cv_results_`` de una búsqueda ajustada.
    index : int
        Posición del candidato (p. ej. ``best_index_``).

    Returns
    -------
    dict
        ``{métrica: {"mean": float, "std": float}}`` para cada métrica de
        ``SCORING``.
    """
    means = {m: cv_results[f"mean_test_{m}"][index] for m in SCORING}
    stds = {m: cv_results[f"std_test_{m}"][index] for m in SCORING}
    return _summary(means, stds)


def evaluate_baseline(x, y):
    """Evaluar el baseline con validación cruzada y ajustarlo sobre train.

    Parameters
    ----------
    x : pandas.DataFrame
        Features de train.
    y : pandas.Series
        Segmentos de train.

    Returns
    -------
    dict
        ``{"estimator", "params", "cv"}``, con la misma forma que cada modelo
        de ``train_models``; ``params`` es vacío.
    """
    pipeline = build_pipeline(BASELINE_NAME)
    scores = cross_validate(pipeline, x, y, scoring=SCORING, cv=_make_cv())
    means = {m: scores[f"test_{m}"].mean() for m in SCORING}
    stds = {m: scores[f"test_{m}"].std() for m in SCORING}
    pipeline.fit(x, y)
    return {"estimator": pipeline, "params": {}, "cv": _summary(means, stds)}


def _log_result(name, result):
    """Registrar en INFO los parámetros y el ``REFIT_METRIC`` medio ± desvío."""
    stats = result["cv"][REFIT_METRIC]
    logger.info(
        "%s: params=%s, %s CV=%.3f ± %.3f",
        name,
        result["params"],
        REFIT_METRIC,
        stats["mean"],
        stats["std"],
    )


def train_models(x, y, grids=None, n_jobs=-1):
    """Entrenar el baseline y ajustar cada modelo de la grilla.

    Parameters
    ----------
    x : pandas.DataFrame
        Features de train.
    y : pandas.Series
        Segmentos de train.
    grids : dict, optional
        ``{nombre: grilla}``; por defecto ``MODEL_GRIDS``.
    n_jobs : int, default -1
        Procesos en paralelo para ``GridSearchCV``.

    Returns
    -------
    dict
        ``{nombre: {"estimator", "params", "cv"}}``, con el baseline y cada
        modelo; ``estimator`` es el mejor ``Pipeline`` reajustado en train.
    """
    results = {BASELINE_NAME: evaluate_baseline(x, y)}
    for name, grid in (MODEL_GRIDS if grids is None else grids).items():
        search = tune_model(name, x, y, grid=grid, n_jobs=n_jobs)
        results[name] = {
            "estimator": search.best_estimator_,
            "params": search.best_params_,
            "cv": cv_summary(search.cv_results_, search.best_index_),
        }
    for name, result in results.items():
        _log_result(name, result)
    return results


def select_best(results):
    """Elegir el modelo real con mayor media de ``REFIT_METRIC`` en la CV.

    Parameters
    ----------
    results : dict
        Salida de ``train_models``.

    Returns
    -------
    str
        Nombre del mejor modelo; el baseline nunca se elige.

    Raises
    ------
    ValueError
        Si no hay ningún modelo además del baseline.
    """
    candidates = {
        name: result["cv"][REFIT_METRIC]["mean"]
        for name, result in results.items()
        if name != BASELINE_NAME
    }
    if not candidates:
        raise ValueError("No hay modelos para elegir además del baseline.")
    best = max(candidates, key=candidates.get)
    logger.info("Mejor modelo: %s (%s CV=%.3f)", best, REFIT_METRIC, candidates[best])
    return best


def save_model(estimator, path=MODEL_PATH):
    """Guardar un estimador ajustado con joblib comprimido, creando las carpetas.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        Modelo (``Pipeline``) ajustado.
    path : str or pathlib.Path, default MODEL_PATH
        Ruta del archivo ``.joblib``.

    Returns
    -------
    pathlib.Path
        Ruta donde se guardó el modelo.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, path, compress=MODEL_COMPRESSION)
    logger.info("Modelo guardado en %s", path)
    return path
