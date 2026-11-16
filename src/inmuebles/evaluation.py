"""Evaluación final: métricas, matriz de confusión y tabla larga de resultados.

Cada modelo ajustado por ``train_models`` se evalúa una sola vez sobre test
con accuracy y precisión, recall y F1 con promedio ``METRIC_AVERAGE``. La
tabla larga combina esas métricas con las de la validación cruzada de train,
para ver si la CV anticipa el desempeño en test. ``model_weights`` extrae los pesos
de cada modelo (importancias o coeficientes) con nombres de variable legibles.
"""

import logging
import math

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from inmuebles.config import (
    ALL_SEGMENTS,
    CATEGORICAL_FEATURES,
    CONFUSION_AXES,
    CV_ORIGIN,
    IMPORTANCE_COLUMNS,
    METRIC_AVERAGE,
    METRICS_TABLE_COLUMNS,
    SCORING,
    SEGMENT_LABELS,
    TEST_ORIGIN,
)

logger = logging.getLogger(__name__)

# Métricas por clase con el mismo promedio, etiquetas y manejo de 0/0
_AVERAGED_SCORES = {
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
}
# Columnas del reporte por clase
_REPORT_COLUMNS = ["precision", "recall", "f1", "support"]
# Pasos del Pipeline del modelo: preprocesador y clasificador
_PREP_STEP = "prep"
_CLF_STEP = "clf"
# Separador entre el transformador y la columna en get_feature_names_out
_TRANSFORMER_SEPARATOR = "__"
# Columnas de la tabla de pesos (sin el run_id) y su orden
_WEIGHT_COLUMNS = list(IMPORTANCE_COLUMNS[1:])
_WEIGHT_ORDER = ["modelo", "segmento", "variable"]


def _check_inputs(y_true, y_pred):
    """Validar que ``y_true`` e ``y_pred`` no estén vacíos y tengan igual largo.

    Raises
    ------
    ValueError
        Si los largos difieren o las entradas están vacías.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true e y_pred difieren en largo: {len(y_true)} != {len(y_pred)}"
        )
    if len(y_true) == 0:
        raise ValueError("No hay observaciones para evaluar.")


def compute_metrics(y_true, y_pred):
    """Calcular accuracy y precisión, recall y F1 promediados.

    Parameters
    ----------
    y_true : array-like
        Segmentos reales.
    y_pred : array-like
        Segmentos predichos.

    Returns
    -------
    dict
        ``{métrica: float}`` con las claves de ``SCORING``. Una clase nunca
        predicha aporta 0 (sin advertencias).

    Raises
    ------
    ValueError
        Si los largos difieren o las entradas están vacías.
    """
    _check_inputs(y_true, y_pred)
    metrics = {"accuracy": float(accuracy_score(y_true, y_pred))}
    for name, score in _AVERAGED_SCORES.items():
        metrics[name] = float(
            score(
                y_true,
                y_pred,
                labels=list(SEGMENT_LABELS),
                average=METRIC_AVERAGE,
                zero_division=0,
            )
        )
    return metrics


def confusion(y_true, y_pred):
    """Armar la matriz de confusión con todas las etiquetas de segmento.

    Parameters
    ----------
    y_true : array-like
        Segmentos reales.
    y_pred : array-like
        Segmentos predichos.

    Returns
    -------
    pandas.DataFrame
        Conteos con filas = real y columnas = predicho, en el orden de
        ``SEGMENT_LABELS`` (las clases ausentes aparecen en 0).

    Raises
    ------
    ValueError
        Si los largos difieren o las entradas están vacías.
    """
    _check_inputs(y_true, y_pred)
    labels = list(SEGMENT_LABELS)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    index_name, columns_name = CONFUSION_AXES
    return pd.DataFrame(
        matrix,
        index=pd.Index(labels, name=index_name),
        columns=pd.Index(labels, name=columns_name),
    )


def class_report(y_true, y_pred):
    """Calcular precisión, recall, F1 y soporte de cada segmento.

    Parameters
    ----------
    y_true : array-like
        Segmentos reales.
    y_pred : array-like
        Segmentos predichos.

    Returns
    -------
    pandas.DataFrame
        Una fila por segmento (orden de ``SEGMENT_LABELS``) y columnas
        ``precision``, ``recall``, ``f1`` y ``support`` (entero).

    Raises
    ------
    ValueError
        Si los largos difieren o las entradas están vacías.
    """
    _check_inputs(y_true, y_pred)
    labels = list(SEGMENT_LABELS)
    scores = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    report = pd.DataFrame(dict(zip(_REPORT_COLUMNS, scores)), index=labels)
    report["support"] = report["support"].astype(int)
    return report


def evaluate_models(results, x_test, y_test):
    """Evaluar sobre test cada modelo ajustado, baseline incluido.

    Parameters
    ----------
    results : dict
        Salida de ``train_models``: ``{nombre: {"estimator", ...}}``.
    x_test : pandas.DataFrame
        Features de test.
    y_test : pandas.Series
        Segmentos reales de test.

    Returns
    -------
    dict
        ``{nombre: {"predictions", "metrics", "confusion", "report"}}``;
        ``predictions`` es una Series con el índice de ``x_test``.
    """
    evaluations = {}
    for name, result in results.items():
        predictions = pd.Series(
            result["estimator"].predict(x_test), index=x_test.index, name=name
        )
        evaluations[name] = {
            "predictions": predictions,
            "metrics": compute_metrics(y_test, predictions),
            "confusion": confusion(y_test, predictions),
            "report": class_report(y_test, predictions),
        }
        logger.info("%s: f1 test=%.3f", name, evaluations[name]["metrics"]["f1"])
    return evaluations


def metrics_table(results, evaluations):
    """Armar la tabla larga de métricas de validación cruzada y de test.

    Parameters
    ----------
    results : dict
        Salida de ``train_models`` (media y desvío de la CV).
    evaluations : dict
        Salida de ``evaluate_models`` (métricas de test).

    Returns
    -------
    pandas.DataFrame
        Columnas ``METRICS_TABLE_COLUMNS``, ordenada por modelo, origen y
        métrica (en el orden de ``SCORING``); ``desvio`` es NaN en test.
    """
    rows = []
    for name in sorted(results):
        for metric in SCORING:
            stats = results[name]["cv"][metric]
            rows.append((name, metric, CV_ORIGIN, stats["mean"], stats["std"]))
        for metric in SCORING:
            value = evaluations[name]["metrics"][metric]
            rows.append((name, metric, TEST_ORIGIN, value, math.nan))
    return pd.DataFrame(rows, columns=list(METRICS_TABLE_COLUMNS))


def _category_text(value):
    """Escribir una categoría; un número entero va sin decimales (``14.0``)."""
    try:
        number = float(value)
    except ValueError:
        return value
    return str(int(number)) if number.is_integer() else value


def _clean_name(raw):
    """Limpiar un nombre de ``get_feature_names_out`` (``cat__comuna_14.0``).

    Se quita el prefijo del transformador y una columna one-hot de la feature
    categórica ``col`` pasa a ``col=valor`` (p. ej. ``comuna=14``).
    """
    name = raw.split(_TRANSFORMER_SEPARATOR, 1)[-1]
    for column in CATEGORICAL_FEATURES:
        prefix = f"{column}_"
        if name.startswith(prefix):
            return f"{column}={_category_text(name.removeprefix(prefix))}"
    return name


def _coef_pairs(clf):
    """Emparejar cada clase de ``classes_`` con su fila de ``coef_``.

    Raises
    ------
    ValueError
        Si ``coef_`` no tiene una fila por clase (p. ej. un problema binario,
        donde hay una sola fila para dos clases).
    """
    if clf.coef_.shape[0] != len(clf.classes_):
        raise ValueError(
            f"coef_ tiene {clf.coef_.shape[0]} filas para {len(clf.classes_)} "
            "clases: se espera una fila de coeficientes por clase."
        )
    return [(str(label), row) for label, row in zip(clf.classes_, clf.coef_)]


def _weight_rows(name, pipeline):
    """Armar las filas de pesos de un modelo; vacías si no tiene pesos."""
    clf = pipeline.named_steps[_CLF_STEP]
    if not hasattr(clf, "feature_importances_") and not hasattr(clf, "coef_"):
        return []
    raw = pipeline.named_steps[_PREP_STEP].get_feature_names_out()
    variables = [_clean_name(column) for column in raw]
    if hasattr(clf, "feature_importances_"):
        pairs = [(ALL_SEGMENTS, clf.feature_importances_)]
    else:
        pairs = _coef_pairs(clf)
    return [
        (name, variable, segment, float(value))
        for segment, values in pairs
        for variable, value in zip(variables, values)
    ]


def model_weights(results):
    """Armar la tabla de pesos (importancias o coeficientes) de cada modelo.

    Parameters
    ----------
    results : dict
        Salida de ``train_models``: ``{nombre: {"estimator", ...}}``.

    Returns
    -------
    pandas.DataFrame
        Columnas ``modelo``, ``variable``, ``segmento`` y ``valor``, ordenada
        por modelo, segmento y variable, con ``RangeIndex``. Un bosque aporta
        una importancia por variable con segmento ``ALL_SEGMENTS``; un modelo
        lineal, un coeficiente por clase (de ``classes_``) y variable. Los
        modelos sin pesos (el baseline) no aparecen.
    """
    rows = [
        row
        for name, result in results.items()
        for row in _weight_rows(name, result["estimator"])
    ]
    weights = pd.DataFrame(rows, columns=_WEIGHT_COLUMNS)
    weights["valor"] = weights["valor"].astype(float)
    return weights.sort_values(_WEIGHT_ORDER, ignore_index=True)
