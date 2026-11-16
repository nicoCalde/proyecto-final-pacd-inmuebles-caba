"""Evaluación final: métricas, matriz de confusión y tabla larga de resultados.

Cada modelo ajustado por ``train_models`` se evalúa una sola vez sobre test
con accuracy y precisión, recall y F1 con promedio ``METRIC_AVERAGE``. La
tabla larga combina esas métricas con las de la validación cruzada de train,
para comparar ambas y detectar sobreajuste.
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
    CONFUSION_AXES,
    CV_ORIGIN,
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
