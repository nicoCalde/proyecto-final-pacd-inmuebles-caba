"""Variable objetivo: segmento de valor según el precio en USD por m².

El precio por m² se discretiza en tres clases ordenadas (``SEGMENT_LABELS``)
usando los terciles (``SEGMENT_QUANTILES``) aprendidos sobre los precios de
entrenamiento.
"""

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from inmuebles.config import SEGMENT_LABELS, SEGMENT_QUANTILES, TARGET_COLUMN


def _as_price_series(prices):
    """Convertir precios a una Series ``float64`` de valores finitos.

    Parameters
    ----------
    prices : array-like 1-D
        Precios en USD por m². Si es una Series se conserva su índice.

    Returns
    -------
    pandas.Series
        Precios como ``float64``; con ``RangeIndex`` si la entrada no era Series.

    Raises
    ------
    ValueError
        Si la entrada no es unidimensional o algún precio es NaN o infinito.
    """
    index = prices.index if isinstance(prices, pd.Series) else None
    values = np.asarray(prices, dtype=float)
    if values.ndim != 1:
        raise ValueError("Los precios deben ser unidimensionales (1-D).")
    if not np.isfinite(values).all():
        raise ValueError(
            "Los precios deben ser valores finitos (sin NaN ni infinitos)."
        )
    return pd.Series(values, index=index)


class ValueSegmentLabeler:
    """Asignar un segmento de valor según los terciles del precio por m².

    Los umbrales se aprenden con ``fit`` solo sobre los precios de
    entrenamiento y luego se aplican sin cambios a cualquier conjunto con
    ``transform``. Así la definición de las clases no depende del conjunto
    de prueba, lo que evita filtrar información de test al modelo.

    No hereda de ``TransformerMixin`` de scikit-learn porque etiqueta la
    variable objetivo (``y``), no las features (``X``). Un ``Pipeline`` solo
    transforma ``X``, así que esta clase se usa fuera de él.

    Attributes
    ----------
    thresholds_ : tuple of float or None
        Cortes entre segmentos, en orden creciente; ``None`` hasta ajustar.
    """

    def __init__(self):
        self.thresholds_ = None

    def fit(self, prices):
        """Aprender los umbrales de segmento a partir de precios de train.

        Parameters
        ----------
        prices : array-like 1-D
            Precios en USD por m² del conjunto de entrenamiento.

        Returns
        -------
        ValueSegmentLabeler
            El mismo objeto, ya ajustado.

        Raises
        ------
        ValueError
            Si no hay precios, si la entrada no es 1-D, si algún precio es NaN
            o infinito, o si los cuantiles no son estrictamente crecientes
            (p. ej. precios constantes).
        """
        values = _as_price_series(prices).to_numpy()
        if values.size == 0:
            raise ValueError("No hay precios para ajustar los segmentos.")
        quantiles = np.quantile(values, SEGMENT_QUANTILES)
        if np.any(np.diff(quantiles) <= 0):
            raise ValueError(
                "Los umbrales de segmento deben ser estrictamente crecientes."
            )
        self.thresholds_ = tuple(float(q) for q in quantiles)
        return self

    def transform(self, prices):
        """Etiquetar precios con el segmento que les corresponde.

        Un precio igual a un umbral se asigna a la clase superior.

        Parameters
        ----------
        prices : array-like 1-D
            Precios en USD por m² a etiquetar.

        Returns
        -------
        pandas.Series
            Etiquetas de ``SEGMENT_LABELS`` con nombre ``TARGET_COLUMN``; se
            conserva el índice si la entrada es una Series.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            Si se llama antes de ``fit``.
        ValueError
            Si la entrada no es 1-D o algún precio es NaN o infinito.
        """
        if self.thresholds_ is None:
            raise NotFittedError("Llamar a fit antes de transform.")
        series = _as_price_series(prices)
        positions = np.searchsorted(self.thresholds_, series.to_numpy(), side="right")
        labels = np.array(SEGMENT_LABELS)[positions]
        return pd.Series(labels, index=series.index, name=TARGET_COLUMN)
