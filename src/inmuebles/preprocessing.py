"""Preprocesamiento de features: selección de columnas y ``ColumnTransformer``.

``select_features`` deja solo las columnas de ``FEATURE_COLUMNS`` (sin las de
precio, que filtrarían el target). ``build_preprocessor`` arma el
transformador, sin ajustar, que luego se integra al ``Pipeline`` del modelo.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from inmuebles.config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from inmuebles.data_loading import validate_columns


def select_features(df):
    """Seleccionar las columnas de features, en el orden de ``FEATURE_COLUMNS``.

    Parameters
    ----------
    df : pandas.DataFrame
        Tabla limpia, con la forma de la salida de ``clean``.

    Returns
    -------
    pandas.DataFrame
        Copia nueva con solo ``FEATURE_COLUMNS``; la entrada no se modifica.

    Raises
    ------
    ValueError
        Si falta alguna feature; el mensaje enumera las faltantes.
    """
    validate_columns(df, FEATURE_COLUMNS)
    return df.loc[:, list(FEATURE_COLUMNS)].copy()


def build_preprocessor():
    """Construir el ``ColumnTransformer`` de features, todavía sin ajustar.

    Se devuelve sin ajustar porque se ajusta más adelante dentro del
    ``Pipeline`` del modelo y solo con los datos de train: así las medianas,
    modas, medias, desvíos y categorías no usan información de test (sin
    filtración de datos).

    - Numéricas: ``SimpleImputer`` con la mediana (robusta a valores
      extremos) completa los NaN y ``StandardScaler`` lleva cada columna a
      media 0 y desvío 1, para que la escala no domine a modelos lineales.
    - Categóricas: ``SimpleImputer`` con la moda completa los NaN y
      ``OneHotEncoder`` crea una columna binaria por categoría. Con
      ``handle_unknown="ignore"`` una categoría no vista en train (p. ej. un
      barrio nuevo) se codifica como ceros en lugar de fallar.
    - El resto de las columnas se descarta (``remainder="drop"``).

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Transformador sin ajustar cuya salida es una matriz densa.
    """
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, list(NUMERIC_FEATURES)),
            ("cat", categorical, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )
