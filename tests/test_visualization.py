"""Pruebas de los gráficos, que se arman solo con lo que hay en la base.

Las funciones de hallazgo se prueban con datos armados a mano; los gráficos,
con tablas chicas con la forma de las que devuelve ``read_table``. Las pruebas
de punta a punta arman una base temporal con un flujo real diminuto (grillas
mínimas, ``n_jobs=1``). Nunca se toca ``data/`` ni ``reports/``.
"""

import dataclasses
from contextlib import closing

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgb
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image

from inmuebles.config import (
    ALL_SEGMENTS,
    BASELINE_NAME,
    CATEGORICAL_FEATURES,
    CHART_LIMITATIONS,
    COEF_CONTRAST,
    CONFUSION_AXES,
    CV_ORIGIN,
    CV_TEST_TOLERANCE,
    DATA_SOURCE,
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    FIGURE_DPI,
    FIGURE_FORMAT,
    ID_COLUMN,
    IMPORTANCE_COLUMNS,
    IMPORTANCE_TABLE,
    KM_PER_DEGREE_LAT,
    LOCATION_FEATURES,
    METRICS_DB_COLUMNS,
    MIN_BARRIO_LISTINGS,
    MODEL_COLORS,
    MODEL_GRIDS,
    MODEL_LABELS,
    NEGATIVE_COLOR,
    NEUTRAL_COLOR,
    NUMERIC_FEATURES,
    POSITIVE_COLOR,
    PRICE_BINS,
    REFIT_METRIC,
    RUN_COLUMNS,
    RUN_ID_COLUMN,
    SCORING,
    SEGMENT_COLORS,
    SEGMENT_LABELS,
    TARGET_SOURCE,
    TEST_ORIGIN,
    TOP_BARRIOS,
)
from inmuebles.database import (
    RunArtifacts,
    build_run_tables,
    connect,
    create_schema,
    write_clean,
    write_run,
)
from inmuebles.evaluation import evaluate_models
from inmuebles.modeling import make_xy, select_best, split_data, train_models
from inmuebles.target import ValueSegmentLabeler
from inmuebles.visualization import (
    ChartData,
    barrio_medians,
    build_figures,
    cv_test_gap,
    feature_label,
    format_number,
    importance_by_variable,
    load_chart_data,
    location_share,
    neighbor_error_share,
    north_shift_km,
    numeric_contrast,
    plot_coefficients,
    plot_confusion,
    plot_cv_vs_test,
    plot_importances,
    plot_location_map,
    plot_metrics,
    plot_price_by_barrio,
    plot_price_distribution,
    row_recall,
    save_figures,
    metrics_on_test,
    segment_of,
)

ACC, STD, HIGH = SEGMENT_LABELS
FOREST, LOGISTIC = "random_forest", "logistic_regression"
REAL_MODELS = tuple(MODEL_GRIDS)
RUN_ID = "20260101T000000Z"
LOW, TOP = 2000.0, 3000.0

# Gráficos esperados: uno por limitación y una matriz por cada modelo real
CONFUSION_CHART = "matriz_confusion"
SINGLE_CHARTS = [key for key in CHART_LIMITATIONS if key != CONFUSION_CHART]
EXPECTED_FIGURES = {
    *SINGLE_CHARTS,
    *(f"{CONFUSION_CHART}_{model}" for model in REAL_MODELS),
}

# Gráficos de barras y eje de valores (que empieza en 0)
VALUE_AXIS = {
    "distribucion_precio_m2": "y",
    "precio_m2_por_barrio": "x",
    "metricas_modelos": "y",
    "cv_vs_test": "y",
    "importancia_random_forest": "x",
}
# Gráficos de barras con una etiqueta de valor por barra
LABELED_BARS = (
    "precio_m2_por_barrio",
    "metricas_modelos",
    "cv_vs_test",
    "importancia_random_forest",
    "coeficientes_regresion_logistica",
)

# Matriz a mano (filas = real, columnas = predicho). Fuera de la diagonal hay
# 26 errores, de los cuales 8 + 5 + 6 + 4 = 23 son entre segmentos vecinos.
# Acierto por fila: 50/60, 40/51 y 45/50; el más difícil es Estándar.
MATRIX_VALUES = [[50, 8, 2], [5, 40, 6], [1, 4, 45]]
NEIGHBOR_SHARE = 23 / 26
ROW_RECALL = [50 / 60, 40 / 51, 45 / 50]

# F1 de test de cada modelo en la tabla de métricas armada a mano
TEST_F1 = {BASELINE_NAME: 0.17, LOGISTIC: 0.61, FOREST: 0.74}
DEFAULT_GAP = CV_TEST_TOLERANCE / 3
CV_STD = 0.02

# Importancias a mano del bosque: suman 1. Por variable original: barrio
# 0,10 + 0,08 + 0,07 = 0,25 y comuna 0,02 + 0,01 = 0,03; la ubicación
# (latitud, longitud, barrio y comuna) suma 0,20 + 0,15 + 0,25 + 0,03 = 0,63.
RF_IMPORTANCES = {
    "antig": 0.24,
    "m2total": 0.06,
    "m2cub": 0.04,
    "ambientes": 0.02,
    "banos": 0.01,
    "latitud": 0.20,
    "longitud": 0.15,
    "barrio=PALERMO": 0.10,
    "barrio=BOCA": 0.08,
    "barrio=FLORES": 0.07,
    "comuna=1": 0.02,
    "comuna=14": 0.01,
}
IMPORTANCE_BY_VARIABLE = {
    "barrio": 0.25,
    "antig": 0.24,
    "latitud": 0.20,
    "longitud": 0.15,
    "m2total": 0.06,
    "m2cub": 0.04,
    "comuna": 0.03,
    "ambientes": 0.02,
    "banos": 0.01,
}
LOCATION_SHARE = 0.63
# Contraste a mano Alta - Accesible de cada numérica y de columnas one-hot
CONTRAST = {
    "antig": -1.2,
    "m2total": 0.7,
    "m2cub": -0.3,
    "ambientes": 0.1,
    "banos": 0.4,
    "latitud": 0.9,
    "longitud": -0.05,
}
ONE_HOT_CONTRAST = {"barrio=PALERMO": 5.0, "comuna=14": -4.0}

# Grillas diminutas para que el flujo completo sea rápido.
TINY_GRIDS = {
    "logistic_regression": {"clf__C": [1.0]},
    "random_forest": {"clf__n_estimators": [10], "clf__max_depth": [5]},
}


# --- Ayudantes -------------------------------------------------------------


def _all_text(fig):
    """Juntar el título general, los títulos de los ejes y los textos de la figura."""
    texts = [fig._suptitle.get_text()] if fig._suptitle is not None else []
    for ax in fig.axes:
        texts.extend(ax.get_title(loc) for loc in ("left", "center", "right"))
    texts.extend(text.get_text() for text in fig.texts)
    return " ".join(texts)


def _footnote(fig):
    """Juntar los textos sueltos de la figura (subtítulo y nota al pie)."""
    return " ".join(text.get_text() for text in fig.texts)


def _bars(ax):
    """Devolver las barras (rectángulos) dibujadas en un eje."""
    return [patch for patch in ax.patches if isinstance(patch, Rectangle)]


def _value_labels(ax):
    """Devolver los textos no vacíos de un eje (etiquetas de valor)."""
    return [text.get_text() for text in ax.texts if text.get_text().strip()]


def _rgb(artist_color):
    """Quitar la transparencia de un color RGBA."""
    return tuple(artist_color[:3])


def _same_color(artist_color, configured):
    """Comparar un color dibujado con uno configurado, ignorando el alfa."""
    return _rgb(artist_color) == pytest.approx(to_rgb(configured))


def _vertical_lines(ax):
    """Devolver ``{x: estilo}`` de las líneas verticales de un eje."""
    lines = {}
    for line in ax.lines:
        xs = np.unique(np.asarray(line.get_xdata(), dtype=float))
        if xs.size == 1:
            lines[float(xs[0])] = line.get_linestyle()
    return lines


def _horizontal_lines(ax):
    """Devolver ``{y: estilo}`` de las líneas horizontales de un eje."""
    lines = {}
    for line in ax.lines:
        ys = np.unique(np.asarray(line.get_ydata(), dtype=float))
        xs = np.unique(np.asarray(line.get_xdata(), dtype=float))
        if ys.size == 1 and xs.size > 1:
            lines[float(ys[0])] = line.get_linestyle()
    return lines


def _top_bar(ax):
    """Devolver la barra horizontal que se ve más arriba."""
    bars = _bars(ax)
    if ax.yaxis_inverted():
        return min(bars, key=lambda bar: bar.get_y())
    return max(bars, key=lambda bar: bar.get_y())


def _labeler_segments(values):
    """Etiquetar con la regla de ``ValueSegmentLabeler`` y los umbrales fijos."""
    labeler = ValueSegmentLabeler()
    labeler.thresholds_ = (LOW, TOP)
    return labeler.transform(np.asarray(values, dtype=float)).tolist()


def _run(chosen=FOREST):
    """Armar una fila de ``ejecuciones`` como la devuelve ``read_table``."""
    values = dict.fromkeys(RUN_COLUMNS)
    values.update(
        {
            RUN_ID_COLUMN: RUN_ID,
            "modelo_elegido": chosen,
            "umbral_bajo": LOW,
            "umbral_alto": TOP,
        }
    )
    return pd.Series(values)


def _barrio_rows(name, base, latitud, count):
    """Armar ``count`` avisos de un barrio con mediana ``base`` del precio."""
    offsets = np.linspace(-50.0, 50.0, count)
    return [
        {"barrio": name, TARGET_SOURCE: base + offset, "latitud": latitud}
        for offset in offsets
    ]


def _matrix(values):
    """Armar una matriz con la forma de ``read_confusion``."""
    real, predicted = CONFUSION_AXES
    labels = list(SEGMENT_LABELS)
    return pd.DataFrame(
        values,
        index=pd.Index(labels, name=real),
        columns=pd.Index(labels, name=predicted),
    )


def _test_value(model, metric):
    """Devolver el valor de test de una métrica en la tabla armada a mano."""
    index = list(SCORING).index(metric)
    offset = 0.0 if metric == REFIT_METRIC else 0.01 * (index + 1)
    return TEST_F1[model] + offset


def _metrics(gaps=None):
    """Armar la tabla larga de métricas como la devuelve ``read_table``.

    La media de CV es la de test menos el ``gap`` de cada modelo, así que la
    diferencia test - CV es exactamente ``gap``.
    """
    gaps = {**dict.fromkeys(TEST_F1, DEFAULT_GAP), **(gaps or {})}
    rows = []
    for model in TEST_F1:
        for metric in SCORING:
            test = _test_value(model, metric)
            rows.append((RUN_ID, model, metric, CV_ORIGIN, test - gaps[model], CV_STD))
            rows.append((RUN_ID, model, metric, TEST_ORIGIN, test, np.nan))
    df = pd.DataFrame(rows, columns=list(METRICS_DB_COLUMNS))
    return df.sort_values(["modelo", "metrica", "origen"], ignore_index=True)


def _weights(coef_sign=1.0):
    """Armar la tabla de pesos como la devuelve ``read_table``.

    Las importancias son ``RF_IMPORTANCES``. Los coeficientes de Alta y de
    Accesible son ``+c/2`` y ``-c/2`` (con ``c = coef_sign * CONTRAST``) y
    los de Estándar cierran la suma 0 entre clases, como en la regresión
    multinomial; las columnas one-hot tienen coeficientes grandes que el
    gráfico de contraste debe ignorar.
    """
    rows = [
        (RUN_ID, FOREST, var, ALL_SEGMENTS, value)
        for var, value in RF_IMPORTANCES.items()
    ]
    high, low = COEF_CONTRAST
    contrast = {var: coef_sign * value for var, value in CONTRAST.items()}
    coefs = {**contrast, **ONE_HOT_CONTRAST}
    for var, value in coefs.items():
        segment_values = {high: value / 2, low: -value / 2, STD: 0.0}
        rows.extend(
            (RUN_ID, LOGISTIC, var, segment, coef)
            for segment, coef in segment_values.items()
        )
    df = pd.DataFrame(rows, columns=list(IMPORTANCE_COLUMNS))
    keys = [RUN_ID_COLUMN, "modelo", "segmento", "variable"]
    return df.sort_values(keys, ignore_index=True)


def _top_contrast(coef_sign=1.0):
    """Devolver ``(variable, valor)`` del mayor contraste numérico en módulo."""
    variable = max(CONTRAST, key=lambda var: abs(CONTRAST[var]))
    return variable, coef_sign * CONTRAST[variable]


# --- Fixtures de datos a mano ----------------------------------------------


@pytest.fixture(scope="module")
def barrio_clean(make_clean):
    """Armar una tabla limpia con barrios de precio y latitud crecientes.

    Hay ``2 * TOP_BARRIOS + 2`` barrios con suficientes avisos (los pares con
    uno extra), cuya mediana sube de a 200 USD/m² y que quedan más al norte
    cuanto más caros; y un barrio carísimo con un aviso menos del mínimo.
    """
    rows = []
    for i in range(2 * TOP_BARRIOS + 2):
        count = MIN_BARRIO_LISTINGS + (i + 1) % 2
        base, latitud = 1000.0 + 200.0 * i, -34.70 + 0.01 * i
        rows.extend(_barrio_rows(f"BARRIO {i:02d}", base, latitud, count))
    rows.extend(_barrio_rows("POCOS AVISOS", 9000.0, -34.53, MIN_BARRIO_LISTINGS - 1))
    return make_clean(rows)


def _shift_rows(alta_lat, accesible_lat):
    """Armar avisos Alta, Estándar y Accesible con latitudes dadas."""
    alta = [{TARGET_SOURCE: 3500.0, "latitud": alta_lat} for _ in range(3)]
    middle = [{TARGET_SOURCE: 2500.0, "latitud": -34.90}]
    accesible = [{TARGET_SOURCE: 1500.0, "latitud": accesible_lat} for _ in range(3)]
    missing = [{TARGET_SOURCE: 3500.0, "latitud": np.nan, "longitud": np.nan}]
    return alta + middle + accesible + missing


@pytest.fixture(scope="module")
def north_clean(make_clean):
    """Armar avisos Alta 0,1° al norte de los Accesibles (y uno sin ubicación)."""
    return make_clean(_shift_rows(-34.55, -34.65))


@pytest.fixture(scope="module")
def south_clean(make_clean):
    """Armar avisos Alta 0,1° al sur de los Accesibles (y uno sin ubicación)."""
    return make_clean(_shift_rows(-34.65, -34.55))


@pytest.fixture(scope="module")
def figures(barrio_clean):
    """Armar una vez cada gráfico con los datos a mano; clave = nombre."""
    run, metrics, weights = _run(), _metrics(), _weights()
    return {
        "distribucion_precio_m2": plot_price_distribution(barrio_clean, run),
        "precio_m2_por_barrio": plot_price_by_barrio(barrio_clean, run),
        "mapa_segmentos": plot_location_map(barrio_clean, run),
        "metricas_modelos": plot_metrics(metrics, run),
        "cv_vs_test": plot_cv_vs_test(metrics, run),
        CONFUSION_CHART: plot_confusion(_matrix(MATRIX_VALUES), FOREST),
        "importancia_random_forest": plot_importances(weights, run),
        "coeficientes_regresion_logistica": plot_coefficients(weights, run),
    }


# --- Formato y reglas de segmento ------------------------------------------


@pytest.mark.parametrize(
    ("value", "decimals", "expected"),
    [
        (2711.4, 0, "2.711"),
        (3453.0, 0, "3.453"),
        (0.7412, 2, "0,74"),
        (2.94, 1, "2,9"),
        (1234567.891, 2, "1.234.567,89"),
        (89.0, 0, "89"),
        (-4.25, 1, "-4,2"),
    ],
)
def test_format_number_uses_spanish_style(value, decimals, expected):
    """Verificar miles con punto y decimales con coma."""
    assert format_number(value, decimals) == expected


def test_format_number_defaults_to_no_decimals():
    """Verificar que por defecto se redondea a entero."""
    assert format_number(2710.6) == "2.711"


@pytest.mark.parametrize(
    ("value", "decimals", "expected"),
    [(-0.0, 0, "0"), (-0.3, 0, "0"), (-0.004, 2, "0,00"), (-0.0, 1, "0,0")],
)
def test_format_number_has_no_negative_zero(value, decimals, expected):
    """Verificar que un valor que se redondea a cero no lleva signo menos."""
    assert format_number(value, decimals) == expected


def test_segment_of_matches_labeler_at_boundaries():
    """Verificar la misma regla que el etiquetador, incluso en los umbrales.

    Un precio igual a un umbral va a la clase superior, como en
    ``ValueSegmentLabeler.transform``.
    """
    values = [LOW - 1, LOW, LOW + 1, TOP - 1, TOP, TOP + 1]
    got = list(segment_of(pd.Series(values), LOW, TOP))
    assert got == _labeler_segments(values)
    assert got == [ACC, STD, STD, STD, HIGH, HIGH]


@pytest.mark.parametrize(
    ("variable", "expected"),
    [
        ("barrio=PALERMO", "Barrio: PALERMO"),
        ("barrio=VILLA GRAL. MITRE", "Barrio: VILLA GRAL. MITRE"),
        ("comuna=14", "Comuna 14"),
    ],
)
def test_feature_label_one_hot_columns(variable, expected):
    """Verificar la etiqueta legible de una columna one-hot."""
    assert feature_label(variable) == expected


@pytest.mark.parametrize("variable", NUMERIC_FEATURES)
def test_feature_label_numeric_features(variable):
    """Verificar que una feature numérica usa su nombre en español."""
    assert feature_label(variable) == FEATURE_LABELS[variable]


# --- Hallazgos -------------------------------------------------------------


def test_barrio_medians_filters_and_orders(barrio_clean):
    """Verificar columnas, barrios con el mínimo de avisos y orden descendente."""
    medians = barrio_medians(barrio_clean, _run())
    assert list(medians.columns) == ["barrio", "mediana", "avisos", "segmento"]
    counts = barrio_clean["barrio"].value_counts()
    assert set(medians["barrio"]) == set(counts[counts >= MIN_BARRIO_LISTINGS].index)
    assert medians["mediana"].is_monotonic_decreasing
    assert medians["avisos"].tolist() == counts[medians["barrio"]].tolist()


def test_barrio_medians_values_and_segments(barrio_clean):
    """Verificar la mediana de cada barrio y su segmento según los umbrales."""
    medians = barrio_medians(barrio_clean, _run())
    expected = barrio_clean.groupby("barrio")[TARGET_SOURCE].median()
    for row in medians.itertuples(index=False):
        assert row.mediana == pytest.approx(expected[row.barrio])
    assert medians["segmento"].tolist() == _labeler_segments(medians["mediana"])


@pytest.mark.parametrize(("fixture", "sign"), [("north_clean", 1), ("south_clean", -1)])
def test_north_shift_km_sign_and_value(request, fixture, sign):
    """Verificar la distancia norte-sur entre Alta y Accesible, sin NaN."""
    clean = request.getfixturevalue(fixture)
    expected = sign * 0.1 * KM_PER_DEGREE_LAT
    assert north_shift_km(clean, _run()) == pytest.approx(expected)


def test_test_metrics_orders_models_and_metrics():
    """Verificar filas en el orden de los modelos y columnas en el de ``SCORING``."""
    table = metrics_on_test(_metrics())
    assert list(table.index) == list(MODEL_LABELS)
    assert list(table.columns) == list(SCORING)
    for model in MODEL_LABELS:
        for metric in SCORING:
            expected = _test_value(model, metric)
            assert table.loc[model, metric] == pytest.approx(expected)


def test_cv_test_gap_values():
    """Verificar media y desvío de CV, valor de test y diferencia test - CV."""
    gaps = {BASELINE_NAME: 0.0, LOGISTIC: -0.05, FOREST: 0.02}
    table = cv_test_gap(_metrics(gaps))
    assert list(table.index) == list(MODEL_LABELS)
    assert list(table.columns) == ["cv", "desvio_cv", "test", "diferencia"]
    for model, gap in gaps.items():
        test = TEST_F1[model]
        row = table.loc[model]
        assert row["cv"] == pytest.approx(test - gap)
        assert row["desvio_cv"] == pytest.approx(CV_STD)
        assert row["test"] == pytest.approx(test)
        assert row["diferencia"] == pytest.approx(gap)


def test_neighbor_error_share_hand_matrix():
    """Verificar la fracción de errores entre segmentos vecinos de la matriz."""
    assert neighbor_error_share(_matrix(MATRIX_VALUES)) == pytest.approx(NEIGHBOR_SHARE)


def test_neighbor_error_share_perfect_matrix_is_zero():
    """Verificar que sin errores la fracción es 0.0."""
    assert neighbor_error_share(_matrix(np.diag([10, 20, 30]))) == 0.0


def test_row_recall_hand_matrix():
    """Verificar el acierto de cada segmento real: diagonal sobre total de fila."""
    recall = row_recall(_matrix(MATRIX_VALUES))
    assert list(recall.index) == list(SEGMENT_LABELS)
    assert recall.tolist() == pytest.approx(ROW_RECALL)


def test_row_recall_row_without_cases_is_nan():
    """Verificar NaN (no una división por cero) en una fila sin casos."""
    recall = row_recall(_matrix([[5, 0, 0], [0, 0, 0], [0, 1, 3]]))
    assert recall[ACC] == pytest.approx(1.0)
    assert np.isnan(recall[STD])
    assert recall[HIGH] == pytest.approx(0.75)


def test_importance_by_variable_sums_one_hot_columns():
    """Verificar la suma por variable original, de mayor a menor."""
    table = importance_by_variable(_weights())
    assert list(table.columns) == ["variable", "valor"]
    assert table["variable"].tolist() == list(IMPORTANCE_BY_VARIABLE)
    assert table["valor"].tolist() == pytest.approx(
        list(IMPORTANCE_BY_VARIABLE.values())
    )
    assert set(table["variable"]) == set(FEATURE_COLUMNS)


def test_location_share_of_importance():
    """Verificar la fracción de importancia de latitud, longitud, barrio y comuna."""
    share = location_share(importance_by_variable(_weights()))
    assert share == pytest.approx(LOCATION_SHARE)
    assert set(LOCATION_FEATURES) >= set(CATEGORICAL_FEATURES)


def test_numeric_contrast_is_alta_minus_accesible():
    """Verificar Alta - Accesible por numérica, sin one-hot, de mayor a menor módulo."""
    contrast = numeric_contrast(_weights())
    assert list(contrast.columns) == ["variable", "valor"]
    assert set(contrast["variable"]) == set(NUMERIC_FEATURES)
    for row in contrast.itertuples(index=False):
        assert row.valor == pytest.approx(CONTRAST[row.variable])
    magnitudes = contrast["valor"].abs()
    assert magnitudes.is_monotonic_decreasing


# --- Reglas comunes de todos los gráficos ----------------------------------


@pytest.mark.parametrize("key", list(CHART_LIMITATIONS))
def test_every_chart_is_a_figure_with_limitation_and_source(figures, key):
    """Verificar Figure, nota al pie con su limitación y la fuente de los datos."""
    fig = figures[key]
    assert isinstance(fig, Figure)
    footnote = _footnote(fig)
    assert "Limitación" in footnote
    assert CHART_LIMITATIONS[key] in footnote
    assert DATA_SOURCE in footnote


@pytest.mark.parametrize("key", list(CHART_LIMITATIONS))
def test_every_chart_has_title_and_axis_labels(figures, key):
    """Verificar que hay título y que el eje principal tiene sus rótulos."""
    fig = figures[key]
    assert _all_text(fig).strip()
    ax = fig.axes[0]
    assert ax.get_xlabel().strip()
    assert ax.get_ylabel().strip()


@pytest.mark.parametrize(("key", "axis"), list(VALUE_AXIS.items()))
def test_bar_charts_value_axis_starts_at_zero(figures, key, axis):
    """Verificar que el eje de valores de las barras empieza en 0."""
    ax = figures[key].axes[0]
    limits = ax.get_xlim() if axis == "x" else ax.get_ylim()
    assert min(limits) == 0


@pytest.mark.parametrize("key", LABELED_BARS)
def test_bar_charts_have_one_value_label_per_bar(figures, key):
    """Verificar una etiqueta de valor por cada barra."""
    ax = figures[key].axes[0]
    assert _bars(ax)
    assert len(_value_labels(ax)) == len(_bars(ax))


def test_plots_do_not_register_pyplot_figures(barrio_clean):
    """Verificar que los gráficos no usan el estado global de ``pyplot``."""
    import matplotlib.pyplot as plt

    before = plt.get_fignums()
    plot_price_distribution(barrio_clean, _run())
    assert plt.get_fignums() == before


# --- Gráficos de los datos -------------------------------------------------


def test_price_distribution_bars_use_segment_colors(figures):
    """Verificar barras coloreadas por el segmento de su centro.

    Son ``PRICE_BINS`` intervalos más uno por cada umbral, que parte la barra
    en la que cae (ambos umbrales están dentro del rango de precios).
    """
    bars = _bars(figures["distribucion_precio_m2"].axes[0])
    assert len(bars) == PRICE_BINS + 2
    centers = [bar.get_x() + bar.get_width() / 2 for bar in bars]
    for bar, segment in zip(bars, _labeler_segments(centers)):
        assert _same_color(bar.get_facecolor(), SEGMENT_COLORS[segment])


def test_price_distribution_bars_do_not_straddle_thresholds(figures, barrio_clean):
    """Verificar que ningún umbral cae dentro de una barra y que se cuentan todos."""
    bars = _bars(figures["distribucion_precio_m2"].axes[0])
    for bar in bars:
        left, right = bar.get_x(), bar.get_x() + bar.get_width()
        assert not any(left < threshold < right for threshold in (LOW, TOP))
    assert sum(bar.get_height() for bar in bars) == len(barrio_clean)


def test_price_distribution_marks_thresholds(figures):
    """Verificar líneas punteadas en ambos umbrales y los umbrales en el título."""
    fig = figures["distribucion_precio_m2"]
    lines = _vertical_lines(fig.axes[0])
    for threshold in (LOW, TOP):
        assert lines[threshold] in ("--", "dashed")
        assert format_number(threshold) in _all_text(fig)


def test_price_distribution_title_has_median(figures, barrio_clean):
    """Verificar que el título dice la mediana del precio por m²."""
    median = format_number(barrio_clean[TARGET_SOURCE].median())
    title = figures["distribucion_precio_m2"]._suptitle.get_text()
    assert f"La mitad de los avisos cuesta menos de {median}" in title


def test_price_by_barrio_shows_top_and_bottom(figures, barrio_clean):
    """Verificar los barrios más caros y más baratos, con el más caro arriba."""
    ax = figures["precio_m2_por_barrio"].axes[0]
    bars = _bars(ax)
    assert len(bars) == 2 * TOP_BARRIOS
    medians = barrio_medians(barrio_clean, _run())["mediana"]
    expected = [*medians.iloc[:TOP_BARRIOS], *medians.iloc[-TOP_BARRIOS:]]
    widths = sorted(bar.get_width() for bar in bars)
    assert widths == pytest.approx(sorted(expected))
    assert _top_bar(ax).get_width() == pytest.approx(medians.max())


def test_price_by_barrio_separates_groups_when_barrios_are_omitted(figures):
    """Verificar una línea punteada horizontal entre los caros y los baratos."""
    ax = figures["precio_m2_por_barrio"].axes[0]
    assert _horizontal_lines(ax) == {TOP_BARRIOS - 0.5: "--"}


def test_price_by_barrio_bars_use_segment_colors(figures):
    """Verificar que cada barra tiene el color del segmento de su mediana."""
    bars = _bars(figures["precio_m2_por_barrio"].axes[0])
    segments = _labeler_segments([bar.get_width() for bar in bars])
    for bar, segment in zip(bars, segments):
        assert _same_color(bar.get_facecolor(), SEGMENT_COLORS[segment])


def test_price_by_barrio_title_has_ratio(figures, barrio_clean):
    """Verificar que el título nombra ambos extremos y cuántas veces más cuesta."""
    medians = barrio_medians(barrio_clean, _run())
    top, bottom = medians.iloc[0], medians.iloc[-1]
    title = _all_text(figures["precio_m2_por_barrio"])
    assert top["barrio"] in title
    assert bottom["barrio"] in title
    assert format_number(top["mediana"] / bottom["mediana"], 1) in title


def test_price_by_barrio_few_barrios_shows_each_once(make_clean):
    """Verificar que con pocos barrios cada uno aparece una sola vez."""
    rows = [
        row
        for i, base in enumerate((1500.0, 2500.0, 3500.0))
        for row in _barrio_rows(f"B{i}", base, -34.6, MIN_BARRIO_LISTINGS)
    ]
    fig = plot_price_by_barrio(make_clean(rows), _run())
    assert len(_bars(fig.axes[0])) == 3
    assert not _horizontal_lines(fig.axes[0])


def test_price_by_barrio_without_enough_listings(make_clean):
    """Verificar una figura sin barras que explica el mínimo de avisos."""
    rows = _barrio_rows("B0", 1500.0, -34.6, MIN_BARRIO_LISTINGS - 1)
    fig = plot_price_by_barrio(make_clean(rows), _run())
    assert isinstance(fig, Figure)
    assert not _bars(fig.axes[0])
    assert format_number(MIN_BARRIO_LISTINGS) in _all_text(fig)


def test_location_map_single_scatter_with_segment_colors(figures, barrio_clean):
    """Verificar una sola dispersión, leyenda en orden y el color de cada punto."""
    ax = figures["mapa_segmentos"].axes[0]
    legend = [text.get_text() for text in ax.get_legend().get_texts()]
    assert legend == list(SEGMENT_LABELS)
    assert len(ax.collections) == 1
    points = ax.collections[0]
    assert len(points.get_offsets()) == len(barrio_clean)
    colors = {_rgb(color) for color in points.get_facecolors()}
    configured = {to_rgb(color) for color in SEGMENT_COLORS.values()}
    assert len(colors) > 1
    assert {tuple(color) for color in colors} <= configured


def test_location_map_point_colors_follow_segments(make_clean):
    """Verificar que cada punto lleva el color del segmento de su precio."""
    prices = [1500.0, 2500.0, 3500.0, 1000.0, 4000.0]
    rows = [
        {TARGET_SOURCE: price, "latitud": -34.6 + 0.01 * i, "longitud": -58.4}
        for i, price in enumerate(prices)
    ]
    ax = plot_location_map(make_clean(rows), _run()).axes[0]
    points = ax.collections[0]
    latitudes = points.get_offsets()[:, 1]
    by_latitude = {row["latitud"]: row[TARGET_SOURCE] for row in rows}
    expected = _labeler_segments([by_latitude[lat] for lat in latitudes])
    for color, segment in zip(points.get_facecolors(), expected):
        assert _same_color(color, SEGMENT_COLORS[segment])


def test_location_map_aspect_and_hidden_coordinates(figures, barrio_clean):
    """Verificar el aspecto 1 / cos(latitud media) y sin números de coordenadas."""
    ax = figures["mapa_segmentos"].axes[0]
    expected = 1 / np.cos(np.radians(barrio_clean["latitud"].mean()))
    assert ax.get_aspect() == pytest.approx(expected)
    ticks = ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks()
    assert not any(tick.label1.get_visible() for tick in ticks)


@pytest.mark.parametrize(
    ("fixture", "direction"), [("north_clean", "al norte"), ("south_clean", "al sur")]
)
def test_location_map_title_has_shift(request, fixture, direction):
    """Verificar el desplazamiento en km (en módulo) y su sentido en el título."""
    clean = request.getfixturevalue(fixture)
    title = _all_text(plot_location_map(clean, _run()))
    assert format_number(0.1 * KM_PER_DEGREE_LAT, 1) in title
    assert direction in title


# --- Gráficos de los modelos -----------------------------------------------


def test_metrics_chart_bars_by_model_color(figures):
    """Verificar que las barras de cada color son las métricas de test del modelo."""
    bars = _bars(figures["metricas_modelos"].axes[0])
    assert len(bars) == len(MODEL_LABELS) * len(SCORING)
    for model in MODEL_LABELS:
        color = MODEL_COLORS[model]
        heights = [
            b.get_height() for b in bars if _same_color(b.get_facecolor(), color)
        ]
        expected = [_test_value(model, metric) for metric in SCORING]
        assert sorted(heights) == pytest.approx(sorted(expected))


def test_metrics_chart_labels_have_two_decimals(figures):
    """Verificar las etiquetas de valor con dos decimales, estilo español."""
    ax = figures["metricas_modelos"].axes[0]
    expected = [
        format_number(_test_value(model, metric), 2)
        for model in MODEL_LABELS
        for metric in SCORING
    ]
    assert sorted(_value_labels(ax)) == sorted(expected)


def test_metrics_chart_title_compares_chosen_and_baseline(figures):
    """Verificar el modelo elegido y su F1 frente al del baseline en el título."""
    title = _all_text(figures["metricas_modelos"])
    assert MODEL_LABELS[FOREST] in title
    assert format_number(TEST_F1[FOREST], 2) in title
    assert format_number(TEST_F1[BASELINE_NAME], 2) in title


def test_metrics_chart_axis_says_which_metrics_are_macro(figures):
    """Verificar que el eje x aclara que precisión, recall y F1 son macro."""
    xlabel = figures["metricas_modelos"].axes[0].get_xlabel()
    assert "precisión, recall y F1: promedio macro" in xlabel


def test_metrics_chart_without_baseline_raises():
    """Verificar ``ValueError`` claro (no ``KeyError``) si falta el baseline."""
    metrics = _metrics()
    metrics = metrics[metrics["modelo"] != BASELINE_NAME]
    with pytest.raises(ValueError, match=BASELINE_NAME):
        plot_metrics(metrics, _run())


def test_cv_vs_test_has_cv_and_test_bars_with_error(figures):
    """Verificar una barra de CV y una de test por modelo, con desvío en la CV."""
    ax = figures["cv_vs_test"].axes[0]
    assert len(_bars(ax)) == 2 * len(MODEL_LABELS)
    assert any(isinstance(c, ErrorbarContainer) for c in ax.containers)


def test_cv_vs_test_cv_labels_show_mean_and_std(figures):
    """Verificar que la etiqueta de cada barra de CV dice media ± desvío."""
    labels = _value_labels(figures["cv_vs_test"].axes[0])
    for model in MODEL_LABELS:
        mean = format_number(TEST_F1[model] - DEFAULT_GAP, 3)
        assert f"{mean}\n± {format_number(CV_STD, 3)}" in labels


def test_cv_vs_test_subtitle_without_double_parentheses(figures):
    """Verificar que el subtítulo explica el desvío y no repite paréntesis."""
    subtitle = figures["cv_vs_test"].axes[0].get_title(loc="left")
    assert "media ± desvío entre folds" in subtitle
    assert "))" not in subtitle


def test_cv_vs_test_title_when_cv_anticipates_test():
    """Verificar el título y la mayor diferencia cuando todas son chicas."""
    gaps = {BASELINE_NAME: 0.002, LOGISTIC: -0.004, FOREST: 0.0095}
    title = plot_cv_vs_test(_metrics(gaps), _run())._suptitle.get_text()
    assert (
        "La CV anticipa bien el F1 de test: difieren a lo sumo 0,01 en todos "
        "los modelos"
    ) in title
    assert "sobreajuste" not in title


@pytest.mark.parametrize(
    ("model", "gap", "direction"),
    [(LOGISTIC, 0.09, "más alto"), (FOREST, -0.07, "más bajo")],
    ids=["test_mas_alto", "test_mas_bajo"],
)
def test_cv_vs_test_title_names_largest_difference(model, gap, direction):
    """Verificar el modelo, el monto y el sentido de la mayor diferencia."""
    title = plot_cv_vs_test(_metrics({model: gap}), _run())._suptitle.get_text()
    amount = format_number(abs(gap), 2)
    expected = f"{MODEL_LABELS[model]}: el F1 de test difiere {amount} del de la CV"
    assert expected in title
    assert f"({direction} en test)" in title
    assert "sobreajuste" not in title


def test_cv_vs_test_limitation_says_it_does_not_measure_overfitting(figures):
    """Verificar que la limitación aclara que la comparación no mide sobreajuste."""
    assert "no mide sobreajuste" in _footnote(figures["cv_vs_test"])


def test_confusion_title_names_hardest_segment(figures):
    """Verificar el modelo, el segmento de menor acierto y su porcentaje."""
    title = figures[CONFUSION_CHART]._suptitle.get_text()
    accuracy = format_number(100 * min(ROW_RECALL))
    expected = (
        f"{MODEL_LABELS[FOREST]}: {STD} es el segmento más difícil "
        f"({accuracy} % de acierto)"
    )
    assert title == expected


def test_confusion_subtitle_has_neighbor_share(figures):
    """Verificar el porcentaje de errores entre segmentos vecinos en el subtítulo."""
    subtitle = figures[CONFUSION_CHART].axes[0].get_title(loc="left")
    assert f"el {format_number(100 * NEIGHBOR_SHARE)} % de los errores" in subtitle


def test_confusion_annotates_row_percentages_and_counts(figures):
    """Verificar el texto exacto de cada celda: porcentaje de fila y conteo."""
    ax = figures[CONFUSION_CHART].axes[0]
    expected = [
        f"{format_number(100 * count / sum(row), 1)} %\n({format_number(count)})"
        for row in MATRIX_VALUES
        for count in row
    ]
    assert [text.get_text() for text in ax.texts] == expected
    assert "real" in ax.get_ylabel().lower()
    assert "predicho" in ax.get_xlabel().lower()
    assert [t.get_text() for t in ax.get_xticklabels()] == list(SEGMENT_LABELS)
    assert [t.get_text() for t in ax.get_yticklabels()] == list(SEGMENT_LABELS)


def test_confusion_perfect_matrix():
    """Verificar que una matriz sin errores da una figura con un título sensato."""
    fig = plot_confusion(_matrix(np.diag([10, 20, 30])), LOGISTIC)
    assert isinstance(fig, Figure)
    title = fig._suptitle.get_text()
    assert title == f"{MODEL_LABELS[LOGISTIC]}: acierta todos los segmentos en test"
    assert "errores" not in fig.axes[0].get_title(loc="left")


def test_importances_one_bar_per_original_variable(figures):
    """Verificar una barra por variable original, la mayor arriba."""
    ax = figures["importancia_random_forest"].axes[0]
    bars = _bars(ax)
    assert len(bars) == len(FEATURE_COLUMNS)
    widths = sorted(bar.get_width() for bar in bars)
    assert widths == pytest.approx(sorted(IMPORTANCE_BY_VARIABLE.values()))
    assert _top_bar(ax).get_width() == pytest.approx(
        max(IMPORTANCE_BY_VARIABLE.values())
    )
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == [FEATURE_LABELS[v] for v in IMPORTANCE_BY_VARIABLE]


def test_importances_highlight_location(figures):
    """Verificar el color del modelo en la ubicación y el neutro en el resto."""
    ax = figures["importancia_random_forest"].axes[0]
    widths = {value: variable for variable, value in IMPORTANCE_BY_VARIABLE.items()}
    for bar in _bars(ax):
        variable = widths[min(widths, key=lambda w: abs(w - bar.get_width()))]
        located = variable in LOCATION_FEATURES
        color = MODEL_COLORS[FOREST] if located else NEUTRAL_COLOR
        assert _same_color(bar.get_facecolor(), color)


def test_importances_title_has_location_share(figures):
    """Verificar el porcentaje de importancia de la ubicación en el título."""
    title = figures["importancia_random_forest"]._suptitle.get_text()
    share = format_number(100 * LOCATION_SHARE)
    expected = f"La ubicación concentra el {share} % de la importancia del "
    assert title == expected + MODEL_LABELS[FOREST]


def test_coefficients_show_numeric_contrast(figures):
    """Verificar una barra por numérica con su contraste, la mayor arriba."""
    ax = figures["coeficientes_regresion_logistica"].axes[0]
    bars = _bars(ax)
    assert len(bars) == len(NUMERIC_FEATURES)
    widths = sorted(bar.get_width() for bar in bars)
    assert widths == pytest.approx(sorted(CONTRAST.values()))
    assert _top_bar(ax).get_width() == pytest.approx(_top_contrast()[1])


def test_coefficients_diverging_colors_and_zero_line(figures):
    """Verificar los colores neutros por signo y la línea en 0."""
    ax = figures["coeficientes_regresion_logistica"].axes[0]
    bars = _bars(ax)
    assert any(bar.get_width() < 0 for bar in bars)
    for bar in bars:
        color = POSITIVE_COLOR if bar.get_width() > 0 else NEGATIVE_COLOR
        assert _same_color(bar.get_facecolor(), color)
    assert 0.0 in _vertical_lines(ax)


@pytest.mark.parametrize(("coef_sign", "direction"), [(1.0, "menor"), (-1.0, "mayor")])
def test_coefficients_title_is_associative(coef_sign, direction):
    """Verificar la variable de mayor contraste y su sentido, sin causalidad."""
    variable, _ = _top_contrast(coef_sign)
    title = plot_coefficients(_weights(coef_sign), _run())._suptitle.get_text()
    high, low = COEF_CONTRAST
    label = FEATURE_LABELS[variable].lower()
    expected = (
        f"Más {label} se asocia con {direction} probabilidad de {high} que de {low}"
    )
    assert title == expected


def test_model_charts_use_spanish_ticks(figures):
    """Verificar coma decimal y el mismo signo menos en ejes y etiquetas."""
    ax = figures["coeficientes_regresion_logistica"].axes[0]
    ax.figure.canvas.draw()
    ticks = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    negatives = [t for t in ticks if t.startswith("-")]
    assert negatives
    assert all("." not in t for t in ticks)
    assert any(label.startswith("-") for label in _value_labels(ax))
    assert not any("\N{MINUS SIGN}" in t for t in ticks + _value_labels(ax))


# --- Datos desde la base y archivos ----------------------------------------


def _artifacts(clean):
    """Entrenar con grillas diminutas y armar los artefactos de una corrida."""
    train, test = split_data(clean)
    labeler = ValueSegmentLabeler().fit(train[TARGET_SOURCE])
    x_train, y_train = make_xy(train, labeler)
    x_test, y_test = make_xy(test, labeler)
    results = train_models(x_train, y_train, grids=TINY_GRIDS, n_jobs=1)
    return RunArtifacts(
        run_id=RUN_ID,
        results=results,
        evaluations=evaluate_models(results, x_test, y_test),
        labeler=labeler,
        best=select_best(results),
        test_ids=test[ID_COLUMN],
        y_test=y_test,
        n_train=len(train),
    )


@pytest.fixture(scope="module")
def chart_db(tmp_path_factory, learnable_clean):
    """Escribir una base temporal con la tabla limpia y una corrida real."""
    path = tmp_path_factory.mktemp("graficos") / "inmuebles.db"
    with closing(connect(path)) as conn:
        create_schema(conn)
        write_clean(conn, learnable_clean)
        write_run(conn, build_run_tables(_artifacts(learnable_clean)))
    return path


@pytest.fixture(scope="module")
def chart_data(chart_db):
    """Leer de la base los datos de la corrida más reciente."""
    with closing(connect(chart_db)) as conn:
        return load_chart_data(conn)


@pytest.fixture(scope="module")
def built(chart_data):
    """Armar una sola vez todos los gráficos de la corrida."""
    return build_figures(chart_data)


def test_load_chart_data_reads_latest_run(chart_data, learnable_clean):
    """Verificar la corrida, la tabla limpia, métricas, pesos y matrices."""
    assert isinstance(chart_data, ChartData)
    assert isinstance(chart_data.run, pd.Series)
    assert chart_data.run[RUN_ID_COLUMN] == RUN_ID
    assert len(chart_data.clean) == len(learnable_clean)
    assert set(chart_data.metrics["modelo"]) == set(MODEL_LABELS)
    assert set(chart_data.weights["modelo"]) == set(REAL_MODELS)
    assert set(REAL_MODELS) <= set(chart_data.confusions)
    for matrix in chart_data.confusions.values():
        assert list(matrix.index) == list(SEGMENT_LABELS)


def test_load_chart_data_is_frozen(chart_data):
    """Verificar que los datos de los gráficos no se pueden reasignar."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        chart_data.run = None


def test_load_chart_data_explicit_run_id(chart_db):
    """Verificar que se puede pedir una corrida por su ``run_id``."""
    with closing(connect(chart_db)) as conn:
        data = load_chart_data(conn, run_id=RUN_ID)
    assert data.run[RUN_ID_COLUMN] == RUN_ID


def test_load_chart_data_unknown_run_raises(chart_db):
    """Verificar ``ValueError`` si el ``run_id`` no existe."""
    with closing(connect(chart_db)) as conn:
        with pytest.raises(ValueError):
            load_chart_data(conn, run_id="no_existe")


def test_load_chart_data_without_runs_raises(tmp_path, learnable_clean):
    """Verificar ``ValueError`` si la base no tiene corridas."""
    with closing(connect(tmp_path / "sin_corridas.db")) as conn:
        create_schema(conn)
        write_clean(conn, learnable_clean)
        with pytest.raises(ValueError):
            load_chart_data(conn)


def _table_names(conn):
    """Devolver los nombres de las tablas de la base."""
    sql = "SELECT name FROM sqlite_master WHERE type = 'table'"
    return {row[0] for row in conn.execute(sql)}


def test_load_chart_data_old_db_without_weights_table_raises(chart_db, tmp_path):
    """Verificar ``ValueError`` que remite a entrenar.py, sin crear la tabla."""
    path = tmp_path / "vieja.db"
    path.write_bytes(chart_db.read_bytes())
    with closing(connect(path)) as conn:
        conn.execute(f"DROP TABLE {IMPORTANCE_TABLE}")
        conn.commit()
        with pytest.raises(ValueError, match="entrenar.py"):
            load_chart_data(conn)
        assert IMPORTANCE_TABLE not in _table_names(conn)


def test_build_figures_names(built):
    """Verificar un gráfico por nombre y una matriz por cada modelo real."""
    assert set(built) == EXPECTED_FIGURES
    assert all(isinstance(fig, Figure) for fig in built.values())


def test_save_figures_writes_files(built, tmp_path, caplog):
    """Verificar archivos no vacíos con su nombre, carpeta creada y logs INFO."""
    out = tmp_path / "reportes" / "figuras"
    with caplog.at_level("INFO"):
        paths = save_figures(built, out)
    expected = {f"{name}.{FIGURE_FORMAT}" for name in EXPECTED_FIGURES}
    assert {path.name for path in paths} == expected
    assert {path.name for path in out.iterdir()} == expected
    assert all(path.stat().st_size > 0 for path in paths)
    infos = [record for record in caplog.records if record.levelname == "INFO"]
    assert len(infos) >= len(built)


# --- Casos agregados tras las pruebas de mutación ------------------------


def test_metrics_tables_drop_unknown_models():
    """Verificar que las tablas de métricas solo incluyen los modelos conocidos."""
    metrics = _metrics()
    extra = metrics[metrics["modelo"] == BASELINE_NAME].assign(modelo="aaa_otro")
    metrics = pd.concat([metrics, extra], ignore_index=True)
    assert list(metrics_on_test(metrics).index) == list(MODEL_LABELS)
    assert list(cv_test_gap(metrics).index) == list(MODEL_LABELS)


def test_cv_vs_test_title_at_tolerance_boundary():
    """Verificar que una diferencia igual a la tolerancia cuenta como coincidencia."""
    metrics = _metrics()
    refit = metrics["metrica"] == REFIT_METRIC
    metrics.loc[refit & (metrics["origen"] == TEST_ORIGIN), "media"] = CV_TEST_TOLERANCE
    metrics.loc[refit & (metrics["origen"] == CV_ORIGIN), "media"] = 0.0
    title = plot_cv_vs_test(metrics, _run())._suptitle.get_text()
    assert "La CV anticipa bien" in title


def test_location_share_is_relative_to_total():
    """Verificar que la participación de la ubicación se divide por el total."""
    table = pd.DataFrame({"variable": ["barrio", "m2total"], "valor": [1.0, 3.0]})
    assert location_share(table) == pytest.approx(0.25)


@pytest.mark.parametrize(
    "key", ["distribucion_precio_m2", "precio_m2_por_barrio", "mapa_segmentos"]
)
def test_segment_legend_colors(figures, key):
    """Verificar que las leyendas muestran cada segmento con su color fijo."""
    legend = figures[key].axes[0].get_legend()
    texts = [text.get_text() for text in legend.get_texts()]
    assert texts == list(SEGMENT_LABELS)
    for text, handle in zip(texts, legend.legend_handles):
        assert to_rgb(handle.get_facecolor()) == to_rgb(SEGMENT_COLORS[text])


def _offsets(fig):
    """Devolver las coordenadas de los puntos del mapa."""
    return fig.axes[0].collections[0].get_offsets().data


def test_location_map_sample_is_reproducible(learnable_clean):
    """Verificar que la muestra del mapa es la misma en cada ejecución."""
    first = _offsets(plot_location_map(learnable_clean, _run()))
    second = _offsets(plot_location_map(learnable_clean, _run()))
    np.testing.assert_array_equal(first, second)


def test_location_map_points_are_shuffled(learnable_clean):
    """Verificar que los puntos se dibujan mezclados y ningún segmento tapa a otro."""
    located = learnable_clean.dropna(subset=["latitud", "longitud"])
    drawn = _offsets(plot_location_map(located, _run()))
    in_order = located[["longitud", "latitud"]].to_numpy()
    assert len(drawn) == len(in_order)
    assert not np.array_equal(drawn, in_order)


def test_confusion_color_scale_and_cell_text():
    """Verificar la escala fija de 0 a 100 y el texto blanco en celdas oscuras."""
    matrix = pd.DataFrame(
        [[9, 1, 0], [1, 8, 1], [0, 2, 8]],
        index=list(SEGMENT_LABELS),
        columns=list(SEGMENT_LABELS),
    )
    ax = plot_confusion(matrix, FOREST).axes[0]
    assert ax.images[0].get_clim() == (0, 100)
    colors = {text.get_text().split(" %")[0]: text.get_color() for text in ax.texts}
    assert colors["90,0"] == "white"
    assert colors["10,0"] != "white"


@pytest.fixture(scope="module")
def two_run_db(tmp_path_factory, chart_db, learnable_clean):
    """Copiar la base de prueba y agregarle una segunda corrida."""
    path = tmp_path_factory.mktemp("dos_corridas") / "inmuebles.db"
    path.write_bytes(chart_db.read_bytes())
    other = dataclasses.replace(_artifacts(learnable_clean), run_id="otra_corrida")
    with closing(connect(path)) as conn:
        write_run(conn, build_run_tables(other))
    return path


@pytest.mark.parametrize("run_id", [RUN_ID, "otra_corrida"])
def test_load_chart_data_reads_only_the_requested_run(two_run_db, run_id):
    """Verificar que con dos corridas se leen solo los datos de la pedida."""
    with closing(connect(two_run_db)) as conn:
        data = load_chart_data(conn, run_id=run_id)
    assert data.run[RUN_ID_COLUMN] == run_id
    assert set(data.metrics[RUN_ID_COLUMN]) == {run_id}
    assert set(data.weights[RUN_ID_COLUMN]) == {run_id}
    assert not data.weights.duplicated(["modelo", "variable", "segmento"]).any()


def test_load_chart_data_run_without_weights_raises(chart_db, tmp_path):
    """Verificar el error claro si la corrida no tiene filas en ``importancias``."""
    path = tmp_path / "sin_pesos.db"
    path.write_bytes(chart_db.read_bytes())
    with closing(connect(path)) as conn:
        with conn:
            conn.execute(f"DELETE FROM {IMPORTANCE_TABLE}")
        with pytest.raises(ValueError, match="entrenar.py"):
            load_chart_data(conn)


def test_save_figures_uses_figure_dpi(built, tmp_path):
    """Verificar que los PNG se guardan con la resolución configurada."""
    paths = save_figures(built, tmp_path)
    with Image.open(paths[0]) as image:
        assert image.info["dpi"] == pytest.approx((FIGURE_DPI, FIGURE_DPI), abs=0.5)
