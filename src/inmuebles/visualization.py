"""Gráficos del proyecto, armados solo con lo que hay en la base SQLite.

Cada gráfico responde una pregunta y su título dice el hallazgo, calculado con
los datos (números con formato español). Lleva un subtítulo con lo que se
muestra y una nota al pie con su limitación (``CHART_LIMITATIONS``) y la
fuente. Las barras arrancan en 0 y llevan su valor, y cada segmento y cada
modelo conservan su color en todos los gráficos.

Las funciones de hallazgo (``barrio_medians``, ``north_shift_km``,
``metrics_on_test``, ``cv_test_gap``, ``row_recall``,
``neighbor_error_share``, ``importance_by_variable``, ``location_share``,
``numeric_contrast``) son puras y se prueban por separado. Los gráficos usan
``matplotlib.figure.Figure`` directamente, sin ``pyplot`` ni estado global,
así que funcionan sin pantalla.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from inmuebles.config import (
    ALL_SEGMENTS,
    BASELINE_NAME,
    CHART_LIMITATIONS,
    CLEAN_TABLE,
    COEF_CONTRAST,
    COEF_MODEL,
    CV_ORIGIN,
    CV_TEST_TOLERANCE,
    DATA_SOURCE,
    FEATURE_LABELS,
    FIGURE_DPI,
    FIGURE_FORMAT,
    IMPORTANCE_MODEL,
    IMPORTANCE_TABLE,
    KM_PER_DEGREE_LAT,
    LOCATION_FEATURES,
    MAP_SAMPLE,
    METRIC_LABELS,
    METRICS_TABLE,
    MIN_BARRIO_LISTINGS,
    MODEL_COLORS,
    MODEL_LABELS,
    NEGATIVE_COLOR,
    NEUTRAL_COLOR,
    NUMERIC_FEATURES,
    POSITIVE_COLOR,
    PRICE_BINS,
    RANDOM_STATE,
    REFIT_METRIC,
    RUN_ID_COLUMN,
    RUNS_TABLE,
    SCORING,
    SEGMENT_COLORS,
    SEGMENT_LABELS,
    TARGET_SOURCE,
    TEST_ORIGIN,
    TOP_BARRIOS,
)
from inmuebles.database import latest_run_id, read_confusion, read_table
from inmuebles.target import ValueSegmentLabeler

logger = logging.getLogger(__name__)

# Nombres de archivo (sin extensión) de cada gráfico
PRICE_CHART = "distribucion_precio_m2"
BARRIO_CHART = "precio_m2_por_barrio"
MAP_CHART = "mapa_segmentos"
METRICS_CHART = "metricas_modelos"
GAP_CHART = "cv_vs_test"
CONFUSION_CHART = "matriz_confusion"
IMPORTANCE_CHART = "importancia_random_forest"
COEF_CHART = "coeficientes_regresion_logistica"

# Segmentos extremos y unidad del precio
LOW_SEGMENT, HIGH_SEGMENT = SEGMENT_LABELS[0], SEGMENT_LABELS[-1]
PRICE_UNIT = "USD/m²"
# Columnas de las tablas de hallazgos
BARRIO_COLUMNS = ["barrio", "mediana", "avisos", "segmento"]
GAP_COLUMNS = ["cv", "desvio_cv", "test", "diferencia"]
WEIGHT_COLUMNS = ["variable", "valor"]
# Separador de las columnas one-hot en la tabla de pesos (``barrio=PALERMO``)
ONE_HOT_SEPARATOR = "="

# Estilo: tamaños (pulgadas), fuentes, colores neutros y decimales
_FIGSIZE = (10, 6)
_MAP_FIGSIZE = (9, 8)
_TITLE_SIZE = 14
_SUBTITLE_SIZE = 10
_NOTE_SIZE = 8
_LABEL_SIZE = 8
_CELL_SIZE = 12
_MUTED = "#555555"
_LINE_COLOR = "#333333"
_BAR_WIDTH = 0.8
_CV_ALPHA = 0.45
_MAP_POINT_SIZE = 2
_MAP_ALPHA = 0.5
_CONFUSION_CMAP = "Blues"
_DARK_CELL = 50  # porcentaje desde el cual el texto de la celda va en blanco
_PERCENT = 100
_METRIC_DECIMALS = 2
_GAP_DECIMALS = 3  # CV y test difieren en milésimas: se muestran 3 decimales
_IMPORTANCE_DECIMALS = 3
_IMPORTANCE_TICK_DECIMALS = 2
_CELL_DECIMALS = 1
_RATIO_DECIMALS = 1
_TICK_DECIMALS = 1
_KM_DECIMALS = 1
_VALUE_MARGIN = 0.15  # margen del eje de valores para que entren las etiquetas


# --- Formato -----------------------------------------------------------------


def format_number(value, decimals=0):
    """Formatear un número al estilo español: miles con punto, decimales con coma.

    El signo menos es el guion ASCII ``-``, igual en etiquetas y ejes; un
    valor que se redondea a cero se escribe sin signo (nunca ``-0``).

    Parameters
    ----------
    value : float
        Número a formatear.
    decimals : int, default 0
        Cantidad de decimales.

    Returns
    -------
    str
        Por ejemplo, ``1234567.891`` con 2 decimales da ``"1.234.567,89"``.
    """
    if round(value, decimals) == 0:
        value = 0.0  # evita el cero negativo ("-0" o "-0,00")
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def segment_of(values, low, high):
    """Asignar el segmento de cada valor con los umbrales de una corrida.

    Reutiliza ``ValueSegmentLabeler.transform`` con los umbrales ya
    aprendidos, así que la regla es la misma: un valor igual a un umbral va
    al segmento superior.

    Parameters
    ----------
    values : array-like 1-D
        Precios en USD por m² (sin NaN).
    low, high : float
        Umbrales de la corrida (``umbral_bajo`` y ``umbral_alto``).

    Returns
    -------
    pandas.Series
        Etiquetas de ``SEGMENT_LABELS``; conserva el índice si ``values`` es
        una Series.
    """
    labeler = ValueSegmentLabeler()
    labeler.thresholds_ = (float(low), float(high))
    return labeler.transform(values)


def feature_label(variable):
    """Devolver el nombre legible de una variable de la tabla de pesos.

    Parameters
    ----------
    variable : str
        Nombre limpio, p. ej. ``antig``, ``barrio=PALERMO`` o ``comuna=14``.

    Returns
    -------
    str
        ``FEATURE_LABELS`` para las numéricas, ``"Barrio: PALERMO"`` o
        ``"Comuna 14"`` para las columnas one-hot.
    """
    column, separator, value = variable.partition(ONE_HOT_SEPARATOR)
    label = FEATURE_LABELS.get(column, column)
    if not separator:
        return label
    if column == "barrio":
        return f"{label}: {value}"
    return f"{label} {value}"


def _thresholds(run):
    """Devolver los umbrales ``(bajo, alto)`` de la corrida."""
    return float(run["umbral_bajo"]), float(run["umbral_alto"])


def _model_label(model):
    """Devolver el nombre en español de un modelo."""
    return MODEL_LABELS.get(model, model)


def _ordered_models(models):
    """Ordenar modelos según ``MODEL_LABELS`` (los desconocidos se omiten)."""
    present = set(models)
    return [model for model in MODEL_LABELS if model in present]


def _sorted_by_magnitude(table):
    """Ordenar una tabla ``variable``/``valor`` de mayor a menor ``|valor|``."""
    order = table["valor"].abs().sort_values(ascending=False, kind="stable").index
    return table.loc[order, WEIGHT_COLUMNS].reset_index(drop=True)


# --- Hallazgos -----------------------------------------------------------------


def barrio_medians(clean, run):
    """Calcular la mediana del precio por m² de cada barrio con avisos suficientes.

    Parameters
    ----------
    clean : pandas.DataFrame
        Tabla limpia (Tabla 1).
    run : pandas.Series
        Fila de la corrida, con ``umbral_bajo`` y ``umbral_alto``.

    Returns
    -------
    pandas.DataFrame
        Columnas ``barrio``, ``mediana``, ``avisos`` y ``segmento`` (el de la
        mediana), solo barrios con al menos ``MIN_BARRIO_LISTINGS`` avisos,
        de la mediana más alta a la más baja.
    """
    stats = clean.groupby("barrio")[TARGET_SOURCE].agg(mediana="median", avisos="size")
    stats = stats[stats["avisos"] >= MIN_BARRIO_LISTINGS].reset_index()
    stats = stats.sort_values(
        ["mediana", "barrio"], ascending=[False, True], ignore_index=True
    )
    stats["segmento"] = segment_of(stats["mediana"], *_thresholds(run)).to_numpy()
    return stats[BARRIO_COLUMNS]


def _located(clean):
    """Quitar los avisos sin latitud o longitud."""
    return clean.dropna(subset=["latitud", "longitud"])


def north_shift_km(clean, run):
    """Medir cuántos km más al norte están, en promedio, los avisos Alta.

    Parameters
    ----------
    clean : pandas.DataFrame
        Tabla limpia (Tabla 1); se ignoran los avisos sin ubicación.
    run : pandas.Series
        Fila de la corrida, con ``umbral_bajo`` y ``umbral_alto``.

    Returns
    -------
    float
        (latitud media de Alta - latitud media de Accesible) por
        ``KM_PER_DEGREE_LAT``; negativo si Alta queda más al sur.
    """
    located = _located(clean)
    segments = segment_of(located[TARGET_SOURCE], *_thresholds(run))
    means = located["latitud"].groupby(segments).mean()
    return float((means[HIGH_SEGMENT] - means[LOW_SEGMENT]) * KM_PER_DEGREE_LAT)


def metrics_on_test(metrics):
    """Armar la tabla de métricas de test: un modelo por fila.

    Parameters
    ----------
    metrics : pandas.DataFrame
        Tabla larga de métricas, como la devuelve ``read_table``.

    Returns
    -------
    pandas.DataFrame
        Filas en el orden de ``MODEL_LABELS`` y columnas en el de ``SCORING``.
    """
    test = metrics[metrics["origen"] == TEST_ORIGIN]
    table = test.pivot(index="modelo", columns="metrica", values="media")
    return table.reindex(index=_ordered_models(table.index), columns=list(SCORING))


def cv_test_gap(metrics):
    """Comparar la métrica de selección en la CV y en test, por modelo.

    Parameters
    ----------
    metrics : pandas.DataFrame
        Tabla larga de métricas, como la devuelve ``read_table``.

    Returns
    -------
    pandas.DataFrame
        Columnas ``cv`` (media), ``desvio_cv``, ``test`` y ``diferencia``
        (test - cv) de ``REFIT_METRIC``; filas en el orden de ``MODEL_LABELS``.
    """
    refit = metrics[metrics["metrica"] == REFIT_METRIC].set_index(["modelo", "origen"])
    cv = refit.xs(CV_ORIGIN, level="origen")
    test = refit.xs(TEST_ORIGIN, level="origen")
    table = pd.DataFrame(
        {"cv": cv["media"], "desvio_cv": cv["desvio"], "test": test["media"]}
    )
    table["diferencia"] = table["test"] - table["cv"]
    return table.reindex(_ordered_models(table.index))[GAP_COLUMNS]


def row_recall(matrix):
    """Calcular el acierto de cada segmento real (recall por fila).

    Parameters
    ----------
    matrix : pandas.DataFrame
        Matriz de confusión (filas = real, columnas = predicho) en el orden
        de ``SEGMENT_LABELS``, como la devuelve ``read_confusion``.

    Returns
    -------
    pandas.Series
        Diagonal sobre el total de cada fila, con el índice de ``matrix``;
        NaN en una fila sin casos.
    """
    values = matrix.to_numpy(dtype=float)
    totals = values.sum(axis=1)
    recall = np.divide(
        np.diag(values), totals, out=np.full(len(values), np.nan), where=totals > 0
    )
    return pd.Series(recall, index=matrix.index)


def neighbor_error_share(matrix):
    """Calcular la fracción de errores que caen entre segmentos vecinos.

    Parameters
    ----------
    matrix : pandas.DataFrame
        Matriz de confusión (filas = real, columnas = predicho) en el orden
        de ``SEGMENT_LABELS``, como la devuelve ``read_confusion``.

    Returns
    -------
    float
        Casos fuera de la diagonal con ``|i - j| == 1`` sobre todos los casos
        fuera de la diagonal; 0.0 si no hay errores.
    """
    values = matrix.to_numpy()
    rows, columns = np.indices(values.shape)
    errors = values[rows != columns].sum()
    if errors == 0:
        return 0.0
    return float(values[np.abs(rows - columns) == 1].sum() / errors)


def importance_by_variable(weights):
    """Sumar las importancias del Random Forest por variable original.

    La importancia por impureza es aditiva, así que la de ``barrio`` (o
    ``comuna``) es la suma de sus columnas one-hot.

    Parameters
    ----------
    weights : pandas.DataFrame
        Tabla de pesos, como la devuelve ``read_table``.

    Returns
    -------
    pandas.DataFrame
        Columnas ``variable`` (original, sin ``=valor``) y ``valor``, de la
        mayor importancia a la menor (a igual valor, por nombre).
    """
    mask = (weights["modelo"] == IMPORTANCE_MODEL) & (
        weights["segmento"] == ALL_SEGMENTS
    )
    rows = weights[mask]
    original = rows["variable"].str.partition(ONE_HOT_SEPARATOR)[0].rename("variable")
    totals = rows["valor"].groupby(original).sum().reset_index()
    return totals.sort_values(
        ["valor", "variable"], ascending=[False, True], ignore_index=True
    )[WEIGHT_COLUMNS]


def location_share(importances):
    """Calcular la fracción de la importancia que aportan las variables de ubicación.

    Parameters
    ----------
    importances : pandas.DataFrame
        Salida de ``importance_by_variable``.

    Returns
    -------
    float
        Suma de ``LOCATION_FEATURES`` sobre el total (entre 0 y 1).
    """
    located = importances["variable"].isin(LOCATION_FEATURES)
    return float(importances.loc[located, "valor"].sum() / importances["valor"].sum())


def numeric_contrast(weights):
    """Calcular el contraste de coeficientes Alta - Accesible de cada numérica.

    Los coeficientes multinomiales suman 0 entre clases, así que la diferencia
    entre dos clases es el cambio en el log-odds de una frente a la otra por
    cada desvío estándar de la variable (que entra estandarizada).

    Parameters
    ----------
    weights : pandas.DataFrame
        Tabla de pesos, como la devuelve ``read_table``.

    Returns
    -------
    pandas.DataFrame
        Columnas ``variable`` y ``valor`` (coeficiente del primer segmento de
        ``COEF_CONTRAST`` menos el del segundo), una fila por variable de
        ``NUMERIC_FEATURES``, de mayor a menor ``|valor|``.
    """
    high, low = COEF_CONTRAST
    mask = (weights["modelo"] == COEF_MODEL) & weights["variable"].isin(
        NUMERIC_FEATURES
    )
    table = weights[mask].pivot(index="variable", columns="segmento", values="valor")
    contrast = (table[high] - table[low]).rename("valor").reset_index()
    return _sorted_by_magnitude(contrast)


# --- Estructura común de los gráficos --------------------------------------------


def _new_chart(figsize=_FIGSIZE):
    """Crear una figura (sin ``pyplot``) con un eje principal."""
    fig = Figure(figsize=figsize, layout="constrained")
    return fig, fig.add_subplot()


def _finish(fig, ax, key, title, subtitle):
    """Poner título (hallazgo), subtítulo y nota al pie con limitación y fuente."""
    fig.suptitle(title, x=0.01, ha="left", fontsize=_TITLE_SIZE, fontweight="bold")
    ax.set_title(subtitle, loc="left", fontsize=_SUBTITLE_SIZE, color=_MUTED)
    note = f"Limitación: {CHART_LIMITATIONS[key]}\n{DATA_SOURCE}"
    fig.supxlabel(note, x=0.01, ha="left", fontsize=_NOTE_SIZE, color=_MUTED)
    return fig


def _segment_legend(ax, **kwargs):
    """Agregar una leyenda con el color de cada segmento, de menor a mayor."""
    handles = [Patch(color=SEGMENT_COLORS[label]) for label in SEGMENT_LABELS]
    ax.legend(handles, SEGMENT_LABELS, title="Segmento", frameon=False, **kwargs)


def _spanish_ticks(axis, decimals=0):
    """Escribir los números de un eje con formato español."""
    axis.set_major_formatter(FuncFormatter(lambda v, _: format_number(v, decimals)))


def _label_bars(ax, container, decimals):
    """Poner el valor de cada barra con formato español."""
    labels = [format_number(value, decimals) for value in container.datavalues]
    ax.bar_label(container, labels=labels, padding=2, fontsize=_LABEL_SIZE)


def _run_subtitle(text, run):
    """Agregar el ``run_id`` de la corrida al subtítulo."""
    return f"{text} · corrida {run[RUN_ID_COLUMN]}"


# --- Gráficos de los datos -----------------------------------------------------


def _price_edges(prices, low, high):
    """Armar los bordes del histograma con los umbrales como bordes.

    Son ``PRICE_BINS`` intervalos de igual ancho más los umbrales que caen
    dentro del rango, así que ninguna barra queda partida entre dos segmentos.
    """
    edges = np.linspace(prices.min(), prices.max(), PRICE_BINS + 1)
    inside = [value for value in (low, high) if edges[0] < value < edges[-1]]
    return np.unique(np.concatenate([edges, inside]))


def _price_axes(ax, thresholds):
    """Marcar los umbrales y rotular los ejes del histograma de precios."""
    for threshold in thresholds:
        ax.axvline(threshold, color=_LINE_COLOR, linestyle="--", linewidth=1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(f"Precio publicado ({PRICE_UNIT})")
    ax.set_ylabel("Cantidad de avisos")
    _spanish_ticks(ax.xaxis)
    _spanish_ticks(ax.yaxis)
    _segment_legend(ax, loc="upper right")


def _price_title(prices, low, high):
    """Armar el título con la mediana del precio y los dos umbrales."""
    return (
        f"La mitad de los avisos cuesta menos de {format_number(prices.median())} "
        f"{PRICE_UNIT}; los segmentos cortan en {format_number(low)} "
        f"y {format_number(high)}"
    )


def plot_price_distribution(clean, run):
    """Graficar el histograma del precio por m² coloreado por segmento.

    Parameters
    ----------
    clean : pandas.DataFrame
        Tabla limpia (Tabla 1).
    run : pandas.Series
        Fila de la corrida, con ``umbral_bajo`` y ``umbral_alto``.

    Returns
    -------
    matplotlib.figure.Figure
        ``PRICE_BINS`` barras de igual ancho, partidas en los umbrales (cada
        barra queda entera en un segmento y lleva su color), y líneas
        punteadas en los dos umbrales.
    """
    low, high = _thresholds(run)
    prices = clean[TARGET_SOURCE].dropna()
    counts, edges = np.histogram(prices, bins=_price_edges(prices, low, high))
    colors = [SEGMENT_COLORS[s] for s in segment_of(edges[:-1], low, high)]
    fig, ax = _new_chart()
    ax.bar(edges[:-1], counts, width=np.diff(edges), color=colors, align="edge")
    _price_axes(ax, (low, high))
    subtitle = "Avisos por precio en USD/m²; las líneas marcan los terciles de train"
    return _finish(fig, ax, PRICE_CHART, _price_title(prices, low, high), subtitle)


def _extremes(medians):
    """Elegir los ``TOP_BARRIOS`` más caros y más baratos, sin repetir."""
    if len(medians) <= 2 * TOP_BARRIOS:
        return medians
    return pd.concat([medians.head(TOP_BARRIOS), medians.tail(TOP_BARRIOS)])


def _barrio_title(shown):
    """Armar el título con la razón entre el barrio más caro y el más barato."""
    if shown.empty:
        return f"Ningún barrio llega a {format_number(MIN_BARRIO_LISTINGS)} avisos"
    top, bottom = shown.iloc[0], shown.iloc[-1]
    ratio = format_number(top["mediana"] / bottom["mediana"], _RATIO_DECIMALS)
    return (
        f"En {top['barrio']} el m² cuesta {ratio} veces más "
        f"que en {bottom['barrio']}"
    )


def _barrio_subtitle(omitted):
    """Describir qué barrios se muestran y cuántos se omiten entre ambos grupos."""
    if not omitted:
        return (
            f"Barrios con al menos {format_number(MIN_BARRIO_LISTINGS)} avisos; "
            "color = segmento de su mediana"
        )
    return (
        f"Los {TOP_BARRIOS} barrios más caros y los {TOP_BARRIOS} más baratos; "
        f"la línea punteada separa ambos grupos\n({omitted} barrios omitidos). "
        "Color = segmento de su mediana"
    )


def _barrio_axes(ax, omitted):
    """Ordenar, separar y rotular los ejes del gráfico de barrios."""
    ax.invert_yaxis()
    if omitted:
        ax.axhline(TOP_BARRIOS - 0.5, color=_LINE_COLOR, linestyle="--", linewidth=1)
    ax.margins(x=_VALUE_MARGIN)
    ax.set_xlim(left=0)
    ax.set_xlabel(f"Mediana del precio publicado ({PRICE_UNIT})")
    _spanish_ticks(ax.xaxis)
    ax.set_ylabel("Barrio")
    _segment_legend(ax, loc="lower right")


def plot_price_by_barrio(clean, run):
    """Graficar la mediana del precio por m² de los barrios extremos.

    Parameters
    ----------
    clean : pandas.DataFrame
        Tabla limpia (Tabla 1).
    run : pandas.Series
        Fila de la corrida, con ``umbral_bajo`` y ``umbral_alto``.

    Returns
    -------
    matplotlib.figure.Figure
        Barras horizontales (el más caro arriba) coloreadas por el segmento
        de la mediana, con una línea punteada entre los más caros y los más
        baratos si se omiten barrios intermedios; sin barras si ningún barrio
        tiene avisos suficientes.
    """
    medians = barrio_medians(clean, run)
    shown = _extremes(medians)
    omitted = len(medians) - len(shown)
    fig, ax = _new_chart()
    colors = [SEGMENT_COLORS[s] for s in shown["segmento"]]
    bars = ax.barh(shown["barrio"], shown["mediana"], color=colors)
    _label_bars(ax, bars, 0)
    _barrio_axes(ax, omitted)
    subtitle = _barrio_subtitle(omitted)
    return _finish(fig, ax, BARRIO_CHART, _barrio_title(shown), subtitle)


def _map_title(shift):
    """Armar el título del mapa con el desplazamiento norte-sur en km."""
    direction = "al norte" if shift >= 0 else "al sur"
    km = format_number(abs(shift), _KM_DECIMALS)
    return (
        f"Los avisos de segmento {HIGH_SEGMENT} están, en promedio, {km} km más "
        f"{direction} que los {LOW_SEGMENT}s"
    )


def _map_aspect(latitudes):
    """Calcular el aspecto que iguala 1 km en ambos ejes a esa latitud.

    Un grado de longitud mide ``cos(latitud)`` veces un grado de latitud.
    """
    if latitudes.empty:
        return 1.0
    return float(1 / np.cos(np.radians(latitudes.mean())))


def _map_axes(ax, latitudes):
    """Fijar el aspecto del mapa, ocultar las coordenadas y poner la leyenda."""
    ax.set_aspect(_map_aspect(latitudes))
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    ax.set_xlabel("Longitud (oeste a la izquierda, este a la derecha)")
    ax.set_ylabel("Latitud (norte arriba)")
    _segment_legend(ax, loc="lower left")


def plot_location_map(clean, run):
    """Graficar la ubicación de una muestra de avisos, coloreada por segmento.

    Parameters
    ----------
    clean : pandas.DataFrame
        Tabla limpia (Tabla 1); se omiten los avisos sin ubicación.
    run : pandas.Series
        Fila de la corrida, con ``umbral_bajo`` y ``umbral_alto``.

    Returns
    -------
    matplotlib.figure.Figure
        Una sola dispersión longitud/latitud de hasta ``MAP_SAMPLE`` avisos
        en orden aleatorio reproducible (ningún segmento tapa a otro), con
        el color de su segmento y aspecto proporcional a la distancia.
    """
    located = _located(clean)
    sample = located.sample(n=min(MAP_SAMPLE, len(located)), random_state=RANDOM_STATE)
    segments = segment_of(sample[TARGET_SOURCE], *_thresholds(run))
    fig, ax = _new_chart(_MAP_FIGSIZE)
    ax.scatter(
        sample["longitud"],
        sample["latitud"],
        s=_MAP_POINT_SIZE,
        c=[SEGMENT_COLORS[segment] for segment in segments],
        alpha=_MAP_ALPHA,
        linewidths=0,
    )
    _map_axes(ax, sample["latitud"])
    subtitle = f"Muestra de {format_number(len(sample))} avisos; color = segmento"
    title = _map_title(north_shift_km(clean, run))
    return _finish(fig, ax, MAP_CHART, title, subtitle)


# --- Gráficos de los modelos ---------------------------------------------------


def _metrics_title(table, chosen):
    """Armar el título que compara el modelo elegido con el baseline.

    Raises
    ------
    ValueError
        Si faltan las métricas de test del modelo elegido o del baseline.
    """
    missing = [model for model in (chosen, BASELINE_NAME) if model not in table.index]
    if missing:
        raise ValueError(
            f"Faltan las métricas de test de {', '.join(missing)}: el título "
            "compara el modelo elegido con el baseline."
        )
    best = format_number(table.loc[chosen, REFIT_METRIC], _METRIC_DECIMALS)
    base = format_number(table.loc[BASELINE_NAME, REFIT_METRIC], _METRIC_DECIMALS)
    metric = METRIC_LABELS[REFIT_METRIC]
    return (
        f"{_model_label(chosen)} logra {metric} {best} en test, "
        f"frente a {base} del baseline"
    )


def _grouped_bars(ax, table):
    """Dibujar una barra por modelo (fila) dentro de cada métrica (columna)."""
    positions = np.arange(len(table.columns))
    width = _BAR_WIDTH / len(table)
    for i, (model, values) in enumerate(table.iterrows()):
        offset = (i - (len(table) - 1) / 2) * width
        bars = ax.bar(
            positions + offset,
            values.to_numpy(),
            width,
            color=MODEL_COLORS[model],
            label=_model_label(model),
        )
        _label_bars(ax, bars, _METRIC_DECIMALS)


def _metrics_axes(ax, metrics):
    """Rotular los ejes y la leyenda del gráfico de métricas."""
    positions = np.arange(len(metrics))
    ax.set_xticks(positions, labels=[METRIC_LABELS[m] for m in metrics])
    ax.set_ylim(0, 1)
    _spanish_ticks(ax.yaxis, _TICK_DECIMALS)
    ax.set_xlabel("Métrica (precisión, recall y F1: promedio macro)")
    ax.set_ylabel("Valor en test")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)


def plot_metrics(metrics, run):
    """Graficar las métricas de test de cada modelo, agrupadas por métrica.

    Parameters
    ----------
    metrics : pandas.DataFrame
        Tabla larga de métricas, como la devuelve ``read_table``.
    run : pandas.Series
        Fila de la corrida, con ``modelo_elegido``.

    Returns
    -------
    matplotlib.figure.Figure
        Barras verticales (x = métrica, una por modelo) con eje de 0 a 1.

    Raises
    ------
    ValueError
        Si faltan las métricas de test del modelo elegido o del baseline.
    """
    table = metrics_on_test(metrics)
    title = _metrics_title(table, run["modelo_elegido"])
    fig, ax = _new_chart()
    _grouped_bars(ax, table)
    _metrics_axes(ax, table.columns)
    subtitle = _run_subtitle("Métricas sobre el conjunto de test", run)
    return _finish(fig, ax, METRICS_CHART, title, subtitle)


def _gap_title(gap):
    """Armar el título según la mayor diferencia, en módulo, entre test y CV."""
    metric = METRIC_LABELS[REFIT_METRIC]
    distance = gap["diferencia"].abs()
    amount = format_number(distance.max(), _METRIC_DECIMALS)
    if distance.max() <= CV_TEST_TOLERANCE:
        return (
            f"La CV anticipa bien el {metric} de test: difieren a lo sumo "
            f"{amount} en todos los modelos"
        )
    worst = distance.idxmax()
    direction = "más bajo" if gap.loc[worst, "diferencia"] < 0 else "más alto"
    return (
        f"{_model_label(worst)}: el {metric} de test difiere {amount} del de "
        f"la CV ({direction} en test)"
    )


def _gap_bars(ax, gap):
    """Dibujar por modelo una barra clara de CV (media ± desvío) y una de test."""
    positions = np.arange(len(gap))
    width = _BAR_WIDTH / 2
    colors = [MODEL_COLORS[model] for model in gap.index]
    cv_bars = ax.bar(
        positions - width / 2,
        gap["cv"],
        width,
        yerr=gap["desvio_cv"],
        capsize=4,
        color=colors,
        alpha=_CV_ALPHA,
    )
    cv_labels = [
        f"{format_number(mean, _GAP_DECIMALS)}\n± {format_number(std, _GAP_DECIMALS)}"
        for mean, std in zip(gap["cv"], gap["desvio_cv"])
    ]
    ax.bar_label(cv_bars, labels=cv_labels, padding=2, fontsize=_LABEL_SIZE)
    test_bars = ax.bar(positions + width / 2, gap["test"], width, color=colors)
    _label_bars(ax, test_bars, _GAP_DECIMALS)


def _gap_axes(ax, models):
    """Rotular los ejes y la leyenda del gráfico de CV contra test."""
    ax.set_xticks(np.arange(len(models)), labels=[_model_label(m) for m in models])
    ax.margins(y=_VALUE_MARGIN)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Modelo")
    ax.set_ylabel(f"{METRIC_LABELS[REFIT_METRIC]} (promedio macro)")
    _spanish_ticks(ax.yaxis, _TICK_DECIMALS)
    handles = [
        Patch(color=_MUTED, alpha=_CV_ALPHA, label="CV (media ± desvío)"),
        Patch(color=_MUTED, label="Test"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1), frameon=False)


def plot_cv_vs_test(metrics, run):
    """Graficar la métrica de selección en la CV (con desvío) y en test.

    Parameters
    ----------
    metrics : pandas.DataFrame
        Tabla larga de métricas, como la devuelve ``read_table``.
    run : pandas.Series
        Fila de la corrida.

    Returns
    -------
    matplotlib.figure.Figure
        Por modelo, una barra clara de CV con su media ± desvío entre folds
        (en la etiqueta y como barra de error) y una barra sólida de test.
    """
    gap = cv_test_gap(metrics)
    fig, ax = _new_chart()
    _gap_bars(ax, gap)
    _gap_axes(ax, gap.index)
    metric = METRIC_LABELS[REFIT_METRIC]
    subtitle = _run_subtitle(
        f"{metric} macro: CV en train, media ± desvío entre folds (barra clara), "
        "y test (barra sólida)",
        run,
    )
    return _finish(fig, ax, GAP_CHART, _gap_title(gap), subtitle)


def _row_percentages(matrix):
    """Pasar los conteos a porcentaje de cada fila (fila sin casos = 0)."""
    values = matrix.to_numpy(dtype=float)
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(
        _PERCENT * values, totals, out=np.zeros_like(values), where=totals > 0
    )


def _confusion_title(matrix, model):
    """Armar el título con el segmento de menor acierto (recall por fila)."""
    recall = row_recall(matrix)
    if not (recall < 1).any():
        return f"{_model_label(model)}: acierta todos los segmentos en test"
    hardest = recall.idxmin()
    accuracy = format_number(_PERCENT * recall[hardest])
    return (
        f"{_model_label(model)}: {hardest} es el segmento más difícil "
        f"({accuracy} % de acierto)"
    )


def _confusion_subtitle(matrix):
    """Describir la matriz y, si hay errores, cuántos son entre vecinos."""
    text = "Porcentaje de cada segmento real (conteo entre paréntesis), en test"
    values = matrix.to_numpy()
    if np.trace(values) == values.sum():
        return text
    share = format_number(_PERCENT * neighbor_error_share(matrix))
    return f"{text};\nel {share} % de los errores es entre segmentos vecinos"


def _annotate_cells(ax, matrix, percentages):
    """Escribir en cada celda el porcentaje de la fila y el conteo."""
    counts = matrix.to_numpy()
    for (i, j), pct in np.ndenumerate(percentages):
        color = "white" if pct >= _DARK_CELL else _LINE_COLOR
        text = (
            f"{format_number(pct, _CELL_DECIMALS)} %\n"
            f"({format_number(counts[i, j])})"
        )
        ax.text(j, i, text, ha="center", va="center", color=color, fontsize=_CELL_SIZE)


def plot_confusion(matrix, model):
    """Graficar la matriz de confusión en porcentaje de cada fila.

    Parameters
    ----------
    matrix : pandas.DataFrame
        Conteos (filas = real, columnas = predicho), como los devuelve
        ``read_confusion``.
    model : str
        Nombre del modelo.

    Returns
    -------
    matplotlib.figure.Figure
        Mapa de calor del porcentaje de cada fila (1 decimal), con los
        conteos anotados.
    """
    percentages = _row_percentages(matrix)
    fig, ax = _new_chart()
    ax.imshow(percentages, cmap=_CONFUSION_CMAP, vmin=0, vmax=_PERCENT)
    _annotate_cells(ax, matrix, percentages)
    ticks = np.arange(len(SEGMENT_LABELS))
    ax.set_xticks(ticks, labels=list(SEGMENT_LABELS))
    ax.set_yticks(ticks, labels=list(SEGMENT_LABELS))
    ax.set_xlabel("Segmento predicho")
    ax.set_ylabel("Segmento real")
    title = _confusion_title(matrix, model)
    return _finish(fig, ax, CONFUSION_CHART, title, _confusion_subtitle(matrix))


def _weight_bars(ax, table, colors, decimals):
    """Dibujar barras horizontales de pesos, la primera fila arriba."""
    labels = [feature_label(variable) for variable in table["variable"]]
    bars = ax.barh(labels, table["valor"], color=colors)
    _label_bars(ax, bars, decimals)
    ax.invert_yaxis()
    ax.margins(x=_VALUE_MARGIN)
    ax.set_ylabel("Variable")


def _location_legend(ax):
    """Agregar la leyenda que distingue la ubicación del resto de las variables."""
    handles = [
        Patch(color=MODEL_COLORS[IMPORTANCE_MODEL], label="Ubicación"),
        Patch(color=NEUTRAL_COLOR, label="Otras variables"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)


def plot_importances(weights, run):
    """Graficar la importancia del Random Forest por variable original.

    Parameters
    ----------
    weights : pandas.DataFrame
        Tabla de pesos, como la devuelve ``read_table``.
    run : pandas.Series
        Fila de la corrida.

    Returns
    -------
    matplotlib.figure.Figure
        Barras horizontales, la mayor arriba y el eje desde 0; las variables
        de ubicación con el color del modelo y el resto en gris.
    """
    importances = importance_by_variable(weights)
    colors = [
        MODEL_COLORS[IMPORTANCE_MODEL] if v in LOCATION_FEATURES else NEUTRAL_COLOR
        for v in importances["variable"]
    ]
    fig, ax = _new_chart()
    _weight_bars(ax, importances, colors, _IMPORTANCE_DECIMALS)
    ax.set_xlim(left=0)
    ax.set_xlabel("Importancia (reducción media de impureza)")
    _spanish_ticks(ax.xaxis, _IMPORTANCE_TICK_DECIMALS)
    _location_legend(ax)
    subtitle = _run_subtitle(
        "Importancia por variable original (barrio y comuna suman sus columnas "
        "one-hot)",
        run,
    )
    return _finish(fig, ax, IMPORTANCE_CHART, _importance_title(importances), subtitle)


def _importance_title(importances):
    """Armar el título con la fracción de la importancia que aporta la ubicación."""
    share = format_number(_PERCENT * location_share(importances))
    return (
        f"La ubicación concentra el {share} % de la importancia del "
        f"{_model_label(IMPORTANCE_MODEL)}"
    )


def _coefficient_title(contrast):
    """Armar el título (asociativo) con la variable de mayor contraste."""
    high, low = COEF_CONTRAST
    label = feature_label(contrast["variable"].iloc[0]).lower()
    direction = "mayor" if contrast["valor"].iloc[0] > 0 else "menor"
    return (
        f"Más {label} se asocia con {direction} probabilidad de {high} " f"que de {low}"
    )


def plot_coefficients(weights, run):
    """Graficar el contraste Alta - Accesible de las variables numéricas.

    Parameters
    ----------
    weights : pandas.DataFrame
        Tabla de pesos, como la devuelve ``read_table``.
    run : pandas.Series
        Fila de la corrida.

    Returns
    -------
    matplotlib.figure.Figure
        Barras divergentes (``POSITIVE_COLOR`` y ``NEGATIVE_COLOR``), la de
        mayor módulo arriba, y una línea vertical en 0.
    """
    contrast = numeric_contrast(weights)
    colors = [POSITIVE_COLOR if v > 0 else NEGATIVE_COLOR for v in contrast["valor"]]
    fig, ax = _new_chart()
    _weight_bars(ax, contrast, colors, _METRIC_DECIMALS)
    ax.axvline(0, color=_LINE_COLOR, linewidth=1)
    high, low = COEF_CONTRAST
    ax.set_xlabel(f"Diferencia de log-odds {high} − {low} por desvío estándar")
    _spanish_ticks(ax.xaxis, _TICK_DECIMALS)
    subtitle = _run_subtitle(
        "Regresión logística: variables numéricas estandarizadas", run
    )
    return _finish(fig, ax, COEF_CHART, _coefficient_title(contrast), subtitle)


# --- Datos desde la base y archivos ----------------------------------------------


@dataclass(frozen=True)
class ChartData:
    """Datos de una corrida para armar los gráficos, leídos de la base.

    Attributes
    ----------
    run : pandas.Series
        Fila de ``ejecuciones`` de la corrida.
    clean : pandas.DataFrame
        Tabla limpia (Tabla 1).
    metrics : pandas.DataFrame
        Métricas de CV y test de la corrida.
    weights : pandas.DataFrame
        Importancias y coeficientes de la corrida.
    confusions : dict of str to pandas.DataFrame
        Matriz de confusión de cada modelo con predicciones.
    """

    run: pd.Series
    clean: pd.DataFrame
    metrics: pd.DataFrame
    weights: pd.DataFrame
    confusions: dict


def _read_run(conn, run_id):
    """Leer la fila de la corrida ``run_id`` (la más reciente si es None).

    Raises
    ------
    ValueError
        Si no hay corridas o no existe ``run_id``.
    """
    if run_id is None:
        run_id = latest_run_id(conn)
    if run_id is None:
        raise ValueError("No hay corridas en la base. Ejecutar scripts/entrenar.py.")
    runs = read_table(conn, RUNS_TABLE, run_id=run_id)
    if runs.empty:
        raise ValueError(f"No existe la corrida {run_id!r} en la base.")
    return runs.iloc[0]


def _has_table(conn, table):
    """Indicar si la base tiene la tabla ``table`` (sin crear nada)."""
    sql = "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?"
    return conn.execute(sql, (table,)).fetchone() is not None


def _read_weights(conn, run_id):
    """Leer los pesos de la corrida, que no pueden faltar.

    Raises
    ------
    ValueError
        Si la base no tiene la tabla de pesos (es anterior a ella) o la
        corrida no tiene pesos.
    """
    if not _has_table(conn, IMPORTANCE_TABLE):
        raise ValueError(
            f"La base no tiene la tabla {IMPORTANCE_TABLE} (es de una versión "
            "anterior). Volver a ejecutar scripts/entrenar.py."
        )
    weights = read_table(conn, IMPORTANCE_TABLE, run_id=run_id)
    if weights.empty:
        raise ValueError(
            f"La corrida {run_id!r} no tiene pesos en {IMPORTANCE_TABLE}. "
            "Volver a ejecutar scripts/entrenar.py."
        )
    return weights


def load_chart_data(conn, run_id=None):
    """Leer de la base todo lo que necesitan los gráficos de una corrida.

    Parameters
    ----------
    conn : sqlite3.Connection
        Conexión abierta; no se modifica la base.
    run_id : str, optional
        Corrida a graficar; por defecto, la más reciente.

    Returns
    -------
    ChartData
        Corrida, tabla limpia, métricas, pesos y matrices de confusión.

    Raises
    ------
    ValueError
        Si no hay corridas, no existe ``run_id``, la base no tiene la tabla
        de pesos o la corrida no tiene pesos.
    """
    run = _read_run(conn, run_id)
    run_id = run[RUN_ID_COLUMN]
    weights = _read_weights(conn, run_id)
    metrics = read_table(conn, METRICS_TABLE, run_id=run_id)
    models = _ordered_models(metrics["modelo"])
    confusions = {model: read_confusion(conn, run_id, model) for model in models}
    logger.info("Datos de la corrida %s leídos: %d modelos", run_id, len(models))
    return ChartData(
        run=run,
        clean=read_table(conn, CLEAN_TABLE),
        metrics=metrics,
        weights=weights,
        confusions=confusions,
    )


def build_figures(data):
    """Armar todos los gráficos de una corrida.

    Parameters
    ----------
    data : ChartData
        Salida de ``load_chart_data``.

    Returns
    -------
    dict of str to matplotlib.figure.Figure
        ``{nombre de archivo: figura}``: un gráfico por tema y una matriz de
        confusión (``matriz_confusion_<modelo>``) por cada modelo real.
    """
    figures = {
        PRICE_CHART: plot_price_distribution(data.clean, data.run),
        BARRIO_CHART: plot_price_by_barrio(data.clean, data.run),
        MAP_CHART: plot_location_map(data.clean, data.run),
        METRICS_CHART: plot_metrics(data.metrics, data.run),
        GAP_CHART: plot_cv_vs_test(data.metrics, data.run),
        IMPORTANCE_CHART: plot_importances(data.weights, data.run),
        COEF_CHART: plot_coefficients(data.weights, data.run),
    }
    for model, matrix in data.confusions.items():
        if model != BASELINE_NAME:
            figures[f"{CONFUSION_CHART}_{model}"] = plot_confusion(matrix, model)
    return figures


def save_figures(figures, out_dir):
    """Guardar cada figura como ``out_dir/<nombre>.<FIGURE_FORMAT>``.

    Parameters
    ----------
    figures : dict of str to matplotlib.figure.Figure
        Salida de ``build_figures``.
    out_dir : str or pathlib.Path
        Carpeta de salida; se crea si no existe.

    Returns
    -------
    list of pathlib.Path
        Rutas de los archivos escritos.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fig in figures.items():
        path = out_dir / f"{name}.{FIGURE_FORMAT}"
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info("Gráfico guardado: %s", path)
        paths.append(path)
    return paths
