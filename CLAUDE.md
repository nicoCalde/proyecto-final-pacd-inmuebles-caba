# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

This is the final project for *Programación Avanzada para la Ciencia de Datos* (Licenciatura en Ciencia de Datos, UNICABA). It implements **Option A, "Mercado Inmobiliario"**: classifying apartments for sale in CABA into value segments.

The official assignment is `consigna/Consigna_Proyecto_Final.pdf`, and it is the source of truth for requirements. Never modify it.

The assignment, the grading and the presentation are in Spanish. Write user-facing deliverables in Spanish: README, plot titles and labels, slides and commit messages. Code identifiers and this file stay in English.

Development follows an approved, staged implementation plan with a TDD cycle (Red → Implementation → Green → independent validation). Work one stage at a time. **The user makes every commit and push themselves.** Never run `git commit` or `git push`.

## Assignment requirements

1. **Model (scikit-learn)**: preprocessing (imputation, scaling, categorical encoding) must live inside a `sklearn.pipeline.Pipeline`. Compare at least two classifiers and tune them with `GridSearchCV` or `RandomizedSearchCV`.
2. **Evaluation**: report Accuracy, Precision, Recall, F1 and the confusion matrix for each model.
3. **Database** (SQLite) with these tables:
   - Table 1: clean, preprocessed input data
   - Table 2: predictions and global metrics
   - Table 3 (extra credit): model configuration
4. **Visualization**: matplotlib or seaborn.
5. **Deliverables**: a Git repo with structured code, a DB loading script, `requirements.txt` and `README.md`. The DB must be accessible to reviewers. There is also a deck of 6 to 8 slides.
6. **Rubric** (25% each): code quality and Git history; DB and visualizations; modeling and the methodological justification; the oral presentation.

## Methodological decisions (approved)

`docs/decisiones.md` is the decision log, written in Spanish. Each decision gets one row with its reason. Add a row whenever a new decision is made.

- **Target**: a value segment from the **terciles of `PRECIOUSDM`** (USD per m²): Accesible / Estándar / Alta. The tercile thresholds are fitted **on the train split only**.
- **No leakage**: no price-derived column may be used as a feature. That rules out `PRECIOUSD`, `PRECIOARS`, `PRECIOUSDM`, `PRECIOARSM` and `CAMBIO`. A test enforces this.
- **Models**: `LogisticRegression` and `RandomForestClassifier`, tuned with `GridSearchCV`, `StratifiedKFold(5)`, `scoring="f1_macro"` and `random_state=42`.
- **DB**: SQLite through the stdlib `sqlite3` plus `pandas.to_sql`/`read_sql`. The schema is explicit DDL with PK and FK constraints and `PRAGMA foreign_keys=ON`. Writes are idempotent and run inside transactions. No ORM, and no `SELECT *` in production code.

## Planned layout

The layout is planned, not built yet. Create each piece only in its stage.

- `src/inmuebles/`: the package. Modules: `config`, `data_loading`, `cleaning`, `target`, `preprocessing`, `modeling`, `evaluation`, `database`, `visualization`.
- `scripts/`: thin orchestrators (`cargar_bd.py`, `entrenar.py`, `generar_graficos.py`) with `main()` and `argparse`. They hold no logic of their own.
- `tests/`: pytest tests, one file per module, with shared fixtures in `conftest.py`.
- `reports/figures/`: the generated PNGs.
- `data/inmuebles.db`: the SQLite database that gets delivered.

## Datasets

The raw data lives in `datasets/`. It comes from BA Data "Departamentos en venta", license CC-BY-2.5-AR.

The files are **ESRI shapefiles, not CSV**. Each year is a folder of `.shp`/`.shx`/`.dbf`/`.prj`/`.cpg`/… files. The tabular data is in the `.dbf`, and the `.cpg` declares UTF-8 encoding. The geometries are points in the Gauss-Krüger Buenos Aires projection (`.prj`).

The schemas differ across years:

| Year | Rows | Usable fields |
| --- | --- | --- |
| 2017 | 72,124 | `M2TOTAL, M2CUB, AMBIENTES, PRECIOUSD, PRECIOUSDM, MES, BARRIO, Comuna`. No age, no bathrooms, no lat/lon columns. |
| 2018 | 84,813 | `OPERACION, TIPO, ANTIG_`(string), `M2TOTAL, M2CUB, AMBIENTES, BANOS, DIRECCION, CAMBIO`(string), prices, `LATITUD, LONGITUD, MES, BARRIO_1, COMUNA_1` |
| 2019 | 88,640 | Same as 2018, but with `ANTIG` (float), an extra `AGE` (for example `EST`), `BARRIOS_1` and `COMUNAS` |
| 2020 | 156,258 | Completely different: `Direccion, PropiedadS, Dolares, Pesos, DolaresM2, PesosM2, Ambientes, Cotizacion, Trimestre, Barrio, Comunas` |

What the schema differences imply:

- 2018 and 2019 share the richest schema, but their column names need harmonizing (`ANTIG_`→`ANTIG`, `BARRIO_1`/`BARRIOS_1`, `COMUNA_1`/`COMUNAS`), and some types need coercing.
- **GitHub rejects files over 100 MB.** The 2018 `.dbf` is 159 MB and the 2019 `.dbf` is 133 MB, so they cannot be pushed as is.

Decisions:

- **The project uses only 2018 and 2019.** Their folders are `datasets/Departamentos en Venta 2018/` and `datasets/Departamentos en venta 2019/`; note that the capitalization differs between the two.
- 2017 and 2020 were deleted: 2017 is missing features and 2020 has an incompatible schema. The table above records them only to justify that choice.
- **`datasets/` is git-ignored** because the raw files are too large for GitHub. Reviewers get the clean data from Table 1 in `data/inmuebles.db`, and the README explains how to download the raw files. Tests must use small fixtures in `tests/`, never `datasets/`.

Read the `.dbf` attribute table with `pyshp` (the `shapefile` module).

## Environment and tooling

- Windows, using Git Bash or PowerShell. The system Python 3.14 has no data libraries. Use a uv virtualenv: `uv venv --python 3.12`, then `uv pip install -r requirements.txt`.
- When printing accented text under Git Bash, set `PYTHONIOENCODING=utf-8`.
- Style rules:
  - PEP 8 first, then the Google Python Style Guide where PEP 8 is silent.
  - Docstrings use the **numpydoc** convention.
  - Tooling is `black`, `flake8`, `flake8-docstrings`, `pep8-naming` and `pytest`.
- Code size targets: units of about 15 statements or fewer, McCabe complexity of about 5 or less, and about 5 parameters or fewer per function. No magic values: constants belong in `config`.
