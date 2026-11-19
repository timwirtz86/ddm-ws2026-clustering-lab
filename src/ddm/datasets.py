"""Course datasets: registry and loaders.

The raw files in ``data/raw/`` are kept exactly as published by their source,
with their original quirks (whitespace-separated values, ``?`` for missing
values, ``;`` as separator, categories stored as integer codes). The loaders
below only *parse* the files into a DataFrame with readable column names and
``NaN`` for missing values. They deliberately do **not** clean, encode or scale
anything: that is your job in the notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata of one course dataset."""

    name: str
    title: str
    file: str
    rows: int
    question: str
    label_candidates: tuple[str, ...]
    source: str
    license: str
    citation: str
    pitfalls: tuple[str, ...] = field(default_factory=tuple)


DATASETS: dict[str, DatasetInfo] = {
    "penguins": DatasetInfo(
        name="penguins",
        title="Palmer Archipelago penguins",
        file="penguins.csv",
        rows=344,
        question="Do body measurements alone separate the penguins into species?",
        label_candidates=("species", "sex"),
        source="https://allisonhorst.github.io/palmerpenguins/",
        license="CC0 1.0",
        citation=(
            "Horst AM, Hill AP, Gorman KB (2020). palmerpenguins: Palmer "
            "Archipelago (Antarctica) penguin data. doi:10.5281/zenodo.3960218"
        ),
        pitfalls=(
            "Measurements in mm and g live on very different scales.",
            "A few rows have missing measurements; 'sex' has more gaps.",
            "'species' holds the known groups: leave it out of the features.",
        ),
    ),
    "wholesale": DatasetInfo(
        name="wholesale",
        title="Wholesale customers (Portugal)",
        file="wholesale_customers.csv",
        rows=440,
        question="Which customer segments exist in the annual spending profiles?",
        label_candidates=("channel", "region"),
        source="https://archive.ics.uci.edu/dataset/292/wholesale+customers",
        license="CC BY 4.0",
        citation=(
            "Cardoso, M. (2013). Wholesale customers [Dataset]. UCI Machine "
            "Learning Repository. https://doi.org/10.24432/C5030X"
        ),
        pitfalls=(
            "Spending is extremely right-skewed: a few huge customers dominate.",
            "'channel' and 'region' are integer codes, but nominal attributes.",
        ),
    ),
    "auto_mpg": DatasetInfo(
        name="auto_mpg",
        title="Auto MPG (cars 1970-1982)",
        file="auto-mpg.data",
        rows=398,
        question="Which kinds of cars exist, and do they follow their origin?",
        label_candidates=("origin",),
        source="https://archive.ics.uci.edu/dataset/9/auto+mpg",
        license="CC BY 4.0",
        citation=(
            "Quinlan, R. (1993). Auto MPG [Dataset]. UCI Machine Learning "
            "Repository. https://doi.org/10.24432/C5859H"
        ),
        pitfalls=(
            "Whitespace-separated file without header; '?' marks missing horsepower.",
            "'origin' is coded 1/2/3 (USA/Europe/Japan): nominal, not numeric.",
            "'car_name' is free text; the brand can be extracted from it.",
        ),
    ),
    "students": DatasetInfo(
        name="students",
        title="Student performance (math course, Portugal)",
        file="student-mat.csv",
        rows=395,
        question="Which student profiles exist, and how do they relate to grades?",
        label_candidates=("school", "G3"),
        source="https://archive.ics.uci.edu/dataset/320/student+performance",
        license="CC BY 4.0",
        citation=(
            "Cortez, P. (2008). Student Performance [Dataset]. UCI Machine "
            "Learning Repository. https://doi.org/10.24432/C5TG7T"
        ),
        pitfalls=(
            "33 attributes of every type: binary, nominal, ordinal and numeric.",
            "Separator is ';'. Codes are explained in data/raw/student.txt.",
            "Grades G1-G3 are strongly correlated: they count three times.",
        ),
    ),
}

_AUTO_COLUMNS = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
    "car_name",
]


def list_datasets() -> pd.DataFrame:
    """Return an overview table of all course datasets."""
    rows = [
        {
            "name": d.name,
            "title": d.title,
            "rows": d.rows,
            "question": d.question,
            "label candidates": ", ".join(d.label_candidates),
            "license": d.license,
        }
        for d in DATASETS.values()
    ]
    return pd.DataFrame(rows).set_index("name")


def dataset_info(name: str) -> DatasetInfo:
    """Return the metadata of one dataset (source, license, pitfalls, ...)."""
    try:
        return DATASETS[name]
    except KeyError:
        known = ", ".join(DATASETS)
        raise KeyError(f"Unknown dataset '{name}'. Known datasets: {known}") from None


def load_dataset(name: str) -> pd.DataFrame:
    """Parse one raw dataset into a DataFrame.

    Only parsing happens here: readable column names, ``NaN`` for missing
    values. No type conversion, no encoding, no scaling.
    """
    info = dataset_info(name)
    path = RAW_DIR / info.file

    if name == "penguins":
        df = pd.read_csv(path, na_values=["NA"])
    elif name == "wholesale":
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"delicassen": "delicatessen"})
    elif name == "auto_mpg":
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=_AUTO_COLUMNS,
            na_values=["?"],
            quotechar='"',
        )
    elif name == "students":
        df = pd.read_csv(path, sep=";")
    else:  # pragma: no cover - guarded by dataset_info
        raise KeyError(name)

    if len(df) != info.rows:
        raise ValueError(
            f"{name}: expected {info.rows} rows, found {len(df)}. "
            "Is data/raw/ complete and unchanged?"
        )
    return df


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Small helper: dtype, missing values and number of distinct values per column."""
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "distinct": df.nunique(),
            "example": df.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan),
        }
    )
