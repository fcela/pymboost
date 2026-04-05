from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import polars as pl


def _is_pandas_dataframe(data) -> bool:
    return data.__class__.__module__.startswith("pandas.") and data.__class__.__name__ == "DataFrame"


def _is_polars_dataframe(data) -> bool:
    return data.__class__.__module__.startswith("polars.") and data.__class__.__name__ == "DataFrame"


def get_raw_column(data, key: str):
    if _is_polars_dataframe(data):
        values = data.get_column(key).to_numpy()
    elif _is_pandas_dataframe(data):
        values = data.loc[:, key].to_numpy()
    elif isinstance(data, Mapping):
        values = np.asarray(data[key])
    else:
        try:
            values = np.asarray(data[key])
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise TypeError(
                "data must be a mapping, pandas DataFrame, or polars DataFrame"
            ) from exc
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"'{key}' must be one-dimensional")
    return array


def get_column(data, key: str) -> np.ndarray:
    array = np.asarray(get_raw_column(data, key), dtype=np.float64)
    return array


def to_formulaic_data(data):
    if _is_polars_dataframe(data) or _is_pandas_dataframe(data):
        return data
    if isinstance(data, Mapping):
        return pl.DataFrame({key: np.asarray(value) for key, value in data.items()})
    return data
