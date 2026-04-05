```{code-cell} ipython3
:tags: [remove-input]
# Shared setup for every book chapter. Chapters `{include}` this snippet
# instead of repeating the same path manipulation, imports, and rpy2
# helpers. Page-specific imports still go in the chapter body.
import sys
from pathlib import Path

_book_dir = next(
    c for c in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
    if (c / "book" / "book_utils.py").exists()
) / "book"
if str(_book_dir) not in sys.path:
    sys.path.insert(0, str(_book_dir))

import book_utils
ROOT = book_utils.configure()

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
from rpy2 import robjects as ro  # noqa: E402

from book_utils import (  # noqa: E402
    NEUTRAL_COLOR,
    PYTHON_COLOR,
    R_COLOR,
    coefficient_bar,
    parity_scatter,
    r_assign_dataframe,
    r_load_library,
    r_named_vector,
    r_numeric,
    risk_path_chart,
)
```
