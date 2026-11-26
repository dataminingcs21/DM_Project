"""Small helpers shared by the data mining and experiment scripts (09-17).

* ``markdown_table`` renders a DataFrame as a GitHub table without extra dependencies.
* ``save_figure`` writes a matplotlib figure (Agg backend, no window) and closes it.
* ``write_report`` writes a Markdown report and its JSON twin next to each other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # must run before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nids.utils.io_utils import write_json  # noqa: E402


def _cell(value: Any, digits: int) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(df: pd.DataFrame, digits: int = 4, max_rows: int | None = None, index: bool = False) -> str:
    """A GitHub-flavoured Markdown table for ``df`` (floats rounded to ``digits``)."""
    frame = df.reset_index() if index else df
    if max_rows is not None and len(frame) > max_rows:
        frame = frame.head(max_rows)
    header = "| " + " | ".join(_cell(str(c), digits) for c in frame.columns) + " |"
    align = "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(frame[c]) else "---" for c in frame.columns) + "|"
    rows = ["| " + " | ".join(_cell(v, digits) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, align, *rows])


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 130) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def write_report(directory: str | Path, name: str, markdown: str, summary: dict[str, Any]) -> tuple[Path, Path]:
    """Write ``<name>.md`` and ``<name>.json`` into ``directory``; return both paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    md_path = directory / f"{name}.md"
    md_path.write_text(markdown, encoding="utf-8")
    json_path = write_json(summary, directory / f"{name}.json")
    return md_path, json_path


def class_names(codec: Any) -> list[str]:
    """Class names of a ``LabelCodec`` in code order."""
    return [str(c) for c in codec.decode(np.arange(codec.n_classes))]
