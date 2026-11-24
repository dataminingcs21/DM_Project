"""Check a typed flow (or an uploaded table of flows) before it reaches the models.

Everything here is decided from ``feature_schema.json`` (built from the TRAINING rows only):
which features the models need, which values are impossible, and which values lie outside
what the models have seen.  Nothing is invented and nothing is repaired silently.

What counts as what
-------------------
Errors (the flow cannot be predicted; the row is marked invalid)
    * a feature the models use is missing or blank (``missing="error"``, the default);
    * a value is not a plain number (text, a boolean, ``1,000``, ``1_000``);
    * a value is negative where the cleaning step treated negatives as impossible
      (``settings.NON_NEGATIVE_COLUMNS``: durations, rates, header lengths, ...);
    * a value breaks a protocol limit listed in ``HARD_LIMITS`` (a TCP/UDP port above 65535);
    * +/- infinity or NaN in a column where the training data never had such values.

Warnings (the flow is predicted, but the user is told)
    * a value outside the minimum..maximum seen in training (the models extrapolate; this is
      NOT evidence of an attack);
    * a fractional value in a column whose training values were all whole numbers;
    * a missing / infinite value in a column where the training data also had such values
      (the pipeline replaces it by the training median, exactly as in training);
    * a column that is not a known feature (with a close-name suggestion);
    * ``missing="median"``: a missing value filled by the training median (named, never hidden).

Info
    * columns that were given but are not used by the models (constant or correlated with a kept
      column in training): they are ignored, so they need not be supplied at all.

The models use only ``model_feature_names``; the other raw columns do not influence them, so
they are not required.  The returned frame has ALL raw columns in the pipeline's order (unused
ones are NaN and are imputed and dropped by the pipeline), ready for ``pipeline.transform``.

What it does not do: transform, scale or predict.
"""

from __future__ import annotations

import difflib
import sys
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np
import pandas as pd

from nids.preprocessing import feature_schema as feature_schema_module
from nids.utils.logger import get_logger

logger = get_logger(__name__)

ERROR, WARNING, INFO = "error", "warning", "info"
MISSING_POLICIES = ("error", "median")
MAX_LISTED_ISSUES = 500  # the counts are always complete; only the listed detail is capped

# Values that are impossible because of how the protocol is defined, not because of what the training data
# contained: {feature name: (lowest, highest, reason)}.  Deliberately short: a limit belongs here only if it is
# true by definition.  A name that is not a feature of the schema is ignored.  Not part of the cleaning or
# preprocessing settings, so changing it never makes saved data or models stale.
HARD_LIMITS: dict[str, tuple[float, float, str]] = {
    "Destination Port": (0.0, 65535.0, "a TCP/UDP port is a 16-bit number (0 to 65535)"),
}


@dataclass(frozen=True)
class Issue:
    """One finding.  ``row`` is 1-based (1 = first row given); ``None`` for a finding about a column."""

    level: str
    code: str
    message: str
    field: str | None = None
    row: int | None = None
    value: str | None = None

    def __str__(self) -> str:
        where = f"row {self.row}: " if self.row is not None else ""
        return f"[{self.level}] {where}{self.message}"


@dataclass
class ValidationResult:
    """Outcome of a validation.

    ``frame``       all raw feature columns in the pipeline's order, float64; cells that are missing,
                    non-finite or invalid are NaN
    ``valid_mask``  True for rows without any error
    ``issues``      the findings (at most ``MAX_LISTED_ISSUES`` listed)
    ``counts``      number of findings per code, complete even when the list is capped
    ``column_ok``   False when a column-level error makes the whole input unusable
    """

    frame: pd.DataFrame
    valid_mask: np.ndarray
    issues: list[Issue] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    truncated: int = 0
    column_ok: bool = True
    missing_policy: str = "error"
    required: list[str] = field(default_factory=list)
    unused_given: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return int(len(self.frame))

    @property
    def n_valid(self) -> int:
        return int(self.valid_mask.sum())

    @property
    def ok(self) -> bool:
        """True when there is at least one row and every row can be predicted."""
        return self.column_ok and self.n_rows > 0 and bool(self.valid_mask.all())

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == WARNING]

    @property
    def valid_frame(self) -> pd.DataFrame:
        """Only the rows without errors (empty when a column-level error makes the input unusable)."""
        return self.frame[self.valid_mask] if self.column_ok else self.frame.iloc[0:0]

    def summary(self) -> dict[str, Any]:
        """JSON-safe overview for logs and the interface."""
        return {
            "n_rows": self.n_rows,
            "n_valid_rows": self.n_valid if self.column_ok else 0,
            "ok": self.ok,
            "missing_policy": self.missing_policy,
            "errors": sum(n for c, n in self.counts.items() if c in _ERROR_CODES),
            "warnings": sum(n for c, n in self.counts.items() if c in _WARNING_CODES),
            "counts": dict(self.counts),
            "issues_listed": len(self.issues),
            "issues_not_listed": self.truncated,
            "unexpected_columns": list(self.unexpected),
            "unused_columns_given": list(self.unused_given),
        }


_ERROR_CODES = {"no_rows", "duplicate_column", "missing_column", "missing", "non_finite", "not_a_number", "negative", "impossible_value"}
_WARNING_CODES = {
    "missing_column_filled", "missing_filled", "missing_like_training", "non_finite_filled", "non_finite_like_training",
    "out_of_range", "not_whole", "unexpected_column",
}


# ---------------------------------------------------------------------------
# Reading cells
# ---------------------------------------------------------------------------
def _to_numbers(series: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(values, blank, bad)`` for one column.

    ``values`` float64 (NaN where blank or bad); ``blank`` marks missing cells (None, NaN, empty text,
    the text "nan"); ``bad`` marks cells that are present but are not a plain number.
    Infinity is returned as infinity (the caller decides what it means).
    """
    n = len(series)
    values = np.full(n, np.nan)
    blank = np.zeros(n, dtype=bool)
    bad = np.zeros(n, dtype=bool)
    if pd.api.types.is_bool_dtype(series.dtype):
        bad[:] = True
        return values, blank, bad
    if pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_complex_dtype(series.dtype):
        arr = series.to_numpy(dtype="float64", na_value=np.nan)
        blank = np.isnan(arr)
        values = np.where(blank, np.nan, arr)
        return values, blank, bad
    for i, cell in enumerate(series.astype(object).to_numpy()):
        if cell is None or cell is pd.NA or (isinstance(cell, float) and cell != cell):
            blank[i] = True
        elif isinstance(cell, (bool, np.bool_)):
            bad[i] = True
        elif isinstance(cell, (int, float, np.integer, np.floating)):
            try:
                values[i] = float(cell)
            except OverflowError:
                bad[i] = True
        elif isinstance(cell, str):
            text = cell.strip()
            if text == "":
                blank[i] = True
                continue
            if "_" in text:  # Python would read "1_000" as 1000; a typed flow should not
                bad[i] = True
                continue
            try:
                number = float(text)
            except ValueError:
                bad[i] = True
                continue
            if np.isnan(number):
                blank[i] = True
            else:
                values[i] = number
        else:
            bad[i] = True
    return values, blank, bad


def _shown(cell: Any) -> str:
    text = repr(cell) if isinstance(cell, str) else str(cell)
    return text if len(text) <= 40 else text[:37] + "..."


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------
def required_features(schema: Mapping[str, Any]) -> list[str]:
    """The features the models use, in model order."""
    return list(schema["model_feature_names"])


def field_specs(schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One description per model feature (for building the input form): name, typical values, limits."""
    by_name = {f["name"]: f for f in schema["features"]}
    keys = ("name", "position", "min", "p01", "median", "p99", "max", "all_whole_numbers", "must_be_non_negative", "non_negative_reason", "n_missing_or_infinite_in_train")
    return [{k: by_name[name].get(k) for k in keys} for name in required_features(schema)]


def typical_flow(schema: Mapping[str, Any]) -> dict[str, float]:
    """Each model feature set to its training median: a valid starting point for a form.

    It is a per-feature median, not a real flow, and the combination may not occur in real traffic.
    """
    return {s["name"]: s["median"] for s in field_specs(schema) if s["median"] is not None}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class _Collector:
    """Counts every finding; lists at most ``MAX_LISTED_ISSUES`` of them."""

    def __init__(self) -> None:
        self.issues: list[Issue] = []
        self.counts: Counter[str] = Counter()
        self.truncated = 0

    def add(self, level: str, code: str, message: str, field_name: str | None = None, row: int | None = None, value: str | None = None) -> None:
        self.counts[code] += 1
        if len(self.issues) < MAX_LISTED_ISSUES:
            self.issues.append(Issue(level, code, message, field_name, row, value))
        else:
            self.truncated += 1

    def add_rows(self, level: str, code: str, rows: np.ndarray, field_name: str, text: Any, cells: pd.Series) -> None:
        """One finding per row in ``rows`` (positions); ``text(shown_value)`` builds the message."""
        self.counts[code] += int(len(rows))
        room = MAX_LISTED_ISSUES - len(self.issues)
        for position in rows[: max(room, 0)]:
            shown = _shown(cells.iloc[int(position)])
            self.issues.append(Issue(level, code, text(shown), field_name, int(position) + 1, shown))
        self.truncated += max(int(len(rows)) - max(room, 0), 0)


def validate_frame(data: pd.DataFrame, schema: Mapping[str, Any] | None = None, missing: str = "error") -> ValidationResult:
    """Validate a table of flows (one row per flow, one column per feature name).

    ``missing="error"`` (default): a blank required value is an error.
    ``missing="median"``: it is filled by the training median and reported as a warning.
    """
    if missing not in MISSING_POLICIES:
        raise ValueError(f"missing must be one of {MISSING_POLICIES}, got {missing!r}")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame (use validate_flow for a dict)")
    schema = schema if schema is not None else feature_schema_module.load_feature_schema()

    raw = list(schema["raw_feature_names"])
    required = set(required_features(schema))
    specs = {f["name"]: f for f in schema["features"]}
    n = int(len(data))
    out_frame = pd.DataFrame(np.full((n, len(raw)), np.nan), columns=raw, index=data.index)
    valid = np.ones(n, dtype=bool)
    log = _Collector()
    column_ok = True

    def finish(unused: list[str] | None = None, unexpected: list[str] | None = None) -> ValidationResult:
        result = ValidationResult(
            frame=out_frame, valid_mask=valid, issues=log.issues, counts=dict(log.counts), truncated=log.truncated,
            column_ok=column_ok, missing_policy=missing, required=sorted(required, key=raw.index),
            unused_given=unused or [], unexpected=unexpected or [],
        )
        logger.debug("Input validated: %d row(s), %d valid, %s", result.n_rows, result.n_valid if column_ok else 0, dict(log.counts) or "no findings")
        return result

    if n == 0:
        log.add(ERROR, "no_rows", "no flow was given")
        return finish()

    names = [str(c) for c in data.columns]
    seen: set[str] = set()
    keep: list[tuple[str, int]] = []  # (name, column position) of the first occurrence of each name
    for position, name in enumerate(names):
        if name in seen:
            log.add(ERROR, "duplicate_column", f"the column '{name}' appears more than once", name)
            column_ok = False
        else:
            seen.add(name)
            keep.append((name, position))
    unexpected = [name for name, _ in keep if name not in specs]
    for name in unexpected:
        close = difflib.get_close_matches(name, raw, n=1, cutoff=0.8)
        hint = f" (did you mean '{close[0]}'?)" if close else ""
        log.add(WARNING, "unexpected_column", f"'{name}' is not a known feature and is ignored{hint}", name)

    absent = [name for name in raw if name in required and name not in seen]
    for name in absent:
        if missing == "error":
            log.add(ERROR, "missing_column", f"the feature '{name}' is required by the models and was not given", name)
            column_ok = False
        else:
            log.add(WARNING, "missing_column_filled", f"the feature '{name}' was not given; the training median is used", name)

    unused_given: list[str] = []
    for name, position in keep:
        if name not in specs:
            continue
        spec = specs[name]
        j = raw.index(name)
        cells = data.iloc[:, position]
        values, blank, bad = _to_numbers(cells)
        infinite = ~blank & ~bad & ~np.isfinite(values)
        finite = ~blank & ~bad & ~infinite

        if name not in required:
            unused_given.append(name)
            out_frame.iloc[:, j] = np.where(finite, values, np.nan)
            continue

        seen_in_training = (spec.get("n_missing_or_infinite_in_train") or 0) > 0
        rows = np.flatnonzero(bad)
        if rows.size:
            log.add_rows(ERROR, "not_a_number", rows, name, lambda v, name=name: f"'{name}': {v} is not a number", cells)
            valid[rows] = False
        for mask, kind, what in ((blank, "missing", "is missing"), (infinite, "non_finite", "is infinite")):
            rows = np.flatnonzero(mask)
            if not rows.size:
                continue
            if seen_in_training:
                log.add_rows(WARNING, f"{kind}_like_training", rows, name,
                             lambda v, name=name, what=what: f"'{name}' {what}; the training data also had such values here, so the training median is used", cells)
            elif missing == "median" and kind == "missing":
                log.add_rows(WARNING, "missing_filled", rows, name, lambda v, name=name: f"'{name}' is missing; the training median is used", cells)
            else:
                log.add_rows(ERROR, kind, rows, name,
                             lambda v, name=name, what=what: f"'{name}' {what}" + ("" if kind == "missing" else " (the training data never had infinite values in this feature)"), cells)
                valid[rows] = False

        good = finite.copy()
        if spec.get("must_be_non_negative"):
            rows = np.flatnonzero(finite & (values < 0))
            if rows.size:
                reason = spec.get("non_negative_reason") or "negative values are impossible here"
                log.add_rows(ERROR, "negative", rows, name, lambda v, name=name, reason=reason: f"'{name}' = {v} is negative: {reason}", cells)
                valid[rows] = False
                good[rows] = False
        limits = HARD_LIMITS.get(name)
        if limits is not None:
            rows = np.flatnonzero(good & ((values < limits[0]) | (values > limits[1])))
            if rows.size:
                log.add_rows(ERROR, "impossible_value", rows, name,
                             lambda v, name=name, limits=limits: f"'{name}' = {v} is impossible: {limits[2]}", cells)
                valid[rows] = False
                good[rows] = False
        out_frame.iloc[:, j] = np.where(good, values, np.nan)

        low, high = spec.get("min"), spec.get("max")
        if low is not None and high is not None:
            outside = np.flatnonzero(good & ((values < low) | (values > high)))
            if outside.size:
                log.add_rows(WARNING, "out_of_range", outside, name,
                             lambda v, name=name, low=low, high=high: f"'{name}' = {v} is outside the range seen in training ({low:g} to {high:g}); the models extrapolate (this alone is not evidence of an attack)", cells)
        if spec.get("all_whole_numbers"):
            fractional = np.flatnonzero(good & (values != np.round(values)))
            if fractional.size:
                log.add_rows(WARNING, "not_whole", fractional, name,
                             lambda v, name=name: f"'{name}' = {v} is not a whole number; every training value of this feature was whole", cells)

    if unused_given:
        log.add(INFO, "unused_given", f"{len(unused_given)} column(s) given are not used by the models and are ignored: " + ", ".join(unused_given[:5]) + (" ..." if len(unused_given) > 5 else ""))
    return finish(unused_given, unexpected)


def validate_flow(values: Mapping[str, Any], schema: Mapping[str, Any] | None = None, missing: str = "error") -> ValidationResult:
    """Validate one flow given as ``{feature name: value}``."""
    if not isinstance(values, Mapping):
        raise TypeError("values must be a mapping {feature name: value}")
    result = validate_frame(pd.DataFrame([dict(values)], dtype=object), schema=schema, missing=missing)
    result.issues = [replace(i, row=None) for i in result.issues]  # one flow: "row 1" would only confuse
    return result


def issues_table(result: ValidationResult) -> pd.DataFrame:
    """The listed findings as a table (level, row, field, code, message, value)."""
    return pd.DataFrame(
        [{"level": i.level, "row": i.row, "field": i.field, "code": i.code, "message": i.message, "value": i.value} for i in result.issues],
        columns=["level", "row", "field", "code", "message", "value"],
    )


# ---------------------------------------------------------------------------
# Demonstration on the real schema:  python -m nids.prediction.input_validator
# ---------------------------------------------------------------------------
def main() -> int:
    try:
        schema = feature_schema_module.load_feature_schema()
    except (FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1
    specs = field_specs(schema)
    print(f"Schema: {schema['n_raw_features']} raw features, {schema['n_model_features']} used by the models, computed on {schema['n_train_rows']:,} training rows.")
    print(f"Features that must be non-negative: {sum(1 for s in specs if s['must_be_non_negative'])} of the {len(specs)} the models use; "
          f"whole-number features: {sum(1 for s in specs if s['all_whole_numbers'])}.")
    print()

    typical = typical_flow(schema)
    cases: list[tuple[str, dict[str, Any], str]] = [("typical flow (training medians)", dict(typical), "error")]
    first = specs[0]["name"]
    bad_text = {**typical, first: "abc"}
    cases.append((f"'{first}' typed as text", bad_text, "error"))
    cases.append((f"'{first}' left out", {k: v for k, v in typical.items() if k != first}, "error"))
    cases.append((f"'{first}' left out, missing='median'", {k: v for k, v in typical.items() if k != first}, "median"))
    far = next((s for s in specs if s["max"] is not None and s["name"] not in HARD_LIMITS), None)
    if far:
        cases.append((f"'{far['name']}' ten times the training maximum", {**typical, far["name"]: (far["max"] or 1.0) * 10 + 10}, "error"))
    neg = next((s for s in specs if s["must_be_non_negative"]), None)
    if neg:
        cases.append((f"'{neg['name']}' = -5", {**typical, neg["name"]: -5}, "error"))
    for limited, (_, high_limit, _) in HARD_LIMITS.items():
        if limited in typical:
            cases.append((f"'{limited}' above its protocol limit", {**typical, limited: high_limit + 1}, "error"))
    cases.append(("a misspelled extra column", {**typical, first + "x": 1}, "error"))

    for title, flow, policy in cases:
        result = validate_flow(flow, schema, missing=policy)
        print(f"- {title}: ok={result.ok}, errors={result.summary()['errors']}, warnings={result.summary()['warnings']}")
        for issue in result.issues:
            print(f"    {issue}")
    return 0


if __name__ == "__main__":
    sys.exit(main())