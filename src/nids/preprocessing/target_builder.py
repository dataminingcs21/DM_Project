"""Build the target (what the models must predict) from the cleaned labels.

Two modes, chosen by ``settings.TARGET_MODE``:

* ``multiclass``: BENIGN plus each attack type (15 classes in CIC-IDS2017).
  With ``RARE_CLASS_POLICY = "merge"`` the classes below the technical minimum
  are merged into one class (``RARE_CLASS_MERGE_LABEL``); with ``"keep"`` they stay.
* ``binary``: BENIGN versus ``settings.MALICIOUS_LABEL`` ("MALICIOUS").

Labels are then turned into integers by :class:`LabelCodec`.  The order is fixed
and documented: **BENIGN is always index 0, the other classes follow in
alphabetical order**.  This matters because the majority-vote tie rule prefers
an attack over BENIGN and then the lowest index.

Nothing here learns from the data (class names are not a leak), so it is safe to
run before the split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from config import settings
from nids.utils.logger import get_logger

logger = get_logger("preprocessing.target_builder")

ATTACK_LABEL = settings.MALICIOUS_LABEL  # the single positive class in binary mode ("MALICIOUS")


# ---------------------------------------------------------------------------
# Label <-> integer codec
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LabelCodec:
    """Fixed mapping between class names and integer codes (saved with joblib)."""

    classes: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("class names must be unique")
        if not self.classes:
            raise ValueError("at least one class is required")

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def benign_index(self) -> int | None:
        return self.classes.index(settings.BENIGN_LABEL) if settings.BENIGN_LABEL in self.classes else None

    def encode(self, labels: pd.Series | list[str] | np.ndarray) -> np.ndarray:
        """Names -> integer codes (int64). Unknown names raise ValueError (never guessed)."""
        values = pd.Series(labels)
        lookup = {name: i for i, name in enumerate(self.classes)}
        codes = values.map(lookup)
        if codes.isna().any():
            unknown = sorted(values[codes.isna()].astype(str).unique())
            raise ValueError(f"unknown class name(s): {unknown}. Known: {list(self.classes)}")
        return codes.to_numpy(dtype="int64")

    def decode(self, codes: np.ndarray | list[int]) -> np.ndarray:
        """Integer codes -> names. Out-of-range codes raise ValueError."""
        arr = np.asarray(codes, dtype="int64")
        if arr.size and (arr.min() < 0 or arr.max() >= self.n_classes):
            raise ValueError(f"class codes must be in 0..{self.n_classes - 1}")
        return np.asarray(self.classes, dtype=object)[arr]

    def to_dict(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.classes)}


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def to_binary(labels: pd.Series) -> pd.Series:
    """BENIGN stays BENIGN; every other label becomes ``ATTACK_LABEL`` (``settings.MALICIOUS_LABEL``)."""
    return labels.where(labels == settings.BENIGN_LABEL, ATTACK_LABEL)


def merge_rare_classes(
    labels: pd.Series,
    min_samples: int | None = None,
    merge_label: str | None = None,
) -> tuple[pd.Series, list[str]]:
    """Merge the classes with fewer than ``min_samples`` rows into one class.

    BENIGN is never merged.  Returns the new labels and the merged class names.
    Raises ValueError if the merged class would itself be below the minimum
    (merging would then not solve the problem).
    """
    min_samples = settings.MIN_CLASS_SAMPLES if min_samples is None else min_samples
    merge_label = settings.RARE_CLASS_MERGE_LABEL if merge_label is None else merge_label

    counts = labels.value_counts()
    rare = sorted(c for c, n in counts.items() if n < min_samples and c != settings.BENIGN_LABEL)
    if not rare:
        return labels, []
    merged_total = int(counts[rare].sum()) + int(counts.get(merge_label, 0))
    if merged_total < min_samples:
        raise ValueError(
            f"the merged class '{merge_label}' would have only {merged_total} rows (minimum {min_samples}); "
            "merging does not help. Remove these classes or change the settings."
        )
    return labels.where(~labels.isin(rare), merge_label), rare


def build_target(
    labels: pd.Series,
    mode: str | None = None,
    rare_policy: str | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Turn the cleaned labels into the final class names for the chosen mode.

    Returns ``(target, report)``; ``target`` has the same index as ``labels``.
    """
    mode = settings.TARGET_MODE if mode is None else mode
    rare_policy = settings.RARE_CLASS_POLICY if rare_policy is None else rare_policy
    if mode not in settings.ALLOWED_TARGET_MODES:
        raise ValueError(f"mode must be one of {settings.ALLOWED_TARGET_MODES}, got {mode!r}")
    if rare_policy not in settings.ALLOWED_RARE_CLASS_POLICIES:
        raise ValueError(f"rare_policy must be one of {settings.ALLOWED_RARE_CLASS_POLICIES}, got {rare_policy!r}")
    if labels.isna().any():
        raise ValueError("labels contain missing values; run the cleaner first")
    if settings.BENIGN_LABEL not in set(labels.unique()):
        raise ValueError(f"no '{settings.BENIGN_LABEL}' rows found: check the labels")

    labels = labels.astype(str)
    counts_before = labels.value_counts().to_dict()
    merged: list[str] = []

    if mode == "binary":
        target = to_binary(labels)
    elif rare_policy == "merge":
        target, merged = merge_rare_classes(labels)
    else:
        target = labels

    counts_after = target.value_counts()
    below_min = sorted(c for c, n in counts_after.items() if n < settings.MIN_CLASS_SAMPLES)
    report = {
        "mode": mode,
        "rare_class_policy": rare_policy if mode == "multiclass" else "not applicable (binary)",
        "n_rows": int(len(target)),
        "n_classes": int(counts_after.size),
        "counts_before": {str(k): int(v) for k, v in counts_before.items()},
        "counts_after": {str(k): int(v) for k, v in counts_after.items()},
        "merged_classes": merged,
        "classes_below_min_class_samples": below_min,
        "min_class_samples_rule": int(settings.MIN_CLASS_SAMPLES),
    }
    if below_min:
        logger.warning("Classes below %d rows: %s", settings.MIN_CLASS_SAMPLES, below_min)
    logger.info("Target built: mode=%s, %d classes, %d rows", mode, report["n_classes"], report["n_rows"])
    return target, report


def make_codec(target: pd.Series) -> LabelCodec:
    """Fixed class order: BENIGN first, then the other classes alphabetically."""
    names = sorted(set(target.astype(str)) - {settings.BENIGN_LABEL})
    ordered = ([settings.BENIGN_LABEL] if settings.BENIGN_LABEL in set(target) else []) + names
    return LabelCodec(tuple(ordered))