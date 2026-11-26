"""Association-rule mining on network flows (FP-Growth).

Each flow becomes a *transaction*, a set of items such as ``port=HTTP``, ``Flow Duration=high``,
``PSH Flag=1`` and ``class=DoS Hulk``.  Two kinds of rules are mined:

1. **General rules** (``mine_rules``): FP-Growth (mlxtend) on all transactions, then every rule
   with enough confidence.  These describe how flow properties co-occur.
2. **Class association rules** (``class_rules``): ``{items} -> class=C``.  A global minimum support
   would hide the tiny classes (Heartbleed has 11 flows), so frequent itemsets are mined
   *inside each class* (support relative to the class), and their confidence and lift are then
   measured on all transactions.  This is the CBA idea (classification based on associations).

The rule list can be used as a simple, readable classifier (``RuleClassifier``): the first rule
(highest confidence, then support) whose items are all present decides; otherwise the default
class (BENIGN).  It is meant to show what the rules capture, not to compete with XGBoost.

Discretisation (``ItemEncoder``): bin edges are learnt on the mining rows only.  A value of
exactly 0 gets its own item (many flow features are 0 for most flows); the other values are cut
into ``settings.RULE_BINS`` quantile bins (``low`` / ``mid`` / ``high``); a feature with at most
``max_exact_values`` distinct non-zero values (e.g. Down/Up Ratio) keeps its exact values.  Flag counts are 0/1
and only their presence becomes an item.  Destination ports are named by service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth

from config import settings
from nids.utils.logger import get_logger

logger = get_logger("mining.association")

CLASS_PREFIX = "class="

# Interpretable features used as items (raw cleaned columns, original units).
RULE_FEATURES: tuple[str, ...] = (
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "Down/Up Ratio",
)
FLAG_FEATURES: tuple[str, ...] = (
    "FIN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
)
PORT_COLUMN = "Destination Port"

SERVICE_PORTS: dict[int, str] = {
    21: "FTP",
    22: "SSH",
    53: "DNS",
    80: "HTTP",
    88: "Kerberos",
    123: "NTP",
    137: "NetBIOS",
    139: "SMB",
    389: "LDAP",
    443: "HTTPS",
    444: "444",
    445: "SMB",
    465: "SMTPS",
    8080: "HTTP-alt",
}
BIN_NAMES = {2: ("low", "high"), 3: ("low", "mid", "high"), 4: ("low", "mid-low", "mid-high", "high")}


def port_item(port: int) -> str:
    if port in SERVICE_PORTS:
        return f"port={SERVICE_PORTS[port]}"
    if port < 1024:
        return "port=other well-known"
    if port < 49152:
        return "port=registered"
    return "port=ephemeral"


# ---------------------------------------------------------------------------
# Flows -> transactions
# ---------------------------------------------------------------------------
@dataclass
class ItemEncoder:
    """Turns raw flows into a one-hot item table. ``fit`` learns quantile edges; ``transform`` applies them."""

    features: tuple[str, ...] = RULE_FEATURES
    flags: tuple[str, ...] = FLAG_FEATURES
    n_bins: int = field(default_factory=lambda: settings.RULE_BINS)
    max_exact_values: int = 4
    edges_: dict[str, np.ndarray] = field(default_factory=dict)
    exact_: set[str] = field(default_factory=set)

    def fit(self, df: pd.DataFrame) -> "ItemEncoder":
        if self.n_bins not in BIN_NAMES:
            raise ValueError(f"n_bins must be one of {sorted(BIN_NAMES)}, got {self.n_bins}")
        missing = [c for c in (*self.features, *self.flags, PORT_COLUMN) if c not in df.columns]
        if missing:
            raise KeyError(f"columns missing for rule mining: {missing}")
        self.edges_ = {}
        self.exact_ = set()
        for col in self.features:
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype="float64")
            nonzero = values[np.isfinite(values) & (values != 0)]
            if np.unique(nonzero).size <= self.max_exact_values:  # few distinct values: use them as items
                self.exact_.add(col)
                self.edges_[col] = np.array([])
                continue
            inner = np.quantile(nonzero, np.linspace(0, 1, self.n_bins + 1)[1:-1]) if nonzero.size else np.array([])
            self.edges_[col] = np.unique(inner)
        return self

    def _bin(self, col: str, values: np.ndarray) -> np.ndarray:
        if col in self.exact_:
            out = np.array([f"{col}={v:g}" for v in values], dtype=object)
            out[~np.isfinite(values)] = f"{col}=missing"
            return out
        edges = self.edges_[col]
        # normal case: n_bins - 1 distinct edges -> low/mid/high; repeated quantiles leave fewer bins
        names = BIN_NAMES[self.n_bins] if len(edges) == self.n_bins - 1 else tuple(f"bin{i + 1}" for i in range(len(edges) + 1))
        labels = np.array([f"{col}={n}" for n in names], dtype=object)
        out = labels[np.searchsorted(edges, values, side="right")]
        out[values == 0] = f"{col}=0"
        out[~np.isfinite(values)] = f"{col}=missing"
        return out

    def transform(self, df: pd.DataFrame, with_class: bool = True) -> pd.DataFrame:
        """One boolean column per item (class items only when ``with_class`` and a Label column exists)."""
        if not self.edges_:
            raise RuntimeError("ItemEncoder is not fitted")
        columns: dict[str, np.ndarray] = {}

        def add(item_values: np.ndarray) -> None:
            for item in pd.unique(item_values):
                columns[str(item)] = item_values == item

        add(np.array([port_item(int(p)) for p in df[PORT_COLUMN].to_numpy()], dtype=object))
        for col in self.features:
            add(self._bin(col, pd.to_numeric(df[col], errors="coerce").to_numpy(dtype="float64")))
        for col in self.flags:
            columns[f"{col.replace(' Count', '')}=1"] = df[col].to_numpy() > 0
        if with_class and settings.LABEL_COLUMN in df.columns:
            add((CLASS_PREFIX + df[settings.LABEL_COLUMN].astype(str)).to_numpy(dtype=object))
        items = pd.DataFrame(columns, index=df.index)
        return items.reindex(sorted(items.columns), axis=1)


# ---------------------------------------------------------------------------
# Mining
# ---------------------------------------------------------------------------
def _format(itemset: frozenset) -> str:
    return "{" + ", ".join(sorted(itemset)) + "}"


def mine_rules(
    items: pd.DataFrame,
    min_support: float | None = None,
    min_confidence: float | None = None,
    max_len: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """FP-Growth frequent itemsets and the association rules above ``min_confidence``."""
    min_support = settings.RULE_MIN_SUPPORT if min_support is None else min_support
    min_confidence = settings.RULE_MIN_CONFIDENCE if min_confidence is None else min_confidence
    max_len = settings.RULE_MAX_LEN if max_len is None else max_len
    itemsets = fpgrowth(items, min_support=min_support, use_colnames=True, max_len=max_len)
    logger.info("FP-Growth: %d frequent itemsets (min support %.4f)", len(itemsets), min_support)
    if itemsets.empty:
        return itemsets, pd.DataFrame()
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets=len(items))
    rules = rules[["antecedents", "consequents", "support", "confidence", "lift", "leverage", "conviction"]]
    return itemsets, rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)


def rules_with_class_consequent(rules: pd.DataFrame) -> pd.DataFrame:
    """Only rules ``{features} -> {class=C}`` (exactly one class item on the right, none on the left)."""
    if rules.empty:
        return rules
    is_class = lambda s: any(i.startswith(CLASS_PREFIX) for i in s)  # noqa: E731
    keep = rules["consequents"].map(lambda s: len(s) == 1 and is_class(s)) & ~rules["antecedents"].map(is_class)
    return rules[keep].reset_index(drop=True)


def class_rules(
    items: pd.DataFrame,
    within_class_support: float = 0.3,
    min_confidence: float | None = None,
    max_len: int = 3,
    top_per_class: int = 5,
) -> pd.DataFrame:
    """Class association rules mined inside each class, measured on all transactions."""
    min_confidence = settings.RULE_MIN_CONFIDENCE if min_confidence is None else min_confidence
    class_cols = [c for c in items.columns if c.startswith(CLASS_PREFIX)]
    feature_items = items.drop(columns=class_cols)
    matrix = feature_items.to_numpy(dtype=bool)
    col_index = {c: i for i, c in enumerate(feature_items.columns)}
    n_total = len(items)
    rows = []
    for class_col in class_cols:
        in_class = items[class_col].to_numpy(dtype=bool)
        n_class = int(in_class.sum())
        if n_class == 0:
            continue
        frequent = fpgrowth(feature_items[in_class], min_support=within_class_support, use_colnames=True, max_len=max_len)
        candidates = []
        for itemset, class_support in zip(frequent["itemsets"], frequent["support"]):
            covered = matrix[:, [col_index[i] for i in itemset]].all(axis=1)
            coverage = int(covered.sum())
            hits = int((covered & in_class).sum())
            confidence = hits / coverage
            if confidence < min_confidence:
                continue
            candidates.append(
                {
                    "class": class_col[len(CLASS_PREFIX):],
                    "rule": f"{_format(itemset)} -> {class_col}",
                    "antecedent": frozenset(itemset),
                    "n_items": len(itemset),
                    "class_coverage": float(class_support),  # share of the class the rule covers (recall)
                    "confidence": float(confidence),  # share of covered flows that are in the class (precision)
                    "lift": float(confidence / (n_class / n_total)),
                    "support": hits / n_total,
                    "class_rows": n_class,
                }
            )
        candidates.sort(key=lambda r: (-r["confidence"], -r["class_coverage"], r["n_items"]))
        rows.extend(candidates[:top_per_class])
        logger.info("%s: %d candidate rules, kept %d", class_col, len(candidates), min(top_per_class, len(candidates)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rules as a classifier
# ---------------------------------------------------------------------------
@dataclass
class RuleClassifier:
    """Ordered rule list: the first rule whose antecedent matches decides; otherwise ``default``."""

    rules: pd.DataFrame  # needs columns antecedent (frozenset of items), class, confidence, class_coverage
    default: str = settings.BENIGN_LABEL

    def predict(self, items: pd.DataFrame) -> np.ndarray:
        ordered = self.rules.sort_values(["confidence", "class_coverage"], ascending=False)
        predicted = np.full(len(items), None, dtype=object)
        for antecedent, cls in zip(ordered["antecedent"], ordered["class"]):
            if any(i not in items.columns for i in antecedent):
                continue
            match = items[list(antecedent)].to_numpy(dtype=bool).all(axis=1) & (predicted == None)  # noqa: E711
            predicted[match] = cls
        predicted[predicted == None] = self.default  # noqa: E711
        return predicted.astype(str)


def describe_rules(rules: pd.DataFrame) -> pd.DataFrame:
    """Rules table with readable antecedent/consequent strings (for reports)."""
    if rules.empty:
        return rules
    out = rules.copy()
    for col in ("antecedents", "consequents"):
        if col in out.columns:
            out[col] = out[col].map(_format)
    return out.drop(columns=[c for c in ("antecedent",) if c in out.columns])


def rule_summary(rules: pd.DataFrame) -> dict[str, Any]:
    return {"n_rules": int(len(rules)), "classes_with_rules": sorted(rules["class"].unique().tolist()) if "class" in rules else []}
