"""Step 11: association-rule mining (FP-Growth) - which flow properties go together, and which point to an attack?

Usage (from the project root, with the virtual environment active):

    python scripts/11_association_rules.py
    python scripts/11_association_rules.py --sample-size 20000 --min-support 0.01   # quick run

Data: the CLEANED data in its original units (bytes, microseconds, ports), so that rules read like
``{port=FTP, Init_Win_bytes_backward=low} -> class=FTP-Patator``.  Rows are split with the same seed
and stratification as the main pipeline; only the TRAINING part is used to learn the bins and mine
rules.  A class-capped sample keeps the small classes visible.  The rule list is then used as a
simple classifier on a class-capped sample of the TEST part.

Outputs (reports/mining/association/): association_report.md / .json, general_rules.csv,
class_rules.csv, rule_classifier_per_class.csv

Exit codes: 0 done; 1 data missing.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from config import settings
from nids.experiments.common import per_class_f1
from nids.mining import association as R
from nids.mining.reporting import markdown_table, write_report
from nids.preprocessing.splitter import stratified_split
from nids.training.data_access import class_capped_indices
from nids.utils.io_utils import read_parquet
from nids.utils.logger import configure_logging, get_logger

logger = get_logger("scripts.association_rules")
OUT = settings.MINING_REPORT_DIR / "association"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sample-size", type=int, default=settings.MINING_SAMPLE_SIZE)
    parser.add_argument("--min-support", type=float, default=settings.RULE_MIN_SUPPORT)
    parser.add_argument("--min-confidence", type=float, default=settings.RULE_MIN_CONFIDENCE)
    parser.add_argument("--class-support", type=float, default=0.3, help="minimum support inside a class for class rules")
    args = parser.parse_args(argv)
    configure_logging()

    columns = [*R.RULE_FEATURES, *R.FLAG_FEATURES, R.PORT_COLUMN, settings.LABEL_COLUMN]
    try:
        df = read_parquet(settings.CLEANED_DATA_FILE, columns=columns, hint="run scripts/03_clean_data.py")
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    labels = df[settings.LABEL_COLUMN].astype(str)
    codes = pd.factorize(labels, sort=True)[0]
    train_pos, test_pos = stratified_split(labels.to_numpy())  # identical to the main pipeline's split
    train = df.iloc[train_pos[class_capped_indices(codes[train_pos], args.sample_size)]]
    test = df.iloc[test_pos[class_capped_indices(codes[test_pos], args.sample_size // 2, seed=settings.RANDOM_SEED + 1)]]
    print(f"Mining sample: {len(train):,} training flows; rule-classifier test sample: {len(test):,} test flows")

    encoder = R.ItemEncoder().fit(train)
    items = encoder.transform(train)
    print(f"{items.shape[1]} distinct items, {items.sum(axis=1).mean():.1f} items per flow")

    itemsets, rules = R.mine_rules(items, min_support=args.min_support, min_confidence=args.min_confidence)
    general = R.describe_rules(rules[~rules["consequents"].map(lambda s: any(i.startswith(R.CLASS_PREFIX) for i in s)) & ~rules["antecedents"].map(lambda s: any(i.startswith(R.CLASS_PREFIX) for i in s))])
    to_class = R.describe_rules(R.rules_with_class_consequent(rules))
    car = R.class_rules(items, within_class_support=args.class_support, min_confidence=args.min_confidence)

    OUT.mkdir(parents=True, exist_ok=True)
    general.head(5000).to_csv(OUT / "general_rules.csv", index=False)  # top 5,000 by lift (the full list is huge)
    to_class.to_csv(OUT / "fpgrowth_class_rules.csv", index=False)
    R.describe_rules(car).to_csv(OUT / "class_rules.csv", index=False)

    # rules as a classifier on held-out test flows
    test_items = encoder.transform(test, with_class=False)
    classifier = R.RuleClassifier(car)
    predicted = classifier.predict(test_items)
    truth = test[settings.LABEL_COLUMN].astype(str).to_numpy()
    class_list = sorted(set(truth) | set(car["class"]))
    pc = per_class_f1(pd.Series(truth).map({c: i for i, c in enumerate(class_list)}).to_numpy(), pd.Series(predicted).map({c: i for i, c in enumerate(class_list)}).to_numpy(), class_list)
    pc.to_csv(OUT / "rule_classifier_per_class.csv", index=False)
    macro = float(f1_score(truth, predicted, labels=class_list, average="macro", zero_division=0))

    car_view = car[["class", "rule", "class_coverage", "confidence", "lift"]].copy()
    car_view["rule"] = car_view["rule"].str.replace(R.CLASS_PREFIX, "", regex=False)
    best_per_class = car_view.groupby("class", sort=False).head(2)
    no_rule = sorted(set(truth) - set(car["class"]))

    md = [
        "# Association rules (FP-Growth)",
        "",
        f"Transactions: {len(train):,} class-capped flows of the **training** part (same stratified split and seed as the "
        f"main pipeline), {items.shape[1]} items. Numeric features are cut into {settings.RULE_BINS} quantile bins "
        "learnt on these flows (`=0` is its own item), flags become `<flag>=1`, destination ports are named by service.",
        "",
        "## 1. General rules between flow properties",
        "",
        f"FP-Growth: {len(itemsets):,} frequent itemsets at minimum support {args.min_support:g}, "
        f"{len(general):,} rules with confidence >= {args.min_confidence:g} and no class item. Top 20 by lift:",
        "",
        markdown_table(general.head(20), digits=3),
        "",
        "The highest-lift rules mostly restate one quantity through two features (a total length of 0 implies a mean "
        "length of 0); they confirm the redundancy the correlation filter of step 4 removes, rather than reveal new behaviour. "
        "The class rules below are the informative part.",
        "",
        "## 2. Rules that point to a class",
        "",
        f"A global minimum support hides small classes, so the itemsets were also mined **inside each class** "
        f"(at least {args.class_support:.0%} of that class) and kept when their confidence on all flows is >= {args.min_confidence:g}. "
        "`class_coverage` = share of the class the rule covers (recall); `confidence` = share of covered flows that "
        "really are that class (precision); `lift` = confidence / overall share of the class.",
        "",
        markdown_table(best_per_class, digits=3),
        "",
        f"Classes with no rule at this confidence: {', '.join(no_rule) if no_rule else 'none'}. "
        "Their flows share their typical values with other classes, so no short rule separates them.",
        "",
        f"Rules found by plain FP-Growth with a class on the right: {len(to_class):,} (file `fpgrowth_class_rules.csv`).",
        "",
        "## 3. The rules as a classifier",
        "",
        f"{len(car)} class rules, ordered by confidence then coverage; first matching rule wins; no match -> BENIGN. "
        f"Scored on {len(test):,} class-capped **test** flows (never used for mining).",
        "",
        f"Macro F1: **{macro:.3f}** (XGBoost on all features: about 0.92 on the full test set). Per class:",
        "",
        markdown_table(pc, digits=3),
        "",
        "## Limits",
        "",
        "* Rules depend on the binning; three quantile bins are coarse by design (readable rules).",
        "* The sample is class-capped: support and lift are measured on it, not on real traffic proportions.",
        "* Port items make many rules easy (FTP-Patator uses port 21); this is the same lab-setup effect as the "
        "Destination Port feature (see `reports/experiments/port_ablation/`).",
    ]
    summary = {
        "transactions": len(train),
        "items": int(items.shape[1]),
        "frequent_itemsets": int(len(itemsets)),
        "general_rules": int(len(general)),
        "class_rules": int(len(car)),
        "classes_without_rule": no_rule,
        "rule_classifier_macro_f1": macro,
        "rule_classifier_test_rows": len(test),
        "bin_edges": {k: v.tolist() for k, v in encoder.edges_.items()},
    }
    md_path, _ = write_report(OUT, "association_report", "\n".join(md), summary)
    print(markdown_table(best_per_class, digits=3))
    print(f"\nRule classifier macro F1 on {len(test):,} test flows: {macro:.3f}")
    print(f"Report: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
