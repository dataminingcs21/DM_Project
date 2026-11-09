"""Train and evaluate a K-Nearest Neighbors (KNN) classifier on CIC-IDS2017.

KNN has no training phase: it stores the training rows and, for every test flow,
searches its k nearest neighbours.  Prediction cost grows with
(training rows x test rows), so this script works on a stratified SAMPLE of the
training rows (--train-rows) and can also sample the test rows (--test-rows).

Everything else (cleaning, scaling, optional top-k / PCA, optional sampler,
metrics, saving) is done by src/training.py, so the result lands in the same
results/tables/experiments.csv as the SVM runs.

Examples
    python -m src.train_knn --max-rows 50000                       # quick check
    python -m src.train_knn --k 5
    python -m src.train_knn --k 11 --weights distance --p 1        # Manhattan distance
    python -m src.train_knn --train-rows 100000 --test-rows 100000
    python -m src.train_knn --pca 0.95                             # faster, often as good
    python -m src.train_knn --imbalance undersample
    python -m src.train_knn --task multiclass --imbalance smote
    python -m src.train_knn --tune                                 # search k / weights / p
"""
import argparse

import numpy as np
from sklearn.model_selection import train_test_split

from . import config as C
from .training import load_xy, run_experiment, tune

K_GRID = [3, 5, 7, 11, 15]
WEIGHTS_GRID = ["uniform", "distance"]
P_GRID = [1, 2]                      # 1 = Manhattan, 2 = Euclidean
DEFAULT_TRAIN_ROWS = 50_000


def build_config(k=5, weights="uniform", p=2, task="binary", imbalance="none",
                 max_rows=None, top_k=None, pca=None, seed=None):
    """Default config + the KNN choices.  Returns the config dict."""
    cfg = C.load_config()
    cfg["model"] = {"name": "knn", "params": {"n_neighbors": k, "weights": weights, "p": p}}
    cfg["task"] = task
    cfg["imbalance"] = imbalance
    cfg["max_rows"] = max_rows
    cfg["features"]["top_k"] = top_k
    cfg["features"]["pca"] = pca
    if seed is not None:
        cfg["seed"] = seed
    cfg["run_name"] = f"knn_{task}_k{k}_{weights}_p{p}_{imbalance}"
    return cfg


def search_space(imbalance="none"):
    """Parameters tried by --tune.  Names get the 'clf__' prefix only when the
    estimator sits inside an imblearn pipeline (undersample / smote)."""
    pre = "clf__" if imbalance in ("undersample", "smote") else ""
    return {pre + "n_neighbors": K_GRID, pre + "weights": WEIGHTS_GRID, pre + "p": P_GRID}


def sample_rows(X, y, n, seed):
    """Stratified sample of n rows (everything if n is None or large enough)."""
    if n is None or n >= len(X):
        return X, y
    try:
        idx, _ = train_test_split(np.arange(len(X)), train_size=int(n), stratify=y, random_state=seed)
    except ValueError:                      # a class too small to stratify: plain random sample
        idx = np.random.default_rng(seed).choice(len(X), size=int(n), replace=False)
    idx = np.sort(idx)
    return X[idx], y[idx]


def limit_rows(data, train_rows, test_rows, seed):
    """Return a copy of the data dict with fewer training (and optionally test) rows."""
    d = dict(data)
    d["X_train"], d["y_train"] = sample_rows(data["X_train"], data["y_train"], train_rows, seed)
    d["X_test"], d["y_test"] = sample_rows(data["X_test"], data["y_test"], test_rows, seed)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=5, help="number of neighbours")
    ap.add_argument("--weights", choices=WEIGHTS_GRID, default="uniform", help="distance = closer neighbours count more")
    ap.add_argument("--p", type=int, choices=P_GRID, default=2, help="1 = Manhattan, 2 = Euclidean")
    ap.add_argument("--task", choices=["binary", "multiclass", "family"], default="binary")
    ap.add_argument("--imbalance", default="none", choices=["none", "undersample", "smote"],
                    help="KNN has no class_weight option")
    ap.add_argument("--train-rows", type=int, default=DEFAULT_TRAIN_ROWS,
                    help=f"training rows kept (default {DEFAULT_TRAIN_ROWS}); 0 = all rows (very slow)")
    ap.add_argument("--test-rows", type=int, default=None,
                    help="evaluate on a stratified sample of the test rows (default: all test rows)")
    ap.add_argument("--max-rows", type=int, default=None, help="debug: sample of the whole dataset")
    ap.add_argument("--top-k", type=int, default=None, help="keep the k most important features")
    ap.add_argument("--pca", type=float, default=None, help="PCA: 0.95 = keep 95 percent of variance")
    ap.add_argument("--tune", action="store_true", help="random search over k / weights / p first")
    ap.add_argument("--n-iter", type=int, default=6, help="--tune: number of settings tried")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--rebuild", action="store_true", help="redo the cleaning cache")
    a = ap.parse_args(argv)

    C.ensure_dirs()
    cfg = build_config(a.k, a.weights, a.p, a.task, a.imbalance, a.max_rows, a.top_k, a.pca, a.seed)
    if a.pca is not None and a.pca >= 1:
        cfg["features"]["pca"] = int(a.pca)

    data = load_xy(cfg, rebuild=a.rebuild)
    data = limit_rows(data, a.train_rows or None, a.test_rows, cfg["seed"])
    print(f"[train_knn] training rows: {len(data['X_train'])}, test rows: {len(data['X_test'])}")

    if a.tune:
        best, table = tune(cfg, search_space(a.imbalance), n_iter=a.n_iter, data=data,
                           sample=min(len(data["X_train"]), 20_000),
                           out_file=C.ROOT / "configs" / "best_knn_params.json")
        print("Best parameters:", best)
        print(table[["params", "mean_test_score"]].head(5).to_string(index=False))
        for key, val in best.items():
            cfg["model"]["params"][key.split("__")[-1]] = val
        cfg["run_name"] = "knn_{}_k{}_{}_p{}_{}".format(
            a.task, cfg["model"]["params"]["n_neighbors"], cfg["model"]["params"]["weights"],
            cfg["model"]["params"]["p"], a.imbalance)

    res = run_experiment(cfg, data=data)
    row = res["row"]
    keys = ["run", "n_train", "n_test", "n_features_used", "recall", "precision", "f1",
            "pr_auc", "macro_f1", "false_alarm_rate", "fit_seconds", "predict_seconds_per_1k"]
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k in keys})
    print(f"Model saved in {C.MODELS_DIR / (res['name'] + '.joblib')}")


if __name__ == "__main__":
    main()
