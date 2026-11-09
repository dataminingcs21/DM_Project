"""Train and evaluate a Support Vector Machine on CIC-IDS2017.

Three variants (pick with --variant):

    linear    LinearSVC.  Fast, works on all rows.                       (default)
    nystroem  Approximate RBF kernel (Nystroem map + LinearSVC).  Works on all rows.
    rbf       Exact RBF-kernel SVC.  Slow: it is trained on a stratified sample of the
              TRAINING rows (--train-rows, default 30000).  The test rows are never cut.

Everything goes through the same leak-free pipeline as the other models
(filters -> scaler -> optional top-k / PCA / LDA -> optional sampler -> SVM), so
the result is comparable with the rest of results/tables/experiments.csv.

Examples
    python -m src.train_svm --max-rows 50000                      # quick check
    python -m src.train_svm --variant linear
    python -m src.train_svm --variant nystroem --n-components 500 --C 1
    python -m src.train_svm --variant rbf --train-rows 30000 --C 10 --gamma 0.01
    python -m src.train_svm --variant linear --imbalance class_weight
    python -m src.train_svm --variant rbf --task multiclass --imbalance undersample
    python -m src.train_svm --variant linear --tune               # search C first
"""
import argparse

from . import config as C
from .training import load_xy, run_experiment, tune

MODEL_OF = {"linear": "linear_svm", "nystroem": "svm_nystroem", "rbf": "svm_rbf"}
DEFAULT_TRAIN_ROWS = {"rbf": 30_000}
C_GRID = [0.01, 0.1, 1, 10, 100]
GAMMA_GRID = [0.001, 0.01, 0.1, 1]


def build_config(variant, task="binary", C_value=None, gamma=None, n_components=None,
                 imbalance="none", train_rows=None, max_rows=None, top_k=None, pca=None, seed=None):
    """Default config + the SVM choices.  Returns the config dict."""
    cfg = C.load_config()
    params = {}
    if C_value is not None:
        params["C"] = C_value
    if gamma is not None and variant in ("rbf", "nystroem"):
        params["gamma"] = gamma
    if n_components is not None and variant == "nystroem":
        params["n_components"] = n_components
    cfg["model"] = {"name": MODEL_OF[variant], "params": params}
    cfg["task"] = task
    cfg["imbalance"] = imbalance
    cfg["max_rows"] = max_rows
    cfg["train_max_rows"] = train_rows or DEFAULT_TRAIN_ROWS.get(variant)
    if cfg["train_max_rows"] is None and variant == "rbf":
        cfg["train_max_rows"] = DEFAULT_TRAIN_ROWS["rbf"]
    if cfg["train_max_rows"] is not None:
        cfg["cv_max_rows"] = min(cfg.get("cv_max_rows") or cfg["train_max_rows"], cfg["train_max_rows"])
    cfg["features"]["top_k"] = top_k
    cfg["features"]["pca"] = pca
    if seed is not None:
        cfg["seed"] = seed
    cfg["run_name"] = f"svm_{variant}_{task}_{imbalance}" + (f"_C{C_value}" if C_value is not None else "")
    return cfg


def search_space(variant):
    """Parameters tried by --tune (names follow the pipeline: clf__...)."""
    if variant == "linear":
        return {"clf__C": C_GRID}
    if variant == "rbf":
        return {"clf__C": C_GRID, "clf__gamma": GAMMA_GRID}
    return {"clf__linearsvc__C": C_GRID, "clf__nystroem__gamma": GAMMA_GRID}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=list(MODEL_OF), default="linear")
    ap.add_argument("--task", choices=["binary", "multiclass", "family"], default="binary")
    ap.add_argument("--C", type=float, default=None, help="regularisation (larger = less regularised)")
    ap.add_argument("--gamma", type=float, default=None, help="RBF width (rbf / nystroem)")
    ap.add_argument("--n-components", type=int, default=None, help="nystroem: number of kernel features")
    ap.add_argument("--imbalance", default="none", choices=["none", "class_weight", "undersample", "smote"])
    ap.add_argument("--train-rows", type=int, default=None, help="cap on training rows (rbf default 30000)")
    ap.add_argument("--max-rows", type=int, default=None, help="debug: work on a stratified sample of the whole dataset")
    ap.add_argument("--top-k", type=int, default=None, help="keep the k most important features")
    ap.add_argument("--pca", type=float, default=None, help="PCA: 0.95 = keep 95 percent of variance")
    ap.add_argument("--tune", action="store_true", help="random search over C (and gamma) before the final fit")
    ap.add_argument("--n-iter", type=int, default=6, help="--tune: number of settings tried")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--rebuild", action="store_true", help="redo the cleaning cache")
    a = ap.parse_args(argv)

    C.ensure_dirs()
    cfg = build_config(a.variant, a.task, a.C, a.gamma, a.n_components, a.imbalance,
                       a.train_rows, a.max_rows, a.top_k, a.pca, a.seed)
    if a.pca is not None and a.pca >= 1:
        cfg["features"]["pca"] = int(a.pca)
    data = load_xy(cfg, rebuild=a.rebuild)

    if a.tune:
        sample = cfg["train_max_rows"] or 50_000
        best, table = tune(cfg, search_space(a.variant), n_iter=a.n_iter, data=data,
                           sample=min(sample, 50_000), out_file=C.ROOT / "configs" / "best_svm_params.json")
        print("Best parameters:", best)
        print(table[["params", "mean_test_score"]].head(5).to_string(index=False))
        for key, val in best.items():
            cfg["model"]["params"][key.split("__")[-1]] = val

    res = run_experiment(cfg, data=data)
    row = res["row"]
    keys = ["run", "n_train", "n_test", "n_features_used", "recall", "precision", "f1",
            "pr_auc", "macro_f1", "false_alarm_rate", "fit_seconds", "predict_seconds_per_1k"]
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k in keys})
    print(f"Model saved in {C.MODELS_DIR / (res['name'] + '.joblib')}")


if __name__ == "__main__":
    main()