# `src/nids/models/registry.py`

**Location:** `DM_Project/src/nids/models/registry.py`
**Roadmap step:** File 41 (Stage 5, models)

## Purpose

One place that knows all 11 model classes. Other modules ask it for a model by key (`"random_forest"`) and never import a model file themselves.

## Why it is needed

The trainer (File 46), the evaluation, the tuning step, the majority vote and the interface all loop over "the 11 models". Without a registry each would keep its own list and could drift apart. The registry also makes a missing model visible instead of silent.

## Dependencies

`config/model_params.py` (the key list and order), `nids.models.base`, the 11 model files (imported lazily).

## Functions

| Function | What it does |
|---|---|
| `all_model_keys()` | the 11 keys in the order of the project brief |
| `display_name(key)` | the readable name, e.g. `"SVC (RBF kernel)"` |
| `get_model_class(key)` | the class; `ValueError` for an unknown key, `ModelUnavailableError` if it cannot be imported |
| `create_model(key, params=None, class_weight_mode=None)` | a new **unfitted** model (defaults of File 28 updated with `params`) |
| `availability(keys=None)` | per key: `available`, `error`, `display_name` |
| `create_all_models(keys=None, class_weight_mode=None, skip_unavailable=False)` | dict key to unfitted model; an unavailable one raises, unless `skip_unavailable=True` (the caller must then report who is missing) |
| `ModelUnavailableError` | subclass of `ImportError` |

## Rules it enforces

- **Lazy import:** a problem with one model's library (for example `xgboost` not installed) makes only that model unavailable. It does not break the other ten.
- **No silent drift:** the registry keys must equal `model_params.MODEL_KEYS` (checked at import), and each class's `key` must equal its registry key (checked when the class is requested). An empty placeholder file or a wrong class name gives a `ModelUnavailableError` that names the file and class.
- **Missing models are reported, never hidden:** this is what lets the majority vote (File 60) state which models voted and which did not.

## How it was checked (sandbox)

All 11 available; `create_all_models` returns 11 unfitted models in brief order with matching keys; `create_model("knn", params={"n_neighbors": 3})` applies the override; an unknown key (`"kmeans"`) gives `ValueError`; with `xgboost` made un-importable, `availability()` reports exactly `xgboost` as unavailable with the reason, `create_all_models(skip_unavailable=True)` returns 10 models, and without `skip_unavailable` it raises `ModelUnavailableError`.

## How to test on your real data

No data needed. PowerShell, from `DM_Project\`, venv active:

```powershell
python -c "from nids.models import registry as r; a = r.availability(); [print(k.ljust(20), v['display_name'].ljust(22), 'available =', v['available']) for k, v in a.items()]; ms = r.create_all_models(); print(len(ms), 'models created, all unfitted:', all(not m.is_fitted for m in ms.values()))"
```

**Expected output:** 11 lines, in this order: `knn`, `decision_tree`, `random_forest`, `gaussian_nb`, `svm`, `bernoulli_nb`, `logistic_regression`, `sgd`, `adaboost`, `gradient_boosting`, `xgboost`, each with its display name and `available = True`, then `11 models created, all unfitted: True`. Warning lines about the GPU are not expected, because nothing is fitted.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `available=False` with `ModuleNotFoundError ... xxx_model` | a model file is empty or misplaced | paste its code, check the length is not 0 |
| `available=False` with `ModuleNotFoundError: xgboost` | library missing in the venv | `pip install xgboost==3.1.3` |
| `ValueError: registry and model_params.MODEL_KEYS differ` | a key was added to one list only | add it to both |

## Decisions and limitations

- **No auto-discovery of files:** the list is explicit, so a stray file cannot become a model.
- **Tests:** `tests/test_model_registry.py` (File 42) covers the registry and all 11 models together.