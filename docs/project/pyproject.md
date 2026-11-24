# `pyproject.toml`

**Location:** `DM_Project/pyproject.toml` (project root)
**Roadmap step:** File 03 (Stage 0, setup). Also the step where the folder skeleton and the 14 empty `__init__.py` files are created.

## Purpose

1. Makes the project importable from anywhere: after `pip install -e .`, scripts, tests and Streamlit can run `import nids` and `import config` without editing `sys.path`.
2. Holds the pytest configuration (test folder and the `real_data` marker used by the roadmap).

## Contents

| Section | What it does |
|---|---|
| `[build-system]` | Uses `setuptools` as the build tool (downloaded automatically by `pip`) |
| `[project]` | Project name (`nids-project`), version, description, minimum Python 3.11. Runtime dependencies are **not** listed here; `requirements.txt` is the single source of truth |
| `[tool.setuptools]` `package-dir` | Declares two import roots: `nids` lives in `src/nids/`, `config` lives in `config/` at the project root |
| `packages = { find = ... }` | Finds every sub-package of `nids` and `config` automatically. Only folders containing `__init__.py` are found |
| `[tool.pytest.ini_options]` | Tests are in `tests/`; `-ra` prints a summary of skipped tests; `-p no:cacheprovider` turns off pytest's `.pytest_cache` folder (added after a Windows machine refused access to it; the cache is not needed here); registers the marker `real_data` (tests that need the real dataset artifacts and are skipped when they are absent) |

## Folder skeleton created in this step

```
DM_Project/
├── config/                 (+ __init__.py)
├── src/nids/               (+ __init__.py)
│   ├── utils/ data/ eda/ cleaning/ preprocessing/ models/
│   └── training/ evaluation/ optimization/ persistence/ prediction/ voting/   (each + __init__.py)
├── scripts/  app/  tests/
├── data/raw/               (the 8 CSV files go here)
└── docs/                   (project/ config/ scripts/ app/ tests/ and src/nids/<each sub-package>/)
```

Folders such as `data/interim`, `data/cleaned`, `data/preprocessed`, `artifacts/`, `reports/` and `logs/` are created later by `ensure_directories()` in `config/settings.py` (File 04).

## Inputs and outputs

- **Input:** the folders `config/` and `src/nids/`, each with `__init__.py` files.
- **Output:** an *editable* installation of the project in `.venv`. Editing a `.py` file takes effect immediately; no reinstall is needed when you change or add modules.

## Dependencies

`requirements.txt` installed in an activated `.venv` (File 02). Internet access the first time (to fetch the build tool).

## How to create and verify

From the project root, in the activated `.venv` (PowerShell):

```powershell
# 1. Folder skeleton
$pkgs = "utils","data","eda","cleaning","preprocessing","models","training","evaluation","optimization","persistence","prediction","voting"
$dirs = @("config","scripts","app","tests","data\raw","docs\project","docs\config","docs\scripts","docs\app","docs\tests","src\nids")
foreach ($p in $pkgs) { $dirs += "src\nids\$p"; $dirs += "docs\src\nids\$p" }
foreach ($d in $dirs) { New-Item -ItemType Directory -Force $d | Out-Null }

# 2. Empty __init__.py files (never overwrites an existing file)
$inits = @("config","src\nids") + ($pkgs | ForEach-Object { "src\nids\$_" })
foreach ($d in $inits) { $f = Join-Path $d "__init__.py"; if (-not (Test-Path $f)) { New-Item -ItemType File $f | Out-Null } }

# 3. Count them (expect 14)
(Get-ChildItem config, src -Recurse -Filter __init__.py | Measure-Object).Count

# 4. Move the dataset CSVs into data\raw (leaves the .complete folder behind)
Move-Item dataset\*.csv data\raw\
(Get-ChildItem data\raw -Filter *.csv | Measure-Object).Count      # expect 8

# 5. Create pyproject.toml (this file), then install the project in editable mode
pip install -e .

# 6. Prove it is importable from ANY folder
cd $env:TEMP
python -c "import nids, config, nids.data, nids.preprocessing, nids.voting; print(nids.__file__); print(config.__file__)"
cd <back to DM_Project>

# 7. pytest reads the configuration
python -m pytest --collect-only
```

**Expected result:**
- Step 3 prints `14`, and step 4 prints `8`.
- `pip install -e .` ends with `Successfully installed nids-project-0.1.0`.
- Step 6 prints two paths ending in `...\DM_Project\src\nids\__init__.py` and `...\DM_Project\config\__init__.py`.
- Step 7 prints `configfile: pyproject.toml`, `testpaths: tests` and `no tests collected`. That message is normal at this stage (pytest exits with code 5 when there are no tests; it is not an error).

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'nids'` | Not installed, or the wrong Python is running | Activate `.venv` (prompt shows `(.venv)`), re-run `pip install -e .`, check `where python` points into `.venv` |
| `ModuleNotFoundError: No module named 'nids.data'` (or another sub-package) | That folder has no `__init__.py`, so it is not found as a package | Re-run step 2 of the commands above |
| Error mentioning `package directory 'config' does not exist` (or `src/nids`) | `pip install -e .` ran before the folders existed | Run steps 1–2 first |
| `Move-Item: Cannot find path ... dataset\*.csv` | The CSVs are not in `DM_Project\dataset\` | Use the actual folder; the target is always `data\raw\` |
| Build fails while downloading `setuptools` | No internet access, or a proxy | Retry on a working connection |
| `pytest` says `unknown marker real_data` | Running pytest from a different folder, so it does not read this file | Run pytest from the project root |
| `ImportError` for `config` from another project | Another installed package named `config` | None expected in this environment; we checked `site-packages` and found no conflict. If it ever happens, run `pip list` to find it |

## Decisions and limitations

- **Two import roots** (`nids` in `src/`, `config` at the root) keep the architecture from Deliverable 2 unchanged. We tested that an editable install exposes both from any folder.
- **No runtime dependencies in this file.** Having them in both places invites mismatched versions.
- **`requires-python = ">=3.11"`** is a floor, not a guarantee. The environment we verified is Python 3.13.
- **Generic package name `config`.** It could clash with another package of the same name if one were ever installed. No clash exists in the pinned environment.
- **Editable install creates a `nids_project.egg-info` folder** (or similar) in the project; `.gitignore` already ignores `*.egg-info/`.