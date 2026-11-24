# `app/streamlit_app.py`

**Location:** `DM_Project/app/streamlit_app.py`
**Roadmap step:** File 63. It is delivered together with a small update to `app/ui_components.py` (File 62): the model table on the "About the models" tab is now tall enough to show all 11 rows (found while looking at a screenshot).

## Role
The page you start with `streamlit run`. It only composes the building blocks of `ui_components.py`; prediction, input checking and voting are done by `nids.prediction.predictor`. Nothing is trained or tuned, and nothing typed in the page is stored anywhere.

## What the page offers
| Where | What |
|---|---|
| Sidebar | status (`Models loaded: N of 11`, how many tuned and baseline versions, any model that did not load and why); the **missing-value setting**: refuse a flow with a missing value (default) or fill it with the training median and warn |
| Tab "Single flow" | start from the training medians or from a real example flow of a class you pick (up to 5 per class); the 40 fields grouped by name prefix; **Predict**. The result shows the predicted class, how many models voted and agreed, every model's vote with its version, the tally, input warnings, and what the test set says about that class. If you change a field after predicting, the page tells you the result is out of date |
| Tab "Upload a CSV file" | a button to download 71 real example flows as a CSV (to try the upload), the uploader, **Predict this file**, counts of predicted and refused rows, a chart of predicted classes, the result per row (with the reason a row was refused) and a download of the result as CSV |
| Tab "About the models" | which version of each model is used and why, how each was trained, the test macro F1 of the version in use, the three majority votes on the test set, classes with too few test flows, the most frequent mistakes, and the limits |

If the models cannot be loaded (preprocessing, training or tuning not done, a settings change), the page shows the reason and the exact commands to run instead of a stack trace.

## Dependencies
`streamlit` 1.52.2, `app/ui_components.py` (File 62), `nids.prediction.predictor` and `input_validator`, `config.settings`. Needs the files from `04_preprocess.py`, `05_train.py` and `07_optimize.py`; the quality panels also use the reports of `06_evaluate.py --stage tuned` (the page works without them).

## Command (PowerShell, from `DM_Project\`, venv active)
```powershell
streamlit run app\streamlit_app.py
```
Streamlit prints a Local URL (normally http://localhost:8501) and opens it in your browser. The first load takes roughly 10 to 60 seconds while the 11 models load (they are then kept in memory until you stop the server with Ctrl+C).

## Expected output
In the terminal: `You can now view your Streamlit app in your browser.` and the Local URL. In the browser:
- title **Network intrusion detection**, three tabs, sidebar `Models loaded: 11 of 11` and `7 tuned and 4 baseline versions`;
- **Predict** on the default (median) fields gives a predicted class with metrics such as `Votes for the result: 7 of 11`; the median flow is not a real flow, so its class means little;
- choosing a class, **Load this real example**, **Predict** normally gives that class back, but not always (the test-set recall of each class is shown under "What the test set says about this class");
- typing `-5` into `Flow Duration` gives `'Flow Duration' = '-5' is negative: a duration cannot be negative` and no prediction; `abc` gives `is not a number`; `Destination Port` 70000 is refused as impossible;
- the example CSV, uploaded back, gives 71 rows predicted, 0 refused.

## What I checked in the sandbox
A headless Streamlit session and a real server with a real browser (screenshots) over a small synthetic project: the page loads; prediction on the medians; the out-of-date notice after a field change and its removal after predicting again; a blank value refused, and filled with a warning when you choose the median setting; loading a real example and predicting it; the models tab; the failure page when nothing is trained (shows the reason and the commands, no tab); the server answers its health check and serves the page. The CSV upload itself cannot be driven by the headless test tool; the code path (reading, predicting, showing a table with refused rows) was exercised through its functions, and File 64 tests it again.

## Common errors
| Problem | Cause | Fix |
|---|---|---|
| `The models could not be loaded: ...` with a list of commands | training or preprocessing files missing or stale | run the listed commands, then reload |
| `streamlit : The term 'streamlit' is not recognized` | the venv is not active | activate `nids`, or `python -m streamlit run app\streamlit_app.py` |
| `ModuleNotFoundError: nids` or `config` | the project is not installed in the venv | `pip install -e .` from `DM_Project\` |
| the page is slow after changing a file | Streamlit reruns the script; the models stay cached | press `R` in the browser, or restart the server |
| Port 8501 in use | another Streamlit app is running | `streamlit run app\streamlit_app.py --server.port 8502` |

## Limits
- One user at a time is the intended use (a project demonstration, not a production service); the loaded models are shared by all browser sessions of the same server.
- Uploads are limited to 50,000 rows and CSV files.
- A prediction is a majority vote of models trained on one dataset; it is not proof of an attack (the page says so).