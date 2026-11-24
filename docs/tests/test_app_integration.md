# tests/test_app_integration.py  (File 64, Checkpoint 6)

## 1. Role
Proves that the application works as one piece: the plain helper functions of `app/ui_components.py`, the whole page `app/streamlit_app.py` run without a browser, and the upload path.

## 2. Why it is needed
Files 62 and 63 were checked by hand. A page that is only checked by hand breaks silently when a later change touches the predictor, the validator or the page. These tests make that visible, and they check that the page shows what the predictor returns and says so honestly (refusals, warnings, stale results, missing models).

## 3. Dependencies
`streamlit` (its `streamlit.testing.v1.AppTest`; the file is skipped if streamlit is not installed), `app/ui_components.py`, `app/streamlit_app.py`, the predictor, validator, preprocessing and model code. Synthetic data only: a small complete project (preprocessing, 11 baseline and 11 tuned models, two small evaluation JSON files) is built once in a temporary folder; the `settings` paths point there for this module only. No real file is touched and no real dataset column is used.

## 4. Code
`tests/test_app_integration.py`, about 900 lines, 121 tests in these groups:

| Group | What it checks |
|---|---|
| Field grouping, value formatting, field help, state, samples | grouping by first word, `Other` last, exact-minimum group size, number to text and back, example flows |
| `read_uploaded_csv` | padded headers, `Label` ignored, row limit (and exactly at the limit), empty file, header-only file, non-text bytes, missing cells |
| Result table, CSV bytes, wording | 1-based rows, refusal reason column, agreement bands (all / majority / plurality only), no change to the batch |
| Reliability and model panel | reads the saved reports, sentences use their numbers, low-support caution, damaged report is not silently accepted |
| Page loads | no exception, title, three tabs, a field per feature filled with its median, nothing predicted before the button |
| Single flow | prediction equals the predictor's, vote table equals the predictor's, real example of every class, benign vs attack headline, bad values (negative, text, port 655360), warning for a far value, blank field refused or filled by choice, stale notice, result kept, refused flow clears the old result |
| Failures | models missing, damaged pipeline, one damaged model (others vote, page says so), no example file, no evaluation reports |
| Upload | the three upload steps in a small page, and the real upload tab with the upload widget replaced by a fixed in-memory file (AppTest cannot upload): predict only on the button, missing-value choice applied, a result for another setting is hidden, unreadable file gives a message |
| Consistency | the same flow typed in the form and uploaded in a file gets the same answer |

## 5. How it works
`AppTest.from_file` runs the real page script in-process, finds widgets by label, sets values, clicks buttons and reads what was drawn (errors, metrics, tables). `AppTest.from_function` runs a small page built for the test. Caches of the page (`st.cache_resource`, `st.cache_data`) are cleared before and after every test. Expected answers come from the predictor itself (File 60 tests the voting rule independently), so these tests check what the page shows, not the voting rule again.

How strong the tests are was checked by mutation: 14 deliberate changes to the app code (agreement threshold, `Label` not dropped, row numbers from 0, group size, stale notice removed, the missing-value setting not passed to the single flow or to the upload, result shown for another setting, benign/attack colours swapped, refusal column moved, headers not stripped) were each caught. One change was not caught: removing the reset of the example number when another class has fewer examples. In the installed Streamlit version the widget is rebuilt anyway, so that line cannot fail there; it stays as a guard for other versions.

## 6. Command
```powershell
python -m pytest tests\test_app_integration.py -q
python -m pytest -q          # the whole suite
```

## 7. Expected output
`121 passed` for this file. For the whole suite, the earlier 543 passed + 1 skipped (GPU test) plus these 121: expect **664 passed, 1 skipped**, with the same two ConvergenceWarnings from the registry tests. A few streamlit lines such as "No runtime found, using MemoryCacheStorageManager" or "missing ScriptRunContext" may appear in captured output; they are harmless.

## 8. Common errors
- `ModuleNotFoundError: streamlit`: the file is skipped; install streamlit in the venv.
- `ModuleNotFoundError: nids`: run `pip install -e .` (see docs/project/pyproject.md).
- A test that uses the Streamlit API fails after a Streamlit upgrade (widget or element names change, for example charts appear as `arrow_vega_lite_chart`): adjust that one element name.
- Slow first run: the module builds 22 small models once (about 10 s).