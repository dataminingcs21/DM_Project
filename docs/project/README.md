# README.md (root file)

## 1. Role
The entry document of the project: what it does, what the results mean and do not mean, how to install, run, test and use it.

## 2. Why it is needed
A reader who has only the project folder must be able to run it and must not be misled by the scores. The limits come first on purpose.

## 3. Dependencies
None to run. Every number in it comes from a report written by the scripts: the cleaning, preprocessing and training reports, `reports/optimization/tuning_report.md`, `reports/evaluation/evaluation_report_tuned.md` and the checkpoint documents in `docs/project/`.

## 4. Code
`README.md` in the project root (Markdown, no code).

## 5. How it works
Sections: limits; data; models and the selection rule; results; install; run; app; comparison files; tests; layout; not done. When a script is rerun and a number changes, update the matching line (the vote table and the mistakes list come from `evaluation_report_tuned.md`; the data figures from the checkpoint documents).

## 6. Command
None. Open it in any Markdown viewer or on the repository page.

## 7. Expected output
Not applicable. To check it, compare the vote table with `reports/evaluation/model_comparison_tuned.md` and the test count with `python -m pytest -q` (664 passed, 1 skipped in the reference run).

## 8. Common errors
- A figure in the README that differs from a new report: the pipeline was rerun; update the README.
- Paths use Windows backslashes; on Linux use forward slashes.