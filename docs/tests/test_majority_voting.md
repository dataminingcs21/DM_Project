# `tests/test_majority_voting.py`

**Roadmap step:** File 54. Tests for `src/nids/voting/majority_voting.py` (File 53). Synthetic data only; it never reads your dataset or saved models.

## What it checks (49 tests)
| Group | Checks |
|---|---|
| Rule | clear majority; one equal vote per model; tie prefers attack over BENIGN; tie between attacks takes the lowest code; three-way tie; `lowest_index` rule; BENIGN wins only with the most votes; order of models irrelevant; repeatable; counts add up; unanimity; labels only, no probabilities |
| Reference | the vectorised vote equals a plain-Python reference of the written rule on 2,000 random flows, for both tie rules and for 1, 2, 3, 6 and 11 voters |
| Missing models | none missing when complete; missing models listed with a default or a given reason; `require_all` refuses; custom expected keys; JSON-safe summary |
| Input checks | no predictions, float labels, out-of-range codes, different lengths, 2-D arrays, unknown rule, bad `benign_index` or class count, zero rows, default rule from settings |
| With models | `vote_models`; a model that raises is isolated and reported; `not_loaded` reasons merged; all models failing raises; `continue_on_error=False` re-raises |
| Explanation | voters and tally; display names; clear winner; tie-break winner; row out of range |

## Command (PowerShell, from `DM_Project\`)
```powershell
python -m pytest tests\test_majority_voting.py -q
```
**Expected:** `49 passed` in well under a second.

Full suite (optional, to confirm nothing else changed): `python -m pytest -q` (earlier result 349 passed, 1 skipped; now 398 passed, 1 skipped).

## Sandbox check beyond passing
I broke the vote on purpose four ways (BENIGN not removed in ties, BENIGN counted double, tie resolved to the highest code, `require_all` disabled): the tests failed each time. The original file passes again.

## Common errors
| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: nids.voting.majority_voting` | File 53 not pasted | paste it |
| `ImportError ... tests` | wrong folder | run from `DM_Project\` |