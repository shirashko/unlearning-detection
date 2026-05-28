# Toy target evaluation fixtures

Minimal inputs for smoke-testing `run_target_evaluation.py` without a full audit run.

## Files

| Path | Role |
|------|------|
| `audit/judge_response.json` | Fake audit verdict (`likely_unlearned_concept`) |
| `labeled.json` | 2 forget + 2 retain text samples |
| `run_toy_evaluation.sh` | Rerun script (calls Gemini; needs API key) |

Generated outputs go to `out/` (gitignored).

## Rerun (live API)

From the repository root:

```bash
export GOOGLE_API_KEY="..."   # or set in .env

bash tests/toy_target_eval/run_toy_evaluation.sh
```

Optional overrides:

```bash
EVAL_MODEL=gemini-2.5-flash MAX_SAMPLES_PER_SET=2 bash tests/toy_target_eval/run_toy_evaluation.sh
```

## Rerun (offline, no API)

```bash
python3 -m pytest tests/test_target_evaluation_smoke.py -v
```
