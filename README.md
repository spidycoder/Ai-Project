# LLM Evaluation Platform

A CI-integrated evaluation harness for LLM/RAG systems: golden datasets,
multi-scorer grading (deterministic exact-match, cheap similarity pre-check,
calibrated LLM-as-judge), statistically-gated regression detection, and
judge/human agreement calibration.

This exists to answer a question most AI projects never actually answer:
**how do you know your system works, and keeps working after you change a
prompt or model?** "I tested it and it looked good" is not an evaluation
pipeline. This is.

## Why this design

- **Pluggable target system** (`eval_platform/targets/`) — the runner never
  knows or cares whether it's hitting a mock, an HTTP endpoint (your real
  RAG assistant, an LLM gateway, anything), or a CLI tool. Swap targets
  without touching the runner, scorers, or storage.
- **Multiple scorers, not one score.** Exact-match is free and has zero
  bias — always prefer it when a deterministic ground truth exists.
  Bag-of-words similarity is a near-zero-cost sanity check that catches
  wildly off-topic answers before spending judge-model tokens. LLM-as-judge
  handles everything that needs semantic/rubric judgment — and its scores
  are never trusted blindly (see Calibration below).
- **SQLite, not Postgres**, deliberately — every run is durable, queryable,
  and versioned with zero setup cost for anyone cloning this repo or for
  CI. Swap the connection layer for Postgres if concurrent writers or
  retention/partitioning ever become real requirements; that's a config
  change, not a rewrite, because `storage.py` is the only place that knows
  about SQL.
- **Regression detection is statistical, not a naive diff.** Two mean
  scores differing is not evidence of a regression — it might be noise
  from a small test set. `detect_regression()` bootstraps a 95% confidence
  interval on the score delta between two runs and only flags a regression
  when that interval is entirely below zero.
- **The judge is graded too.** `calibrate_judge()` measures Cohen's kappa
  between the LLM judge's scores and human labels on the same test cases.
  An uncalibrated judge is not an evaluation system — it's an unverified
  opinion generator wearing a lab coat.

## Verified end-to-end (numbers below are from an actual run in this repo, not invented)

```
$ python -m eval_platform.cli run --dataset datasets/golden_set.jsonl --target mock --label baseline --no-judge
{
  "run_id": "78948829",
  "n_cases": 16,
  "composite_score": 0.439,
  "category_scores": {
    "robustness": 0.792,
    "factual_accuracy": 0.644,
    "out_of_scope": 0.0,
    "refusal_appropriateness": 0.0,
    "instruction_following": 0.0
  }
}
```

Note the three categories scoring exactly 0.0: `out_of_scope`,
`refusal_appropriateness`, and `instruction_following` test cases have no
`expected_output`, only a `rubric` — so with `--no-judge`, no scorer applies
to them at all (both exact-match and similarity return `None`), and an
unscored case defaults to 0. **This is the concrete, observed reason
LLM-as-judge exists in this pipeline**: it's the only scorer that can grade
rubric-based cases at all. Run without `--no-judge` (needs
`GEMINI_API_KEY`) to see those categories actually scored.

Regression detection, verified against a deliberately degraded candidate
(`--mock-error-rate 0.6` vs. the `0.15` baseline):

```
$ python -m eval_platform.cli compare --baseline 78948829 --candidate 87945fb0
REGRESSION DETECTED: candidate is significantly worse than baseline
(95% CI on delta: [-0.597, -0.187])
```

...and confirmed to **not** false-flag an equivalent candidate run
(identical error rate, different seed of randomness): delta `0.0`, 95% CI
`[-0.274, 0.275]`, `is_regression: false`, exit code `0`.

## Quickstart

```bash
pip install -r requirements.txt
pytest -q                                    # 12 unit tests, no API key required

# Baseline run against the built-in mock RAG target (no API key needed)
python -m eval_platform.cli run --dataset datasets/golden_set.jsonl --target mock --label baseline --no-judge

# Simulate a regression and confirm detection
python -m eval_platform.cli run --dataset datasets/golden_set.jsonl --target mock --label bad --no-judge --mock-error-rate 0.6
python -m eval_platform.cli compare --baseline <baseline_run_id> --candidate <bad_run_id>   # exits 1 on regression

python -m eval_platform.cli list
```

### Evaluating with the LLM judge

```bash
cp .env.example .env    # fill in GEMINI_API_KEY (free tier: https://aistudio.google.com/apikey)
python -m eval_platform.cli run --dataset datasets/golden_set.jsonl --target mock --label with-judge
```

### Calibrating the judge against human labels (do this before trusting it for CI gating)

```bash
python -m eval_platform.cli calibrate --dataset datasets/golden_set.jsonl --human-labels datasets/human_labels.jsonl
```

`datasets/human_labels.jsonl` ships with an initial pass of labels I
assigned by hand against the mock target's actual (deterministic) output —
they're a starting point, not ground truth handed down from elsewhere.
Before relying on the kappa number for anything real, review them yourself
(or better, have someone else independently label a batch and check
inter-rater agreement between the two of *you* first) — a judge is only as
trustworthy as the labels it's being measured against.

### Pointing this at your real RAG Document Assistant

`eval_platform/targets/http_target.py` is a generic adapter: it POSTs
`{"input": "..."}` to `EVAL_TARGET_URL` and expects
`{"output": "...", "tokens_in": int, "tokens_out": int, "cost_usd": float}`
back. Wrap your RAG assistant's inference behind a small HTTP endpoint
matching that contract (or edit the two marked lines in `http_target.py` to
match its actual request/response shape), then:

```bash
export EVAL_TARGET_URL=http://localhost:8000/query
python -m eval_platform.cli run --dataset datasets/golden_set.jsonl --target http --label rag-v1
```

This replaces the resume's previously unverified "95%+ accurate" RAG claim
with a reproducible, versioned benchmark.

## Repository structure

```
eval_platform/
  types.py           dataclasses: TestCase, TargetResponse, ScoreResult, TestCaseResult
  dataset.py          JSONL golden-dataset loader
  targets/
    base.py           TargetSystem interface
    mock_rag.py        zero-dependency mock target with a deterministic, configurable injected-error rate
    http_target.py      generic HTTP adapter for a real deployed system
  scorers/
    base.py           Scorer interface
    exact_match.py       deterministic substring match against expected_output
    text_similarity.py    bag-of-words cosine similarity (cheap sanity check)
    llm_judge.py         rubric-based LLM-as-judge (Gemini), degrades gracefully with no API key
  runner.py           parallel (thread pool) execution engine
  storage.py          SQLite-backed results store, versioned by run
  stats.py            bootstrap regression detection + Cohen's kappa
  calibration.py       judge-vs-human agreement measurement
  cli.py              run / compare / calibrate / list commands
datasets/
  golden_set.jsonl     16 test cases across 5 categories (factual accuracy,
                        instruction following, refusal appropriateness,
                        robustness to noisy input, out-of-scope handling)
  human_labels.jsonl    initial human labels for judge calibration
tests/                 12 unit tests covering scorers, stats, and end-to-end runs
.github/workflows/eval.yml   CI: runs unit tests + a mock-target eval on every PR
```

## Roadmap

- **V2**: category-weighted composite scoring (not all categories are
  equally important); paired bootstrap (resample by matched test-case
  index rather than independently) for a tighter regression test; a small
  web dashboard over the SQLite DB showing score trends per category
  across runs.
- **V3**: wire `eval.yml` to compare against a committed baseline
  `run_id` and actually fail PRs on regression (currently proves the
  pipeline runs in CI; the gate itself needs a trusted baseline to compare
  against first); multi-judge ensemble (average 2-3 judge models) to
  reduce single-judge bias; adversarial/robustness test-case generation.

## Known limitations (stated explicitly, not discovered by an interviewer)

- `text_similarity.py` uses bag-of-words cosine similarity as a
  zero-dependency stand-in for real embedding similarity — it's a cheap
  pre-filter, not a primary quality signal. Swap in a real embedding API
  for a more accurate similarity score.
- The bootstrap in `detect_regression` resamples the two score lists
  independently. If your dataset guarantees the same test-case ordering
  across runs, a *paired* bootstrap (resampling matched pairs) gives a
  tighter, more powerful test — noted as a V2 item above.
- `human_labels.jsonl` is a single rater's initial pass, not a
  cross-validated ground truth — see the calibration section above.
