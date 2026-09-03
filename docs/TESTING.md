# Testing

## Main suite (public repo, CI baseline)

```bash
python3 -m venv .venv-pub
.venv-pub/bin/pip install -r requirements-dev.txt
.venv-pub/bin/python -m pytest            # markers and config come from pytest.ini
```

- Green baseline: **~1,900 tests** (count drifts slightly with machine and environment).
- No paid external calls: the default run skips tests marked `paid` (live-provider, network, or remote-store tests) via `-m "not paid"`.
- `faiss` has no wheel for some interpreters: the retrieval layer treats it as optional (`FAISS_AVAILABLE` fallback), and vector-retrieval tests skip themselves when faiss is absent. On Python 3.11/3.12, `faiss-cpu` installs normally and the coverage is fuller.

## Engine contracts (no large data needed)

```bash
.venv-pub/bin/python -m pytest tests/test_mastery.py tests/test_selector.py \
  tests/test_planner.py tests/test_recommender.py tests/test_memory.py \
  tests/test_event_log.py -q
```

## Paid / network tests (excluded by default)

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-pub/bin/python -B -m pytest -q --timeout=120 \
  --override-ini="addopts="
```

This runs live-provider tests and incurs API cost; it is not part of the public CI baseline.
