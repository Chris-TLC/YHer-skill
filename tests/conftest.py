"""Collection safety gates for the repository test suite."""

from __future__ import annotations

_PAID_INTEGRATION_SCRIPTS = (
    "test_llm_v3.py",
    "test_cache_v3.py",
    "test_t6_regression.py",
)

# These legacy integration scripts execute at module import time, so they are
# never valid pytest modules. Run one directly only when a paid check is
# explicitly authorized; future pytest-native integrations must use @paid.
collect_ignore = list(_PAID_INTEGRATION_SCRIPTS)
