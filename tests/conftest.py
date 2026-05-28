"""Pytest configuration: ensure repo root is on ``sys.path``."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: live Gemini API (requires GOOGLE_API_KEY).",
    )
