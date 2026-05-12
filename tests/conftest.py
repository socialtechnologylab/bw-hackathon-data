"""Shared pytest fixtures."""

import os
from datetime import UTC, datetime

import pytest


@pytest.fixture
def dt():
    """Factory for UTC datetimes: dt(2025, 5, 1, 12) → 2025-05-01T12:00:00+00:00."""

    def make(*args):
        return datetime(*args, tzinfo=UTC)

    return make


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless BW_DATA_NETWORK=1."""
    if os.environ.get("BW_DATA_NETWORK") == "1":
        return
    skip_integration = pytest.mark.skip(reason="set BW_DATA_NETWORK=1 to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
