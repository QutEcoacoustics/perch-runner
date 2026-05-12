# Top-level conftest — shared plugins and fixtures.
import socket

import pytest

pytest_plugins = [
  "tests.shared_fixtures.helpers"
]

_original_connect = socket.socket.connect


@pytest.fixture(autouse=True)
def _block_network(request):
    """Block all network access for all tests."""

    def _blocked(*args, **kwargs):
        raise ConnectionError(
            "Network access blocked in tests. "
            "Models must be pre-cached."
        )

    socket.socket.connect = _blocked
    try:
        yield
    finally:
        socket.socket.connect = _original_connect



