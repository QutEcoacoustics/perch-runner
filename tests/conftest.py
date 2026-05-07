# Top-level conftest — shared plugins and fixtures.
import socket

import pytest

pytest_plugins = [
  "tests.shared_fixtures.helpers"
]

_original_connect = socket.socket.connect


@pytest.fixture(autouse=True)
def _block_network(request):
    """Block all network access unless the test is marked with @pytest.mark.allow_network."""
    if "allow_network" in request.keywords:
        yield
        return

    def _blocked(*args, **kwargs):
        raise ConnectionError(
            "Network access blocked in tests. "
            "Models must be pre-cached. Mark with @pytest.mark.allow_network to allow downloads."
        )

    socket.socket.connect = _blocked
    try:
        yield
    finally:
        socket.socket.connect = _original_connect



