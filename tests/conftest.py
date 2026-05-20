# Top-level conftest — shared plugins and fixtures.
import socket

import pytest

_original_connect = socket.socket.connect
_original_getaddrinfo = socket.getaddrinfo


@pytest.fixture(autouse=True)
def _block_network(request):
    """Block all network access for all tests."""

    error_message = (
        "Network access blocked in tests. "
        "Models must be pre-cached."
    )

    def _blocked(*args, **kwargs):
        raise ConnectionError(error_message)

    def _blocked_getaddrinfo(*args, **kwargs):
        # Mirror DNS resolution failure while preserving a deterministic message.
        raise socket.gaierror(-2, error_message)

    socket.socket.connect = _blocked
    socket.getaddrinfo = _blocked_getaddrinfo
    try:
        yield
    finally:
        socket.socket.connect = _original_connect
        socket.getaddrinfo = _original_getaddrinfo



