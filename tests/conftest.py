"""
Shared test configuration.

The suite must run offline and fast. A test that reaches the network is slow,
flaky in CI, and — for the SSRF tests especially — liable to pass for the wrong
reason: a blocked host that returns 404 raises the same exception type as a host
that was correctly refused. The autouse fixture below makes any accidental
outbound request an immediate, loud failure.

Mark a test `@pytest.mark.network` to opt out.
"""

from __future__ import annotations

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "network: test makes real outbound requests")


@pytest.fixture(autouse=True)
def no_network(request, monkeypatch):
    if request.node.get_closest_marker("network"):
        return

    import socket

    import requests

    def blocked(*args, **kwargs):
        raise AssertionError(
            "This test attempted a real network request. Either mock it, or mark "
            "the test with @pytest.mark.network."
        )

    monkeypatch.setattr(requests.Session, "request", blocked)
    monkeypatch.setattr(requests.Session, "get", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
