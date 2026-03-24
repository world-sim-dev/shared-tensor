from __future__ import annotations

import pytest

from shared_tensor import SharedTensorProvider
from shared_tensor.errors import SharedTensorProviderError


def test_provider_registers_named_endpoints() -> None:
    provider = SharedTensorProvider(execution_mode="local")

    def add(a: int, b: int) -> int:
        return a + b

    provider.register(add, name="add")
    assert provider.invoke_local("add", args=(2, 3)) == 5
    assert provider.list_endpoints()["add"]["cache"] is False


def test_provider_rejects_duplicate_names() -> None:
    provider = SharedTensorProvider(execution_mode="local")

    provider.register(lambda: None, name="dup")
    with pytest.raises(SharedTensorProviderError):
        provider.register(lambda: None, name="dup")


def test_share_decorator_local_mode_returns_original_function() -> None:
    provider = SharedTensorProvider(execution_mode="local")

    @provider.share(name="trim")
    def trim(value: str) -> str:
        return value.strip()

    assert trim("  x ") == "x"
    assert provider.invoke_local("trim", args=("  x ",)) == "x"


def test_client_mode_wrapper_calls_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SharedTensorProvider(execution_mode="client")

    calls = []

    class FakeClient:
        def call(self, endpoint, *args, **kwargs):
            calls.append((endpoint, args, kwargs))
            return "ok"

        def close(self):
            return None

    provider._client = FakeClient()

    @provider.share(name="wrapped")
    def wrapped(value: int) -> int:
        return value + 1

    assert wrapped(3) == "ok"
    assert calls == [("wrapped", (3,), {})]
