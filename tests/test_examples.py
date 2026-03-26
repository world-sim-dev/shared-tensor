from __future__ import annotations

import importlib

import pytest

from shared_tensor import SharedTensorProvider


def test_examples_server_modules_expose_server_mode_providers() -> None:
    model_service = importlib.import_module("examples.model_service")
    basic_service = importlib.import_module("examples.basic_service")
    async_service = importlib.import_module("examples.async_service")

    assert isinstance(model_service.provider, SharedTensorProvider)
    assert model_service.provider.execution_mode == "server"

    assert isinstance(basic_service.provider, SharedTensorProvider)
    assert basic_service.provider.execution_mode == "server"

    assert isinstance(async_service.provider, SharedTensorProvider)
    assert async_service.provider.execution_mode == "server"
    assert async_service.provider.list_endpoints()["build_delayed_model"]["execution"] == "task"
    assert async_service.provider.list_endpoints()["clone_tensor_async"]["execution"] == "direct"


def test_zero_branch_example_uses_auto_mode_provider() -> None:
    zero_branch_env = importlib.import_module("examples.zero_branch_env")

    assert isinstance(zero_branch_env.provider, SharedTensorProvider)
    assert zero_branch_env.provider.execution_mode in {"local", "client", "server"}
    assert zero_branch_env.provider.auto_mode is True


@pytest.mark.parametrize(
    ("enabled", "role", "expected_mode"),
    [
        (None, None, "local"),
        ("1", None, "client"),
        ("1", "server", "server"),
        ("1", "client", "client"),
    ],
)
def test_zero_branch_example_follows_environment_resolution(
    monkeypatch: pytest.MonkeyPatch,
    enabled: str | None,
    role: str | None,
    expected_mode: str,
) -> None:
    if enabled is None:
        monkeypatch.delenv("SHARED_TENSOR_ENABLED", raising=False)
    else:
        monkeypatch.setenv("SHARED_TENSOR_ENABLED", enabled)
    if role is None:
        monkeypatch.delenv("SHARED_TENSOR_ROLE", raising=False)
    else:
        monkeypatch.setenv("SHARED_TENSOR_ROLE", role)

    zero_branch_env = importlib.import_module("examples.zero_branch_env")
    zero_branch_env = importlib.reload(zero_branch_env)

    assert isinstance(zero_branch_env.provider, SharedTensorProvider)
    assert zero_branch_env.provider.execution_mode == expected_mode
    assert zero_branch_env.provider.auto_mode is True
    zero_branch_env.provider.close()
