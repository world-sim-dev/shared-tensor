# CLAUDE.md

## Mission

Keep `shared_tensor` small, explicit, and production-maintainable.

The library exists to expose named PyTorch endpoints over a localhost-first RPC transport for same-host, same-GPU object sharing. Every change should make the runtime behavior easier to reason about, not more magical.

## Product Boundaries

- Default deployment target is a trusted same-host internal environment.
- Public-internet hardening is out of scope unless a future change adds explicit auth, rate limiting, payload validation, and a documented security model.
- The public API is endpoint-oriented. Arbitrary remote imports and arbitrary code execution are forbidden.
- Generic Python-object RPC is out of scope.

## Runtime Rules

- `torch` is a hard dependency.
- Register endpoints explicitly with `register()` or `@share()`.
- Supported transport payloads are CUDA `torch.Tensor` and CUDA `torch.nn.Module` values, including tuple/list/dict containers used for args and kwargs.
- Same-host CUDA transport is allowed only when PyTorch IPC supports it. Never fake GPU sharing by silently copying to CPU.
- CPU tensors, CPU modules, and arbitrary Python objects must fail fast with a capability error.
- Empty `args` and `kwargs` may use a tiny control encoding only for no-argument RPC calls.
- Errors must be typed and actionable: configuration, protocol, serialization, capability, remote execution, and task lifecycle failures stay distinct.

## API Principles

- Prefer explicit endpoint names over import paths.
- Prefer a small stable surface over broad compatibility shims.
- Keep sync and async APIs aligned: `call` for direct execution, `submit/status/result/cancel` for tasks.
- Compatibility helpers may exist, but new docs and tests must use the modern API.

## Testing Standard

- `pytest` is the single source of truth.
- Unit tests cover endpoint registration, protocol handling, caching, serialization rules, and error mapping.
- Integration tests cover real server/client behavior for sync and async flows.
- GPU tests must be isolated with a `gpu` marker and skip cleanly when CUDA is unavailable.
- Non-GPU tests should still verify control-path behavior such as empty-call RPC, endpoint discovery, caching, and task lifecycle semantics.
- Examples should be smoke-testable and reflect supported behavior only.

## Packaging And Tooling

- Python baseline is `3.10+`.
- Keep dependency surface minimal.
- Do not add repo-local scripts that duplicate package CLI behavior.
- Version metadata must stay consistent between the package and `pyproject.toml`.

## Anti-Patterns To Reject

- Dynamic `module:function` execution from untrusted client input
- Environment-variable-driven hidden mode switches for core behavior
- Silent device migration or automatic `.cuda()` calls in serialization
- Generic pickle transport for non-empty RPC payloads
- Example code or tests that depend on unsupported CPU payload transport
