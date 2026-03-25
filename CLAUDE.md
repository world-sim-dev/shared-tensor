# CLAUDE.md

## Mission

Keep `shared_tensor` narrow, explicit, and production-maintainable.

This repository exists to move CUDA `torch.Tensor` and CUDA `torch.nn.Module` objects across processes on the same host and the same GPU using native PyTorch IPC semantics. Every change should make that path more reliable and easier to reason about.

## Product Definition

- `torch` is a hard dependency.
- The only supported data plane is same-host, same-GPU CUDA IPC.
- Public API is endpoint-oriented and explicitly registered.
- Trusted internal environments are in scope. Public-internet hardening is out of scope.
- Generic Python-object RPC is out of scope.

## Runtime Rules

- Supported payloads are CUDA `torch.Tensor` and CUDA `torch.nn.Module`.
- `args` and `kwargs` may contain only supported payloads, wrapped in `tuple`, `list`, or `dict[str, ...]`.
- Empty `args` and `kwargs` may use the control encoding for no-argument calls only.
- CPU tensors, CPU modules, plain Python values, and `mps` payloads must fail fast.
- Never fake GPU sharing by copying to CPU.
- Never auto-call `.cuda()` during transport.
- Never deserialize CUDA IPC payloads in the wrong process just for convenience.

## API Principles

- Prefer explicit endpoint names over import-path execution.
- Keep sync and async interfaces aligned.
- Keep compatibility helpers small and secondary to the main API.
- Remove design complexity instead of hiding it behind more configuration.

## Testing Standard

- `pytest` is the source of truth.
- Non-GPU runs must cover protocol, provider, sync, async, and control-path behavior.
- GPU runs must validate real cross-process CUDA IPC behavior.
- GPU tests should skip cleanly when CUDA is unavailable.
- Examples and docs must only show supported behavior.

## Packaging

- Python baseline is `3.10+`.
- Keep runtime dependencies minimal.
- Package version must stay aligned across code and metadata.
- Release docs should describe the single supported production path, not speculative future modes.

## Reject These Changes

- Dynamic arbitrary `module:function` execution from untrusted input
- Generic pickle RPC for non-empty application payloads
- Silent CPU fallback
- Hidden runtime mode switches for core transport semantics
- New non-CUDA transport layers added as side paths to the core product
