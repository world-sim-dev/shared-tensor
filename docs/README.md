# Docs

This directory documents the actual runtime design of `shared_tensor`.

Recommended reading order:

- `overview.md`: scope, terminology, and the main runtime modes
- `architecture.md`: control plane, data plane, and component boundaries
- `diagrams.md`: flow charts for requests, tasks, cache, and managed refs
- `lifecycle.md`: object lifetime, task lifetime, cache lifetime, and release semantics
- `autostart.md`: auto mode, thread-backed local server behavior, and same-process short-circuiting
- `patterns.md`: canonical usage patterns and parameter combinations

The docs are intentionally design-first. They describe the behavior the tests enforce today.
