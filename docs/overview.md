# Overview

`shared_tensor` is a narrow SDK for one specific problem:

- same host
- same GPU
- trusted local processes
- native PyTorch CUDA IPC object sharing

It is not a general RPC framework.

## Supported Payloads

- CUDA `torch.Tensor`
- CUDA `torch.nn.Module`
- container wrappers around those values where the transport contract allows them

## Explicitly Out Of Scope

- CPU tensor transport
- CPU module transport
- plain Python object RPC results
- cross-host transport
- MPS transport
- automatic device migration

## Runtime Modes

There are three runtime modes at the provider layer:

- `local`: call the registered function directly in-process
- `server`: own endpoints locally and expose them over UDS RPC
- `client`: turn shared functions into local RPC calls

There is also `execution_mode="auto"`, which resolves to one of the above based on:

- provider `enabled`
- `SHARED_TENSOR_ENABLED`
- `SHARED_TENSOR_ROLE`

## Two Planes

`shared_tensor` has two separate planes:

```text
control plane: Unix Domain Socket RPC
data plane:    native torch CUDA IPC serialization
```

That separation is the core design.

- RPC decides what to execute and where.
- torch IPC decides how CUDA objects are reopened across processes.

## Primary Production Shape

The preferred production topology is still explicit two-process deployment:

```text
client process --UDS RPC--> server process
client process <--CUDA IPC-- server process
```

Auto-start exists for convenience and local embedding. It is not the architectural center of the library.
