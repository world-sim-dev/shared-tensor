# Shared Tensor

`shared_tensor` 只做一件事：在同一台机器上，用原生 PyTorch 语义，把 CUDA `torch.Tensor` 和 CUDA `torch.nn.Module` 在两个进程之间传过去用。

不做这些事：

- 不支持 CPU tensor 传输
- 不支持普通 Python 对象 RPC
- 不支持 macOS `mps`
- 不做跨机器分布式传输

## Scope

- 同机
- 同 GPU
- 信任环境
- native torch
- 端点显式注册

## Payload Rules

- 支持 CUDA `torch.Tensor`
- 支持 CUDA `torch.nn.Module`
- 支持由它们组成的 `tuple` / `list` / `dict[str, ...]`，用于 `args` 和 `kwargs`
- 空 `args` / `kwargs` 允许走一个极小的 control path，只用于无参调用

下面这些会直接失败：

- CPU tensor
- CPU module
- 普通 `dict` / `list` / `int` / `str` 等 Python 对象
- `mps` tensor / `mps` module

## Install

需要 Python `3.10+` 和可用的 CUDA 版 PyTorch。

```bash
pip install -e ".[dev,test]"
```

## Minimal Example

服务端 provider：

```python
import torch

from shared_tensor import SharedTensorProvider

provider = SharedTensorProvider(execution_mode="client")


@provider.share(name="load_model")
def load_model() -> torch.nn.Module:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    return torch.nn.Linear(4, 2, device="cuda")


@provider.share(name="echo_tensor")
def echo_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_cuda:
        raise RuntimeError("CUDA tensor required")
    return tensor
```

启动服务：

```bash
shared-tensor-server --provider my_service:provider --host 127.0.0.1 --port 2537
```

客户端进程直接拿 CUDA 对象：

```python
import torch

from shared_tensor import SharedTensorClient

with SharedTensorClient(port=2537) as client:
    model = client.call("load_model")
    x = torch.ones(1, 4, device="cuda")
    y = model(x)
    z = client.call("echo_tensor", x)
```

## MPS

不能。这个项目当前产品定义就是 CUDA-only IPC。

- `mps` 不是 CUDA
- 当前运行时校验按 CUDA 写死
- 目标能力是 PyTorch CUDA 跨进程对象共享，不是泛 GPU 抽象层

所以在 macOS 上你最多只能做非目标场景的单进程实验，不能把它当成这个仓库的核心能力验证。

## Development

推荐环境：

```bash
conda create -y -n shared-tensor-dev python=3.11
conda activate shared-tensor-dev
pip install -e ".[dev,test]"
```

默认测试：

```bash
python -m pytest -m "not gpu"
```

有 CUDA 时再跑：

```bash
python -m pytest -m gpu
```

## Repo Notes

- [CLAUDE.md](/Users/mapix/workspace/shared-tensor/CLAUDE.md) 是仓库约束
- [examples/basic_service.py](/Users/mapix/workspace/shared-tensor/examples/basic_service.py) 是最小同步示例
- [examples/async_service.py](/Users/mapix/workspace/shared-tensor/examples/async_service.py) 是异步示例
- [examples/model_service.py](/Users/mapix/workspace/shared-tensor/examples/model_service.py) 是模型 handoff 示例
