# Shared Tensor Examples

这个目录包含完整的使用示例，展示 shared-tensor 库的各种功能。所有示例都可以独立运行。

## 📋 示例列表

### 1. `simple_test.py` - 基础功能演示
演示基础的JSON-RPC功能和远程函数执行。

**使用方法:**
```bash
cd shared-tensor
python3 examples/simple_test.py
```

**功能:**
- JSON-RPC协议测试
- 基础服务器/客户端交互
- 远程函数调用

### 2. `example_usage.py` - 典型使用场景
展示如何使用 `@provider.share` 装饰器进行函数共享。

**使用方法:**
```bash
# 1. 启动服务器（在另一个终端）
python3 scripts/run_server.py

# 2. 运行示例
python3 examples/example_usage.py
```

**功能:**
- 函数装饰器使用
- 远程函数调用
- 参数序列化/反序列化

### 3. `demo_async_vs_sync.py` - 异步vs同步演示
对比异步和同步执行模式的差异。

**使用方法:**
```bash
# 1. 启动服务器（在另一个终端）
python3 scripts/run_server.py

# 2. 运行演示
python3 examples/demo_async_vs_sync.py
```

**功能:**
- 异步任务执行
- 性能对比
- 并发处理演示

### 4. `example_models.py` - PyTorch模型示例
展示PyTorch模型的创建、训练和序列化。

**使用方法:**
```bash
python3 examples/example_models.py
```

**功能:**
- PyTorch模型定义
- GPU/CPU兼容性
- 模型训练和推理

## 🚀 快速开始

### 1. 启动服务器
```bash
# 基本启动
python3 scripts/run_server.py

# 自定义配置
python3 scripts/run_server.py --host 0.0.0.0 --port 9000 --debug
```

### 2. 运行示例
```bash
# 运行所有独立示例
python3 examples/simple_test.py
python3 examples/example_models.py

# 运行需要服务器的示例
python3 examples/example_usage.py
python3 examples/demo_async_vs_sync.py
```

## 📝 依赖要求

### 基础依赖
- Python 3.7+
- 项目核心模块 (`shared_tensor`)

### 可选依赖
- **PyTorch**: `example_models.py` 需要PyTorch
- **CUDA**: GPU相关功能需要CUDA支持

### 检查环境
```bash
# 检查依赖
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 🔧 故障排除

### 常见问题

1. **模块导入错误**
   ```
   ModuleNotFoundError: No module named 'shared_tensor'
   ```
   **解决方案**: 确保在项目根目录运行，或者设置PYTHONPATH

2. **服务器连接失败**
   ```
   ConnectionRefusedError: [Errno 111] Connection refused
   ```
   **解决方案**: 先启动服务器 `python3 scripts/run_server.py`

3. **PyTorch未安装**
   ```
   ModuleNotFoundError: No module named 'torch'
   ```
   **解决方案**: 安装PyTorch或跳过PyTorch相关示例

## 📚 更多信息

- **测试**: 参见 `tests/` 目录
- **文档**: 参见项目根目录的README
- **API参考**: 查看 `shared_tensor/` 模块源码

