# Shared Tensor 测试套件

这个目录包含 shared-tensor 项目的完整测试套件，重新组织为清晰的模块化结构。

## 目录结构

```
shared-tensor/
├── scripts/                       # 脚本工具
│   └── run_server.py             # 服务器启动脚本
├── examples/                      # 独立运行示例
│   ├── README.md                 # 示例使用说明
│   ├── simple_test.py            # 基础功能演示
│   ├── example_usage.py          # 典型使用场景
│   ├── demo_async_vs_sync.py     # 异步vs同步演示
│   └── example_models.py         # PyTorch模型示例
├── tests/                         # 测试套件
│   ├── __init__.py               # 测试包初始化和配置
│   ├── README.md                 # 本文档
│   ├── run_tests.py              # 统一测试运行器
│   ├── test_structure.py         # 测试结构验证
│   ├── start_test_server.py      # 测试专用服务器
│   ├── unit/                     # 单元测试
│   │   ├── __init__.py
│   │   └── test_function_paths.py # 函数路径解析测试
│   ├── integration/              # 集成测试
│   │   ├── __init__.py
│   │   ├── test_async_system.py  # 异步系统集成测试
│   │   ├── test_client.py        # 客户端集成测试
│   │   └── test_jsonrpc_integration.py # JSON-RPC集成测试
│   └── pytorch_tests/            # PyTorch 相关测试
│       ├── __init__.py
│       ├── test_models.py        # 测试模型定义
│       ├── test_tensor_serialization.py # 基础tensor序列化测试
│       ├── test_gpu_tensors.py   # GPU tensor测试
│       ├── test_model_serialization.py # 模型序列化测试
│       ├── test_remote_execution.py # 远程执行测试
│       └── test_gpu_summary.py   # GPU tensor支持功能总结测试
└── shared_tensor/                 # 核心模块
    └── ...
```

## 测试分类

### 1. 单元测试 (unit/)
测试独立组件的功能，专注于单个模块或类的行为。

**文件:**
- `test_function_paths.py`: 函数路径解析和导入功能测试

### 2. 集成测试 (integration/)
测试组件间的交互和系统级功能。

**文件:**
- `test_async_system.py`: 异步任务系统的完整测试
- `test_client.py`: 客户端与服务器的交互测试
- `test_jsonrpc_integration.py`: JSON-RPC协议集成测试

### 3. PyTorch测试 (pytorch_tests/)
专门针对PyTorch功能的测试，包括GPU支持。

**重要说明**: 目录名为`pytorch_tests`而不是`torch`是为了避免与PyTorch的`torch`模块产生名称冲突，确保测试能正确导入PyTorch库。

**文件:**
- `test_models.py`: 测试用模型定义（供其他测试使用）
- `test_tensor_serialization.py`: 基础tensor序列化功能测试
- `test_gpu_tensors.py`: GPU tensor特定功能测试
- `test_model_serialization.py`: PyTorch模型序列化测试
- `test_remote_execution.py`: 远程PyTorch函数执行测试
- `test_gpu_summary.py`: GPU tensor支持功能总结测试

## 相关目录

### Examples (../examples/)
位于项目顶级目录，包含可独立运行的完整示例。这些示例已从测试体系中分离，作为独立的演示和教程。

**参见**: `../examples/README.md` 了解详细使用说明

## 服务器支持

项目提供两种服务器启动方式：

### 1. 生产/开发服务器 (`scripts/run_server.py`)
位于scripts目录，适用于：
- 开发和调试
- 生产部署
- 手动演示和交互测试

```bash
# 基本启动
python3 scripts/run_server.py

# 自定义配置
python3 scripts/run_server.py --host 0.0.0.0 --port 9000 --debug
```

### 2. 测试专用服务器 (`tests/start_test_server.py`)
位于tests目录，适用于：
- 自动化测试
- CI/CD环境
- 临时测试场景

```bash
# 启动测试服务器
python3 tests/start_test_server.py

# 限时运行
python3 tests/start_test_server.py --timeout 30
```

## 运行测试

### 使用统一测试运行器

```bash
# 查看环境信息
python3 tests/run_tests.py --env-info

# 运行所有测试
python3 tests/run_tests.py

# 运行特定类别的测试
python3 tests/run_tests.py --category torch
python3 tests/run_tests.py --category unit
python3 tests/run_tests.py --category integration
python3 tests/run_tests.py --category examples

# 只运行PyTorch相关测试
python3 tests/run_tests.py --torch-only

# 运行特定测试模块
python3 tests/run_tests.py --test tests.torch.test_tensor_serialization

# 详细输出
python3 tests/run_tests.py --verbose
```

### 直接运行单个测试文件

```bash
# 进入tests目录
cd tests

# 运行特定测试
python3 torch/test_tensor_serialization.py
python3 torch/test_gpu_tensors.py
python3 integration/test_async_system.py
```

### 使用unittest发现

```bash
# 从项目根目录运行
python3 -m unittest discover tests

# 运行特定模块
python3 -m unittest tests.torch.test_tensor_serialization
```

## 测试配置

测试环境通过 `tests/__init__.py` 中的配置进行管理：

```python
TEST_CONFIG = {
    'timeout': 30,           # 默认测试超时时间（秒）
    'retry_count': 3,        # 失败测试重试次数
    'gpu_tests_enabled': True,   # 启用GPU测试（如果CUDA可用）
    'remote_tests_enabled': False,  # 启用远程测试（需要服务器）
    'verbose': True,         # 详细测试输出
}
```

## 依赖管理

测试会自动检测以下依赖项的可用性：

- **PyTorch**: 如果不可用，PyTorch相关测试会被跳过
- **CUDA**: 如果不可用，GPU测试会被跳过  
- **服务器**: 如果服务器未运行，远程执行测试会被跳过

## 测试数据和模型

### 测试模型 (pytorch_tests/test_models.py)

该文件包含用于测试的PyTorch模型定义：

- `SimpleModel`: 基础的全连接模型
- `DynamicTrainingModel`: 可配置大小的训练模型
- `GPUTrainingModel`: 专为GPU测试优化的模型
- `ComplexModel`: 复杂的多层模型
- `CNNModel`: 卷积神经网络模型
- `ModelWithOptimizer`: 包含优化器的模型包装器

### 模型工厂函数

- `create_simple_model()`: 创建简单模型
- `create_dynamic_model(input_size, hidden_size, output_size)`: 创建动态大小模型
- `create_gpu_model(...)`: 创建GPU模型
- `create_complex_model(...)`: 创建复杂模型

## GPU测试支持

GPU测试具有完整的功能覆盖：

### 功能测试
- 多种数据类型GPU tensor序列化
- 多GPU环境兼容性
- 设备信息保持验证
- 大型tensor性能测试

### 模型测试
- GPU模型序列化
- 模型状态保持
- 设备迁移兼容性

### 远程执行测试
- 远程GPU tensor创建
- 远程GPU操作
- 远程GPU模型训练

## 远程执行测试

远程执行测试需要服务器运行：

```bash
# 启动服务器
python3 run_server.py

# 在另一个终端运行远程测试
python3 tests/run_tests.py --test tests.torch.test_remote_execution
```

## 故障排除

### 常见问题

1. **PyTorch不可用**: 所有PyTorch测试会被跳过
2. **CUDA不可用**: GPU测试会被跳过，但CPU测试继续运行
3. **服务器未运行**: 远程执行测试会被跳过
4. **导入错误**: 检查Python路径和依赖安装

### 调试技巧

1. 使用 `--verbose` 获取详细输出
2. 使用 `--env-info` 检查环境配置
3. 运行 `test_structure.py` 验证测试结构
4. 单独运行失败的测试文件进行调试

## 贡献指南

### 添加新测试

1. 确定测试类别（unit/integration/torch/examples）
2. 在相应目录创建测试文件
3. 遵循现有的命名约定
4. 添加适当的跳过条件（如PyTorch/CUDA依赖）
5. 更新 `run_tests.py` 中的测试配置

### 测试最佳实践

1. 使用描述性的测试名称
2. 添加适当的文档字符串
3. 使用 `setUp()` 和 `tearDown()` 管理测试环境
4. 为可选依赖添加跳过装饰器
5. 包含错误情况测试
6. 保持测试独立性

## 性能基准

测试套件包括性能基准：

- 大型tensor序列化性能
- GPU vs CPU执行时间对比
- 远程执行延迟测量
- 内存使用量评估

## 维护

测试套件的维护包括：

- 定期更新依赖兼容性
- 添加新功能的测试覆盖
- 性能回归检测
- 文档更新
- 清理过时测试

## 未来改进

计划的改进包括：

- [ ] 增加代码覆盖率报告
- [ ] 集成持续集成(CI)支持
- [ ] 添加性能回归测试
- [ ] 支持更多PyTorch版本
- [ ] 添加内存泄漏检测
- [ ] 实现测试并行执行
