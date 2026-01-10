# OnnxRunner 模块文档
## 📘 简介
ONNXRunner 是一个依托于 Linger 主体功能开发的轻量级推理模块，用于直接加载 ONNX 模型并基于给定输入（`torch.Tensor`）执行推理。 它同时支持中间 Tensor 的自动 dump，用于结果验证与调试。

## ⚙️ 安装与依赖
OnnxRunner 随 Linger 一起发布，无需单独安装。

## 🚀 使用示例

```python
import torch
from linger.checker.onnxrunner import OnnxRunner
# 创建 OnnxRunner 实例
model = OnnxRunner("data.ignore/conv_linear.onnx", dump=True, dump_format='all')
# 创建随机输入
input_tensor = torch.randn(1, 3, 64, 64)
# 获取推理结果
output = model.run([input_tensor])
```

## 🧩 参数说明
| 参数名           | 类型     | 默认值     | 说明                                         |
| ------------- | ------ | ------- | ------------------------------------------ |
| `model_path`  | `str`  | 无       | ONNX 模型文件路径，必须提供                                |
| `dump`        | `bool` | `False` | 是否启用中间 Tensor dump 功能                       |
| `dump_format` | `str`  | `'all'` | dump 输出格式，可选 `'all'` / `'int'` / `'float'` |

## 🧩 API-run()
### 函数定义

```python
def run(self, data, special_key='None', out_type="list"):
```

### 参数说明

| 参数名           | 类型                                    | 默认值      | 说明                                                                                            |
| ------------- | ------------------------------------- | -------- | --------------------------------------------------------------------------------------------- |
| `data`        | `torch.Tensor` 或 `List[torch.Tensor]` | 必填       | 模型输入数据。当模型有多个输入时，可传入张量列表，顺序与模型输入节点定义一致。                                                       |
| `special_key` | `str` 或 `None`                        | `'None'` | 用于指定一个中间Tensor，便于导出该节点的输出结果。若设置此值且 `out_type="dict"`，会在输出字典中额外包含 `{special_key: tensor}` 键值对。 |
| `out_type`    | `str`                                 | `"list"` | 控制输出格式：<br> - `"list"`：输出为按节点顺序排列的张量列表；<br> - `"dict"`：输出为以节点名为键的字典形式。                        |

### 返回值说明
| 返回类型                      | 说明                                                            |
| ------------------------- | ------------------------------------------------------------- |
| `List[torch.Tensor]`      | 默认返回，依次包含模型输出节点的结果。                                           |
| `Dict[str, torch.Tensor]` | 当 `out_type="dict"` 时返回，以节点名为键。若指定了 `special_key`，该键也会包含在结果中。 |

### 示例
#### 1️⃣ 默认使用 list 输出
```python
input_tensor = torch.randn(1, 3, 64, 64)
output_list = model.run([input_tensor])
print(type(output_list))  # <class 'list'>
```
#### 2️⃣ 使用 dict 输出
```python
output_dict = model.run([input_tensor], out_type="dict")
print(output_dict.keys())  # dict_keys(['output_0'])
```
#### 3️⃣ 导出中间层结果
```python
output_with_mid = model.run([input_tensor], special_key="1123", out_type="dict")
print(output_with_mid.keys())  # dict_keys(['Conv_3', 'output_0'])
```
## 🧪 Dump 功能说明
当 `dump=True` 时，OnnxRunner 会在`./dump`下生成推理过程的中间结果，用于算子级别验证与调试。
`dump_format` 参数用于控制 **dump 的数据类型与保存路径**，支持以下三种模式：
| dump_format | 说明 | 输出内容 | Dump 路径 |
|--------------|------|-----------|------------|
| `'all'` | 同时导出浮点与定点结果 | 浮点结果与定点结果 | `./dump/float/` 和 `./dump/int/` |
| `'float'` | 仅导出浮点结果 | 浮点张量（FP32） | `./dump/float/` |
| `'int'` | 仅导出定点结果 | 定点张量（INT8 / INT16 / INT32） | `./dump/int/` |
---