# DPO (Direct Preference Optimization) 训练代码

## 概述

这是一个完整的DPO训练实现，用于基于人类偏好数据训练语言模型。DPO是一种直接偏好优化方法，通过比较两个回答（chosen vs rejected）来训练模型产生更符合人类偏好的回答。

## 主要特性

- **数据格式转换**: 自动将原始问答数据转换为DPO训练格式
- **完整的DPO实现**: 包含损失函数、训练器和数据集类
- **LoRA微调**: 使用PEFT进行高效微调
- **自动设备检测**: 自动检测并使用GPU/CPU
- **模型保存**: 自动保存训练好的模型

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据格式

### 输入数据格式
```json
{
  "query": "用户问题",
  "base_response": "基础回答（较差的回答）",
  "expanded_answer": "扩展回答（更好的回答）"
}
```

### DPO训练格式
```json
{
  "prompt": "用户问题",
  "chosen": "更好的回答",
  "rejected": "较差的回答"
}
```

## 使用方法

### 1. 准备数据
将您的问答数据按照上述格式组织，放入`main()`函数中的`raw_data`列表。

### 2. 配置模型
在`main()`函数中修改`model_name`变量，选择适合的基础模型：
```python
model_name = "THUDM/chatglm2-6b"  # 或其他可用模型
```

### 3. 运行训练
```bash
python DPO.py
```

## 核心组件

### 1. 数据转换 (`convert_to_dpo_format`)
- 将原始数据转换为DPO训练格式
- 自动将`expanded_answer`作为chosen，`base_response`作为rejected

### 2. DPO数据集 (`DPODataset`)
- 处理tokenization和padding
- 支持自定义最大长度
- 返回训练所需的张量格式

### 3. DPO损失函数 (`DPOLoss`)
- 实现标准的DPO损失计算
- 可调节的beta参数控制偏好强度

### 4. DPO训练器 (`DPOTrainer`)
- 完整的训练循环实现
- 支持LoRA微调
- 自动保存训练好的模型

## 训练参数

- **beta**: DPO温度参数，控制偏好强度（默认0.1）
- **learning_rate**: 学习率（默认5e-5）
- **max_length**: 最大序列长度（默认512）
- **batch_size**: 批次大小（默认2）
- **num_epochs**: 训练轮数（默认3）

## 输出文件

- `dpo_training_data.json`: 转换后的DPO训练数据
- `./dpo_trained_model/`: 训练好的模型文件夹

## 注意事项

1. **模型兼容性**: 确保选择的基础模型与您的硬件兼容
2. **内存需求**: DPO训练需要较大内存，建议使用GPU
3. **数据质量**: 确保chosen回答确实比rejected回答质量更高
4. **模型大小**: 大模型训练时间较长，建议先用小模型测试

## 扩展功能

- 支持自定义损失函数
- 可添加验证集评估
- 支持早停和模型检查点
- 可集成wandb等日志工具

## 故障排除

如果遇到模型加载错误，请尝试：
1. 检查模型名称是否正确
2. 确认网络连接正常
3. 尝试使用其他可用的预训练模型
4. 检查依赖包版本兼容性 