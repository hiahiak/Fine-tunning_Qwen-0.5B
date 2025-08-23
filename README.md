# LoRA 微调 Qwen 项目说明

本项目基于 Qwen-0.5B-Chat 模型，使用 LoRA（Low-Rank Adaptation）方法进行高效微调，实现了对个人知识（hiahia相关信息）的定制化问答。

> **注意：本项目涉及大模型权重文件（如 .pth、.safetensors），请勿直接推送到 GitHub。建议在 `.gitignore` 中忽略大文件，详见下文“推送与大文件管理”部分。**
## 推送与大文件管理

**强烈建议：不要将大模型权重（如 `best_model.pth`、`model.safetensors`）等大文件直接推送到 GitHub，否则会导致 push 失败或仓库异常膨胀。**

### .gitignore 推荐配置
在项目根目录下的 `.gitignore` 文件中添加：

```
*.pth
*.safetensors
output/
model/
__pycache__/
```

### 移除已被 git 跟踪的大文件
如果你之前已经 add/commit 过大文件，需用如下命令移除索引：

```bash
git rm --cached output/best_model.pth
git rm --cached model/model.safetensors
```

然后重新 commit 并 push。

### 推送注意事项
- 首次推送前，确保 `.gitignore` 已生效且大文件未被跟踪。
- 若历史中已包含大文件且 push 失败，可考虑删除 `.git` 重新初始化仓库（适合个人项目/不需保留历史时），或用 BFG 等工具清理历史。
- 大文件建议用网盘等方式单独分享。

## 目录结构

```
Fine-tunning_Qwen/
├── config.json           # 微调超参数配置
├── data.py               # 问答数据集（自定义hiahia相关知识）
├── inject.py             # LoRA注入与冻结参数逻辑
├── LoRALinear.py         # LoRA线性层实现
├── loss_curve.png        # 最近一次训练损失曲线
├── train.py              # 主训练与推理脚本
├── output/               # 输出目录（保存最佳模型和损失曲线）
│   ├── best_model.pth    # 验证集上表现最优的模型权重
│   └── loss_curve.png    # 训练/验证损失曲线
├── model/                # Qwen-0.5B-Chat原始模型及分词器文件
└── __pycache__/          # Python缓存文件
```

## 主要功能
- 支持基于LoRA的高效参数微调，仅训练部分注意力层，极大减少显存和计算需求。
- 数据集可自定义，适合个人知识、专属问答等场景。
- 训练过程自动划分训练/验证集，自动保存最佳模型。
- 支持训练损失与验证损失曲线可视化。
- 推理时自动加载最佳模型，支持自定义问题测试。

## 快速开始

### 1. 安装依赖
请确保已安装 PyTorch、transformers 及相关依赖。

```bash
pip install torch transformers matplotlib
```

### 2. 数据准备
编辑 `data.py`，补充/修改 `qa_data` 列表，格式如下：
```python
qa_data = [
    {"instruction": "谁是hiahia？", "input": "", "output": "hiahia是一个真实的人，2006年出生，来自广东河源。"},
    # ...更多问答...
]
```

### 3. 配置参数
编辑 `config.json`，可调整学习率、epoch、LoRA层、目标层等参数：
```json
{
    "model_name": "Qwen/Qwen-0.5B-Chat",
    "lora_rank": 8,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "v_proj"],
    "target_layers": [20, 21, 22, 23],
    "lr": 5e-5,
    "epoch": 10
}
```

### 4. 运行训练
```bash
python train.py
```
训练过程会自动保存验证集上表现最优的模型到 `output/best_model.pth`，并绘制损失曲线。

### 5. 推理测试
训练结束后，`train.py` 会自动加载最佳模型并对示例问题进行推理，输出结果。

## 关键文件说明
- `inject.py`：负责将 LoRA 注入到 Qwen 模型指定层，并冻结非LoRA参数。
- `LoRALinear.py`：LoRA线性层的PyTorch实现。
- `train.py`：训练主流程，包含数据加载、训练、验证、推理、可视化等。
- `data.py`：自定义问答数据集。
- `config.json`：所有超参数配置。

## 常见问题
- **模型权重文件很大？**
  > `best_model.pth` 包含了整个Qwen模型的权重，属于正常现象。**不建议上传到 GitHub**，部署时可考虑只保存LoRA权重以减小体积。
- **推送时报错/断开？**
  > 99% 是因为大文件未被正确忽略或历史中有大文件，详见上文“推送与大文件管理”。
- **如何扩充知识？**
  > 直接在 `data.py` 中添加更多问答对，内容越丰富，模型表现越好。
- **如何自定义推理问题？**
  > 修改 `train.py` 最后推理部分的 `messages` 内容即可。

