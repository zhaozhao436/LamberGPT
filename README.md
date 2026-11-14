# LamberGPT


## 文件

- `train_clm_pt_full.py`：全参数预训练，支持继续预训练和随机初始化。
- `train_clm_pt_lora.py`：LoRA 继续预训练。
- `train_sft_lora.py`：LoRA 指令微调，只计算回答部分的损失。
- `train.py`：参数解析、模型加载、Trainer、保存和评估。
- `data.py`：文本切块、指令编码、动态补齐。

## 安装

建议 Python 3.10 或 3.11。根据训练设备安装对应的 PyTorch，然后安装依赖：

```sh
pip install -r requirements.txt
```


