# MultiModalDTA-LM (train-only)

本仓库仅保留论文实验所用的 **训练代码与模型实现**，用于在不同数据集上复现实验流程。  
由于 `dataset/` 体积过大，**不随仓库提供**；请自行准备数据并按脚本中的路径配置运行。

## 内容包含什么
- `train/model/`：论文中用到的模型实现（1D/2D/3D 以及多模态版本）
- `train/` 下的多个 `train_*.py`：不同数据集对应的训练脚本（Davis/KIBA/BindingDB/ATOM3D/FDavis/Metz/ToxCast/IEDB2016/BD2017/IL6 等）
- `train/util.py`：训练/数据处理/评估等通用工具函数（脚本会直接 import 使用）
- `logs/`：训练日志输出目录（如果脚本启用 TensorBoard/日志记录会写到这里）

## 仓库结构（简化）
```text
.
└─ train/
   ├─ model/
   │  ├─ model_1D.py
   │  ├─ model_2D.py
   │  ├─ model_3D.py
   │  ├─ model_3D_si30.py
   │  ├─ model_3D_si60.py
   │  ├─ model_multimodal.py
   │  ├─ model_multimodal_lm.py
   │  ├─ ...（其它多模态/变体）
   │  └─ Uni-Core-main/          # 依赖/第三方模块（若脚本引用）
   ├─ logs/
   ├─ util.py
   ├─ train_davis.py
   ├─ train_kiba.py
   ├─ train_bindingdb_multimodal_lm.py
   ├─ train_atom3D_multimodal_si30_lm.py
   ├─ train_atom3D_multimodal_si60_lm.py
   ├─ train_bd2017_multimodal_lm.py
   ├─ train_iedb2016_multimodal_lm.py
   ├─ train_il6_aai_multimodal_lm.py
   └─ ...（更多 train_*.py）
