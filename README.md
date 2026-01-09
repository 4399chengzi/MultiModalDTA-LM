# MultiModalDTA-LM (train-only)

本仓库仅保留论文实验所用的**训练代码**与**模型实现**（仅 `train/` 目录）。  
由于数据集体积较大，`dataset/` **不随仓库提供**；请在本地自行准备数据，并在对应训练脚本中配置数据路径后运行。

---

## 目录结构

```text
.
└─ train/
   ├─ logs/                         # 训练日志输出目录（是否生成取决于脚本）
   ├─ model/                        # 模型实现（1D/2D/3D/多模态）
   │  ├─ __init__.py
   │  ├─ model_1D.py
   │  ├─ model_2D.py
   │  ├─ model_3D.py
   │  ├─ model_3D_si30.py
   │  ├─ model_3D_si60.py
   │  ├─ model_multimodal.py
   │  ├─ model_multimodal_lm.py
   │  └─ Uni-Core-main/             # 如脚本引用到的第三方/依赖模块（保留原结构）
   ├─ util.py                       # 通用工具（训练/评估/数据处理等，供各脚本 import）
   ├─ test.py                       # 简单测试/调试脚本（如有）
   ├─ train_davis.py
   ├─ train_kiba.py
   ├─ train_bindingdb_multimodal_lm.py
   ├─ train_fdavis.py
   ├─ train_fdavis_multimodal_lm.py
   ├─ train_metz_multimodal_lm.py
   ├─ train_toxcast_multimodal_lm.py
   ├─ train_atom3D_1D.py
   ├─ train_atom3D_2D.py
   ├─ train_atom3D_3D.py
   ├─ train_atom3D_3D_si30.py
   ├─ train_atom3D_3D_si60.py
   ├─ train_atom3D_multimodal_si30.py
   ├─ train_atom3D_multimodal_si30_lm.py
   ├─ train_atom3D_multimodal_si60.py
   ├─ train_atom3D_multimodal_si60_lm.py
   ├─ train_bd2017_multimodal_lm.py
   ├─ train_iedb2016_multimodal_lm.py
   └─ train_il6_aai_multimodal_lm.py
