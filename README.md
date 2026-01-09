# MultiModalDTA-LM (train-only)

本仓库仅保留论文实验所用的**训练代码**与**模型实现**（仅 `train/` 目录）。  
由于数据集体积较大，`dataset/` **不随仓库提供**；你需要在本地自行准备数据，并在对应训练脚本中配置数据路径后运行。

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
环境与依赖

本仓库是 PyTorch 训练脚本集合；具体依赖会随你启用的模态变化（是否使用图结构、3D、分子工具等）。
建议使用：

Python 3.8+

PyTorch（CUDA/CPU 视你机器配置）

常见科学计算依赖：numpy、scipy、scikit-learn、tqdm、pandas

可能按脚本需要额外安装：

图/分子图：dgl 或 torch-geometric

分子处理：rdkit

其它依赖以各 train_*.py 顶部 import 报错提示为准（缺啥装啥）

数据集说明（不随仓库提供）

dataset/ 不上传，你需要在本地准备数据并让脚本能找到它。

一个常用的本地组织方式如下（示例）：

D:\SimbaDTA\SimbaDTA\
├─ train\                 # 本仓库代码（GitHub 上只有这个）
└─ dataset\               # 你本地准备的数据（不上传）


如果你的数据放在其它位置，请到对应的 train_*.py 中修改数据路径（不同脚本可能有不同的默认路径/缓存路径）。
若脚本依赖已处理好的缓存（例如 .npz / .pt / .lmdb），请按你的实验流程提前生成并放到脚本期望的位置。

快速运行（示例）

在仓库根目录执行（Windows PowerShell / CMD / Linux bash 都可以）：

Davis
python train/train_davis.py

KIBA
python train/train_kiba.py

BindingDB（多模态 LM）
python train/train_bindingdb_multimodal_lm.py

ATOM3D（SI-30 / SI-60，多模态 LM）
python train/train_atom3D_multimodal_si30_lm.py
python train/train_atom3D_multimodal_si60_lm.py

BD2017 / IEDB2016 / IL6（示例）
python train/train_bd2017_multimodal_lm.py
python train/train_iedb2016_multimodal_lm.py
python train/train_il6_aai_multimodal_lm.py


训练输出（日志/权重/结果文件）保存位置以脚本内配置为准，常见会写到 train/logs/ 或脚本指定的输出目录。

备注

本仓库用于复现实验训练流程与模型结构参考。

数据集与大体积文件不上传到 GitHub，建议用本地磁盘/网盘/对象存储管理数据与权重。

如果你希望他人开箱即用，建议在各 train_*.py 中把“数据路径/输出路径/超参数”集中成一个配置区，或增加一个统一的配置文件。
