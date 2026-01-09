import os                                   # 操作系统相关（路径、文件存在性等）
from pathlib import Path                    # 更安全/清晰的路径拼接
import time                                 # 计时
import logging                              # 打日志
import csv                                  # 训练指标写入CSV
import matplotlib                           # 服务器环境无GUI时必须先设置Agg后端
from torch.utils.tensorboard import SummaryWriter
matplotlib.use("Agg")                       # 使用非交互后端，防止无显示环境报错
import matplotlib.pyplot as plt             # 画训练曲线PNG
import torch                                # PyTorch 主库
import torch.nn as nn                       # 常用神经网络模块
import torch.optim as optim                 # 优化器
import torch.utils.data as Data             # DataLoader、random_split等
from model.model_1D import SimbaDTA         # 你的模型定义（保持不变）
import util

# ========== 可配置区域 ==========
DATA_DB_PATH = r"../dataset/ATOM3D/raw/pdbbind_2019-refined-set/data"  # Windows绝对路径，r""避免转义
DATASET_NAME = "atom3d_raw"                    # 数据集名（只用于结果目录命名）
OUT_DIR = "../dataset/ATOM3D/processed_3d"
SMILES_MAXLEN = 128                         # 药物SMILES最大长度（沿用fdavis设置，可按需调整）
PROSEQ_MAXLEN = 64                       # 蛋白序列最大长度（沿用fdavis设置，可按需调整）
TRAIN_RATIO = 0.8                          # 训练集比例（其余作为测试集；可换 0.9/0.1 等）
EPOCHS = 600
# 训练轮数
BATCH_SIZE = 32                            # batch大小
ACCUM_STEPS = 32                           # 梯度累积步数（显存不足时提高此值）
LR = 1e-3                                  # 学习率
NUM_WORKERS = 4                            # DataLoader并行读取进程数（Windows可先4）
PIN_MEMORY = True                          # pin_memory加速CPU->GPU拷贝
LOG_LEVEL = logging.INFO                   # 日志等级

# ========== 主程序入口 ==========
if __name__ == "__main__":
    # 1) 基础日志配置
    logging.basicConfig(level=LOG_LEVEL)   # 配置全局日志等级
    util.seed_torch()                      # 设定随机种子（你的util中应实现，确保可复现实验）

    # 2) 结果目录与文件
    result_root = Path(f"../result/{DATASET_NAME}")   # 结果根目录：./result/atom3d
    run_dir     = result_root / "runs"               # TensorBoard日志目录
    fig_dir     = result_root / "figs"               # 训练曲线PNG目录
    model_path  = result_root / "model.pth"          # 最优模型权重保存路径
    ckpt_path   = result_root / "checkpoint.pth.tar" # 断点恢复文件
    out_txt     = result_root / "output_metrics.txt" # 最终测试指标txt
    csv_file    = result_root / "metrics.csv"        # 训练过程指标CSV

    # 3) 确保以上目录存在
    for p in [run_dir, fig_dir, result_root]:
        p.mkdir(parents=True, exist_ok=True)         # 若目录不存在则创建

    # 4) 初始化TensorBoard记录器
    writer = SummaryWriter(log_dir=str(run_dir))     # 指向 runs 目录

    # 5) 读取数据（从LMDB拉取smiles/seq/score；util中应、。，、。，
    # 提供LoadData_atom3d）
    drug_seqs, target_seqs, affi = util.LoadData_atom3d_3d(
        root=DATA_DB_PATH,
        out_dir=OUT_DIR,
        out_name="train_3d_unimol2_small",
        model_size="unimol2_small"
    )

    print("Ligand embeddings shape :", drug_seqs.shape)
    print("Pocket embeddings shape :", target_seqs.shape)
    print("Affinity array shape    :", affi.shape)
    print(affi)

    # print(len(drug_seqs), len(target_seqs), len(affi))
    # print(drug_seqs[0], target_seqs[0][:50], affi[0])
    # print(drug_seqs[1], target_seqs[1][:50], affi[1])
    # print(drug_seqs[2], target_seqs[2][:50], affi[2])
    logging.info(f"Loaded ATOM3D(LBA) samples: {len(affi)}")           # 打印样本数量
