# -*- coding: utf-8 -*-                                      # 行：声明源码文件使用 UTF-8 编码，防止中文注释乱码
# =========================================================
# IL-6 抗原-抗体 1D LM 训练脚本（带早停）
#
# 任务：二分类（binder / non-binder）
# 模态：仅用 1D LM 特征（VHH / 抗原均为 ESM-2 向量）
# 指标：Precision, Recall, F1, PR-AUC
#
# 模型：MultiModalPPI_LM（只启用 1D 分支，2D/3D 参数先占位）
# 数据：LoadData_il6_aai_lm_1d（你 util 里已经实现的导入函数）
# =========================================================

from pathlib import Path                                    # 行：路径处理工具
import logging                                              # 行：日志模块
import time                                                 # 行：计时用
import csv                                                  # 行：写入 CSV 指标
from typing import Dict, Any                                # 行：类型注解

import numpy as np                                          # 行：数值运算
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter           # 行：TensorBoard 日志

import util                                                 # 行：你自己的工具包（含 seed、checkpoint 等）
from util import LoadData_il6_aai_lm_1d                     # 行：IL-6 AAI 1D LM 数据加载函数

# ========== 设备/日志/随机种子 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 行：有 GPU 就用 GPU
torch.backends.cudnn.benchmark = True                                   # 行：固定输入尺寸可开启 benchmark 加速
LOG_LEVEL = logging.INFO                                                # 行：日志等级
SANITY_CHECK = True                                                     # 行：是否做一次 batch 形状检查

# ========== 可配置区域 ==========
DATA_IL6_AAI = r"../dataset/AVIDa-hIL6"             # 行：IL-6 AAI 原始数据目录（内含 il6_aai_dataset.csv）
DATASET_NAME = "il6_aai_lm_1d"                      # 行：任务名，用于结果目录命名
OUT_DIR      = f"../result/{DATASET_NAME}"          # 行：结果根目录（日志、模型、曲线都放这里）

BATCH_SIZE   = 64                                   # 行：batch 大小，可按显存调整
NUM_WORKERS  = 0                                    # 行：DataLoader 的 worker 数（Win 下可改小）
PIN_MEMORY   = bool(torch.cuda.is_available())      # 行：GPU 训练时开启 pin_memory 提升数据拷贝速度

EPOCHS         = 600                                # 行：最大 epoch 数（配合早停，不一定跑满）
LR             = 1e-4                               # 行：学习率（LM 向量一般用偏小的 LR）
WEIGHT_DECAY   = 1e-4                               # 行：L2 权重衰减
USE_AMP        = False                              # 行：是否使用混合精度训练
ACCUM_STEPS    = 2                                  # 行：梯度累积步数（1 表示不用累积）
GRAD_CLIP_NORM = 0.5                                # 行：梯度裁剪阈值，防止梯度爆炸
USE_SCHEDULER  = True                               # 行：是否使用余弦退火学习率调度
VAL_AS_BEST    = True                               # 行：早停时以 val_PR_AUC 为最优（True），否则可以切到 val_F1

# ---- 早停相关参数 ----
EARLY_STOP_PATIENCE  = 40                           # 行：连续多少个 epoch 指标无提升就提前停止
EARLY_STOP_MIN_DELTA = 0.0                          # 行：认为“有提升”的最小改善幅度（针对 PR-AUC/F1）

# ========== IL-6 AAI 1D LM Dataset 封装 ==========
class IL6AAILMDataset(Data.Dataset):                # 行：自定义 Dataset，封装 1D LM 特征
    """封装 IL-6 AAI 1D LM 特征（仅 1D 模态：VHH + 抗原 LM 向量）。"""

    def __init__(self, pkg: Dict[str, Any]):
        # 这里对齐你 LoadData_il6_aai_lm_1d 的返回字段
        self.ids      = pkg["ids"]                  # 行：样本 ID（如 il6_0 ..）
        self.y        = pkg["y"]                    # 行：标签（0/1）
        self.vhh_seq  = pkg["vhh_seq"]              # 行：VHH 序列
        self.ag_seq   = pkg["ag_seq"]               # 行：抗原序列
        self.ag_label = pkg["ag_label"]             # 行：抗原名字（31 种之一）
        self.ab_lm    = pkg["ab_lm"]                # 行：VHH LM 向量 [N, D_ab]
        self.ag_lm    = pkg["ag_lm"]                # 行：抗原 LM 向量 [N, D_ag]

    def __len__(self):
        return len(self.y)                          # 行：样本数

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 行：返回单一样本的字典形式，方便 collate_fn 拼 batch
        return {
            "id"      : self.ids[idx],
            "y"       : float(self.y[idx]),
            "vhh_seq" : self.vhh_seq[idx],
            "ag_seq"  : self.ag_seq[idx],
            "ag_label": self.ag_label[idx],
            "ab_lm"   : self.ab_lm[idx],
            "ag_lm"   : self.ag_lm[idx],
        }

# ========== collate：合并 batch ==========
def il6_aai_lm_collate(batch: list) -> Dict[str, Any]:
    """把单样本 list 合成一个 batch 张量/列表。"""
    ids      = [b["id"]       for b in batch]       # 行：ID 列表
    vhh_seq  = [b["vhh_seq"]  for b in batch]       # 行：VHH 序列列表
    ag_seq   = [b["ag_seq"]   for b in batch]       # 行：抗原序列列表
    ag_label = [b["ag_label"] for b in batch]       # 行：抗原名字列表

    # 行：LM 向量堆成 [B, D] 的 float32 Tensor
    ab_lm = torch.as_tensor(
        np.stack([b["ab_lm"] for b in batch], axis=0),
        dtype=torch.float32
    )
    ag_lm = torch.as_tensor(
        np.stack([b["ag_lm"] for b in batch], axis=0),
        dtype=torch.float32
    )
    # 行：标签拼成 [B]
    y = torch.as_tensor(
        [b["y"] for b in batch],
        dtype=torch.float32
    )

    return {
        "ids"     : ids,
        "vhh_seq" : vhh_seq,
        "ag_seq"  : ag_seq,
        "ag_label": ag_label,
        "ab_lm"   : ab_lm,
        "ag_lm"   : ag_lm,
        "y"       : y,
    }

# ========== 分类指标：P / R / F1 / PR-AUC ==========
def _binary_classification_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
):
    """
    y_true:  [N] ∈ {0,1}
    y_score: [N] ∈ [0,1]   （预测概率）
    threshold: 判为正类的阈值（默认 0.5）

    返回：precision, recall, f1, pr_auc
    """
    # ---- 阈值化得到离散预测标签 ----
    y_pred = (y_score >= threshold).astype(np.int64)          # 行：大于等于阈值 → 1，否则 0

    # ---- 统计 TP / FP / FN / TN ----
    tp = np.sum((y_true == 1) & (y_pred == 1))                # 行：真正例
    fp = np.sum((y_true == 0) & (y_pred == 1))                # 行：假正例
    fn = np.sum((y_true == 1) & (y_pred == 0))                # 行：假负例
    tn = np.sum((y_true == 0) & (y_pred == 0))                # 行：真负例（这里不直接用）

    # ---- Precision / Recall / F1 ----
    precision = tp / (tp + fp + 1e-8)                         # 行：精确率
    recall    = tp / (tp + fn + 1e-8)                         # 行：召回率
    f1        = 2 * precision * recall / (precision + recall + 1e-8)  # 行：F1 值

    # ---- PR-AUC（Average Precision 简易实现）----
    # 1. 按 y_score 从大到小排序
    order = np.argsort(-y_score)                              # 行：降序排序索引
    y_true_sorted  = y_true[order]
    y_score_sorted = y_score[order]

    # 2. 到每个排序位置 k 的累积 TP 数
    cum_tp = np.cumsum(y_true_sorted)                         # 行：前缀真正例计数
    ranks  = np.arange(1, len(y_true_sorted) + 1)             # 行：排名位置 1..N
    precision_at_k = cum_tp / (ranks + 1e-8)                  # 行：每个位置的 precision(k)

    total_pos = np.sum(y_true_sorted == 1)                    # 行：正样本总数
    if total_pos == 0:
        pr_auc = 0.0                                          # 行：没有正样本时 AP=0
    else:
        # 只在真实为正的位置累积 precision(k)，再除以正样本数
        pr_auc = float(
            np.sum(precision_at_k[y_true_sorted == 1]) / (total_pos + 1e-8)
        )

    return float(precision), float(recall), float(f1), float(pr_auc)

# ========== 评估函数：P / R / F1 / PR-AUC ==========
@torch.no_grad()
def evaluate_il6_aai(
    model: torch.nn.Module,
    loader: Data.DataLoader,
    device: torch.device,
    criterion: nn.Module = None,
):
    """
    若给定 criterion：返回 (precision, recall, f1, pr_auc, avg_loss)
    否则         ：返回 (precision, recall, f1, pr_auc)
    """
    model.eval()                                              # 行：eval 模式（关闭 dropout 等）
    obs, score_list = [], []                                  # 行：存放真实标签和预测概率
    loss_sum, n_batches = 0.0, 0                              # 行：累计 loss 和 batch 数

    for b in loader:
        ab_lm = b["ab_lm"].to(device, non_blocking=True)      # 行：VHH LM 特征
        ag_lm = b["ag_lm"].to(device, non_blocking=True)      # 行：抗原 LM 特征
        y_true = b["y"].to(device, non_blocking=True)         # 行：真实标签（0/1）

        with torch.cuda.amp.autocast(enabled=(USE_AMP and device.type == "cuda")):
            # MultiModalPPI_LM 前向接口：prot1_lm / prot2_lm
            logits = model(prot1_lm=ab_lm, prot2_lm=ag_lm)    # 行：输出 logits [B]
            y_prob = torch.sigmoid(logits)                    # 行：sigmoid → 概率 [0,1]
            if criterion is not None:
                loss_val = criterion(logits, y_true)          # 行：BCEWithLogitsLoss 用 logits
                if torch.isfinite(loss_val):                  # 行：过滤 NaN/Inf
                    loss_sum += float(loss_val)
                    n_batches += 1

        obs.extend(y_true.view(-1).tolist())                  # 行：收集真实值
        score_list.extend(y_prob.view(-1).tolist())           # 行：收集预测概率

    y_true_np  = np.array(obs, dtype=np.float32)              # 行：转 numpy
    y_score_np = np.array(score_list, dtype=np.float32)

    precision, recall, f1, pr_auc = _binary_classification_metrics(
        y_true_np, y_score_np, threshold=0.5
    )

    if criterion is None:
        return precision, recall, f1, pr_auc                   # 行：只返回四个指标
    return precision, recall, f1, pr_auc, (loss_sum / max(1, n_batches))  # 行：附带平均 loss

# ========== 主程序入口 ==========
if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL)                      # 行：初始化日志配置
    util.seed_torch()                                         # 行：固定随机种子，保证可复现

    result_root = Path(OUT_DIR)                               # 行：结果根目录 Path 对象
    result_root.mkdir(parents=True, exist_ok=True)            # 行：不存在则创建目录

    # —— 加载 IL-6 AAI 1D LM 数据 —— #
    kw_common = dict(
        out_1d="../dataset/AVIDa-hIL6/processed_lm_1d",       # 行：1D LM 缓存目录（你可根据习惯修改）
        logspace_trans=False,                                 # 行：0/1 分类不做 log 变换
        esm_ab_model_name="facebook/esm2_t33_650M_UR50D",     # 行：VHH 侧 LM 模型名
        esm_ag_model_name="facebook/esm2_t33_650M_UR50D",     # 行：抗原侧 LM 模型名
        lm_batch_size=32,                                     # 行：LM 编码 batch_size
        use_safetensors=True,                                 # 行：优先使用 safetensors
    )
    # 行：分别加载 train / val / test 三个划分
    mm_train = LoadData_il6_aai_lm_1d(DATA_IL6_AAI, split="train", **kw_common)["train"]
    mm_val   = LoadData_il6_aai_lm_1d(DATA_IL6_AAI, split="val",   **kw_common)["val"]
    mm_test  = LoadData_il6_aai_lm_1d(DATA_IL6_AAI, split="test",  **kw_common)["test"]

    # 行：从训练集里读出 LM 向量维度
    d_ab_lm = mm_train["ab_lm"].shape[1]                      # 行：VHH LM 维度
    d_ag_lm = mm_train["ag_lm"].shape[1]                      # 行：抗原 LM 维度
    logging.info(f"[IL6_AAI|LM] ab_lm_dim={d_ab_lm}, ag_lm_dim={d_ag_lm}")

    # —— Dataset / DataLoader 构建 —— #
    train_ds = IL6AAILMDataset(mm_train)                      # 行：训练集 Dataset
    val_ds   = IL6AAILMDataset(mm_val)                        # 行：验证集 Dataset
    test_ds  = IL6AAILMDataset(mm_test)                       # 行：测试集 Dataset

    train_loader = Data.DataLoader(                           # 行：训练 DataLoader
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=il6_aai_lm_collate
    )
    val_loader = Data.DataLoader(                             # 行：验证 DataLoader
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=il6_aai_lm_collate
    )
    test_loader = Data.DataLoader(                            # 行：测试 DataLoader
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=il6_aai_lm_collate
    )

    logging.info(f"[IL6_AAI] sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # —— Sanity Check：看一眼 batch 形状 —— #
    if SANITY_CHECK:
        b = next(iter(train_loader))
        print(">>> Batch keys:", list(b.keys()))
        print("ab_lm:", tuple(b["ab_lm"].shape),
              "| ag_lm:", tuple(b["ag_lm"].shape),
              "| y:", tuple(b["y"].shape))
        print("[OK] IL6_AAI LM loaders ready.")

    # —— 导入模型：使用你写好的 MultiModalPPI_LM（这里只用 1D 分支）—— #
    from model.model_multimodal_lm_ppi import MultiModalPPI_LM          # 行：导入 PPI 模型

    model = MultiModalPPI_LM(
        d_prot1_lm=d_ab_lm, d_prot2_lm=d_ag_lm,             # 行：prot1=VHH，prot2=Ag
        prot1_node_dim=33, prot1_edge_dim=3,                # 行：2D/3D 分支现在不用，先给默认
        prot2_node_dim=33, prot2_edge_dim=3,
        gcn_hidden=128, gcn_out=128,
        d3_prot1=512, d3_prot2=512,
        d_model=256, d_attn=256, n_heads=4,
        add_interactions_3d=True
    ).to(DEVICE)

    # —— 损失/优化器/调度器/AMP —— #
    criterion = nn.BCEWithLogitsLoss()                      # 行：二分类损失，内部自带 sigmoid
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)  # 行：AdamW 优化器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-5
    ) if USE_SCHEDULER else None                            # 行：余弦退火 + 重启 学习率调度
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE.type == "cuda"))  # 行：AMP 缩放器

    # —— 日志与权重路径 —— #
    result_dir  = result_root / "train_logs"                # 行：保存 checkpoint / csv 等
    run_dir     = result_root / "runs"                      # 行：TensorBoard 目录
    model_best  = result_dir / "best_model.pth"             # 行：最优模型权重文件
    ckpt_path   = result_dir / "checkpoint.pth.tar"         # 行：最新 checkpoint 文件
    csv_file    = result_dir / "metrics.csv"                # 行：记录训练过程指标
    final_txt   = result_dir / "final_test_metrics.txt"     # 行：最终测试指标文本

    result_dir.mkdir(parents=True, exist_ok=True)           # 行：创建结果子目录
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir))            # 行：初始化 TensorBoard 写入器

    # —— 如果有 checkpoint 则尝试恢复 —— #
    start_epoch = 0                                         # 行：默认从 epoch 0 开始
    if ckpt_path.exists():
        start_epoch = util.load_checkpoint(str(ckpt_path), model, optimizer)
        logging.info(f"[Resume|IL6_AAI] from epoch: {start_epoch}")

    # —— CSV 表头（P / R / F1 / PR-AUC）—— #
    if not csv_file.exists():
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "val_loss",
                "train_P", "train_R", "train_F1", "train_PR_AUC",
                "val_P",   "val_R",   "val_F1",   "val_PR_AUC",
                "test_P",  "test_R",  "test_F1",  "test_PR_AUC",
                "lr"
            ])

    # ========== 训练主循环 ==========
    best_score = -1.0                                      # 行：记录最优 val_PR_AUC（越大越好）
    epochs_no_improve = 0                                  # 行：连续“无提升”的 epoch 计数
    t0 = time.time()                                       # 行：记录起始时间

    for epoch in range(start_epoch, EPOCHS):
        model.train()                                      # 行：切换到训练模式
        optimizer.zero_grad(set_to_none=True)              # 行：清空梯度

        running_loss, n_batches, last_bidx = 0.0, 0, -1

        for last_bidx, b in enumerate(train_loader):
            ab_lm = b["ab_lm"].to(DEVICE, non_blocking=True)   # 行：VHH LM 特征
            ag_lm = b["ag_lm"].to(DEVICE, non_blocking=True)   # 行：抗原 LM 特征
            y_true = b["y"].to(DEVICE, non_blocking=True)      # 行：真实标签

            with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE.type == "cuda")):
                logits = model(prot1_lm=ab_lm, prot2_lm=ag_lm) # 行：前向计算，输出 logits [B]
                loss   = criterion(logits, y_true)             # 行：BCEWithLogitsLoss

            if not torch.isfinite(loss):                       # 行：NaN/Inf 检查
                logging.warning(f"[NaN|IL6_AAI] loss at epoch={epoch}, batch={last_bidx}: {float(loss)}")
                optimizer.zero_grad(set_to_none=True)
                continue

            if scaler.is_enabled():                            # 行：AMP 情况下的反向传播
                scaler.scale(loss / ACCUM_STEPS).backward()
            else:
                (loss / ACCUM_STEPS).backward()

            # 行：梯度累积到 ACCUM_STEPS 再更新一次
            if (last_bidx + 1) % ACCUM_STEPS == 0:
                if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)             # 行：AMP 需先 unscale 再裁剪
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

                if scaler.is_enabled():
                    scaler.step(optimizer); scaler.update()    # 行：带 AMP 的优化步
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss)                        # 行：累计 loss
            n_batches    += 1

        # ---- 处理最后一个不足 ACCUM_STEPS 的 batch ----
        if (last_bidx + 1) % ACCUM_STEPS != 0 and n_batches > 0:
            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            if scaler.is_enabled():
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_loss = running_loss / max(1, n_batches)         # 行：平均训练损失
        writer.add_scalar("Loss/train", train_loss, epoch)    # 行：写入 TensorBoard

        # —— 在 train/val/test 上评估：P / R / F1 / PR-AUC —— #
        train_P, train_R, train_F1, train_AUC, _ = evaluate_il6_aai(
            model, train_loader, DEVICE, criterion
        )
        val_P, val_R, val_F1, val_AUC, val_loss = evaluate_il6_aai(
            model, val_loader, DEVICE, criterion
        )
        test_P, test_R, test_F1, test_AUC = evaluate_il6_aai(
            model, test_loader, DEVICE, criterion=None
        )

        writer.add_scalar("Loss/val", val_loss, epoch)        # 行：写入验证集 loss

        # 行：各类指标写入 TensorBoard
        for tag, val in [
            ("train_P", train_P),
            ("train_R", train_R),
            ("train_F1", train_F1),
            ("train_PR_AUC", train_AUC),
            ("val_P", val_P),
            ("val_R", val_R),
            ("val_F1", val_F1),
            ("val_PR_AUC", val_AUC),
            ("test_P", test_P),
            ("test_R", test_R),
            ("test_F1", test_F1),
            ("test_PR_AUC", test_AUC),
        ]:
            writer.add_scalar(f"Metrics/{tag}", val, epoch)

        # —— 学习率调度器步进 —— #
        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]          # 行：取当前学习率
        writer.add_scalar("LR", current_lr, epoch)

        # —— 写入 CSV —— #
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, train_loss, val_loss,
                train_P, train_R, train_F1, train_AUC,
                val_P,   val_R,   val_F1,   val_AUC,
                test_P,  test_R,  test_F1,  test_AUC,
                current_lr
            ])

        # —— 以 val_PR_AUC（或 val_F1）选 best model & 早停 —— #
        score_now = val_AUC if VAL_AS_BEST else val_F1        # 行：越大越好
        if score_now - EARLY_STOP_MIN_DELTA > best_score and torch.isfinite(torch.tensor(score_now)):
            best_score = score_now
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_best)        # 行：保存当前最优模型权重
            logging.info(
                f"[BEST|IL6_AAI] epoch={epoch+1}, "
                f"{'val_PR_AUC' if VAL_AS_BEST else 'val_F1'}={best_score:.4f} -> saved: {model_best}"
            )
        else:
            epochs_no_improve += 1
            logging.info(
                f"[EARLY-STOP|IL6_AAI] epoch={epoch+1}: no improvement on "
                f"{'val_PR_AUC' if VAL_AS_BEST else 'val_F1'} for {epochs_no_improve} epoch(s) "
                f"(best={best_score:.4f}, now={score_now:.4f})"
            )
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                logging.info(
                    f"[EARLY-STOP|IL6_AAI] Stop training at epoch {epoch+1}, "
                    f"no improvement for {EARLY_STOP_PATIENCE} epochs."
                )
                break

        logging.info(
            f"[IL6_AAI] Epoch {epoch+1:04d} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"train(P={train_P:.4f}, R={train_R:.4f}, F1={train_F1:.4f}, AUC={train_AUC:.4f}) | "
            f"val(P={val_P:.4f}, R={val_R:.4f}, F1={val_F1:.4f}, AUC={val_AUC:.4f}) | "
            f"test(P={test_P:.4f}, R={test_R:.4f}, F1={test_F1:.4f}, AUC={test_AUC:.4f}) | "
            f"lr={current_lr:.3e} | time={(time.time()-t0)/60:.2f} min"
        )

        util.save_checkpoint(epoch + 1, model, optimizer, filename=str(ckpt_path))  # 行：保存最新 checkpoint

    # ========== 最优模型做最终测试 ==========
    best_model = MultiModalPPI_LM(                         # 行：重新构建同结构模型
        d_prot1_lm=d_ab_lm, d_prot2_lm=d_ag_lm,
        prot1_node_dim=33, prot1_edge_dim=3,
        prot2_node_dim=33, prot2_edge_dim=3,
        gcn_hidden=128, gcn_out=128,
        d3_prot1=512, d3_prot2=512,
        d_model=256, d_attn=256, n_heads=4,
        add_interactions_3d=True
    ).to(DEVICE)
    best_model.load_state_dict(torch.load(model_best, map_location=DEVICE))  # 行：加载最优权重
    best_model.eval()

    with torch.no_grad():
        final_P, final_R, final_F1, final_AUC = evaluate_il6_aai(  # 行：只计算四项指标
            best_model, test_loader, DEVICE, criterion=None
        )

    logging.info(
        f"[FINAL|IL6_AAI] test_P={final_P:.4f}, "
        f"test_R={final_R:.4f}, test_F1={final_F1:.4f}, test_PR_AUC={final_AUC:.4f}"
    )
    with open(final_txt, "w", encoding="utf-8") as f:       # 行：把最终结果写到 txt
        f.write(f"test_P: {final_P:.4f}\n")
        f.write(f"test_R: {final_R:.4f}\n")
        f.write(f"test_F1: {final_F1:.4f}\n")
        f.write(f"test_PR_AUC: {final_AUC:.4f}\n")

    writer.close()                                          # 行：关闭 TensorBoard 写入器
