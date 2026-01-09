# ======================================
# BD2017 仅 1D（LM 版）训练脚本（论文同口径评估）
# 任务：MHC-I peptide binding（二分类评估） + 连续标签回归训练（与原作者一致：MSE + PCC监控）
#
# 模型：MultiModalPPA_LM（位于 model/model_multimodal_lm_ppa.py）
# 数据：LoadData_bd2017_lm_1d（返回 pep_lm, mhc_lm, y(连续), y_bin(0/1), pep_len）
#
# 评估指标（整体 & 分桶）：
#   - AUC
#   - PCC（pred vs 连续 y）
#   - PPV(Precision)
#   - Sensitivity(Recall)
#   - F1-score
#   - AUPRC
# 论文分桶：8-mer / 9-mer / 10-mer / 11-mer / ≥12-mer
# ======================================

from pathlib import Path                                  # 行：路径处理（创建目录/拼路径）
import logging                                           # 行：日志打印
import time                                              # 行：计时
import csv                                               # 行：保存训练过程指标到 CSV
from typing import Dict, Any, Optional, Tuple            # 行：类型注解

import numpy as np                                       # 行：数值处理
import torch                                             # 行：PyTorch 主库
import torch.nn as nn                                    # 行：神经网络模块（loss 等）
import torch.optim as optim                              # 行：优化器
import torch.utils.data as Data                          # 行：Dataset/DataLoader
from torch.utils.tensorboard import SummaryWriter        # 行：TensorBoard

import util                                              # 行：你项目里的工具函数（seed/checkpoint等）
from util import LoadData_bd2017_lm_1d                   # 行：✅ BD2017 1D-LM 数据加载函数


# ========== 设备/日志/随机种子 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 行：有 GPU 用 GPU，否则 CPU
torch.backends.cudnn.benchmark = True                                  # 行：固定 shape 时可加速
LOG_LEVEL = logging.INFO                                               # 行：日志等级
SANITY_CHECK = True                                                    # 行：是否打印一次 batch 形状检查


# ========== 可配置区域 ==========
DATA_BD2017    = r"../dataset/BD2017"                 # 行：BD2017 原始目录（含 ba_data.txt / MHC_pseudo.dat）
DATASET_NAME   = "bd2017_lm_1d_ppa_binary"            # 行：任务名（用于结果目录命名）
OUT_DIR        = f"../result/{DATASET_NAME}"          # 行：输出根目录

BATCH_SIZE     = 128                                  # 行：批量大小
NUM_WORKERS    = 4                                    # 行：DataLoader 进程数（Windows 可改 0/2）
PIN_MEMORY     = bool(torch.cuda.is_available())      # 行：GPU 时 pin_memory 搬运更快

EPOCHS         = 200                                  # 行：最大 epoch（原作者 20 epoch 只是示例，你可更大+早停）
LR             = 3e-4                                 # 行：学习率
WEIGHT_DECAY   = 1e-4                                 # 行：AdamW 权重衰减
USE_AMP        = True                                 # 行：混合精度（加速）
ACCUM_STEPS    = 1                                    # 行：梯度累积（显存紧张才开）
GRAD_CLIP_NORM = 1.0                                  # 行：梯度裁剪阈值

USE_SCHEDULER  = True                                 # 行：是否用调度器（这里用 cosine 重启）
VAL_AS_BEST    = True                                 # 行：用 val 指标挑 best（否则用 test）

# ---- 早停相关参数（按 AUC 或 PCC 监控都行）----
EARLY_STOP_PATIENCE  = 30                              # 行：连续多少 epoch 无提升就停
EARLY_STOP_MIN_DELTA = 0.0                             # 行：认为“有提升”的最小幅度

# ---- best 选择策略：论文表最核心是 AUC + PCC ----
BEST_KEY = "AUC"                                       # 行：best 监控指标：AUC 或 PCC（二选一）


# ========== 指标函数（AUC/AUPRC/Precision/Recall/F1/PCC）==========
def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """行：计算 PCC（避免常数数组导致 NaN）"""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    # 行：手写 PCC
    vx = x - x.mean()
    vy = y - y.mean()
    denom = (np.sqrt((vx * vx).sum()) * np.sqrt((vy * vy).sum())) + 1e-12
    return float((vx * vy).sum() / denom)


def _try_sklearn_auc(y_true_bin: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """行：优先用 sklearn 算 ROC-AUC / PR-AUC；没有 sklearn 就返回 nan"""
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc = float(roc_auc_score(y_true_bin, y_score))
        auprc = float(average_precision_score(y_true_bin, y_score))
        return auc, auprc
    except Exception:
        return float("nan"), float("nan")


def _bin_metrics(y_true_bin: np.ndarray, y_score: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    """
    行：由二分类标签 + 连续 score 计算：
      PPV(Precision), Sensitivity(Recall), F1
    """
    y_true_bin = np.asarray(y_true_bin, dtype=np.int64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    y_pred = (y_score >= thr).astype(np.int64)

    tp = int(((y_true_bin == 1) & (y_pred == 1)).sum())
    tn = int(((y_true_bin == 0) & (y_pred == 0)).sum())
    fp = int(((y_true_bin == 0) & (y_pred == 1)).sum())
    fn = int(((y_true_bin == 1) & (y_pred == 0)).sum())

    precision = tp / max(1, tp + fp)                         # 行：PPV
    recall = tp / max(1, tp + fn)                            # 行：Sensitivity
    f1 = (2 * precision * recall) / max(1e-12, precision + recall)

    acc = (tp + tn) / max(1, tp + tn + fp + fn)              # 行：可选：ACC
    return {
        "PPV": float(precision),
        "Sensitivity": float(recall),
        "F1": float(f1),
        "ACC": float(acc),
        "TP": float(tp), "TN": float(tn), "FP": float(fp), "FN": float(fn),
    }


# ========== BD2017 Dataset（仅 1D-LM）==========
class BD2017LMDataset(Data.Dataset):
    """
    行：封装 BD2017 的 1D-LM 特征：
        pep_lm: [N, D_pep]
        mhc_lm: [N, D_mhc]
        y     : [N] 连续标签（用于 PCC / MSE训练）
        y_bin : [N] 0/1 标签（用于 AUC/AUPRC/PPV/Recall/F1）
        pep_len: [N] 肽长度（用于论文分桶）
    """
    def __init__(self, pkg: Dict[str, Any]):
        self.ids = pkg["ids"]                                 # 行：样本 ID
        self.pep_seq = pkg.get("pep_seq", None)               # 行：peptide 序列（可选）
        self.mhc_seq = pkg.get("mhc_seq", None)               # 行：mhc pseudo 序列（可选）
        self.pep_len = pkg.get("pep_len", None)               # 行：peptide 长度（可选，但我们推荐必须有）

        self.y = pkg["y"]                                     # 行：连续标签 float32
        self.y_bin = pkg.get("y_bin", None)                   # 行：二值标签 int64（若没有我们会由 y 推断）
        self.pep_lm = pkg["pep_lm"]                            # 行：pep 向量
        self.mhc_lm = pkg["mhc_lm"]                            # 行：mhc 向量

        # 行：兜底：如果没有 y_bin，就把 y>=0.5 当正类（但论文更推荐用 loader 的阈值逻辑）
        if self.y_bin is None:
            self.y_bin = (np.asarray(self.y) >= 0.5).astype(np.int64)

        # 行：兜底：如果没有 pep_len，就由 pep_seq 计算
        if self.pep_len is None:
            if self.pep_seq is None:
                self.pep_len = np.zeros((len(self.y),), dtype=np.int32)
            else:
                self.pep_len = np.array([len(s) for s in self.pep_seq], dtype=np.int32)

    def __len__(self):
        return len(self.y)                                    # 行：数据集大小

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "id": self.ids[idx],                               # 行：样本 ID
            "y": float(self.y[idx]),                           # 行：连续标签
            "y_bin": int(self.y_bin[idx]),                     # 行：二值标签
            "pep_len": int(self.pep_len[idx]),                 # 行：pep 长度
            "pep_seq": None if self.pep_seq is None else self.pep_seq[idx],  # 行：可选调试
            "mhc_seq": None if self.mhc_seq is None else self.mhc_seq[idx],
            "pep_lm": self.pep_lm[idx],                        # 行：pep 向量
            "mhc_lm": self.mhc_lm[idx],                        # 行：mhc 向量
        }


# ========== collate：合并 batch（仅 1D-LM）==========
def bd2017_lm_collate(batch: list) -> Dict[str, Any]:
    """行：把 list[dict] 合并成 batch dict，LM/标签堆叠成 tensor，文本保留 list。"""
    ids = [b["id"] for b in batch]                             # 行：ID 列表
    pep_seqs = [b["pep_seq"] for b in batch]                   # 行：pep 序列列表
    mhc_seqs = [b["mhc_seq"] for b in batch]                   # 行：mhc 序列列表

    pep_lm = torch.as_tensor(np.stack([b["pep_lm"] for b in batch], axis=0),
                             dtype=torch.float32)              # 行：[B, D_pep]
    mhc_lm = torch.as_tensor(np.stack([b["mhc_lm"] for b in batch], axis=0),
                             dtype=torch.float32)              # 行：[B, D_mhc]

    y = torch.as_tensor([b["y"] for b in batch], dtype=torch.float32)          # 行：[B] 连续
    y_bin = torch.as_tensor([b["y_bin"] for b in batch], dtype=torch.int64)   # 行：[B] 二值
    pep_len = torch.as_tensor([b["pep_len"] for b in batch], dtype=torch.int64)  # 行：[B] 长度

    return {
        "ids": ids,                                            # 行：list[str]
        "pep_seqs": pep_seqs,                                  # 行：list[str|None]
        "mhc_seqs": mhc_seqs,                                  # 行：list[str|None]
        "pep_lm": pep_lm,                                      # 行：tensor
        "mhc_lm": mhc_lm,                                      # 行：tensor
        "y": y,                                                # 行：连续标签
        "y_bin": y_bin,                                        # 行：二值标签
        "pep_len": pep_len,                                    # 行：长度
    }


# ========== 评估函数：整体 + 分桶（论文口径）==========
@torch.no_grad()
def evaluate_bd2017(model: nn.Module,
                    loader: Data.DataLoader,
                    device: torch.device,
                    criterion: Optional[nn.Module] = None,
                    use_amp: bool = True) -> Tuple[Dict[str, Any], Optional[float]]:
    """
    返回：
      metrics_all: dict（包含整体指标 + per-bucket 指标）
      avg_loss: float or None
    """
    model.eval()                                               # 行：评估模式

    ys, ybins, preds, lens = [], [], [], []                    # 行：收集 y / y_bin / pred / pep_len
    loss_sum, n_batches = 0.0, 0                               # 行：loss 累计

    for b in loader:
        pep_lm = b["pep_lm"].to(device, non_blocking=True)      # 行：pep LM 上设备
        mhc_lm = b["mhc_lm"].to(device, non_blocking=True)      # 行：mhc LM 上设备
        y_true = b["y"].to(device, non_blocking=True)           # 行：连续标签
        y_bin = b["y_bin"].to(device, non_blocking=True)        # 行：二值标签
        pep_len = b["pep_len"].to(device, non_blocking=True)    # 行：长度

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            # ✅ 只喂 1D：pep 当 prot1，mhc 当 prot2
            y_hat = model(prot1_lm=pep_lm, prot2_lm=mhc_lm).view(-1)  # 行：输出 [B]

            if criterion is not None:
                loss_val = criterion(y_hat, y_true.view(-1))         # 行：MSE loss（训练口径）
                if torch.isfinite(loss_val):
                    loss_sum += float(loss_val)
                    n_batches += 1

        ys.append(y_true.detach().cpu().view(-1))
        ybins.append(y_bin.detach().cpu().view(-1))
        preds.append(y_hat.detach().cpu().view(-1))
        lens.append(pep_len.detach().cpu().view(-1))

    y = torch.cat(ys, dim=0).numpy().astype(np.float64)              # 行：连续标签
    yb = torch.cat(ybins, dim=0).numpy().astype(np.int64)            # 行：二值标签
    p = torch.cat(preds, dim=0).numpy().astype(np.float64)           # 行：模型输出（回归式 score）
    L = torch.cat(lens, dim=0).numpy().astype(np.int64)              # 行：长度

    # 行：把预测当成“score/prob”，限制到 [0,1]（避免极端值影响 AUC/AUPRC）
    score = np.clip(p, 0.0, 1.0)

    # --- 整体指标 ---
    auc, auprc = _try_sklearn_auc(yb, score)                         # 行：AUC/AUPRC
    pcc = _safe_pearsonr(p, y)                                       # 行：PCC（pred vs 连续 y）
    binm = _bin_metrics(yb, score, thr=0.5)                          # 行：PPV/Sens/F1/ACC

    metrics_all: Dict[str, Any] = {
        "AUC": float(auc),
        "AUPRC": float(auprc),
        "PCC": float(pcc),
        "PPV": float(binm["PPV"]),
        "Sensitivity": float(binm["Sensitivity"]),
        "F1": float(binm["F1"]),
        "ACC": float(binm["ACC"]),
    }

    # --- 分桶指标（论文表格：8/9/10/11/≥12）---
    # 行：定义每个桶的 mask
    bucket_defs = {
        "8-mer": (L == 8),
        "9-mer": (L == 9),
        "10-mer": (L == 10),
        "11-mer": (L == 11),
        ">=12-mer": (L >= 12),
    }

    bucket_metrics = {}                                            # 行：保存每个桶的指标
    for name, mask in bucket_defs.items():
        idx = np.where(mask)[0]
        if idx.size < 2:
            bucket_metrics[name] = {
                "N": int(idx.size),
                "AUC": float("nan"),
                "AUPRC": float("nan"),
                "PCC": float("nan"),
                "PPV": float("nan"),
                "Sensitivity": float("nan"),
                "F1": float("nan"),
            }
            continue

        yb_i = yb[idx]
        y_i = y[idx]
        s_i = score[idx]
        p_i = p[idx]

        auc_i, auprc_i = _try_sklearn_auc(yb_i, s_i)
        pcc_i = _safe_pearsonr(p_i, y_i)
        bm_i = _bin_metrics(yb_i, s_i, thr=0.5)

        bucket_metrics[name] = {
            "N": int(idx.size),
            "AUC": float(auc_i),
            "AUPRC": float(auprc_i),
            "PCC": float(pcc_i),
            "PPV": float(bm_i["PPV"]),
            "Sensitivity": float(bm_i["Sensitivity"]),
            "F1": float(bm_i["F1"]),
        }

    metrics_all["Buckets"] = bucket_metrics                         # 行：挂到总 metrics 里

    avg_loss = None if criterion is None else (loss_sum / max(1, n_batches))
    return metrics_all, avg_loss


# ========== 主程序 ==========
if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL)                             # 行：初始化日志
    util.seed_torch()                                                # 行：固定随机种子（可复现）

    result_root = Path(OUT_DIR)                                      # 行：输出根目录
    result_root.mkdir(parents=True, exist_ok=True)                   # 行：创建输出目录

    # —— 加载 BD2017 仅 1D-LM 数据 —— #
    kw_common = dict(
        out_1d="../dataset/BD2017/processed_lm_1d",                  # 行：缓存目录
        lm_batch_size=64,                                            # 行：ESM 编码 batch（看显存）
        use_safetensors=True,                                        # 行：优先 safetensors
        esm_pep_model_name="facebook/esm2_t33_650M_UR50D",           # 行：pep ESM
        esm_mhc_model_name="facebook/esm2_t33_650M_UR50D",           # 行：mhc ESM
        split_seed=2023,                                             # 行：划分随机种子
        train_ratio=0.8,                                             # 行：train 比例
        val_ratio=0.1,                                               # 行：val 比例
    )

    # 行：分别加载 train/val/test（第一次会编码并缓存，之后直接读 npz）
    mm_train = LoadData_bd2017_lm_1d(data_dir=DATA_BD2017, split="train", **kw_common)["train"]
    mm_val   = LoadData_bd2017_lm_1d(data_dir=DATA_BD2017, split="val",   **kw_common)["val"]
    mm_test  = LoadData_bd2017_lm_1d(data_dir=DATA_BD2017, split="test",  **kw_common)["test"]

    # 行：LM 维度（初始化模型要用）
    d_pep = mm_train["pep_lm"].shape[1]
    d_mhc = mm_train["mhc_lm"].shape[1]
    logging.info(f"[BD2017|LM-1D] pep_lm_dim={d_pep}, mhc_lm_dim={d_mhc}")

    # —— Dataset / DataLoader —— #
    train_ds = BD2017LMDataset(mm_train)
    val_ds   = BD2017LMDataset(mm_val)
    test_ds  = BD2017LMDataset(mm_test)

    train_loader = Data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=bd2017_lm_collate
    )
    val_loader = Data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=bd2017_lm_collate
    )
    test_loader = Data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=bd2017_lm_collate
    )

    logging.info(f"[BD2017] sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # —— Sanity Check —— #
    if SANITY_CHECK:
        b = next(iter(train_loader))
        print(">>> Batch keys:", list(b.keys()))
        print("pep_lm:", tuple(b["pep_lm"].shape),
              "| mhc_lm:", tuple(b["mhc_lm"].shape),
              "| y:", tuple(b["y"].shape),
              "| y_bin:", tuple(b["y_bin"].shape),
              "| pep_len:", tuple(b["pep_len"].shape))
        print("[OK] BD2017 loaders ready.")

    # —— 导入模型 —— #
    from model.model_multimodal_lm_ppa import MultiModalPPA_LM       # 行：✅ 你的新模型

    model = MultiModalPPA_LM(
        d_prot1_lm=d_pep,                                           # 行：pep LM 维度
        d_prot2_lm=d_mhc,                                           # 行：mhc LM 维度
        prot1_node_dim=33, prot1_edge_dim=3,                        # 行：2D参数（不走分支也无妨）
        prot2_node_dim=33, prot2_edge_dim=3,
        gcn_hidden=128, gcn_out=128,
        d3_prot1=512, d3_prot2=512,                                 # 行：3D参数（不走分支也无妨）
        d_model=256, d_attn=256, n_heads=4,
        add_interactions_3d=True
    ).to(DEVICE)

    # —— 损失（与原作者一致：MSE）—— #
    criterion = nn.MSELoss(reduction="mean")                        # 行：回归式训练（对连续 y）
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)  # 行：AdamW

    # —— 调度器（可选）—— #
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    ) if USE_SCHEDULER else None

    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE.type == "cuda"))  # 行：AMP 缩放器

    # —— 日志与权重路径 —— #
    result_dir  = result_root / "train_logs"
    run_dir     = result_root / "runs"
    model_best  = result_dir / "best_model.pth"
    ckpt_path   = result_dir / "checkpoint.pth.tar"
    csv_file    = result_dir / "metrics.csv"
    final_txt   = result_dir / "final_test_metrics.txt"

    result_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    # —— 断点恢复（可选）—— #
    start_epoch = 0
    if ckpt_path.exists():
        start_epoch = util.load_checkpoint(str(ckpt_path), model, optimizer)
        logging.info(f"[Resume] from epoch: {start_epoch}")

    # —— CSV 表头 —— #
    if not csv_file.exists():
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "epoch",
                "train_loss", "val_loss",
                "train_AUC", "train_PCC", "train_PPV", "train_Sensitivity", "train_F1", "train_AUPRC",
                "val_AUC",   "val_PCC",   "val_PPV",   "val_Sensitivity",   "val_F1",   "val_AUPRC",
                "test_AUC",  "test_PCC",  "test_PPV",  "test_Sensitivity",  "test_F1",  "test_AUPRC",
                "lr"
            ])

    # ========== 训练主循环 ==========
    # 行：best_score 越大越好（AUC / PCC 都是越大越好）
    best_score = -1e18
    epochs_no_improve = 0
    t0 = time.time()

    for epoch in range(start_epoch, EPOCHS):
        model.train()                                             # 行：训练模式
        optimizer.zero_grad(set_to_none=True)                     # 行：清梯度（更省显存）

        running_loss, n_batches, last_bidx = 0.0, 0, -1

        for last_bidx, b in enumerate(train_loader):
            pep_lm = b["pep_lm"].to(DEVICE, non_blocking=True)     # 行：pep LM 上设备
            mhc_lm = b["mhc_lm"].to(DEVICE, non_blocking=True)     # 行：mhc LM 上设备
            y_true = b["y"].to(DEVICE, non_blocking=True)          # 行：连续标签

            with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE.type == "cuda")):
                y_hat = model(prot1_lm=pep_lm, prot2_lm=mhc_lm).view(-1)  # 行：输出 [B]
                loss = criterion(y_hat, y_true.view(-1))                  # 行：MSE loss

            if not torch.isfinite(loss):                            # 行：防 NaN/Inf
                logging.warning(f"[NaN] loss at epoch={epoch}, batch={last_bidx}: {float(loss)}")
                optimizer.zero_grad(set_to_none=True)
                continue

            # 行：梯度累积
            if scaler.is_enabled():
                scaler.scale(loss / ACCUM_STEPS).backward()
            else:
                (loss / ACCUM_STEPS).backward()

            # 行：每 ACCUM_STEPS 更新一次参数
            if (last_bidx + 1) % ACCUM_STEPS == 0:
                if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:          # 行：可选梯度裁剪
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss)
            n_batches += 1

        # 行：尾批补一次 step（若没整除 ACCUM_STEPS）
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

        train_loss = running_loss / max(1, n_batches)
        writer.add_scalar("Loss/train", train_loss, epoch)

        # —— 评估：train/val/test（指标：AUC/PCC/PPV/Sens/F1/AUPRC + 分桶）—— #
        train_metrics, _ = evaluate_bd2017(model, train_loader, DEVICE, criterion=criterion, use_amp=USE_AMP)
        val_metrics, val_loss = evaluate_bd2017(model, val_loader, DEVICE, criterion=criterion, use_amp=USE_AMP)
        test_metrics, _ = evaluate_bd2017(model, test_loader, DEVICE, criterion=None, use_amp=USE_AMP)

        writer.add_scalar("Loss/val", val_loss, epoch)

        # 行：写 TensorBoard（整体指标）
        for k in ["AUC", "PCC", "PPV", "Sensitivity", "F1", "AUPRC", "ACC"]:
            writer.add_scalar(f"Train/{k}", train_metrics.get(k, float("nan")), epoch)
            writer.add_scalar(f"Val/{k}",   val_metrics.get(k, float("nan")),   epoch)
            writer.add_scalar(f"Test/{k}",  test_metrics.get(k, float("nan")),  epoch)

        # 行：写 TensorBoard（分桶指标：只写 AUC 和 PCC，和论文表一致）
        for bucket_name, bm in val_metrics["Buckets"].items():
            writer.add_scalar(f"ValBuckets/{bucket_name}_AUC", bm["AUC"], epoch)
            writer.add_scalar(f"ValBuckets/{bucket_name}_PCC", bm["PCC"], epoch)

        # —— 学习率调度 —— #
        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("LR", current_lr, epoch)

        # —— 写 CSV（整体指标）—— #
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch,
                train_loss, val_loss,
                train_metrics["AUC"], train_metrics["PCC"], train_metrics["PPV"], train_metrics["Sensitivity"], train_metrics["F1"], train_metrics["AUPRC"],
                val_metrics["AUC"],   val_metrics["PCC"],   val_metrics["PPV"],   val_metrics["Sensitivity"],   val_metrics["F1"],   val_metrics["AUPRC"],
                test_metrics["AUC"],  test_metrics["PCC"],  test_metrics["PPV"],  test_metrics["Sensitivity"],  test_metrics["F1"],  test_metrics["AUPRC"],
                current_lr
            ])

        # —— best 选择 + 早停 —— #
        score_now = (val_metrics[BEST_KEY] if VAL_AS_BEST else test_metrics[BEST_KEY])

        if (score_now + EARLY_STOP_MIN_DELTA > best_score) and np.isfinite(score_now):
            best_score = float(score_now)
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_best)
            logging.info(
                f"[BEST] epoch={epoch+1}, "
                f"{'val' if VAL_AS_BEST else 'test'}_{BEST_KEY}={best_score:.4f} -> saved: {model_best}"
            )
        else:
            epochs_no_improve += 1
            logging.info(
                f"[EARLY-STOP] epoch={epoch+1}: no improvement on "
                f"{'val' if VAL_AS_BEST else 'test'}_{BEST_KEY} for {epochs_no_improve} epoch(s) "
                f"(best={best_score:.4f}, now={score_now:.4f})"
            )
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                logging.info(f"[EARLY-STOP] Stop at epoch {epoch+1}.")
                break

        # —— 打印每个 epoch 的总结 —— #
        logging.info(
            f"Epoch {epoch+1:04d} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"val(AUC={val_metrics['AUC']:.4f}, PCC={val_metrics['PCC']:.4f}, "
            f"PPV={val_metrics['PPV']:.4f}, Sens={val_metrics['Sensitivity']:.4f}, "
            f"F1={val_metrics['F1']:.4f}, AUPRC={val_metrics['AUPRC']:.4f}) | "
            f"test(AUC={test_metrics['AUC']:.4f}, PCC={test_metrics['PCC']:.4f}, "
            f"AUPRC={test_metrics['AUPRC']:.4f}) | "
            f"lr={current_lr:.3e} | time={(time.time()-t0)/60:.2f} min"
        )

        # 行：每轮保存断点
        util.save_checkpoint(epoch + 1, model, optimizer, filename=str(ckpt_path))

    # ========== 最优模型做最终测试 + 输出分桶表 ==========
    from model.model_multimodal_lm_ppa import MultiModalPPA_LM

    best_model = MultiModalPPA_LM(
        d_prot1_lm=d_pep, d_prot2_lm=d_mhc,
        prot1_node_dim=33, prot1_edge_dim=3,
        prot2_node_dim=33, prot2_edge_dim=3,
        gcn_hidden=128, gcn_out=128,
        d3_prot1=512, d3_prot2=512,
        d_model=256, d_attn=256, n_heads=4,
        add_interactions_3d=True
    ).to(DEVICE)

    best_model.load_state_dict(torch.load(model_best, map_location=DEVICE))
    best_model.eval()

    final_metrics, _ = evaluate_bd2017(best_model, test_loader, DEVICE, criterion=None, use_amp=USE_AMP)

    logging.info(
        f"[FINAL] test_AUC={final_metrics['AUC']:.4f}, test_PCC={final_metrics['PCC']:.4f}, "
        f"test_PPV={final_metrics['PPV']:.4f}, test_Sens={final_metrics['Sensitivity']:.4f}, "
        f"test_F1={final_metrics['F1']:.4f}, test_AUPRC={final_metrics['AUPRC']:.4f}"
    )

    # 行：写最终指标文件（包含分桶）
    with open(final_txt, "w", encoding="utf-8") as f:
        f.write(f"test_AUC: {final_metrics['AUC']:.6f}\n")
        f.write(f"test_PCC: {final_metrics['PCC']:.6f}\n")
        f.write(f"test_PPV: {final_metrics['PPV']:.6f}\n")
        f.write(f"test_Sensitivity: {final_metrics['Sensitivity']:.6f}\n")
        f.write(f"test_F1: {final_metrics['F1']:.6f}\n")
        f.write(f"test_AUPRC: {final_metrics['AUPRC']:.6f}\n\n")

        f.write("=== Buckets (test) ===\n")
        for bn, bm in final_metrics["Buckets"].items():
            f.write(
                f"{bn} | N={bm['N']} | "
                f"AUC={bm['AUC']:.6f} | PCC={bm['PCC']:.6f} | "
                f"PPV={bm['PPV']:.6f} | Sens={bm['Sensitivity']:.6f} | "
                f"F1={bm['F1']:.6f} | AUPRC={bm['AUPRC']:.6f}\n"
            )

    writer.close()  # 行：关闭 TensorBoard writer
