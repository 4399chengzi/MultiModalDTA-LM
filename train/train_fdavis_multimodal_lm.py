# ======================================
# FDavis 1D LM 版训练脚本（带早停）
# 指标：RMSE, CI, Spearman   （不计算 rm2 / AUPR）
# 模态：1D = ESM-2 / ChemBERTa-2；不使用 2D/3D
# ======================================

from pathlib import Path                       # 行：路径处理
import logging                                 # 行：日志
import time                                    # 行：计时
import csv                                     # 行：写 CSV 指标
from typing import Dict, Any                   # 行：类型注解

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter

import util
from util import LoadData_fdavis_lm_1d         # 行：FDavis 1D LM 数据加载函数（你在 util 里实现的那个）

# ========== 设备/日志/随机种子 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 行：优先用 GPU
torch.backends.cudnn.benchmark = True                                   # 行：固定输入时加速卷积
LOG_LEVEL = logging.INFO                                                # 行：日志等级
SANITY_CHECK = True                                                     # 行：是否做一次 batch 形状检查

# ========== 可配置区域 ==========
DATA_FDAVIS   = r"../dataset/fdavis"            # 行：FDavis 原始数据目录（含 affi_info.txt）
DATASET_NAME = "fdavis_lm_1d"                   # 行：结果任务名，用于存放日志/模型
OUT_DIR      = f"../result/{DATASET_NAME}"      # 行：结果根目录

BATCH_SIZE   = 64                               # 行：FDavis 样本中等，batch 可按显存调
NUM_WORKERS  = 4                                # 行：DataLoader 的 worker 数（Windows 可改 0/2）
PIN_MEMORY   = bool(torch.cuda.is_available())  # 行：GPU 时开启 pin_memory 加速拷贝

EPOCHS         = 600                            # 行：最大 epoch 数（配合早停，跑不满也没关系）
LR             = 1e-4                           # 行：保守学习率（LM 特征）
WEIGHT_DECAY   = 1e-4                           # 行：L2 正则
USE_AMP        = False                          # 行：先关 AMP，保证稳定；显存吃紧再改 True
ACCUM_STEPS    = 2                              # 行：梯度累积步数（1 表示不用累积）
GRAD_CLIP_NORM = 0.5                            # 行：梯度裁剪阈值
USE_SCHEDULER  = True                           # 行：是否使用余弦退火调度器
VAL_AS_BEST    = True                           # 行：以 val_RMSE 作为“最优模型”标准

# ---- 早停相关参数 ----
EARLY_STOP_PATIENCE  = 40      # 行：若连续 40 个 epoch val_RMSE 无提升，则提前停止
EARLY_STOP_MIN_DELTA = 0.0     # 行：认为“有提升”的最小幅度（新 RMSE 至少比 best 小 0.0）

# ========== FDavis 1D LM Dataset ==========
class FDavisLMDataset(Data.Dataset):
    """封装 FDavis 1D LM 特征（不包含 2D/3D）。"""
    def __init__(self, pkg: Dict[str, Any]):
        self.ids     = pkg["ids"]        # 行：np.array(object)，样本 ID（例如 fd_0）
        self.y       = pkg["y"]          # 行：np.float32，标签（pKd 等）
        self.smiles  = pkg["smiles"]     # 行：np.array(str)，SMILES
        self.seq     = pkg["seq"]        # 行：np.array(str)，蛋白序列
        self.drug_lm = pkg["drug_lm"]    # 行：np.ndarray[N, D_drug_lm]
        self.prot_lm = pkg["prot_lm"]    # 行：np.ndarray[N, D_prot_lm]

    def __len__(self):
        return len(self.y)               # 行：样本数量

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "id"     : self.ids[idx],    # 行：样本 ID
            "y"      : float(self.y[idx]),
            "smiles" : self.smiles[idx],
            "seq"    : self.seq[idx],
            "drug_lm": self.drug_lm[idx],
            "prot_lm": self.prot_lm[idx],
        }

# ========== collate：合并 batch（只有 1D）==========
def fdavis_lm_collate(batch: list) -> Dict[str, Any]:
    """行：LM 特征/标签堆叠为 tensor；SMILES/序列保留 list。"""
    ids    = [b["id"]     for b in batch]      # 行：收集 ID
    smiles = [b["smiles"] for b in batch]      # 行：收集 SMILES
    seqs   = [b["seq"]    for b in batch]      # 行：收集 序列

    import numpy as np
    drug_lm = torch.as_tensor(                 # 行：把单样本的 drug_lm 堆成 [B, D]
        np.stack([b["drug_lm"] for b in batch], axis=0),
        dtype=torch.float32
    )
    prot_lm = torch.as_tensor(                 # 行：把单样本的 prot_lm 堆成 [B, D]
        np.stack([b["prot_lm"] for b in batch], axis=0),
        dtype=torch.float32
    )
    y = torch.as_tensor([b["y"] for b in batch], dtype=torch.float32)  # 行：标签拼成 [B]

    return {
        "ids"    : ids,
        "smiles" : smiles,
        "seqs"   : seqs,
        "drug_lm": drug_lm,
        "prot_lm": prot_lm,
        "y"      : y,
    }

# ========== FDavis 评估函数：RMSE / CI / Spearman ==========
@torch.no_grad()
def evaluate_fdavis(model: torch.nn.Module,
                    loader: Data.DataLoader,
                    device: torch.device,
                    criterion: nn.Module = None):
    """
    若给定 criterion：返回 (rmse, ci, spearman, avg_loss)
    否则         ：返回 (rmse, ci, spearman)
    """
    model.eval()                             # 行：eval 模式关闭 dropout
    obs, pred = [], []                       # 行：收集真实值和预测值
    loss_sum, n_batches = 0.0, 0            # 行：记录 loss

    for b in loader:
        drug_lm = b["drug_lm"].to(device, non_blocking=True)   # 行：药物 LM 特征
        prot_lm = b["prot_lm"].to(device, non_blocking=True)   # 行：蛋白 LM 特征
        y_true  = b["y"].to(device, non_blocking=True)         # 行：真实标签

        with torch.cuda.amp.autocast(enabled=(USE_AMP and device.type == "cuda")):
            y_hat = model(drug_lm=drug_lm, prot_lm=prot_lm)    # 行：前向得到预测
            if criterion is not None:
                loss_val = criterion(y_hat, y_true)            # 行：计算 MSE 损失
                if torch.isfinite(loss_val):
                    loss_sum += float(loss_val)
                    n_batches += 1

        obs.extend(y_true.view(-1).tolist())   # 行：把真实值放进 list
        pred.extend(y_hat.view(-1).tolist())   # 行：把预测值放进 list

    # 转 numpy 计算指标（调用 util 里的函数）
    rmse     = util.get_RMSE(obs, pred)        # 行：RMSE
    ci       = util.get_cindex(obs, pred)      # 行：C-index（CI）
    spearman = util.get_spearmanr(obs, pred)   # 行：Spearman 相关系数

    if criterion is None:
        return rmse, ci, spearman              # 行：无 loss 情况
    return rmse, ci, spearman, (loss_sum / max(1, n_batches))   # 行：返回平均 loss

# ========== 主程序 ==========
if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL)          # 行：初始化日志配置
    util.seed_torch()                             # 行：固定随机种子，方便复现

    result_root = Path(OUT_DIR)                   # 行：结果目录 Path 对象
    result_root.mkdir(parents=True, exist_ok=True)# 行：若不存在则创建

    # —— 加载 FDavis 1D LM 数据 —— #
    kw_common = dict(
        out_1d="../dataset/fdavis/processed_lm_1d",  # 行：缓存目录，要和 util 中保持一致
        logspace_trans=False,                        # 行：是否对 affinity 做 log 变换（默认 False，可以改）
        esm2_model_name="facebook/esm2_t33_650M_UR50D", # 行：蛋白 LM 模型名
        chemberta_model_name="DeepChem/ChemBERTa-77M-MTR",# 行：药物 LM 模型名
        lm_batch_size=32,                           # 行：LM 编码时的 batch_size
        use_safetensors=True,                       # 行：优先用 safetensors
    )
    mm_train = LoadData_fdavis_lm_1d(
        data_dir=DATA_FDAVIS, split="train", **kw_common
    )["train"]                                      # 行：训练集
    mm_val   = LoadData_fdavis_lm_1d(
        data_dir=DATA_FDAVIS, split="val", **kw_common
    )["val"]                                        # 行：验证集
    mm_test  = LoadData_fdavis_lm_1d(
        data_dir=DATA_FDAVIS, split="test", **kw_common
    )["test"]                                       # 行：测试集

    # LM 维度
    d_drug_lm = mm_train["drug_lm"].shape[1]        # 行：药物 LM 特征维度
    d_prot_lm = mm_train["prot_lm"].shape[1]        # 行：蛋白 LM 特征维度
    logging.info(f"[FDavis|LM] drug_lm_dim={d_drug_lm}, prot_lm_dim={d_prot_lm}")

    # —— Dataset / DataLoader —— #
    train_ds = FDavisLMDataset(mm_train)            # 行：构造训练集 Dataset
    val_ds   = FDavisLMDataset(mm_val)              # 行：构造验证集 Dataset
    test_ds  = FDavisLMDataset(mm_test)             # 行：构造测试集 Dataset

    train_loader = Data.DataLoader(                 # 行：训练 DataLoader
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=fdavis_lm_collate
    )
    val_loader = Data.DataLoader(                   # 行：验证 DataLoader
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=fdavis_lm_collate
    )
    test_loader = Data.DataLoader(                  # 行：测试 DataLoader
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=fdavis_lm_collate
    )

    logging.info(f"[FDavis] sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # —— Sanity Check —— #
    if SANITY_CHECK:
        b = next(iter(train_loader))                # 行：取一个 batch 看看形状
        print(">>> Batch keys:", list(b.keys()))
        print("drug_lm:", tuple(b["drug_lm"].shape),
              "| prot_lm:", tuple(b["prot_lm"].shape),
              "| y:", tuple(b["y"].shape))
        print("[OK] FDavis LM loaders ready.")

    # —— 导入模型 —— #
    # 行：继续用 MultiModalDTA_LM，只启用 LM 分支（forward 只传 drug_lm / prot_lm）
    from model.model_multimodal_lm import MultiModalDTA_LM

    model = MultiModalDTA_LM(
        d_drug_lm=d_drug_lm, d_prot_lm=d_prot_lm,   # 行：LM 维度
        drug_node_dim=70, drug_edge_dim=6,          # 行：2D/3D 不用，这些参数不会用到
        prot_node_dim=33, prot_edge_dim=3,
        gcn_hidden=128, gcn_out=128,
        d3_lig=512, d3_poc=512,
        d_model=256, d_attn=256, n_heads=4,
        add_interactions_3d=True
    ).to(DEVICE)

    # —— 优化器 / 损失 / 调度 / AMP —— #
    criterion = nn.MSELoss(reduction="mean")        # 行：回归任务用 MSE 作为训练损失
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-5
    ) if USE_SCHEDULER else None
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE.type == "cuda"))

    # —— 日志与权重路径 —— #
    result_dir  = result_root / "train_logs"        # 行：存 checkpoint / csv 等
    run_dir     = result_root / "runs"              # 行：TensorBoard 日志目录
    model_best  = result_dir / "best_model.pth"     # 行：最优模型权重路径
    ckpt_path   = result_dir / "checkpoint.pth.tar" # 行：最新 checkpoint
    csv_file    = result_dir / "metrics.csv"        # 行：记录指标的 CSV
    final_txt   = result_dir / "final_test_metrics.txt"  # 行：最终测试结果 txt

    result_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir))    # 行：初始化 TensorBoard 写入器

    # —— 断点恢复（可选）—— #
    start_epoch = 0                                 # 行：起始 epoch
    if ckpt_path.exists():
        start_epoch = util.load_checkpoint(str(ckpt_path), model, optimizer)
        logging.info(f"[Resume|FDavis] from epoch: {start_epoch}")

    # —— CSV 表头（对应 RMSE / CI / Spearman）—— #
    if not csv_file.exists():
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "val_loss",
                "train_RMSE", "train_CI", "train_Spearman",
                "val_RMSE",   "val_CI",   "val_Spearman",
                "test_RMSE",  "test_CI",  "test_Spearman",
                "lr"
            ])

    # ========== 训练主循环 ==========
    best_score = float("inf")    # 行：记录最优 val_RMSE
    epochs_no_improve = 0        # 行：记录连续“无提升”的 epoch 数
    t0 = time.time()             # 行：记录起始时间

    for epoch in range(start_epoch, EPOCHS):
        model.train()                                # 行：训练模式
        optimizer.zero_grad(set_to_none=True)        # 行：清空梯度

        running_loss, n_batches, last_bidx = 0.0, 0, -1

        for last_bidx, b in enumerate(train_loader):
            drug_lm = b["drug_lm"].to(DEVICE, non_blocking=True)
            prot_lm = b["prot_lm"].to(DEVICE, non_blocking=True)
            y_true  = b["y"].to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE.type == "cuda")):
                y_hat = model(drug_lm=drug_lm, prot_lm=prot_lm)   # 行：前向
                loss = criterion(y_hat, y_true)                   # 行：计算 MSE 损失

            if not torch.isfinite(loss):                          # 行：NaN/Inf 检查
                logging.warning(f"[NaN|FDavis] loss at epoch={epoch}, batch={last_bidx}: {float(loss)}")
                optimizer.zero_grad(set_to_none=True)
                continue

            if scaler.is_enabled():
                scaler.scale(loss / ACCUM_STEPS).backward()       # 行：支持混合精度的反向
            else:
                (loss / ACCUM_STEPS).backward()

            if (last_bidx + 1) % ACCUM_STEPS == 0:                # 行：梯度累积到一定步数再 step
                if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)                # 行：AMP 下先 unscale 再裁剪
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

                if scaler.is_enabled():
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss)                            # 行：累计 loss
            n_batches    += 1

        # 尾批（如果最后一个 batch 没凑够 ACCUM_STEPS，这里再 step 一次）
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

        train_loss = running_loss / max(1, n_batches)              # 行：平均训练 loss
        writer.add_scalar("Loss/train", train_loss, epoch)         # 行：写入 TensorBoard

        # —— 评估：train/val/test（算 RMSE / CI / Spearman）—— #
        train_rmse, train_ci, train_sp, _ = evaluate_fdavis(
            model, train_loader, DEVICE, criterion
        )
        val_rmse,   val_ci,   val_sp,   val_loss = evaluate_fdavis(
            model, val_loader,   DEVICE, criterion
        )
        test_rmse,  test_ci,  test_sp             = evaluate_fdavis(
            model, test_loader,  DEVICE, criterion=None
        )

        writer.add_scalar("Loss/val", val_loss, epoch)             # 行：验证集 loss

        # 行：各指标写入 TensorBoard
        for tag, val in [
            ("train_RMSE",   train_rmse),
            ("train_CI",     train_ci),
            ("train_Spearman", train_sp),
            ("val_RMSE",     val_rmse),
            ("val_CI",       val_ci),
            ("val_Spearman", val_sp),
            ("test_RMSE",    test_rmse),
            ("test_CI",      test_ci),
            ("test_Spearman", test_sp),
        ]:
            writer.add_scalar(f"Metrics/{tag}", val, epoch)

        # —— 学习率调度 —— #
        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]               # 行：当前学习率
        writer.add_scalar("LR", current_lr, epoch)

        # —— 写 CSV（对应 RMSE / CI / Spearman）—— #
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, train_loss, val_loss,
                train_rmse, train_ci, train_sp,
                val_rmse,   val_ci,   val_sp,
                test_rmse,  test_ci,  test_sp,
                current_lr
            ])

        # —— 以 val_RMSE 选 best model & 早停 —— #
        score_now = val_rmse if VAL_AS_BEST else test_rmse        # 行：当前早停指标

        if score_now + EARLY_STOP_MIN_DELTA < best_score and torch.isfinite(torch.tensor(score_now)):
            best_score = score_now
            epochs_no_improve = 0                                  # 行：有提升，计数清零
            torch.save(model.state_dict(), model_best)             # 行：保存最优模型
            logging.info(
                f"[BEST|FDavis] epoch={epoch+1}, "
                f"{'val' if VAL_AS_BEST else 'test'}_RMSE={best_score:.4f} "
                f"-> saved: {model_best}"
            )
        else:
            epochs_no_improve += 1                                 # 行：无提升，计数+1
            logging.info(
                f"[EARLY-STOP|FDavis] epoch={epoch+1}: no improvement on "
                f"{'val' if VAL_AS_BEST else 'test'}_RMSE for {epochs_no_improve} epoch(s) "
                f"(best={best_score:.4f}, now={score_now:.4f})"
            )
            if epochs_no_improve >= EARLY_STOP_PATIENCE:           # 行：达到耐心上限，停止训练
                logging.info(
                    f"[EARLY-STOP|FDavis] Stop training at epoch {epoch+1}, "
                    f"no improvement for {EARLY_STOP_PATIENCE} epochs."
                )
                break

        logging.info(
            f"[FDavis] Epoch {epoch+1:04d} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"train(RMSE={train_rmse:.4f}, CI={train_ci:.4f}, Spearman={train_sp:.4f}) | "
            f"val(RMSE={val_rmse:.4f}, CI={val_ci:.4f}, Spearman={val_sp:.4f}) | "
            f"test(RMSE={test_rmse:.4f}, CI={test_ci:.4f}, Spearman={test_sp:.4f}) | "
            f"lr={current_lr:.3e} | time={(time.time()-t0)/60:.2f} min"
        )

        util.save_checkpoint(epoch + 1, model, optimizer, filename=str(ckpt_path))

    # ========== 最优模型做最终测试 ==========
    best_model = MultiModalDTA_LM(                            # 行：重新构造一个同结构的模型
        d_drug_lm=d_drug_lm, d_prot_lm=d_prot_lm,
        drug_node_dim=70, drug_edge_dim=6,
        prot_node_dim=33, prot_edge_dim=3,
        gcn_hidden=128, gcn_out=128,
        d3_lig=512, d3_poc=512,
        d_model=256, d_attn=256, n_heads=4,
        add_interactions_3d=True
    ).to(DEVICE)
    best_model.load_state_dict(torch.load(model_best, map_location=DEVICE))  # 行：加载最优权重
    best_model.eval()

    with torch.no_grad():
        final_rmse, final_ci, final_sp = evaluate_fdavis(      # 行：只算 RMSE/CI/Spearman
            best_model, test_loader, DEVICE, criterion=None
        )

    logging.info(
        f"[FINAL|FDavis] test_RMSE={final_rmse:.4f}, "
        f"test_CI={final_ci:.4f}, test_Spearman={final_sp:.4f}"
    )
    with open(final_txt, "w", encoding="utf-8") as f:
        f.write(f"test_RMSE: {final_rmse:.4f}\n")
        f.write(f"test_CI: {final_ci:.4f}\n")
        f.write(f"test_Spearman: {final_sp:.4f}\n")

    writer.close()
