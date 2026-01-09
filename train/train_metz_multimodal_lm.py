# ======================================
# Metz 1D LM 训练脚本（带早停）
# 指标：CI, MSE, r2m（使用 util.get_rm2）
# 模态：1D = ESM-2 / ChemBERTa-2，只用 LM 特征
# ======================================

from pathlib import Path                     # 行：处理路径
import logging                               # 行：打印/记录日志
import time                                  # 行：计时用
import csv                                   # 行：写 CSV 指标
from typing import Dict, Any                 # 行：类型注解

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter  # 行：TensorBoard 日志

import util
from util import LoadData_metz_lm_1d        # 行：你在 util 里实现的 Metz 1D LM 数据加载函数

# ========== 设备/日志/随机种子 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 行：优先用 GPU
torch.backends.cudnn.benchmark = True                                   # 行：对固定尺寸卷积略有加速
LOG_LEVEL = logging.INFO                                                # 行：日志等级
SANITY_CHECK = True                                                     # 行：是否做 batch 形状检查

# ========== 可配置区域 ==========
DATA_METZ    = r"../dataset/metz"             # 行：Metz 原始数据目录（drug_info / targ_info / affi_info）
DATASET_NAME = "metz_lm_1d"                   # 行：任务名，用于结果存放目录
OUT_DIR      = f"../result/{DATASET_NAME}"    # 行：结果根目录

BATCH_SIZE   = 64                             # 行：batch 大小
NUM_WORKERS  = 4                              # 行：DataLoader 线程数（Win 下可调小）
PIN_MEMORY   = bool(torch.cuda.is_available())# 行：GPU 训练时开启 pin_memory 加速

EPOCHS         = 600                          # 行：最大 epoch 数（配合早停）
LR             = 1e-4                         # 行：学习率
WEIGHT_DECAY   = 1e-4                         # 行：L2 正则
USE_AMP        = False                        # 行：是否使用混合精度
ACCUM_STEPS    = 2                            # 行：梯度累积步数
GRAD_CLIP_NORM = 0.5                          # 行：梯度裁剪阈值
USE_SCHEDULER  = True                         # 行：是否使用余弦退火 lr 调度
VAL_AS_BEST    = True                         # 行：用 val_MSE 还是 test_MSE 选 best，这里用 val

# ---- 早停参数 ----
EARLY_STOP_PATIENCE  = 40                     # 行：连续多少个 epoch 无提升就停止
EARLY_STOP_MIN_DELTA = 0.0                    # 行：认为“有提升”的最小改善幅度

# ========== Metz 1D LM Dataset ==========
class MetzLMDataset(Data.Dataset):
    """封装 Metz 1D LM 特征（仅 1D）。"""
    def __init__(self, pkg: Dict[str, Any]):
        self.ids     = pkg["ids"]             # 行：样本 ID
        self.y       = pkg["y"]               # 行：标签（亲和力）
        self.smiles  = pkg["smiles"]          # 行：SMILES 字符串
        self.seq     = pkg["seq"]             # 行：蛋白序列
        self.drug_lm = pkg["drug_lm"]         # 行：药物 LM 特征
        self.prot_lm = pkg["prot_lm"]         # 行：蛋白 LM 特征

    def __len__(self):
        return len(self.y)                    # 行：样本数

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 行：返回单个样本，供 DataLoader 使用
        return {
            "id"     : self.ids[idx],
            "y"      : float(self.y[idx]),
            "smiles" : self.smiles[idx],
            "seq"    : self.seq[idx],
            "drug_lm": self.drug_lm[idx],
            "prot_lm": self.prot_lm[idx],
        }

# ========== collate：组 batch ==========
def metz_lm_collate(batch: list) -> Dict[str, Any]:
    """把若干单样本拼成一个 batch。"""
    ids    = [b["id"]     for b in batch]     # 行：ID 列表
    smiles = [b["smiles"] for b in batch]     # 行：SMILES 列表
    seqs   = [b["seq"]    for b in batch]     # 行：序列列表

    import numpy as np
    drug_lm = torch.as_tensor(               # 行：把 drug_lm 堆成 [B, D] Tensor
        np.stack([b["drug_lm"] for b in batch], axis=0),
        dtype=torch.float32
    )
    prot_lm = torch.as_tensor(               # 行：把 prot_lm 堆成 [B, D] Tensor
        np.stack([b["prot_lm"] for b in batch], axis=0),
        dtype=torch.float32
    )
    y = torch.as_tensor(                    # 行：标签堆成 [B]
        [b["y"] for b in batch],
        dtype=torch.float32
    )

    return {
        "ids"    : ids,
        "smiles" : smiles,
        "seqs"   : seqs,
        "drug_lm": drug_lm,
        "prot_lm": prot_lm,
        "y"      : y,
    }

# ========== 评估函数：CI / MSE / rm2 ==========
@torch.no_grad()
def evaluate_metz(model: torch.nn.Module,
                  loader: Data.DataLoader,
                  device: torch.device,
                  criterion: nn.Module = None):
    """
    若给定 criterion：返回 (ci, mse, rm2, avg_loss)
    否则         ：返回 (ci, mse, rm2)
    """
    model.eval()                             # 行：eval 模式（关掉 dropout 等）
    obs, pred = [], []                       # 行：保存真实值和预测值
    loss_sum, n_batches = 0.0, 0            # 行：累计 loss

    for b in loader:
        drug_lm = b["drug_lm"].to(device, non_blocking=True)
        prot_lm = b["prot_lm"].to(device, non_blocking=True)
        y_true  = b["y"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(USE_AMP and device.type == "cuda")):
            y_hat = model(drug_lm=drug_lm, prot_lm=prot_lm)   # 行：前向
            if criterion is not None:
                loss_val = criterion(y_hat, y_true)           # 行：MSE
                if torch.isfinite(loss_val):                  # 行：避免 NaN/Inf
                    loss_sum += float(loss_val)
                    n_batches += 1

        obs.extend(y_true.view(-1).tolist())                  # 行：真实值展开进列表
        pred.extend(y_hat.view(-1).tolist())                  # 行：预测值展开进列表

    ci  = util.get_cindex(obs, pred)                          # 行：CI
    mse = util.get_MSE(obs, pred)                             # 行：MSE
    rm2 = util.get_rm2(obs, pred)                             # 行：rm2（论文里的 r2m）

    if criterion is None:
        return ci, mse, rm2
    return ci, mse, rm2, (loss_sum / max(1, n_batches))       # 行：附带平均 loss

# ========== 主程序 ==========
if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL)      # 行：配置日志
    util.seed_torch()                         # 行：固定随机种子（torch/np/random）

    result_root = Path(OUT_DIR)               # 行：结果目录
    result_root.mkdir(parents=True, exist_ok=True)

    # —— 加载 Metz 1D LM 数据 —— #
    kw_common = dict(
        out_1d="../dataset/metz/processed_lm_1d",         # 行：LM 缓存目录
        logspace_trans=False,                             # 行：是否对标签做 log 变换
        esm2_model_name="facebook/esm2_t33_650M_UR50D",   # 行：蛋白 LM 名
        chemberta_model_name="DeepChem/ChemBERTa-77M-MTR",# 行：药物 LM 名
        lm_batch_size=32,                                 # 行：LM 编码时 batch_size
        use_safetensors=True,                             # 行：优先使用 safetensors
    )
    mm_train = LoadData_metz_lm_1d(DATA_METZ, split="train", **kw_common)["train"]
    mm_val   = LoadData_metz_lm_1d(DATA_METZ, split="val",   **kw_common)["val"]
    mm_test  = LoadData_metz_lm_1d(DATA_METZ, split="test",  **kw_common)["test"]

    # 行：LM 特征维度
    d_drug_lm = mm_train["drug_lm"].shape[1]
    d_prot_lm = mm_train["prot_lm"].shape[1]
    logging.info(f"[Metz|LM] drug_lm_dim={d_drug_lm}, prot_lm_dim={d_prot_lm}")

    # —— Dataset / DataLoader —— #
    train_ds = MetzLMDataset(mm_train)
    val_ds   = MetzLMDataset(mm_val)
    test_ds  = MetzLMDataset(mm_test)

    train_loader = Data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=metz_lm_collate
    )
    val_loader = Data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=metz_lm_collate
    )
    test_loader = Data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=metz_lm_collate
    )

    logging.info(f"[Metz] sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # —— Sanity Check：看下 batch 形状 —— #
    if SANITY_CHECK:
        b = next(iter(train_loader))
        print(">>> Batch keys:", list(b.keys()))
        print("drug_lm:", tuple(b["drug_lm"].shape),
              "| prot_lm:", tuple(b["prot_lm"].shape),
              "| y:", tuple(b["y"].shape))
        print("[OK] Metz LM loaders ready.")

    # —— 导入模型 —— #
    from model.model_multimodal_lm import MultiModalDTA_LM

    model = MultiModalDTA_LM(
        d_drug_lm=d_drug_lm, d_prot_lm=d_prot_lm,   # 行：LM 输入维度
        drug_node_dim=70, drug_edge_dim=6,          # 行：2D/3D 分支参数（这里实际上不用）
        prot_node_dim=33, prot_edge_dim=3,
        gcn_hidden=128, gcn_out=128,
        d3_lig=512, d3_poc=512,
        d_model=256, d_attn=256, n_heads=4,
        add_interactions_3d=True
    ).to(DEVICE)

    # —— 损失 / 优化器 / 调度器 / AMP —— #
    criterion = nn.MSELoss(reduction="mean")        # 行：回归任务用 MSE
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-5
    ) if USE_SCHEDULER else None
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE.type == "cuda"))

    # —— 日志与权重路径 —— #
    result_dir  = result_root / "train_logs"        # 行：checkpoint / csv 等
    run_dir     = result_root / "runs"              # 行：TensorBoard 日志目录
    model_best  = result_dir / "best_model.pth"     # 行：最优模型权重
    ckpt_path   = result_dir / "checkpoint.pth.tar" # 行：最新 checkpoint
    csv_file    = result_dir / "metrics.csv"        # 行：记录指标的 CSV
    final_txt   = result_dir / "final_test_metrics.txt"  # 行：最终测试指标

    result_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir))    # 行：TensorBoard writer

    # —— CSV 表头（只写一次）—— #
    if not csv_file.exists():
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "val_loss",
                "train_CI", "train_MSE", "train_r2m",
                "val_CI",   "val_MSE",   "val_r2m",
                "test_CI",  "test_MSE",  "test_r2m",
                "lr"
            ])

    # ========== 训练循环 ==========
    best_score = float("inf")                 # 行：记录最优 val_MSE
    epochs_no_improve = 0                     # 行：早停计数
    t0 = time.time()                          # 行：开始时间

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss, n_batches, last_bidx = 0.0, 0, -1

        for last_bidx, b in enumerate(train_loader):
            drug_lm = b["drug_lm"].to(DEVICE, non_blocking=True)
            prot_lm = b["prot_lm"].to(DEVICE, non_blocking=True)
            y_true  = b["y"].to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE.type == "cuda")):
                y_hat = model(drug_lm=drug_lm, prot_lm=prot_lm)
                loss  = criterion(y_hat, y_true)

            if not torch.isfinite(loss):      # 行：防 NaN/Inf
                logging.warning(f"[NaN|Metz] loss at epoch={epoch}, batch={last_bidx}: {float(loss)}")
                optimizer.zero_grad(set_to_none=True)
                continue

            if scaler.is_enabled():
                scaler.scale(loss / ACCUM_STEPS).backward()
            else:
                (loss / ACCUM_STEPS).backward()

            # 行：梯度累积到 ACCUM_STEPS 次再更新
            if (last_bidx + 1) % ACCUM_STEPS == 0:
                if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

                if scaler.is_enabled():
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss)
            n_batches    += 1

        # 尾 batch：不足 ACCUM_STEPS 也要再更新一次
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
        writer.add_scalar("Loss/train", train_loss, epoch)   # ★ 行：带 epoch

        # —— 评估 train / val / test —— #
        train_ci, train_mse, train_r2m, _ = evaluate_metz(model, train_loader, DEVICE, criterion)
        val_ci,   val_mse,   val_r2m,   val_loss = evaluate_metz(model, val_loader, DEVICE, criterion)
        test_ci,  test_mse,  test_r2m            = evaluate_metz(model, test_loader, DEVICE, criterion=None)

        writer.add_scalar("Loss/val", val_loss, epoch)       # ★ 行：带 epoch

        # ★★★★★ 关键修正：所有指标写 TensorBoard 时都要传 epoch ★★★★★
        for tag, val in [
            ("train_CI",  train_ci),
            ("train_MSE", train_mse),
            ("train_r2m", train_r2m),
            ("val_CI",    val_ci),
            ("val_MSE",   val_mse),
            ("val_r2m",   val_r2m),
            ("test_CI",   test_ci),
            ("test_MSE",  test_mse),
            ("test_r2m",  test_r2m),
        ]:
            writer.add_scalar(f"Metrics/{tag}", val, epoch)  # ★ 行：第三个参数一定是 epoch

        # —— 学习率调度 —— #
        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("LR", current_lr, epoch)

        # —— 写 CSV 一行 —— #
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, train_loss, val_loss,
                train_ci, train_mse, train_r2m,
                val_ci,   val_mse,   val_r2m,
                test_ci,  test_mse,  test_r2m,
                current_lr
            ])

        # —— 以 val_MSE 做 early stopping & best model —— #
        score_now = val_mse if VAL_AS_BEST else test_mse
        if score_now + EARLY_STOP_MIN_DELTA < best_score and torch.isfinite(torch.tensor(score_now)):
            best_score = score_now
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_best)
            logging.info(f"[BEST|Metz] epoch={epoch+1}, val_MSE={best_score:.4f} -> saved: {model_best}")
        else:
            epochs_no_improve += 1
            logging.info(
                f"[EARLY-STOP|Metz] epoch={epoch+1}: no improvement for {epochs_no_improve} epoch(s) "
                f"(best={best_score:.4f}, now={score_now:.4f})"
            )
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                logging.info(
                    f"[EARLY-STOP|Metz] Stop training at epoch {epoch+1}, "
                    f"no improvement for {EARLY_STOP_PATIENCE} epochs."
                )
                break

        logging.info(
            f"[Metz] Epoch {epoch+1:04d} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"train(CI={train_ci:.4f}, MSE={train_mse:.4f}, r2m={train_r2m:.4f}) | "
            f"val(CI={val_ci:.4f}, MSE={val_mse:.4f}, r2m={val_r2m:.4f}) | "
            f"test(CI={test_ci:.4f}, MSE={test_mse:.4f}, r2m={test_r2m:.4f}) | "
            f"lr={current_lr:.3e} | time={(time.time()-t0)/60:.2f} min"
        )

        util.save_checkpoint(epoch + 1, model, optimizer, filename=str(ckpt_path))

    # ========== 用最优模型在 test 上做最终评估 ==========
    best_model = MultiModalDTA_LM(
        d_drug_lm=d_drug_lm, d_prot_lm=d_prot_lm,
        drug_node_dim=70, drug_edge_dim=6,
        prot_node_dim=33, prot_edge_dim=3,
        gcn_hidden=128, gcn_out=128,
        d3_lig=512, d3_poc=512,
        d_model=256, d_attn=256, n_heads=4,
        add_interactions_3d=True
    ).to(DEVICE)
    best_model.load_state_dict(torch.load(model_best, map_location=DEVICE))
    best_model.eval()

    with torch.no_grad():
        final_ci, final_mse, final_r2m = evaluate_metz(
            best_model, test_loader, DEVICE, criterion=None
        )

    logging.info(
        f"[FINAL|Metz] test_CI={final_ci:.4f}, test_MSE={final_mse:.4f}, test_r2m={final_r2m:.4f}"
    )
    with open(final_txt, "w", encoding="utf-8") as f:
        f.write(f"test_CI: {final_ci:.4f}\n")
        f.write(f"test_MSE: {final_mse:.4f}\n")
        f.write(f"test_r2m: {final_r2m:.4f}\n")

    writer.close()
