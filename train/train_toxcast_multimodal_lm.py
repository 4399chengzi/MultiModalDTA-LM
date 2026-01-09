# ======================================
# ToxCast 1D LM 训练脚本（带早停）
# 指标：CI, MSE, r2m
# 模态：1D = ESM-2 / ChemBERTa-2，仅用 LM 特征
# ======================================

from pathlib import Path                       # 行：路径处理工具
import logging                                 # 行：日志模块
import time                                    # 行：计时用
import csv                                     # 行：写入 CSV 指标
from typing import Dict, Any                   # 行：类型注解

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter  # 行：TensorBoard 日志

import util                                    # 行：你自己的工具包（含指标等）
from util import LoadData_toxcast_lm_1d        # 行：ToxCast 1D LM 数据加载函数（你在 util 里实现）

# ========== 设备/日志/随机种子 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 行：有 GPU 就用 GPU
torch.backends.cudnn.benchmark = True                                   # 行：卷积加速（输入尺寸相对固定时效果更好）
LOG_LEVEL = logging.INFO                                                # 行：日志等级
SANITY_CHECK = True                                                     # 行：是否做一次 batch 形状检查

# ========== 可配置区域 ==========
DATA_TOXCAST  = r"../dataset/toxcast"         # 行：ToxCast 原始数据目录（data_train.csv / data_test.csv）
DATASET_NAME  = "toxcast_lm_1d"              # 行：结果任务名，用于存放日志/模型
OUT_DIR       = f"../result/{DATASET_NAME}"  # 行：结果根目录

BATCH_SIZE   = 64                             # 行：batch 大小，可按显存调整
NUM_WORKERS  = 4                              # 行：DataLoader 的 worker 数（Windows 可改小）
PIN_MEMORY   = bool(torch.cuda.is_available())# 行：GPU 训练时开启 pin_memory 提升数据拷贝速度

EPOCHS         = 600                          # 行：最大 epoch 数（配合早停，实际不一定跑满）
LR             = 1e-4                         # 行：学习率（LM 特征一般用小一点）
WEIGHT_DECAY   = 1e-4                         # 行：L2 权重衰减
USE_AMP        = False                        # 行：是否使用混合精度（显存紧张可以改 True）
ACCUM_STEPS    = 2                            # 行：梯度累积步数（1 表示不用累积）
GRAD_CLIP_NORM = 0.5                          # 行：梯度裁剪阈值，防梯度爆炸
USE_SCHEDULER  = True                         # 行：是否使用余弦退火学习率调度
VAL_AS_BEST    = True                         # 行：早停时以 val_MSE 为最优标准

# ---- 早停相关参数 ----
EARLY_STOP_PATIENCE  = 40      # 行：若连续 40 个 epoch 指标无提升，则提前停止
EARLY_STOP_MIN_DELTA = 0.0     # 行：认为“有提升”的最小改善幅度

# ========== ToxCast 1D LM Dataset ==========
class ToxCastLMDataset(Data.Dataset):
    """封装 ToxCast 1D LM 特征（仅 1D 模态）。"""
    def __init__(self, pkg: Dict[str, Any]):
        self.ids     = pkg["ids"]        # 行：样本 ID（如 tr_0 / te_0）
        self.y       = pkg["y"]          # 行：标签（连续值，来自 data_train/test.csv 中的 label）
        self.smiles  = pkg["smiles"]     # 行：SMILES 字符串
        self.seq     = pkg["seq"]        # 行：蛋白序列
        self.drug_lm = pkg["drug_lm"]    # 行：药物 LM 向量 [N, D_drug_lm]
        self.prot_lm = pkg["prot_lm"]    # 行：蛋白 LM 向量 [N, D_prot_lm]

    def __len__(self):
        return len(self.y)               # 行：样本数

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 行：返回单一样本的字典形式，方便 collate_fn 拼 batch
        return {
            "id"     : self.ids[idx],
            "y"      : float(self.y[idx]),
            "smiles" : self.smiles[idx],
            "seq"    : self.seq[idx],
            "drug_lm": self.drug_lm[idx],
            "prot_lm": self.prot_lm[idx],
        }

# ========== collate：合并 batch ==========
def toxcast_lm_collate(batch: list) -> Dict[str, Any]:
    """行：把单样本 list 合成一个 batch 张量/列表。"""
    ids    = [b["id"]     for b in batch]    # 行：ID 列表
    smiles = [b["smiles"] for b in batch]    # 行：SMILES 列表（一般仅用于 debug）
    seqs   = [b["seq"]    for b in batch]    # 行：序列列表

    import numpy as np
    # 行：LM 向量堆成 [B, D] 的 float32 Tensor
    drug_lm = torch.as_tensor(
        np.stack([b["drug_lm"] for b in batch], axis=0),
        dtype=torch.float32
    )
    prot_lm = torch.as_tensor(
        np.stack([b["prot_lm"] for b in batch], axis=0),
        dtype=torch.float32
    )
    # 行：标签拼成 [B]
    y = torch.as_tensor([b["y"] for b in batch], dtype=torch.float32)

    return {
        "ids"    : ids,
        "smiles" : smiles,
        "seqs"   : seqs,
        "drug_lm": drug_lm,
        "prot_lm": prot_lm,
        "y"      : y,
    }

# ========== 评估函数：CI / MSE / r2m ==========
@torch.no_grad()
def evaluate_toxcast(model: torch.nn.Module,
                     loader: Data.DataLoader,
                     device: torch.device,
                     criterion: nn.Module = None):
    """
    若给定 criterion：返回 (ci, mse, r2m, avg_loss)
    否则         ：返回 (ci, mse, r2m)
    """
    model.eval()                               # 行：eval 模式（关闭 dropout 等）
    obs, pred = [], []                         # 行：存放真实值/预测值
    loss_sum, n_batches = 0.0, 0              # 行：累计 loss 和 batch 数

    for b in loader:
        drug_lm = b["drug_lm"].to(device, non_blocking=True)   # 行：药物 LM 特征
        prot_lm = b["prot_lm"].to(device, non_blocking=True)   # 行：蛋白 LM 特征
        y_true  = b["y"].to(device, non_blocking=True)         # 行：真实标签

        with torch.cuda.amp.autocast(enabled=(USE_AMP and device.type == "cuda")):
            y_hat = model(drug_lm=drug_lm, prot_lm=prot_lm)    # 行：前向预测
            if criterion is not None:
                loss_val = criterion(y_hat, y_true)            # 行：MSE 损失
                if torch.isfinite(loss_val):                   # 行：过滤 NaN/Inf
                    loss_sum += float(loss_val)
                    n_batches += 1

        obs.extend(y_true.view(-1).tolist())                   # 行：收集真实值
        pred.extend(y_hat.view(-1).tolist())                   # 行：收集预测值

    # ---- 计算三个指标（和 Metz 一致）----
    ci   = util.get_cindex(obs, pred)                          # 行：CI
    mse  = util.get_MSE(obs, pred)                             # 行：MSE
    rm2  = util.get_rm2(obs, pred)                             # 行：rm2 / r2m

    if criterion is None:
        return ci, mse, rm2                                    # 行：仅返回指标
    return ci, mse, rm2, (loss_sum / max(1, n_batches))        # 行：附带平均 loss

# ========== 主程序 ==========
if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL)          # 行：初始化日志配置
    util.seed_torch()                             # 行：固定随机种子，保证可复现

    result_root = Path(OUT_DIR)                   # 行：结果目录 Path 对象
    result_root.mkdir(parents=True, exist_ok=True)# 行：不存在则创建目录

    # —— 加载 ToxCast 1D LM 数据 —— #
    kw_common = dict(
        out_1d="../dataset/toxcast/processed_lm_1d",   # 行：1D LM 缓存目录（要和 util 中一致）
        logspace_trans=False,                          # 行：这里默认不做 log 变换
        esm2_model_name="facebook/esm2_t33_650M_UR50D",# 行：蛋白 LM 模型
        chemberta_model_name="DeepChem/ChemBERTa-77M-MTR", # 行：药物 LM 模型
        lm_batch_size=32,                              # 行：LM 编码 batch_size
        use_safetensors=True,                          # 行：优先使用 safetensors
    )
    # 行：分别加载 train / val / test 三个划分
    mm_train = LoadData_toxcast_lm_1d(DATA_TOXCAST, split="train", **kw_common)["train"]
    mm_val   = LoadData_toxcast_lm_1d(DATA_TOXCAST, split="val",   **kw_common)["val"]
    mm_test  = LoadData_toxcast_lm_1d(DATA_TOXCAST, split="test",  **kw_common)["test"]

    # 行：从训练集里读出 LM 向量维度
    d_drug_lm = mm_train["drug_lm"].shape[1]
    d_prot_lm = mm_train["prot_lm"].shape[1]
    logging.info(f"[ToxCast|LM] drug_lm_dim={d_drug_lm}, prot_lm_dim={d_prot_lm}")

    # —— Dataset / DataLoader 构建 —— #
    train_ds = ToxCastLMDataset(mm_train)        # 行：训练集 Dataset
    val_ds   = ToxCastLMDataset(mm_val)          # 行：验证集 Dataset
    test_ds  = ToxCastLMDataset(mm_test)         # 行：测试集 Dataset

    train_loader = Data.DataLoader(              # 行：训练 DataLoader
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=toxcast_lm_collate
    )
    val_loader = Data.DataLoader(                # 行：验证 DataLoader
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=toxcast_lm_collate
    )
    test_loader = Data.DataLoader(               # 行：测试 DataLoader
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=toxcast_lm_collate
    )

    logging.info(f"[ToxCast] sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # —— Sanity Check：看一眼 batch 形状 —— #
    if SANITY_CHECK:
        b = next(iter(train_loader))
        print(">>> Batch keys:", list(b.keys()))
        print("drug_lm:", tuple(b["drug_lm"].shape),
              "| prot_lm:", tuple(b["prot_lm"].shape),
              "| y:", tuple(b["y"].shape))
        print("[OK] ToxCast LM loaders ready.")

    # —— 导入模型：沿用 MultiModalDTA_LM，只用 LM 分支 —— #
    from model.model_multimodal_lm import MultiModalDTA_LM

    model = MultiModalDTA_LM(
        d_drug_lm=d_drug_lm, d_prot_lm=d_prot_lm,   # 行：传入 LM 维度
        drug_node_dim=70, drug_edge_dim=6,          # 行：2D/3D 分支虽然不用，但参数要给
        prot_node_dim=33, prot_edge_dim=3,
        gcn_hidden=128, gcn_out=128,
        d3_lig=512, d3_poc=512,
        d_model=256, d_attn=256, n_heads=4,
        add_interactions_3d=True
    ).to(DEVICE)

    # —— 损失/优化器/调度器/AMP —— #
    criterion = nn.MSELoss(reduction="mean")       # 行：回归任务损失函数 MSE
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)  # 行：AdamW 优化器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-5
    ) if USE_SCHEDULER else None                   # 行：余弦退火+重启学习率调度
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE.type == "cuda"))  # 行：AMP 缩放器

    # —— 日志与权重路径 —— #
    result_dir  = result_root / "train_logs"       # 行：保存 checkpoint / csv 等
    run_dir     = result_root / "runs"             # 行：TensorBoard 目录
    model_best  = result_dir / "best_model.pth"    # 行：最优模型权重文件
    ckpt_path   = result_dir / "checkpoint.pth.tar"# 行：最新 checkpoint 文件
    csv_file    = result_dir / "metrics.csv"       # 行：记录训练过程指标
    final_txt   = result_dir / "final_test_metrics.txt"  # 行：最终测试指标

    result_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir))   # 行：初始化 TensorBoard 写入器

    # —— 如果有 checkpoint 则尝试恢复 —— #
    start_epoch = 0                                # 行：默认从 0 开始
    if ckpt_path.exists():
        start_epoch = util.load_checkpoint(str(ckpt_path), model, optimizer)
        logging.info(f"[Resume|ToxCast] from epoch: {start_epoch}")

    # —— CSV 表头（CI / MSE / r2m）—— #
    if not csv_file.exists():
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "val_loss",
                "train_CI", "train_MSE", "train_r2m",
                "val_CI",   "val_MSE",   "val_r2m",
                "test_CI",  "test_MSE",  "test_r2m",
                "lr"
            ])

    # ========== 训练主循环 ==========
    best_score = float("inf")        # 行：记录最优 val_MSE（越小越好）
    epochs_no_improve = 0            # 行：连续“无提升”的 epoch 计数
    t0 = time.time()                 # 行：记录起始时间

    for epoch in range(start_epoch, EPOCHS):
        model.train()                                # 行：切换到训练模式
        optimizer.zero_grad(set_to_none=True)        # 行：清空梯度

        running_loss, n_batches, last_bidx = 0.0, 0, -1

        for last_bidx, b in enumerate(train_loader):
            drug_lm = b["drug_lm"].to(DEVICE, non_blocking=True)
            prot_lm = b["prot_lm"].to(DEVICE, non_blocking=True)
            y_true  = b["y"].to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE.type == "cuda")):
                y_hat = model(drug_lm=drug_lm, prot_lm=prot_lm)   # 行：前向计算
                loss  = criterion(y_hat, y_true)                  # 行：MSE 损失

            if not torch.isfinite(loss):                          # 行：NaN/Inf 检查
                logging.warning(f"[NaN|ToxCast] loss at epoch={epoch}, batch={last_bidx}: {float(loss)}")
                optimizer.zero_grad(set_to_none=True)
                continue

            if scaler.is_enabled():
                scaler.scale(loss / ACCUM_STEPS).backward()       # 行：AMP 下的反向传播
            else:
                (loss / ACCUM_STEPS).backward()

            # 行：梯度累积到 ACCUM_STEPS 再更新
            if (last_bidx + 1) % ACCUM_STEPS == 0:
                if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)                # 行：AMP 时先 unscale 再裁剪
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

                if scaler.is_enabled():
                    scaler.step(optimizer); scaler.update()       # 行：带 AMP 的 step/update
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss)                           # 行：累计 loss
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

        train_loss = running_loss / max(1, n_batches)             # 行：平均训练损失
        writer.add_scalar("Loss/train", train_loss, epoch)        # 行：写入 TensorBoard

        # —— 在 train/val/test 上评估：CI, MSE, r2m —— #
        train_ci, train_mse, train_r2m, _ = evaluate_toxcast(
            model, train_loader, DEVICE, criterion
        )
        val_ci,   val_mse,   val_r2m,   val_loss = evaluate_toxcast(
            model, val_loader,   DEVICE, criterion
        )
        test_ci,  test_mse,  test_r2m            = evaluate_toxcast(
            model, test_loader,  DEVICE, criterion=None
        )

        writer.add_scalar("Loss/val", val_loss, epoch)            # 行：写入验证集 loss

        # 行：各类指标写入 TensorBoard
        for tag, val in [
            ("train_CI",   train_ci),
            ("train_MSE",  train_mse),
            ("train_r2m",  train_r2m),
            ("val_CI",     val_ci),
            ("val_MSE",    val_mse),
            ("val_r2m",    val_r2m),
            ("test_CI",    test_ci),
            ("test_MSE",   test_mse),
            ("test_r2m",   test_r2m),
        ]:
            writer.add_scalar(f"Metrics/{tag}", val, epoch)

        # —— 学习率调度器步进 —— #
        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]              # 行：取当前学习率
        writer.add_scalar("LR", current_lr, epoch)

        # —— 写入 CSV —— #
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, train_loss, val_loss,
                train_ci, train_mse, train_r2m,
                val_ci,   val_mse,   val_r2m,
                test_ci,  test_mse,  test_r2m,
                current_lr
            ])

        # —— 以 val_MSE 选 best model & 早停 —— #
        score_now = val_mse if VAL_AS_BEST else test_mse         # 行：越小越好
        if score_now + EARLY_STOP_MIN_DELTA < best_score and torch.isfinite(torch.tensor(score_now)):
            best_score = score_now
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_best)            # 行：保存最优模型
            logging.info(
                f"[BEST|ToxCast] epoch={epoch+1}, "
                f"{'val' if VAL_AS_BEST else 'test'}_MSE={best_score:.4f} -> saved: {model_best}"
            )
        else:
            epochs_no_improve += 1
            logging.info(
                f"[EARLY-STOP|ToxCast] epoch={epoch+1}: no improvement on "
                f"{'val' if VAL_AS_BEST else 'test'}_MSE for {epochs_no_improve} epoch(s) "
                f"(best={best_score:.4f}, now={score_now:.4f})"
            )
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                logging.info(
                    f"[EARLY-STOP|ToxCast] Stop training at epoch {epoch+1}, "
                    f"no improvement for {EARLY_STOP_PATIENCE} epochs."
                )
                break

        logging.info(
            f"[ToxCast] Epoch {epoch+1:04d} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"train(CI={train_ci:.4f}, MSE={train_mse:.4f}, r2m={train_r2m:.4f}) | "
            f"val(CI={val_ci:.4f}, MSE={val_mse:.4f}, r2m={val_r2m:.4f}) | "
            f"test(CI={test_ci:.4f}, MSE={test_mse:.4f}, r2m={test_r2m:.4f}) | "
            f"lr={current_lr:.3e} | time={(time.time()-t0)/60:.2f} min"
        )

        util.save_checkpoint(epoch + 1, model, optimizer, filename=str(ckpt_path))  # 行：保存最新 checkpoint

    # ========== 最优模型做最终测试 ==========
    best_model = MultiModalDTA_LM(                           # 行：重新构建同结构模型
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
        final_ci, final_mse, final_r2m = evaluate_toxcast(   # 行：只计算三项指标
            best_model, test_loader, DEVICE, criterion=None
        )

    logging.info(
        f"[FINAL|ToxCast] test_CI={final_ci:.4f}, "
        f"test_MSE={final_mse:.4f}, test_r2m={final_r2m:.4f}"
    )
    with open(final_txt, "w", encoding="utf-8") as f:
        f.write(f"test_CI: {final_ci:.4f}\n")
        f.write(f"test_MSE: {final_mse:.4f}\n")
        f.write(f"test_r2m: {final_r2m:.4f}\n")

    writer.close()  # 行：关闭 TensorBoard 写入器
