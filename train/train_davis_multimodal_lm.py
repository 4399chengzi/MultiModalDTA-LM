# ======================================
# Davis 仅 1D（LM 版）训练脚本（带早停）
# 指标：CI, MSE, rm2, AUPR
# 模态：1D = ESM-2/ChemBERTa-2；不使用 2D/3D
# ======================================

from pathlib import Path                          # 行：处理路径（创建目录/拼路径）
import logging                                   # 行：日志打印（info/warning）
import time                                      # 行：计时
import csv                                       # 行：保存训练过程指标到 CSV
from typing import Dict, Any                     # 行：类型注解（更清晰）

import torch                                     # 行：PyTorch 主库
import torch.nn as nn                            # 行：神经网络模块（loss 等）
import torch.optim as optim                      # 行：优化器
import torch.utils.data as Data                  # 行：Dataset/DataLoader
from torch.utils.tensorboard import SummaryWriter # 行：TensorBoard 记录

import util                                      # 行：你项目里的工具函数（seed/metrics/checkpoint）
from util import LoadData_davis_lm_1d            # 行：✅ 改成仅 1D 的 Davis LM 数据加载函数


# ========== 设备/日志/随机种子 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 行：有 GPU 用 GPU，否则 CPU
torch.backends.cudnn.benchmark = True                                  # 行：固定输入形状时可加速卷积（一般没坏处）
LOG_LEVEL = logging.INFO                                               # 行：日志等级
SANITY_CHECK = True                                                    # 行：是否打印一次 batch 形状做检查


# ========== 可配置区域 ==========
DATA_DAVIS    = r"../dataset/davis"              # 行：Davis 原始数据目录（含 ligands_can.txt / proteins.txt / Y）
DATASET_NAME  = "davis_lm_1d"                    # 行：任务名（用于结果目录命名）
OUT_DIR       = f"../result/{DATASET_NAME}"      # 行：输出根目录

BATCH_SIZE   = 64                                # 行：批量大小（Davis 小，可以大一点）
NUM_WORKERS  = 4                                 # 行：DataLoader 进程数（Windows 可改 0/2）
PIN_MEMORY   = bool(torch.cuda.is_available())   # 行：GPU 时 pin_memory 会更快搬数据

EPOCHS         = 600                             # 行：最大 epoch
LR             = 1e-4                            # 行：学习率
WEIGHT_DECAY   = 1e-4                            # 行：AdamW 的 L2 正则
USE_AMP        = False                           # 行：是否混合精度（先关保证稳定）
ACCUM_STEPS    = 2                               # 行：梯度累积步数（等效更大 batch）
GRAD_CLIP_NORM = 0.5                             # 行：梯度裁剪阈值
USE_SCHEDULER  = True                            # 行：是否使用学习率调度器
VAL_AS_BEST    = True                            # 行：用 val_MSE 选 best（否则可用 test_MSE）

# ---- 早停相关参数 ----
EARLY_STOP_PATIENCE  = 40                        # 行：连续多少个 epoch 无提升就停
EARLY_STOP_MIN_DELTA = 0.0                       # 行：认为“有提升”的最小幅度


# ========== Davis Dataset（仅 1D-LM）==========
class DavisLMDataset(Data.Dataset):
    """封装 Davis 1D-LM 特征（drug_lm + prot_lm），不包含 2D/3D。"""
    def __init__(self, pkg: Dict[str, Any]):
        self.ids     = pkg["ids"]                 # 行：样本 ID（Lx_Py）
        self.y       = pkg["y"]                   # 行：标签（pKd）
        self.smiles  = pkg["smiles"]              # 行：SMILES（仅保留，方便 debug）
        self.seq     = pkg["seq"]                 # 行：蛋白序列（仅保留，方便 debug）
        self.drug_lm = pkg["drug_lm"]             # 行：药物 LM 向量 [N, D_d]
        self.prot_lm = pkg["prot_lm"]             # 行：蛋白 LM 向量 [N, D_p]

    def __len__(self):
        return len(self.y)                        # 行：返回数据集大小

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "id"     : self.ids[idx],             # 行：样本 ID
            "y"      : float(self.y[idx]),        # 行：标签（转 python float，collate 再转 tensor）
            "smiles" : self.smiles[idx],          # 行：SMILES
            "seq"    : self.seq[idx],             # 行：序列
            "drug_lm": self.drug_lm[idx],         # 行：药物向量 [D_d]
            "prot_lm": self.prot_lm[idx],         # 行：蛋白向量 [D_p]
        }


# ========== collate：合并 batch（仅 1D-LM）==========
def davis_lm_collate(batch: list) -> Dict[str, Any]:
    """把 list[dict] 合并为一个 batch dict：LM/标签堆叠成 tensor，文本保留 list。"""
    ids    = [b["id"]     for b in batch]         # 行：ID 列表
    smiles = [b["smiles"] for b in batch]         # 行：SMILES 列表
    seqs   = [b["seq"]    for b in batch]         # 行：序列列表

    import numpy as np                            # 行：局部导入，避免全局污染
    drug_lm = torch.as_tensor(                    # 行：堆叠成 [B, D_d]
        np.stack([b["drug_lm"] for b in batch], axis=0),
        dtype=torch.float32
    )
    prot_lm = torch.as_tensor(                    # 行：堆叠成 [B, D_p]
        np.stack([b["prot_lm"] for b in batch], axis=0),
        dtype=torch.float32
    )
    y = torch.as_tensor([b["y"] for b in batch], dtype=torch.float32)  # 行：[B]

    return {
        "ids"    : ids,                           # 行：list[str]
        "smiles" : smiles,                        # 行：list[str]
        "seqs"   : seqs,                          # 行：list[str]
        "drug_lm": drug_lm,                       # 行：tensor[B, D_d]
        "prot_lm": prot_lm,                       # 行：tensor[B, D_p]
        "y"      : y,                             # 行：tensor[B]
    }


# ========== Davis 评估函数：CI / MSE / rm2 / AUPR ==========
@torch.no_grad()
def evaluate_davis(model: torch.nn.Module,
                   loader: Data.DataLoader,
                   device: torch.device,
                   criterion: nn.Module = None):
    """
    若给定 criterion：返回 (ci, mse, rm2, aupr, avg_loss)
    否则         ：返回 (ci, mse, rm2, aupr)
    """
    model.eval()                                  # 行：评估模式（关 dropout）
    obs, pred = [], []                            # 行：收集真实值/预测值用于算指标
    loss_sum, n_batches = 0.0, 0                  # 行：累计 loss 与 batch 数

    for b in loader:                              # 行：遍历 DataLoader 的 batch
        drug_lm = b["drug_lm"].to(device, non_blocking=True)  # 行：药物向量搬到 GPU/CPU
        prot_lm = b["prot_lm"].to(device, non_blocking=True)  # 行：蛋白向量搬到 GPU/CPU
        y_true  = b["y"].to(device, non_blocking=True)        # 行：标签搬到 GPU/CPU

        with torch.cuda.amp.autocast(enabled=(USE_AMP and device.type == "cuda")):  # 行：可选 AMP
            # ✅ 只传 1D-LM；不传 g_lig/g_prot/3D，让模型走“仅 LM”分支
            y_hat = model(drug_lm=drug_lm,
                          prot_lm=prot_lm)

            if criterion is not None:             # 行：如果给了 loss 函数就计算 loss
                loss_val = criterion(y_hat, y_true)
                if torch.isfinite(loss_val):      # 行：防 NaN/Inf
                    loss_sum += float(loss_val)
                    n_batches += 1

        obs.extend(y_true.view(-1).tolist())      # 行：把当前 batch 的真实值加入列表
        pred.extend(y_hat.view(-1).tolist())      # 行：把当前 batch 的预测值加入列表

    # 行：用 util 里已有的函数计算指标
    ci   = util.get_cindex(obs, pred)
    mse  = util.get_MSE(obs, pred)
    rm2  = util.get_rm2(obs, pred)
    aupr = util.get_aupr(obs, pred)

    if criterion is None:
        return ci, mse, rm2, aupr
    return ci, mse, rm2, aupr, (loss_sum / max(1, n_batches))


# ========== 主程序 ==========
if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL)          # 行：设置日志系统
    util.seed_torch()                             # 行：固定随机种子（可复现）

    result_root = Path(OUT_DIR)                   # 行：结果目录 Path
    result_root.mkdir(parents=True, exist_ok=True)  # 行：没有则创建

    # —— 加载 Davis 仅 1D-LM 数据 —— #
    kw_common = dict(
        out_1d="../dataset/davis/processed_lm_1d", # 行：✅ 改成 1D 缓存目录
        logspace_trans=True,                      # 行：Davis 常用 pKd
        esm2_model_name="facebook/esm2_t33_650M_UR50D",
        chemberta_model_name="DeepChem/ChemBERTa-77M-MTR",
        lm_batch_size=32,
        use_safetensors=True,
    )

    # 行：分别加载 train/val/test（第一次会编码并缓存，之后直接读 npz）
    mm_train = LoadData_davis_lm_1d(data_dir=DATA_DAVIS, split="train", **kw_common)["train"]
    mm_val   = LoadData_davis_lm_1d(data_dir=DATA_DAVIS, split="val",   **kw_common)["val"]
    mm_test  = LoadData_davis_lm_1d(data_dir=DATA_DAVIS, split="test",  **kw_common)["test"]

    # 行：LM 维度（后面初始化模型要用）
    d_drug_lm = mm_train["drug_lm"].shape[1]
    d_prot_lm = mm_train["prot_lm"].shape[1]
    logging.info(f"[Davis|LM-1D] drug_lm_dim={d_drug_lm}, prot_lm_dim={d_prot_lm}")

    # —— Dataset / DataLoader —— #
    train_ds = DavisLMDataset(mm_train)           # 行：训练集 Dataset
    val_ds   = DavisLMDataset(mm_val)             # 行：验证集 Dataset
    test_ds  = DavisLMDataset(mm_test)            # 行：测试集 Dataset

    train_loader = Data.DataLoader(              # 行：训练 DataLoader
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=davis_lm_collate
    )
    val_loader = Data.DataLoader(                # 行：验证 DataLoader
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=davis_lm_collate
    )
    test_loader = Data.DataLoader(               # 行：测试 DataLoader
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=davis_lm_collate
    )

    logging.info(f"[Davis] sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # —— Sanity Check —— #
    if SANITY_CHECK:
        b = next(iter(train_loader))             # 行：取一个 batch 看形状
        print(">>> Batch keys:", list(b.keys()))
        print("drug_lm:", tuple(b["drug_lm"].shape),
              "| prot_lm:", tuple(b["prot_lm"].shape),
              "| y:", tuple(b["y"].shape))
        print("[OK] Davis LM-1D loaders ready.")

    # —— 导入模型 —— #
    from model.model_multimodal_lm import MultiModalDTA_LM  # 行：你的总模型

    # 行：仍然用同一个模型类；只要 forward 支持 g_lig/g_prot 可选，就能只喂 LM
    model = MultiModalDTA_LM(
        d_drug_lm=d_drug_lm, d_prot_lm=d_prot_lm,
        drug_node_dim=70, drug_edge_dim=6,       # 行：2D 参数即使不用也可保留（不走该分支就不会用到）
        prot_node_dim=33, prot_edge_dim=3,
        gcn_hidden=128, gcn_out=128,
        d3_lig=512, d3_poc=512,                  # 行：3D 参数同理（不走 3D 分支就不会用到）
        d_model=256, d_attn=256, n_heads=4,
        add_interactions_3d=True
    ).to(DEVICE)

    # —— 优化器 / 损失 / 调度 / AMP —— #
    criterion = nn.MSELoss(reduction="mean")     # 行：回归常用 MSE
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)  # 行：AdamW
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-5
    ) if USE_SCHEDULER else None                 # 行：可选余弦重启
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE.type == "cuda"))  # 行：AMP 缩放器

    # —— 日志与权重路径 —— #
    result_dir  = result_root / "train_logs"     # 行：训练日志目录
    run_dir     = result_root / "runs"           # 行：TensorBoard 目录
    model_best  = result_dir / "best_model.pth"  # 行：最优模型权重
    ckpt_path   = result_dir / "checkpoint.pth.tar"  # 行：断点续训文件
    csv_file    = result_dir / "metrics.csv"     # 行：训练过程指标 CSV
    final_txt   = result_dir / "final_test_metrics.txt"  # 行：最终测试指标

    result_dir.mkdir(parents=True, exist_ok=True)  # 行：创建目录
    run_dir.mkdir(parents=True, exist_ok=True)     # 行：创建目录

    writer = SummaryWriter(log_dir=str(run_dir))   # 行：TensorBoard writer

    # —— 断点恢复（可选）—— #
    start_epoch = 0                                # 行：默认从 0 开始
    if ckpt_path.exists():                         # 行：如果存在 checkpoint
        start_epoch = util.load_checkpoint(str(ckpt_path), model, optimizer)  # 行：恢复模型和优化器
        logging.info(f"[Resume] from epoch: {start_epoch}")

    # —— CSV 表头 —— #
    if not csv_file.exists():                      # 行：首次运行写表头
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "val_loss",
                "train_CI", "train_MSE", "train_rm2", "train_AUPR",
                "val_CI",   "val_MSE",   "val_rm2",   "val_AUPR",
                "test_CI",  "test_MSE",  "test_rm2",  "test_AUPR",
                "lr"
            ])

    # ========== 训练主循环 ==========
    best_score = float("inf")                      # 行：记录最优 val_MSE（越小越好）
    epochs_no_improve = 0                          # 行：连续无提升计数
    t0 = time.time()                               # 行：总计时起点

    for epoch in range(start_epoch, EPOCHS):       # 行：epoch 循环
        model.train()                              # 行：训练模式
        optimizer.zero_grad(set_to_none=True)      # 行：清梯度（set_to_none 更省显存）

        running_loss, n_batches, last_bidx = 0.0, 0, -1  # 行：累计 loss、batch 数、最后 batch idx

        for last_bidx, b in enumerate(train_loader):     # 行：遍历训练集 batch
            drug_lm = b["drug_lm"].to(DEVICE, non_blocking=True)  # 行：药物向量搬设备
            prot_lm = b["prot_lm"].to(DEVICE, non_blocking=True)  # 行：蛋白向量搬设备
            y_true  = b["y"].to(DEVICE, non_blocking=True)        # 行：标签搬设备

            with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE.type == "cuda")):
                y_hat = model(drug_lm=drug_lm,                     # ✅ 只喂 1D
                              prot_lm=prot_lm)
                loss = criterion(y_hat, y_true)                    # 行：MSE loss

            if not torch.isfinite(loss):                           # 行：防 NaN/Inf
                logging.warning(f"[NaN] loss at epoch={epoch}, batch={last_bidx}: {float(loss)}")
                optimizer.zero_grad(set_to_none=True)
                continue

            # 行：梯度累积（把 loss/ACCUM_STEPS 再 backward）
            if scaler.is_enabled():
                scaler.scale(loss / ACCUM_STEPS).backward()
            else:
                (loss / ACCUM_STEPS).backward()

            # 行：每 ACCUM_STEPS 个 batch 更新一次参数
            if (last_bidx + 1) % ACCUM_STEPS == 0:
                if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:          # 行：可选梯度裁剪
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

                if scaler.is_enabled():                            # 行：AMP 情况下 step/update
                    scaler.step(optimizer); scaler.update()
                else:                                              # 行：普通 step
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)              # 行：清梯度

            running_loss += float(loss)                            # 行：累计 loss（未除以 accum）
            n_batches    += 1                                      # 行：batch 计数

        # 行：处理尾批（如果最后没整除 ACCUM_STEPS）
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
        writer.add_scalar("Loss/train", train_loss, epoch)         # 行：写 TensorBoard

        # —— 评估：train/val/test（CI/MSE/rm2/AUPR）—— #
        train_ci, train_mse, train_rm2, train_aupr, _ = evaluate_davis(
            model, train_loader, DEVICE, criterion
        )
        val_ci,   val_mse,   val_rm2,   val_aupr,   val_loss = evaluate_davis(
            model, val_loader, DEVICE, criterion
        )
        test_ci,  test_mse,  test_rm2,  test_aupr = evaluate_davis(
            model, test_loader, DEVICE, criterion=None
        )

        writer.add_scalar("Loss/val", val_loss, epoch)             # 行：记录 val loss

        # 行：写所有指标到 TensorBoard
        for tag, val in [
            ("train_CI",   train_ci),
            ("train_MSE",  train_mse),
            ("train_rm2",  train_rm2),
            ("train_AUPR", train_aupr),
            ("val_CI",     val_ci),
            ("val_MSE",    val_mse),
            ("val_rm2",    val_rm2),
            ("val_AUPR",   val_aupr),
            ("test_CI",    test_ci),
            ("test_MSE",   test_mse),
            ("test_rm2",   test_rm2),
            ("test_AUPR",  test_aupr),
        ]:
            writer.add_scalar(f"Metrics/{tag}", val, epoch)

        # —— 学习率调度 —— #
        if scheduler is not None:
            scheduler.step()                                       # 行：更新 lr（按 epoch）
        current_lr = optimizer.param_groups[0]["lr"]               # 行：取当前 lr
        writer.add_scalar("LR", current_lr, epoch)                 # 行：写 TensorBoard

        # —— 写 CSV —— #
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, train_loss, val_loss,
                train_ci, train_mse, train_rm2, train_aupr,
                val_ci,   val_mse,   val_rm2,   val_aupr,
                test_ci,  test_mse,  test_rm2,  test_aupr,
                current_lr
            ])

        # —— 以 val_MSE 选 best model & 早停 —— #
        score_now = val_mse if VAL_AS_BEST else test_mse           # 行：决定 early-stop 的指标

        # 行：有提升则保存 best
        if score_now + EARLY_STOP_MIN_DELTA < best_score and torch.isfinite(torch.tensor(score_now)):
            best_score = score_now
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_best)             # 行：保存最优权重
            logging.info(
                f"[BEST] epoch={epoch+1}, "
                f"{'val' if VAL_AS_BEST else 'test'}_MSE={best_score:.4f} "
                f"-> saved: {model_best}"
            )
        else:
            epochs_no_improve += 1
            logging.info(
                f"[EARLY-STOP] epoch={epoch+1}: no improvement on "
                f"{'val' if VAL_AS_BEST else 'test'}_MSE for {epochs_no_improve} epoch(s) "
                f"(best={best_score:.4f}, now={score_now:.4f})"
            )
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                logging.info(
                    f"[EARLY-STOP] Stop training at epoch {epoch+1}, "
                    f"no improvement for {EARLY_STOP_PATIENCE} epochs."
                )
                break

        # —— 打印每个 epoch 的总结 —— #
        logging.info(
            f"Epoch {epoch+1:04d} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"train(CI={train_ci:.4f}, MSE={train_mse:.4f}, rm2={train_rm2:.4f}, AUPR={train_aupr:.4f}) | "
            f"val(CI={val_ci:.4f}, MSE={val_mse:.4f}, rm2={val_rm2:.4f}, AUPR={val_aupr:.4f}) | "
            f"test(CI={test_ci:.4f}, MSE={test_mse:.4f}, rm2={test_rm2:.4f}, AUPR={test_aupr:.4f}) | "
            f"lr={current_lr:.3e} | time={(time.time()-t0)/60:.2f} min"
        )

        util.save_checkpoint(epoch + 1, model, optimizer, filename=str(ckpt_path))  # 行：保存断点

    # ========== 最优模型做最终测试 ==========
    best_model = MultiModalDTA_LM(                                 # 行：重新建同结构模型
        d_drug_lm=d_drug_lm, d_prot_lm=d_prot_lm,
        drug_node_dim=70, drug_edge_dim=6,
        prot_node_dim=33, prot_edge_dim=3,
        gcn_hidden=128, gcn_out=128,
        d3_lig=512, d3_poc=512,
        d_model=256, d_attn=256, n_heads=4,
        add_interactions_3d=True
    ).to(DEVICE)

    best_model.load_state_dict(torch.load(model_best, map_location=DEVICE))  # 行：加载 best 权重
    best_model.eval()                                                       # 行：评估模式

    with torch.no_grad():                                                   # 行：不记录梯度
        final_ci, final_mse, final_rm2, final_aupr = evaluate_davis(
            best_model, test_loader, DEVICE, criterion=None
        )

    logging.info(
        f"[FINAL] test_CI={final_ci:.4f}, test_MSE={final_mse:.4f}, "
        f"test_rm2={final_rm2:.4f}, test_AUPR={final_aupr:.4f}"
    )

    with open(final_txt, "w", encoding="utf-8") as f:                       # 行：写最终指标文件
        f.write(f"test_CI: {final_ci:.4f}\n")
        f.write(f"test_MSE: {final_mse:.4f}\n")
        f.write(f"test_rm2: {final_rm2:.4f}\n")
        f.write(f"test_AUPR: {final_aupr:.4f}\n")

    writer.close()                                                          # 行：关闭 TensorBoard writer
