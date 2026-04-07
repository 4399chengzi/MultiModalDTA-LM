# ======================================
# 多模态 ATOM3D(SI-30) 训练脚本（LM 版：1D 用 ESM-2 + ChemBERTa-2 向量）
# 模态：1D(LM) + 3D(Uni-Mol2)，2D 图模态关闭（g_lig/g_prot = None）
# 带：数值稳定设置 + 早停机制
# revise版：统一把模型、结果路径、日志路径都切到 revise 命名
# ======================================

from pathlib import Path                      # 行：路径处理
import logging                                # 行：日志输出
import time                                   # 行：计时
import csv                                    # 行：写入 CSV
from typing import Dict, Any                  # 行：类型注解

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter

import util
from util import LoadData_atom3d_si30_multimodal_lm  # 行：新的 1D+3D 数据加载函数

# —— 关闭 Uni-Mol 冗余日志 —— #
lg = logging.getLogger("unimol_tools")        # 行：获取 unimol_tools 日志器
lg.setLevel(logging.WARNING)                  # 行：只保留 warning 以上日志
lg.propagate = False                          # 行：不向上层传播
for h in list(lg.handlers):                   # 行：清空已有 handler
    lg.removeHandler(h)

# ========== 设备/日志/随机种子 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 行：优先用 GPU
torch.backends.cudnn.benchmark = True                                   # 行：固定尺寸卷积略微加速
LOG_LEVEL = logging.INFO                                                # 行：日志级别
SANITY_CHECK = True                                                     # 行：是否做一次 batch 检查

# ========== revise 版本标记 ==========
REV_TAG = "revise"                                                     # 行：当前修订版本统一后缀
MODEL_MODULE_NAME = "model_multimodal_lm_revise"                       # 行：模型文件名（不带 .py）
RESULT_ROOT_NAME = "result_revise"                                     # 行：新的结果根目录
DATASET_NAME = f"atom3d_si30_mm_lm_{REV_TAG}"                          # 行：结果目录名（带 revise）

# ========== 可配置区域 ==========
DATA_DB_SI30 = r"../dataset/ATOM3D/split-by-sequence-identity-30/data"  # 行：si-30 LMDB 根目录
OUT_DIR = f"../{RESULT_ROOT_NAME}/{DATASET_NAME}"                       # 行：结果根目录，改到 revise 目录

BATCH_SIZE   = 16                    # 行：batch 大小
NUM_WORKERS  = 4                     # 行：DataLoader 线程数
PIN_MEMORY   = bool(torch.cuda.is_available())  # 行：若有 GPU 则开启 pinned memory

EPOCHS         = 600                 # 行：最大训练 epoch 数
LR             = 1e-4                # 行：初始学习率
WEIGHT_DECAY   = 1e-4                # 行：L2 权重衰减
USE_AMP        = False               # 行：是否启用 AMP
ACCUM_STEPS    = 4                   # 行：梯度累积步数
GRAD_CLIP_NORM = 0.5                 # 行：梯度裁剪阈值
USE_SCHEDULER  = True                # 行：是否使用学习率调度器
VAL_AS_BEST    = True                # 行：选择 best_score 的指标来源（True=val，False=test）

# —— 早停相关配置 —— #
EARLY_STOP_PATIENCE = 60             # 行：早停耐心 epoch 数
EARLY_STOP_MIN_DELTA = 1e-4          # 行：认为有“明显提升”的最小 RMSE 改善幅度


# ========== 统计 si-30 拆分 ==========
def count_si30_splits(root_base: str):
    """统计原始 LMDB 中 train/val/test 数量和比例"""
    import atom3d.datasets as da                 # 行：本地导入，避免全局无用依赖
    root = Path(root_base)                       # 行：转成 Path
    counts = {}
    for sp in ("train", "val", "test"):          # 行：遍历三个拆分
        ds = da.LMDBDataset(str(root / sp))      # 行：加载对应 LMDB
        counts[sp] = len(ds)                     # 行：记录样本数
    total = sum(counts.values())                 # 行：总样本数
    ratios = {sp: (counts[sp] / total if total > 0 else 0.0) for sp in counts}  # 行：各拆分占比
    return counts, ratios, total


# ========== 多模态 Dataset（带 LM 特征） ==========
class MultiModalDatasetLM(Data.Dataset):
    """封装 1D(LM) + 3D 特征；2D 图模态已关闭"""
    def __init__(self, pkg: Dict[str, Any]):
        self.ids     = pkg['ids']           # 行：样本 ID 数组
        self.y       = pkg['y']             # 行：标签 pKd
        self.smiles  = pkg['smiles']        # 行：SMILES 列表
        self.seq     = pkg['seq']           # 行：蛋白序列列表
        self.drug_lm = pkg['drug_lm']       # 行：药物 LM 向量 [N, D_drug_lm]
        self.prot_lm = pkg['prot_lm']       # 行：蛋白 LM 向量 [N, D_prot_lm]
        self.g_lig   = pkg.get('g_lig', None)   # 行：药物 2D 图（此处为 None）
        self.g_prot  = pkg.get('g_prot', None)  # 行：蛋白 2D 图（此处为 None）
        self.lig3d   = pkg['lig_3d']        # 行：配体 3D 向量 [N, D3_lig]
        self.poc3d   = pkg['poc_3d']        # 行：口袋 3D 向量 [N, D3_poc]

    def __len__(self):
        return len(self.y)                  # 行：样本数

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """返回单条样本；g_lig/g_prot 直接返回 None"""
        return {
            'id'     : self.ids[idx],
            'y'      : float(self.y[idx]),
            'smiles' : self.smiles[idx],
            'seq'    : self.seq[idx],
            'drug_lm': self.drug_lm[idx],
            'prot_lm': self.prot_lm[idx],
            'g_lig'  : None,               # 行：2D 模态关闭
            'g_prot' : None,               # 行：2D 模态关闭
            'lig3d'  : self.lig3d[idx],
            'poc3d'  : self.poc3d[idx],
        }


# ========== collate：合并 batch ==========
def multimodal_collate_lm(batch: list) -> Dict[str, Any]:
    """
    合并一个 batch：
      - 1D/3D/标签：堆叠成 tensor
      - 2D：已关闭，返回 None
    """
    ids    = [b['id']     for b in batch]   # 行：收集 ID
    smiles = [b['smiles'] for b in batch]   # 行：收集 SMILES
    seqs   = [b['seq']    for b in batch]   # 行：收集序列

    g_lig_batch  = None                     # 行：药物 2D 图 batch
    g_prot_batch = None                     # 行：蛋白 2D 图 batch

    import numpy as np
    drug_lm = torch.as_tensor(
        np.stack([b['drug_lm'] for b in batch], axis=0),
        dtype=torch.float32
    )                                       # 行：药物 LM [B, D_drug_lm]
    prot_lm = torch.as_tensor(
        np.stack([b['prot_lm'] for b in batch], axis=0),
        dtype=torch.float32
    )                                       # 行：蛋白 LM [B, D_prot_lm]
    lig3d = torch.as_tensor(
        np.stack([b['lig3d'] for b in batch], axis=0),
        dtype=torch.float32
    )                                       # 行：配体 3D 向量 [B, Dl]
    poc3d = torch.as_tensor(
        np.stack([b['poc3d'] for b in batch], axis=0),
        dtype=torch.float32
    )                                       # 行：口袋 3D 向量 [B, Dp]
    y = torch.as_tensor(
        [b['y'] for b in batch],
        dtype=torch.float32
    )                                       # 行：真实标签 [B]

    return {
        'ids'    : ids,
        'smiles' : smiles,
        'seqs'   : seqs,
        'drug_lm': drug_lm,
        'prot_lm': prot_lm,
        'g_lig'  : g_lig_batch,
        'g_prot' : g_prot_batch,
        'lig3d'  : lig3d,
        'poc3d'  : poc3d,
        'y'      : y,
    }


# ========== 简单的 NaN 参数检测 ==========
def has_nan_params(model: nn.Module) -> bool:
    """检查模型参数中是否出现 NaN / Inf，用于 debug 数值稳定性"""
    for p in model.parameters():
        if p is not None and torch.is_tensor(p):
            if torch.isnan(p).any() or torch.isinf(p).any():
                return True
    return False


# ========== 统一评估函数（LM 版） ==========
@torch.no_grad()
def evaluate(model: torch.nn.Module,
             loader: Data.DataLoader,
             device: torch.device,
             criterion: nn.Module = None):
    """
    若给定 criterion：返回 (rmse, pearson, spearman, avg_loss)
    否则         ：返回 (rmse, pearson, spearman)
    """
    model.eval()                            # 行：评估模式
    obs, pred, loss_sum, n_batches = [], [], 0.0, 0
    for b in loader:
        drug_lm = b['drug_lm'].to(device, non_blocking=True)
        prot_lm = b['prot_lm'].to(device, non_blocking=True)
        g_lig, g_prot = b['g_lig'], b['g_prot']       # 行：2D 图，此时为 None
        lig3d = b['lig3d'].to(device, non_blocking=True)
        poc3d = b['poc3d'].to(device, non_blocking=True)
        y_true = b['y'].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(USE_AMP and device.type == "cuda")):
            y_hat = model(
                drug_lm=drug_lm, prot_lm=prot_lm,
                g_lig=g_lig, g_prot=g_prot,
                lig3d=lig3d, poc3d=poc3d
            )
            if criterion is not None:
                loss_val = criterion(y_hat, y_true)
                if torch.isfinite(loss_val):
                    loss_sum += float(loss_val)
                    n_batches += 1

        obs.extend(y_true.view(-1).tolist())    # 行：累计真实值
        pred.extend(y_hat.view(-1).tolist())    # 行：累计预测值

    rmse    = util.get_RMSE(obs, pred)          # 行：RMSE
    pearson = util.get_pearsonr(obs, pred)      # 行：皮尔逊
    spear   = util.get_spearmanr(obs, pred)     # 行：斯皮尔曼
    if criterion is None:
        return rmse, pearson, spear
    return rmse, pearson, spear, (loss_sum / max(1, n_batches))


# ========== 主程序 ==========
if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL)        # 行：初始化日志
    util.seed_torch()                           # 行：固定随机种子
    result_root = Path(OUT_DIR)                 # 行：结果根目录 Path
    result_root.mkdir(parents=True, exist_ok=True)

    logging.info(f"[Revision Tag] {REV_TAG}")   # 行：打印当前修订标记
    logging.info(f"[Model Module] {MODEL_MODULE_NAME}")  # 行：打印当前模型文件名
    logging.info(f"[Output Dir] {result_root}")          # 行：打印当前结果目录

    # 原始 LMDB 统计
    counts, ratios, total = count_si30_splits(DATA_DB_SI30)
    logging.info(f"[si-30] counts(raw): {counts} | total={total}")
    logging.info(
        f"[si-30] ratios(raw): "
        f"train={ratios['train']:.4f}, val={ratios['val']:.4f}, test={ratios['test']:.4f}"
    )

    # —— 加载对齐后的 1D+3D+LM 数据 —— #
    kw_common = dict(
        out_mm="../dataset/ATOM3D/processed_mm_si30_lm",  # 行：缓存目录
        unimol2_size="unimol2_small",                     # 行：Uni-Mol2 规格
        contact_threshold=8.0, dis_min=1.0,
        prot_self_loop=False, bond_bidirectional=True,
        prefer_model=None, force_refresh=False,
        use_cuda_for_unimol=True,
    )
    mm_train = LoadData_atom3d_si30_multimodal_lm(
        root_base=DATA_DB_SI30, split="train", **kw_common
    )["train"]
    mm_val = LoadData_atom3d_si30_multimodal_lm(
        root_base=DATA_DB_SI30, split="val", **kw_common
    )["val"]
    mm_test = LoadData_atom3d_si30_multimodal_lm(
        root_base=DATA_DB_SI30, split="test", **kw_common
    )["test"]

    # LM 特征维度
    d_drug_lm = mm_train['drug_lm'].shape[1]     # 行：药物 LM 维度
    d_prot_lm = mm_train['prot_lm'].shape[1]     # 行：蛋白 LM 维度
    logging.info(f"[LM] drug_lm_dim={d_drug_lm}, prot_lm_dim={d_prot_lm}")

    # —— Dataset / DataLoader —— #
    train_ds = MultiModalDatasetLM(mm_train)
    val_ds   = MultiModalDatasetLM(mm_val)
    test_ds  = MultiModalDatasetLM(mm_test)

    train_loader = Data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=multimodal_collate_lm
    )
    val_loader = Data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=multimodal_collate_lm
    )
    test_loader = Data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=multimodal_collate_lm
    )

    logging.info(
        f"[si-30|LM] final sizes: "
        f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    # —— Sanity Check —— #
    if SANITY_CHECK:
        b = next(iter(train_loader))
        print(">>> Batch keys:", list(b.keys()))
        print("drug_lm:", tuple(b['drug_lm'].shape),
              "| prot_lm:", tuple(b['prot_lm'].shape),
              "| lig3d:", tuple(b['lig3d'].shape),
              "| poc3d:", tuple(b['poc3d'].shape),
              "| y:", tuple(b['y'].shape))
        print("g_lig:", b['g_lig'])
        print("g_prot:", b['g_prot'])
        print("[OK] 1D+3D LM loaders ready.")

    # —— 导入 revise 版模型 —— #
    from model.model_multimodal_lm_revise import MultiModalDTA_LM  # 行：导入 revise 版模型

    model = MultiModalDTA_LM(
        d_drug_lm=d_drug_lm, d_prot_lm=d_prot_lm,
        drug_node_dim=70, drug_edge_dim=6,
        prot_node_dim=33, prot_edge_dim=3,
        gcn_hidden=128, gcn_out=128,
        d3_lig=512, d3_poc=512,
        d_model=256, d_attn=256, n_heads=4,
        add_interactions_3d=True
    ).to(DEVICE)

    # 优化器 / 损失 / 调度 / AMP
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = (
        optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-5
        )
        if USE_SCHEDULER else None
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=(USE_AMP and DEVICE.type == "cuda")
    )

    # 日志/权重/CSV 路径（全部切到 revise 目录与 revise 文件名）
    run_dir    = result_root / f"runs_{REV_TAG}"                         # 行：TensorBoard 日志目录
    result_dir = result_root / f"train_logs_{REV_TAG}"                  # 行：训练日志/权重目录
    model_best = result_dir / f"best_model_{REV_TAG}.pth"               # 行：最优模型权重
    ckpt_path  = result_dir / f"checkpoint_{REV_TAG}.pth.tar"           # 行：最新 checkpoint
    csv_file   = result_dir / f"metrics_{REV_TAG}.csv"                  # 行：记录指标的 CSV
    final_txt  = result_dir / f"final_test_metrics_{REV_TAG}.txt"       # 行：最终测试结果
    config_txt = result_dir / f"run_config_{REV_TAG}.txt"               # 行：运行配置记录

    run_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    # 把本次运行配置写到文本，方便后续 revision 对照
    with open(config_txt, "w", encoding="utf-8") as f:
        f.write(f"REV_TAG: {REV_TAG}\n")
        f.write(f"MODEL_MODULE_NAME: {MODEL_MODULE_NAME}\n")
        f.write(f"OUT_DIR: {OUT_DIR}\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"EPOCHS: {EPOCHS}\n")
        f.write(f"LR: {LR}\n")
        f.write(f"WEIGHT_DECAY: {WEIGHT_DECAY}\n")
        f.write(f"USE_AMP: {USE_AMP}\n")
        f.write(f"ACCUM_STEPS: {ACCUM_STEPS}\n")
        f.write(f"GRAD_CLIP_NORM: {GRAD_CLIP_NORM}\n")
        f.write(f"USE_SCHEDULER: {USE_SCHEDULER}\n")
        f.write(f"VAL_AS_BEST: {VAL_AS_BEST}\n")
        f.write(f"EARLY_STOP_PATIENCE: {EARLY_STOP_PATIENCE}\n")
        f.write(f"EARLY_STOP_MIN_DELTA: {EARLY_STOP_MIN_DELTA}\n")

    # 断点恢复
    start_epoch = 0
    if ckpt_path.exists():
        start_epoch = util.load_checkpoint(str(ckpt_path), model, optimizer)
        logging.info(f"[Resume] from epoch: {start_epoch}")

    # CSV 表头
    if not csv_file.exists():
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                'epoch', 'train_loss', 'val_loss',
                'train_RMSE','train_Pearson','train_Spearman',
                'val_RMSE','val_Pearson','val_Spearman',
                'test_RMSE','test_Pearson','test_Spearman',
                'lr'
            ])

    # ========== 训练主循环 ==========
    best_score = float('inf')
    epochs_no_improve = 0
    t0 = time.time()

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss, n_batches, last_bidx = 0.0, 0, -1

        for last_bidx, b in enumerate(train_loader):
            drug_lm = b['drug_lm'].to(DEVICE, non_blocking=True)
            prot_lm = b['prot_lm'].to(DEVICE, non_blocking=True)
            g_lig, g_prot = b['g_lig'], b['g_prot']
            lig3d = b['lig3d'].to(DEVICE, non_blocking=True)
            poc3d = b['poc3d'].to(DEVICE, non_blocking=True)
            y_true = b['y'].to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE.type == "cuda")):
                y_hat = model(
                    drug_lm=drug_lm, prot_lm=prot_lm,
                    g_lig=g_lig, g_prot=g_prot,
                    lig3d=lig3d, poc3d=poc3d
                )
                loss = criterion(y_hat, y_true)

            if not torch.isfinite(loss):
                logging.warning(
                    f"Non-finite loss at epoch={epoch}, batch={last_bidx}: {float(loss)}"
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            if scaler.is_enabled():
                scaler.scale(loss / ACCUM_STEPS).backward()
            else:
                (loss / ACCUM_STEPS).backward()

            if (last_bidx + 1) % ACCUM_STEPS == 0:
                if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

                if has_nan_params(model):
                    logging.error(
                        f"[NaN] detected in params at epoch={epoch}, "
                        f"batch={last_bidx} (before step)"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    break

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if has_nan_params(model):
                    logging.error(
                        f"[NaN] detected in params at epoch={epoch}, "
                        f"batch={last_bidx} (after step)"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    break

            running_loss += float(loss)
            n_batches    += 1

        if (last_bidx + 1) % ACCUM_STEPS != 0 and n_batches > 0 and not has_nan_params(model):
            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_loss = running_loss / max(1, n_batches)
        writer.add_scalar('Loss/train', train_loss, epoch)

        train_rmse, train_p, train_s, _ = evaluate(
            model, train_loader, DEVICE, criterion
        )
        val_rmse, val_p, val_s, val_loss = evaluate(
            model, val_loader, DEVICE, criterion
        )
        writer.add_scalar('Loss/val', val_loss, epoch)
        for tag, val in [
            ('train_RMSE',train_rmse), ('train_Pearson',train_p), ('train_Spearman',train_s),
            ('val_RMSE',val_rmse), ('val_Pearson',val_p), ('val_Spearman',val_s)
        ]:
            writer.add_scalar(f'Metrics/{tag}', val, epoch)

        test_rmse, test_p, test_s = evaluate(
            model, test_loader, DEVICE, criterion=None
        )
        writer.add_scalar('Metrics/test_RMSE',    test_rmse, epoch)
        writer.add_scalar('Metrics/test_Pearson', test_p,    epoch)
        writer.add_scalar('Metrics/test_Spearman',test_s,    epoch)

        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LR', current_lr, epoch)

        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                epoch, train_loss, val_loss,
                train_rmse, train_p, train_s,
                val_rmse,   val_p,   val_s,
                test_rmse,  test_p,  test_s,
                current_lr
            ])

        score_now = val_rmse if VAL_AS_BEST else test_rmse
        improved = (best_score - score_now) > EARLY_STOP_MIN_DELTA

        if improved and torch.isfinite(torch.tensor(score_now)):
            best_score = score_now
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_best)
            logging.info(
                f"[BEST] epoch={epoch+1}, "
                f"{'val' if VAL_AS_BEST else 'test'}_RMSE={best_score:.4f} "
                f"-> saved: {model_best}"
            )
        else:
            epochs_no_improve += 1
            logging.info(
                f"[EARLY-STOP] no improve for {epochs_no_improve} epoch(s); "
                f"best_score={best_score:.4f}"
            )
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                logging.info(
                    f"[EARLY-STOP] patience={EARLY_STOP_PATIENCE} reached at epoch {epoch+1}, stop training."
                )
                util.save_checkpoint(epoch + 1, model, optimizer, filename=str(ckpt_path))
                break

        logging.info(
            f"Epoch {epoch+1:04d} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"train(RMSE={train_rmse:.4f}, P={train_p:.4f}, S={train_s:.4f}) | "
            f"val(RMSE={val_rmse:.4f}, P={val_p:.4f}, S={val_s:.4f}) | "
            f"test(RMSE={test_rmse:.4f}, P={test_p:.4f}, S={test_s:.4f}) | "
            f"lr={current_lr:.3e} | time={(time.time()-t0)/60:.2f} min"
        )

        util.save_checkpoint(epoch + 1, model, optimizer, filename=str(ckpt_path))

    # ========== 用“最优模型”做最终测试 ==========
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
        final_rmse, final_p, final_s = evaluate(
            best_model, test_loader, DEVICE, criterion=None
        )

    logging.info(
        f"[FINAL] test_RMSE={final_rmse:.4f}, "
        f"test_Pearson={final_p:.4f}, test_Spearman={final_s:.4f}"
    )
    with open(final_txt, 'w', encoding='utf-8') as f:
        f.write(f"test_RMSE: {final_rmse:.4f}\n")
        f.write(f"test_Pearson: {final_p:.4f}\n")
        f.write(f"test_Spearman: {final_s:.4f}\n")

    writer.close()