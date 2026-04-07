# ======================================
# 多模态 ATOM3D(SI-60) 训练脚本（LM 版：1D 用 ESM-2 + ChemBERTa-2 向量）
# （数值稳定 revise 版：关闭 AMP、降低 LR、加 NaN 保护，1D+3D，无 2D 图）
# 说明：
#   1. 模型文件改为 model_multimodal_lm_revise.py
#   2. 结果目录改到 ../result_revise/
#   3. 日志、权重、CSV、最终结果统一带 revise 后缀
# ======================================

from pathlib import Path                 # 行：路径处理
import logging                           # 行：日志
import time                              # 行：计时
import csv                               # 行：写入 CSV
from typing import Dict, Any             # 行：类型注解

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter

import util
from util import LoadData_atom3d_si60_multimodal_lm  # 行：SI-60 的 LM 数据加载函数（已在 util 里实现）

# —— 关闭 Uni-Mol 冗余日志，避免控制台刷屏 —— #
lg = logging.getLogger("unimol_tools")   # 行：获取 unimol_tools 日志器
lg.setLevel(logging.WARNING)             # 行：只保留 warning 及以上级别
lg.propagate = False                     # 行：不向上层 logger 传播
for h in list(lg.handlers):              # 行：清空已有 handler，避免重复打印
    lg.removeHandler(h)

# ========== 设备/日志/随机种子 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 行：自动选 GPU/CPU
torch.backends.cudnn.benchmark = True     # 行：对固定输入形状略加速
LOG_LEVEL = logging.INFO                  # 行：日志级别
SANITY_CHECK = True                       # 行：开头做一次 batch 维度检查

# ========== revise 版本标记 ==========
REV_TAG = "revise"                                        # 行：当前修订版统一后缀
MODEL_MODULE_NAME = "model_multimodal_lm_revise"          # 行：当前模型文件名（不带 .py）
RESULT_ROOT_NAME = "result_revise"                        # 行：新的结果根目录名
DATASET_NAME = f"atom3d_si60_mm_lm_{REV_TAG}"             # 行：任务名（带 revise）

# ========== 可配置区域 ==========
DATA_DB_SI60 = r"../dataset/ATOM3D/split-by-sequence-identity-60/data"  # 行：SI-60 LMDB 根目录
OUT_DIR = f"../{RESULT_ROOT_NAME}/{DATASET_NAME}"                        # 行：新的结果根目录

BATCH_SIZE   = 16                  # 行：受限于 3D/LM 维度，16 比较稳
NUM_WORKERS  = 4                   # 行：DataLoader 线程数
PIN_MEMORY   = bool(torch.cuda.is_available())  # 行：GPU 下开 pin_memory

EPOCHS         = 600               # 行：最大训练轮次
LR             = 1e-4              # 行：保守学习率（数值更稳）
WEIGHT_DECAY   = 1e-4              # 行：AdamW 的 L2 正则
USE_AMP        = False             # 行：明确关闭 AMP（先追求稳定）
ACCUM_STEPS    = 4                 # 行：梯度累积步数（等效大 batch）
GRAD_CLIP_NORM = 0.5               # 行：梯度裁剪阈值（防爆）
USE_SCHEDULER  = True              # 行：使用余弦重启调度
VAL_AS_BEST    = True              # 行：以验证集 RMSE 选最优权重

# —— 早停机制参数 —— #
EARLY_STOP_PATIENCE   = 60         # 行：若连续 60 个 epoch 没有提升，则早停
EARLY_STOP_MIN_DELTA  = 1e-4       # 行：只有提升超过该阈值才算“更好”


# ========== 统计 si-60 拆分 ==========
def count_si60_splits(root_base: str):
    """读取 train/val/test 的样本数和占比，仅做信息打印"""
    import atom3d.datasets as da
    root = Path(root_base)
    counts = {}
    for sp in ("train", "val", "test"):
        ds = da.LMDBDataset(str(root / sp))
        counts[sp] = len(ds)
    total = sum(counts.values())
    ratios = {sp: (counts[sp] / total if total > 0 else 0.0) for sp in counts}
    return counts, ratios, total


# ========== 多模态 Dataset（带 LM 特征，2D 图可为 None） ==========
class MultiModalDatasetLM(Data.Dataset):
    """封装 multi-modal + LM 特征；getitem 返回单条样本 dict"""
    def __init__(self, pkg: Dict[str, Any]):
        self.ids     = pkg['ids']        # 行：样本 id（object 数组）
        self.y       = pkg['y']          # 行：标签 pKd（float32）
        self.smiles  = pkg['smiles']     # 行：记录方便排错
        self.seq     = pkg['seq']        # 行：记录方便排错
        self.drug_lm = pkg['drug_lm']    # 行：[N, D_drug_lm]
        self.prot_lm = pkg['prot_lm']    # 行：[N, D_prot_lm]

        # —— 2D 图在 SI-60 1D+3D 配置下是 None，这里做兼容 —— #
        self.g_lig   = pkg.get('g_lig', None)   # 行：药物图，可能为 None
        self.g_prot  = pkg.get('g_prot', None)  # 行：蛋白图，可能为 None
        self.has_2d  = (self.g_lig is not None) and (self.g_prot is not None)  # 行：是否存在 2D 图模态

        self.lig3d   = pkg['lig_3d']     # 行：配体 Uni-Mol2 3D 向量 [N, 512]
        self.poc3d   = pkg['poc_3d']     # 行：口袋 Uni-Mol2 3D 向量 [N, 512]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 行：2D 图若不存在，直接返回 None，后续 collate 会感知
        g_lig_i  = self.g_lig[idx]  if self.has_2d else None
        g_prot_i = self.g_prot[idx] if self.has_2d else None
        return {
            'id'     : self.ids[idx],
            'y'      : float(self.y[idx]),
            'smiles' : self.smiles[idx],
            'seq'    : self.seq[idx],
            'drug_lm': self.drug_lm[idx],
            'prot_lm': self.prot_lm[idx],
            'g_lig'  : g_lig_i,
            'g_prot' : g_prot_i,
            'lig3d'  : self.lig3d[idx],
            'poc3d'  : self.poc3d[idx],
        }


# ========== collate：合并 batch ==========
def multimodal_collate_lm(batch: list) -> Dict[str, Any]:
    """
    合并一个 batch：
      - 1D / 3D：numpy 行堆叠 -> torch.float32
      - 2D 图：若存在则 dgl.batch，否则返回 None
    """
    ids    = [b['id']     for b in batch]
    smiles = [b['smiles'] for b in batch]
    seqs   = [b['seq']    for b in batch]

    # —— 判断是否存在 2D 图（SI-60 1D+3D 时为 None） —— #
    has_2d = (batch[0]['g_lig'] is not None) and (batch[0]['g_prot'] is not None)
    if has_2d:
        import dgl
        g_lig_batch  = dgl.batch([b['g_lig']  for b in batch])
        g_prot_batch = dgl.batch([b['g_prot'] for b in batch])
    else:
        g_lig_batch, g_prot_batch = None, None

    import numpy as np
    drug_lm = torch.as_tensor(np.stack([b['drug_lm'] for b in batch], axis=0), dtype=torch.float32)
    prot_lm = torch.as_tensor(np.stack([b['prot_lm'] for b in batch], axis=0), dtype=torch.float32)
    lig3d   = torch.as_tensor(np.stack([b['lig3d']  for b in batch], axis=0), dtype=torch.float32)
    poc3d   = torch.as_tensor(np.stack([b['poc3d']  for b in batch], axis=0), dtype=torch.float32)
    y       = torch.as_tensor([b['y'] for b in batch], dtype=torch.float32)

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


# ========== NaN/Inf 参数检测 ==========
def has_nan_params(model: nn.Module) -> bool:
    """扫描模型参数里是否出现 NaN/Inf，出现则返回 True"""
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
    model.eval()
    obs, pred, loss_sum, n_batches = [], [], 0.0, 0
    for b in loader:
        drug_lm = b['drug_lm'].to(device, non_blocking=True)
        prot_lm = b['prot_lm'].to(device, non_blocking=True)
        g_lig, g_prot = b['g_lig'], b['g_prot']     # 行：可为 None，模型内部需兼容
        lig3d = b['lig3d'].to(device, non_blocking=True)
        poc3d = b['poc3d'].to(device, non_blocking=True)
        y_true = b['y'].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=False):   # 行：明确关闭 AMP
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

        obs.extend(y_true.view(-1).tolist())
        pred.extend(y_hat.view(-1).tolist())

    rmse    = util.get_RMSE(obs, pred)
    pearson = util.get_pearsonr(obs, pred)
    spear   = util.get_spearmanr(obs, pred)
    if criterion is None:
        return rmse, pearson, spear
    return rmse, pearson, spear, (loss_sum / max(1, n_batches))


# ========== 主程序 ==========
if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL)
    util.seed_torch()
    result_root = Path(OUT_DIR)
    result_root.mkdir(parents=True, exist_ok=True)

    logging.info(f"[Revision Tag] {REV_TAG}")            # 行：打印修订标签
    logging.info(f"[Model Module] {MODEL_MODULE_NAME}")  # 行：打印当前模型文件名
    logging.info(f"[Output Dir] {result_root}")          # 行：打印结果根目录

    # 原始 LMDB 统计（仅打印信息）
    counts, ratios, total = count_si60_splits(DATA_DB_SI60)
    logging.info(f"[si-60] counts(raw): {counts} | total={total}")
    logging.info(
        f"[si-60] ratios(raw): "
        f"train={ratios['train']:.4f}, val={ratios['val']:.4f}, test={ratios['test']:.4f}"
    )

    # —— 加载对齐后的多模态+LM 数据 —— #
    kw_common = dict(
        out_mm="../dataset/ATOM3D/processed_mm_si60_lm",
        unimol2_size="unimol2_small",
        contact_threshold=8.0, dis_min=1.0,
        prot_self_loop=False, bond_bidirectional=True,
        prefer_model=None, force_refresh=False,
        use_cuda_for_unimol=True,
        # 如需更换 ESM/ChemBERTa checkpoint，可在此传 esm2_model_name / chemberta_model_name
    )
    mm_train = LoadData_atom3d_si60_multimodal_lm(root_base=DATA_DB_SI60, split="train", **kw_common)["train"]
    mm_val   = LoadData_atom3d_si60_multimodal_lm(root_base=DATA_DB_SI60, split="val",   **kw_common)["val"]
    mm_test  = LoadData_atom3d_si60_multimodal_lm(root_base=DATA_DB_SI60, split="test",  **kw_common)["test"]

    # LM 特征维度（用于构建模型）
    d_drug_lm = mm_train['drug_lm'].shape[1]
    d_prot_lm = mm_train['prot_lm'].shape[1]
    logging.info(f"[LM] drug_lm_dim={d_drug_lm}, prot_lm_dim={d_prot_lm}")

    # —— Dataset / DataLoader —— #
    train_ds, val_ds, test_ds = (
        MultiModalDatasetLM(mm_train),
        MultiModalDatasetLM(mm_val),
        MultiModalDatasetLM(mm_test),
    )
    train_loader = Data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=multimodal_collate_lm
    )
    val_loader   = Data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=multimodal_collate_lm
    )
    test_loader  = Data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=multimodal_collate_lm
    )

    logging.info(
        f"[si-60|LM] final sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    # —— Sanity Check（看一眼形状） —— #
    if SANITY_CHECK:
        b = next(iter(train_loader))
        print(">>> Batch keys:", list(b.keys()))
        print("drug_lm:", tuple(b['drug_lm'].shape),
              "| prot_lm:", tuple(b['prot_lm'].shape),
              "| lig3d:", tuple(b['lig3d'].shape),
              "| poc3d:", tuple(b['poc3d'].shape),
              "| y:", tuple(b['y'].shape))
        if b['g_lig'] is not None:
            print("g_lig nodes/edges:", b['g_lig'].num_nodes(), b['g_lig'].num_edges())
            print("g_prot nodes/edges:", b['g_prot'].num_nodes(), b['g_prot'].num_edges())
        else:
            print("g_lig / g_prot: None (1D+3D-only setting for SI-60)")
        print("[OK] multi-modal LM loaders ready.")

    # —— 导入 revise 版模型 —— #
    from model.model_multimodal_lm_revise import MultiModalDTA_LM

    model = MultiModalDTA_LM(
        d_drug_lm=d_drug_lm, d_prot_lm=d_prot_lm,     # 行：LM 维度
        drug_node_dim=70, drug_edge_dim=6,            # 行：2D 配体特征维（此处不会用到）
        prot_node_dim=33, prot_edge_dim=3,            # 行：2D 蛋白特征维（此处不会用到）
        gcn_hidden=128, gcn_out=128,                  # 行：GNN 隐层/输出维
        d3_lig=512, d3_poc=512,                       # 行：Uni-Mol2 3D 维度
        d_model=256, d_attn=256, n_heads=4,           # 行：融合模块维度/头数
        add_interactions_3d=True                      # 行：是否加入 3D 交互分支
    ).to(DEVICE)

    # 优化器 / 损失 / 调度
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-5
    ) if USE_SCHEDULER else None

    # 日志/权重/CSV 路径（全部切到 revise 目录与 revise 文件名）
    run_dir     = result_root / f"runs_{REV_TAG}"                     # 行：TensorBoard 日志目录
    result_dir  = result_root / f"train_logs_{REV_TAG}"              # 行：训练日志/权重目录
    model_best  = result_dir / f"best_model_{REV_TAG}.pth"           # 行：最优模型权重
    ckpt_path   = result_dir / f"checkpoint_{REV_TAG}.pth.tar"       # 行：断点文件
    csv_file    = result_dir / f"metrics_{REV_TAG}.csv"              # 行：epoch 级指标 CSV
    final_txt   = result_dir / f"final_test_metrics_{REV_TAG}.txt"   # 行：最终测试指标
    config_txt  = result_dir / f"run_config_{REV_TAG}.txt"           # 行：本次运行配置记录

    run_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    # 记录当前运行配置，方便之后核对 revision 结果
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

    # 断点恢复（如果存在）
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
    best_score = float('inf')        # 行：记录当前最优的验证 RMSE
    epochs_no_improve = 0            # 行：早停计数器
    t0 = time.time()

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss, n_batches, last_bidx = 0.0, 0, -1

        for last_bidx, b in enumerate(train_loader):
            drug_lm = b['drug_lm'].to(DEVICE, non_blocking=True)
            prot_lm = b['prot_lm'].to(DEVICE, non_blocking=True)
            g_lig, g_prot = b['g_lig'], b['g_prot']   # 行：None 直接传给模型
            lig3d = b['lig3d'].to(DEVICE, non_blocking=True)
            poc3d = b['poc3d'].to(DEVICE, non_blocking=True)
            y_true = b['y'].to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=False):
                y_hat = model(
                    drug_lm=drug_lm, prot_lm=prot_lm,
                    g_lig=g_lig, g_prot=g_prot,
                    lig3d=lig3d, poc3d=poc3d
                )
                loss = criterion(y_hat, y_true)

            # NaN/Inf 保护
            if not torch.isfinite(loss):
                logging.warning(
                    f"Non-finite loss at epoch={epoch}, batch={last_bidx}: {float(loss)}"
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            # 梯度累积
            (loss / ACCUM_STEPS).backward()

            if (last_bidx + 1) % ACCUM_STEPS == 0:
                if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

                if has_nan_params(model):
                    logging.error(
                        f"[NaN] detected in params at epoch={epoch}, batch={last_bidx} (before step)"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    break

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if has_nan_params(model):
                    logging.error(
                        f"[NaN] detected in params at epoch={epoch}, batch={last_bidx} (after step)"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    break

            running_loss += float(loss)
            n_batches    += 1

        # 尾批（没有凑齐 ACCUM_STEPS）
        if (last_bidx + 1) % ACCUM_STEPS != 0 and n_batches > 0 and not has_nan_params(model):
            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_loss = running_loss / max(1, n_batches)
        writer.add_scalar('Loss/train', train_loss, epoch)

        # 评估阶段
        train_rmse, train_p, train_s, _ = evaluate(model, train_loader, DEVICE, criterion)
        val_rmse,   val_p,   val_s, val_loss = evaluate(model, val_loader, DEVICE, criterion)
        writer.add_scalar('Loss/val', val_loss, epoch)
        for tag, val in [
            ('train_RMSE',train_rmse), ('train_Pearson',train_p), ('train_Spearman',train_s),
            ('val_RMSE',val_rmse), ('val_Pearson',val_p), ('val_Spearman',val_s)
        ]:
            writer.add_scalar(f'Metrics/{tag}', val, epoch)

        test_rmse, test_p, test_s = evaluate(model, test_loader, DEVICE, criterion=None)
        writer.add_scalar('Metrics/test_RMSE',    test_rmse, epoch)
        writer.add_scalar('Metrics/test_Pearson', test_p,    epoch)
        writer.add_scalar('Metrics/test_Spearman',test_s,    epoch)

        # 学习率调度
        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LR', current_lr, epoch)

        # 写 CSV
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                epoch, train_loss, val_loss,
                train_rmse, train_p, train_s,
                val_rmse,   val_p,   val_s,
                test_rmse,  test_p,  test_s,
                current_lr
            ])

        # 以 val_RMSE 选最优并落盘（早停也是看这个）
        score_now = val_rmse if VAL_AS_BEST else test_rmse
        improved = (best_score - score_now) > EARLY_STOP_MIN_DELTA

        if torch.isfinite(torch.tensor(score_now)) and (improved or best_score == float('inf')):
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
                f"[ES] no improvement for {epochs_no_improve} epoch(s) "
                f"(best={best_score:.4f}, current={score_now:.4f})"
            )

        # 早停判断
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            logging.info(
                f"[EARLY STOP] Triggered at epoch {epoch+1}, "
                f"no improvement in last {EARLY_STOP_PATIENCE} epochs."
            )
            util.save_checkpoint(epoch + 1, model, optimizer, filename=str(ckpt_path))  # 行：早停前保存断点
            break

        logging.info(
            f"Epoch {epoch+1:04d} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"train(RMSE={train_rmse:.4f}, P={train_p:.4f}, S={train_s:.4f}) | "
            f"val(RMSE={val_rmse:.4f}, P={val_p:.4f}, S={val_s:.4f}) | "
            f"test(RMSE={test_rmse:.4f}, P={test_p:.4f}, S={test_s:.4f}) | "
            f"lr={current_lr:.3e} | time={(time.time()-t0)/60:.2f} min"
        )

        # 保存断点，便于续训
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
        final_rmse, final_p, final_s = evaluate(best_model, test_loader, DEVICE, criterion=None)

    logging.info(
        f"[FINAL] test_RMSE={final_rmse:.4f}, test_Pearson={final_p:.4f}, test_Spearman={final_s:.4f}"
    )
    with open(final_txt, 'w', encoding='utf-8') as f:
        f.write(f"test_RMSE: {final_rmse:.4f}\n")
        f.write(f"test_Pearson: {final_p:.4f}\n")
        f.write(f"test_Spearman: {final_s:.4f}\n")

    writer.close()