# ======================================
# 多模态 ATOM3D(SI-30) 训练脚本（精简重构版）
# 数据准备 -> DataLoader -> （可选）Sanity Check -> 训练/验证/测试
# 兼容：Python 3.9, Windows, PyTorch+DGL
# ======================================

from pathlib import Path                                  # 行：跨平台路径
import logging                                            # 行：日志
import time                                               # 行：计时
import csv                                                # 行：写训练过程指标到 CSV
from typing import Dict, Any                              # 行：类型注解（可选）

import torch                                              # 行：PyTorch 主库
import torch.nn as nn                                     # 行：损失/层
import torch.optim as optim                               # 行：优化器
import torch.utils.data as Data                           # 行：Dataset / DataLoader
from torch.utils.tensorboard import SummaryWriter         # 行：TensorBoard 日志
import dgl                                                # 行：DGL 图批处理（dgl.batch）

import util                                               # 行：你的工具库（seed/metrics/ckpt）
from util import LoadData_atom3d_si30_multimodal          # 行：严格三模态数据加载函数

# ——（可选）关闭 Uni-Mol 工具库的冗余日志 —— #
lg = logging.getLogger("unimol_tools")                    # 行：获取 unimol_tools 日志器
lg.setLevel(logging.WARNING)                              # 行：仅 WARNING+
lg.propagate = False                                      # 行：不向上级传播
for h in list(lg.handlers):                                # 行：清理已挂载 handler
    lg.removeHandler(h)

# ========== 设备/日志/随机种子 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 行：GPU 优先
torch.backends.cudnn.benchmark = True                                  # 行：卷积/固定尺寸加速
LOG_LEVEL = logging.INFO                                                # 行：日志级别
SANITY_CHECK = True                                                     # 行：是否打印一批形状

# ========== 可配置区域 ==========
DATA_DB_SI30 = r"../dataset/ATOM3D/split-by-sequence-identity-30/data"  # 行：si-30 数据根目录（到 data 层）
DATASET_NAME = "atom3d_si30_mm"                                         # 行：结果任务名
OUT_DIR = f"../result/{DATASET_NAME}"                                   # 行：结果根目录

# DataLoader 超参（按显存调整）
BATCH_SIZE   = 16                                                       # 行：批大小
NUM_WORKERS  = 4                                                        # 行：加载进程（Windows 可改 0/2）
PIN_MEMORY   = bool(torch.cuda.is_available())                          # 行：GPU 时建议开启

# 训练超参（保持你之前的习惯）
EPOCHS         = 600                                                    # 行：训练轮数
LR             = 3e-4                                                   # 行：初始学习率
WEIGHT_DECAY   = 1e-2                                                   # 行：L2 正则
USE_AMP        = True                                                   # 行：自动混合精度
ACCUM_STEPS    = 4                                                      # 行：梯度累积步数
GRAD_CLIP_NORM = 1.0                                                    # 行：梯度裁剪阈值
USE_SCHEDULER  = True                                                   # 行：是否使用 ReduceLROnPlateau
VAL_AS_BEST    = True                                                   # 行：以 val_RMSE 选最优

# ==========（可选）统计官方 si-30 各拆分原始样本数 ==========
def count_si30_splits(root_base: str):
    """行：统计原始 LMDB 每个拆分的条数，仅信息提示"""
    import atom3d.datasets as da
    root = Path(root_base)
    counts = {}
    for sp in ("train", "val", "test"):
        ds = da.LMDBDataset(str(root / sp))
        counts[sp] = len(ds)
    total = sum(counts.values())
    ratios = {sp: (counts[sp] / total if total > 0 else 0.0) for sp in counts}
    return counts, ratios, total

# ========== 多模态 Dataset ==========
class MultiModalDataset(Data.Dataset):
    """行：将一个 split 的对齐‘多模态包’封装为 Dataset；getitem 返回单条样本 dict"""
    def __init__(self, pkg: Dict[str, Any]):
        self.ids    = pkg['ids']             # 行：np.array(object)
        self.y      = pkg['y']               # 行：np.float32（pKd）
        self.smiles = pkg['smiles']          # 行：np.array(str)
        self.seq    = pkg['seq']             # 行：np.array(str)
        self.g_lig  = pkg['g_lig']           # 行：list[DGLGraph]
        self.g_prot = pkg['g_prot']          # 行：list[DGLGraph]
        self.lig3d  = pkg['lig_3d']          # 行：np.ndarray[N, Dl]
        self.poc3d  = pkg['poc_3d']          # 行：np.ndarray[N, Dp]

    def __len__(self):
        return len(self.y)                   # 行：样本数

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {                             # 行：逐条取样本
            'id'    : self.ids[idx],
            'y'     : float(self.y[idx]),
            'smiles': self.smiles[idx],
            'seq'   : self.seq[idx],
            'g_lig' : self.g_lig[idx],
            'g_prot': self.g_prot[idx],
            'lig3d' : self.lig3d[idx],
            'poc3d' : self.poc3d[idx],
        }

# ========== 多模态 collate（合并成一个 batch）=========
def multimodal_collate(batch: list) -> Dict[str, Any]:
    """行：把若干条样本合并为 1 个 batch；图用 dgl.batch；3D/标签堆叠；1D 文本保留 list"""
    ids    = [b['id']     for b in batch]
    smiles = [b['smiles'] for b in batch]
    seqs   = [b['seq']    for b in batch]

    g_lig_batch  = dgl.batch([b['g_lig']  for b in batch])
    g_prot_batch = dgl.batch([b['g_prot'] for b in batch])

    import numpy as np
    lig3d = torch.as_tensor(np.stack([b['lig3d'] for b in batch], axis=0), dtype=torch.float32)
    poc3d = torch.as_tensor(np.stack([b['poc3d'] for b in batch], axis=0), dtype=torch.float32)
    y     = torch.as_tensor([b['y'] for b in batch], dtype=torch.float32)

    return {
        'ids'   : ids,
        'smiles': smiles,
        'seqs'  : seqs,
        'g_lig' : g_lig_batch,
        'g_prot': g_prot_batch,
        'lig3d' : lig3d,
        'poc3d' : poc3d,
        'y'     : y,
    }

# ========== 1D 分词/编码工具 ==========
def _smiles_tokenize(s: str):
    """行：SMILES 粗分词：优先合并 'Cl'/'Br'，其余按字符"""
    i, toks = 0, []
    while i < len(s):
        if i + 1 < len(s) and s[i:i+2] in ("Cl", "Br"):
            toks.append(s[i:i+2]); i += 2
        else:
            toks.append(s[i]); i += 1
    return toks

def _build_vocab_from_loader(loader: Data.DataLoader, key: str, is_smiles: bool):
    """行：从训练 loader 单次遍历构建词表（含 <PAD>/<UNK>），无需依赖 train_ds"""
    sym = {"<PAD>", "<UNK>"}
    for b in loader:
        items = b[key]                          # 行：list[str]
        for s in items:
            if is_smiles:
                sym.update(_smiles_tokenize(s))
            else:
                sym.update(list(s.upper()))
    stoi = {t: i for i, t in enumerate(sorted(sym))}
    return stoi

def encode_smiles_batch(smiles_list, stoi):
    """行：批量 SMILES → 索引张量（右侧 PAD）"""
    PAD, UNK = stoi["<PAD>"], stoi["<UNK>"]
    toks_list = [_smiles_tokenize(s) for s in smiles_list]
    L = max(len(t) for t in toks_list) if toks_list else 1
    idx = []
    for toks in toks_list:
        row = [stoi.get(t, UNK) for t in toks]
        row += [PAD] * (L - len(row))
        idx.append(row)
    return torch.as_tensor(idx, dtype=torch.long)

def encode_protein_batch(seq_list, stoi):
    """行：批量蛋白序列（大写）→ 索引张量（右侧 PAD）"""
    PAD, UNK = stoi["<PAD>"], stoi["<UNK>"]
    seq_list = [s.upper() for s in seq_list]
    L = max(len(s) for s in seq_list) if seq_list else 1
    idx = []
    for s in seq_list:
        row = [stoi.get(ch, UNK) for ch in s]
        row += [PAD] * (L - len(row))
        idx.append(row)
    return torch.as_tensor(idx, dtype=torch.long)

# ========== 统一评估函数（train/val/test 通用）==========
@torch.no_grad()
def evaluate(model: torch.nn.Module,
             loader: Data.DataLoader,
             smiles_stoi: Dict[str, int],
             prot_stoi: Dict[str, int],
             device: torch.device,
             criterion: nn.Module = None):
    """
    行：统一评估接口
      - 若给定 criterion：返回 (rmse, pearson, spearman, avg_loss)
      - 否则               返回 (rmse, pearson, spearman)
    """
    model.eval()
    obs, pred, loss_sum, n_batches = [], [], 0.0, 0
    for b in loader:
        smiles, seqs = b['smiles'], b['seqs']
        g_lig, g_prot = b['g_lig'], b['g_prot']
        lig3d = b['lig3d'].to(device, non_blocking=True)
        poc3d = b['poc3d'].to(device, non_blocking=True)
        y_true = b['y'].to(device, non_blocking=True)

        drug_idx = encode_smiles_batch(smiles, smiles_stoi).to(device)
        prot_idx = encode_protein_batch(seqs,  prot_stoi).to(device)

        with torch.cuda.amp.autocast(enabled=(USE_AMP and device.type == "cuda")):
            y_hat = model(drug_seq=drug_idx, prot_seq=prot_idx,
                          g_lig=g_lig, g_prot=g_prot,
                          lig3d=lig3d, poc3d=poc3d)
            if criterion is not None:
                loss_sum += float(criterion(y_hat, y_true))
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
    # —— 基础准备 —— #
    logging.basicConfig(level=LOG_LEVEL)
    util.seed_torch()                                          # 行：固定随机种子
    result_root = Path(OUT_DIR)
    result_root.mkdir(parents=True, exist_ok=True)

    # —— 打印原始 LMDB 拆分统计（可选）—— #
    counts, ratios, total = count_si30_splits(DATA_DB_SI30)
    logging.info(f"[si-30] counts(raw): {counts} | total={total}")
    logging.info(f"[si-30] ratios(raw): "
                 f"train={ratios['train']:.4f}, val={ratios['val']:.4f}, test={ratios['test']:.4f}")

    # —— 加载对齐后的三模态数据 —— #
    kw_common = dict(
        out_mm="../dataset/ATOM3D/processed_mm_si30",
        unimol2_size="unimol2_small",
        contact_threshold=8.0, dis_min=1.0,
        prot_self_loop=False, bond_bidirectional=True,
        prefer_model=None, force_refresh=False,
        use_cuda_for_unimol=True
    )
    mm_train = LoadData_atom3d_si30_multimodal(root_base=DATA_DB_SI30, split="train", **kw_common)["train"]
    mm_val   = LoadData_atom3d_si30_multimodal(root_base=DATA_DB_SI30, split="val",   **kw_common)["val"]
    mm_test  = LoadData_atom3d_si30_multimodal(root_base=DATA_DB_SI30, split="test",  **kw_common)["test"]

    # —— 包装 Dataset/DataLoader —— #
    train_ds, val_ds, test_ds = MultiModalDataset(mm_train), MultiModalDataset(mm_val), MultiModalDataset(mm_test)
    train_loader = Data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=multimodal_collate)
    val_loader   = Data.DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=multimodal_collate)
    test_loader  = Data.DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=multimodal_collate)

    logging.info(f"[si-30] final sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # ——（可选）Sanity Check：取一批打印形状 —— #
    if SANITY_CHECK:
        b = next(iter(train_loader))
        print(">>> Batch keys:", list(b.keys()))
        print("ids(len):", len(b['ids']), "| smiles(len):", len(b['smiles']), "| seqs(len):", len(b['seqs']))
        print("lig3d:", tuple(b['lig3d'].shape), "| poc3d:", tuple(b['poc3d'].shape), "| y:", tuple(b['y'].shape))
        print("g_lig nodes/edges:", b['g_lig'].num_nodes(), b['g_lig'].num_edges())
        print("g_prot nodes/edges:", b['g_prot'].num_nodes(), b['g_prot'].num_edges())
        print("[OK] multi-modal loaders ready.")

    # —— 导入模型并实例化 —— #
    from model.model_multimodal import MultiModalDTA  # 行：你的多模态模型（1D+2D+3D+CA）

    # 先用训练 loader 构词表（只扫一遍）
    logging.info("[Vocab] building from train_loader ...")
    smiles_stoi = _build_vocab_from_loader(train_loader, key='smiles', is_smiles=True)
    prot_stoi   = _build_vocab_from_loader(train_loader, key='seqs',   is_smiles=False)
    logging.info(f"[Vocab] drug={len(smiles_stoi)}, target={len(prot_stoi)}")

    # 模型构建（维度与你的数据保持一致）
    model = MultiModalDTA(
        drug_vocab=len(smiles_stoi), target_vocab=len(prot_stoi),
        emb_dim_1d=128, seq_layers=2,            # 行：1D 分支
        drug_node_dim=70, drug_edge_dim=6,       # 行：2D 配体特征
        prot_node_dim=33, prot_edge_dim=3,       # 行：2D 蛋白特征
        gcn_hidden=128, gcn_out=128,             # 行：2D 输出维
        d3_lig=512, d3_poc=512,                  # 行：3D Uni-Mol2 维度
        d_model=256, d_attn=256, n_heads=4,      # 行：统一维度与注意力超参
        add_interactions_3d=True                 # 行：3D 对加入交互项
    ).to(DEVICE)

    # 优化器/损失/调度/Amp
    criterion = nn.MSELoss(reduction='mean')
    # optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # scheduler = (optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
    #                                                   patience=10, verbose=True, min_lr=1e-6)
    #              if USE_SCHEDULER else None)
    # 行：用 CosineAnnealingWarmRestarts，先热身再余弦，避免过早见底
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)  # 行：顺便把 WD 降到 1e-4
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-5
    )


    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE.type == "cuda"))

    # 日志/权重/CSV 路径
    run_dir     = result_root / "runs"
    result_dir  = result_root / "train_logs"
    model_best  = result_dir / "best_model.pth"
    ckpt_path   = result_dir / "checkpoint.pth.tar"
    csv_file    = result_dir / "metrics.csv"
    final_txt   = result_dir / "final_test_metrics.txt"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    # 断点恢复（可选）
    start_epoch = 0
    if ckpt_path.exists():
        start_epoch = util.load_checkpoint(str(ckpt_path), model, optimizer)
        logging.info(f"[Resume] from epoch: {start_epoch}")

    # CSV 表头（若不存在）
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
    t0 = time.time()

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss, n_batches, last_bidx = 0.0, 0, -1

        for last_bidx, b in enumerate(train_loader):
            # —— 取 batch & 文本编码 —— #
            smiles, seqs = b['smiles'], b['seqs']
            g_lig, g_prot = b['g_lig'], b['g_prot']
            lig3d = b['lig3d'].to(DEVICE, non_blocking=True)
            poc3d = b['poc3d'].to(DEVICE, non_blocking=True)
            y_true = b['y'].to(DEVICE, non_blocking=True)

            drug_idx = encode_smiles_batch(smiles, smiles_stoi).to(DEVICE)
            prot_idx = encode_protein_batch(seqs,  prot_stoi).to(DEVICE)

            # —— 前向 & 损失（AMP）—— #
            with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE.type == "cuda")):
                y_hat = model(drug_seq=drug_idx, prot_seq=prot_idx,
                              g_lig=g_lig, g_prot=g_prot,
                              lig3d=lig3d, poc3d=poc3d)
                loss = criterion(y_hat, y_true)

            # —— 非有限损失防护 —— #
            if not torch.isfinite(loss):
                logging.warning(f"Non-finite loss at epoch={epoch}, batch={last_bidx}: {float(loss)}")
                optimizer.zero_grad(set_to_none=True)
                continue

            # —— 反传（带梯度累积/AMP）—— #
            if scaler.is_enabled():
                scaler.scale(loss / ACCUM_STEPS).backward()
            else:
                (loss / ACCUM_STEPS).backward()

            # —— 累积到步：更新 —— #
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

        # —— 处理尾批（< ACCUM_STEPS 也要 step 一次）—— #
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

        # —— 记录训练损失 —— #
        train_loss = running_loss / max(1, n_batches)
        writer.add_scalar('Loss/train', train_loss, epoch)

        # —— 评估（train/val：含 loss）—— #
        train_rmse, train_p, train_s, _   = evaluate(model, train_loader, smiles_stoi, prot_stoi, DEVICE, criterion)
        val_rmse,   val_p,   val_s, val_loss = evaluate(model, val_loader,   smiles_stoi, prot_stoi, DEVICE, criterion)
        writer.add_scalar('Loss/val', val_loss, epoch)
        for tag, val in [('train_RMSE',train_rmse), ('train_Pearson',train_p), ('train_Spearman',train_s),
                         ('val_RMSE',val_rmse), ('val_Pearson',val_p), ('val_Spearman',val_s)]:
            writer.add_scalar(f'Metrics/{tag}', val, epoch)

        # —— 测试（不计 loss）—— #
        test_rmse, test_p, test_s = evaluate(model, test_loader, smiles_stoi, prot_stoi, DEVICE, criterion=None)
        writer.add_scalar('Metrics/test_RMSE',    test_rmse, epoch)
        writer.add_scalar('Metrics/test_Pearson', test_p,    epoch)
        writer.add_scalar('Metrics/test_Spearman',test_s,    epoch)

        # # —— 调度（看 val_RMSE 更稳）—— #
        # if scheduler is not None:
        #     scheduler.step(val_rmse)
        # current_lr = optimizer.param_groups[0]['lr']
        # writer.add_scalar('LR', current_lr, epoch)

        # —— 调度（Warmup+Cosine：每个 epoch 末尾更新一次 LR）——
        if scheduler is not None:
            scheduler.step()  # 行：基于 epoch 进度更新 LR（不需要 val_rmse）
        current_lr = optimizer.param_groups[0]['lr']  # 行：取“更新后”的学习率
        writer.add_scalar('LR', current_lr, epoch)  # 行：记录到 TensorBoard

        # —— 写 CSV —— #
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                epoch, train_loss, val_loss,
                train_rmse, train_p, train_s,
                val_rmse,   val_p,   val_s,
                test_rmse,  test_p,  test_s,
                current_lr
            ])

        # —— 保存“最优权重” —— #
        score_now = val_rmse if VAL_AS_BEST else test_rmse
        if score_now < best_score:
            best_score = score_now
            torch.save(model.state_dict(), model_best)
            logging.info(f"[BEST] epoch={epoch+1}, "
                         f"{'val' if VAL_AS_BEST else 'test'}_RMSE={best_score:.4f} "
                         f"-> saved: {model_best}")

        # —— 控制台日志 —— #
        logging.info(
            f"Epoch {epoch+1:04d} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"train(RMSE={train_rmse:.4f}, P={train_p:.4f}, S={train_s:.4f}) | "
            f"val(RMSE={val_rmse:.4f}, P={val_p:.4f}, S={val_s:.4f}) | "
            f"test(RMSE={test_rmse:.4f}, P={test_p:.4f}, S={test_s:.4f}) | "
            f"lr={current_lr:.3e} | time={(time.time()-t0)/60:.2f} min"
        )

        # —— 断点（可中断续训）—— #
        util.save_checkpoint(epoch + 1, model, optimizer, filename=str(ckpt_path))

    # ========== 用“最优模型”做最终测试 ==========
    best_model = MultiModalDTA(
        drug_vocab=len(smiles_stoi), target_vocab=len(prot_stoi),
        emb_dim_1d=128, seq_layers=2,
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
        final_rmse, final_p, final_s = evaluate(best_model, test_loader, smiles_stoi, prot_stoi, DEVICE, criterion=None)

    logging.info(f"[FINAL] test_RMSE={final_rmse:.4f}, test_Pearson={final_p:.4f}, test_Spearman={final_s:.4f}")
    with open(final_txt, 'w', encoding='utf-8') as f:
        f.write(f"test_RMSE: {final_rmse:.4f}\n")
        f.write(f"test_Pearson: {final_p:.4f}\n")
        f.write(f"test_Spearman: {final_s:.4f}\n")

    writer.close()
