# ======================================
# 多模态 ATOM3D(SI-60) 训练脚本（精简重构版）
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
from util import LoadData_atom3d_si60_multimodal          # 行：严格三模态数据加载函数（si-60 版）

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
DATA_DB_SI60 = r"../dataset/ATOM3D/split-by-sequence-identity-60/data"  # 行：si-60 数据根目录（到 data 层）
DATASET_NAME = "atom3d_si60_mm"                                         # 行：结果任务名（si60）
OUT_DIR = f"../result/{DATASET_NAME}"                                   # 行：结果根目录

# DataLoader 超参（按显存调整）
BATCH_SIZE   = 16                                                       # 行：批大小
NUM_WORKERS  = 4                                                        # 行：加载进程（Windows 可改 0/2）
PIN_MEMORY   = bool(torch.cuda.is_available())                          # 行：GPU 时建议开启

# 训练超参（与 si-30 保持一致，方便横向对比）
EPOCHS         = 600                                                    # 行：训练轮数
LR             = 3e-4                                                   # 行：初始学习率
WEIGHT_DECAY   = 1e-2                                                   # 行：L2 正则
USE_AMP        = True                                                   # 行：自动混合精度
ACCUM_STEPS    = 4                                                      # 行：梯度累积步数
GRAD_CLIP_NORM = 1.0                                                    # 行：梯度裁剪阈值
USE_SCHEDULER  = True                                                   # 行：是否使用 ReduceLROnPlateau
VAL_AS_BEST    = True                                                   # 行：以 val_RMSE 选最优

# ==========（可选）统计官方 si-60 各拆分原始样本数 ==========
def count_si60_splits(root_base: str):
    """行：统计原始 LMDB 每个拆分的条数，仅信息提示"""
    import atom3d.datasets as da                                       # 行：ATOM3D LMDBDataset
    root = Path(root_base)                                             # 行：规范路径
    counts = {}                                                        # 行：计数容器
    for sp in ("train", "val", "test"):                                # 行：三拆分
        ds = da.LMDBDataset(str(root / sp))                            # 行：打开 LMDB
        counts[sp] = len(ds)                                           # 行：记录样本数
    total = sum(counts.values())                                       # 行：总样本数
    ratios = {sp: (counts[sp] / total if total > 0 else 0.0) for sp in counts}  # 行：占比
    return counts, ratios, total                                       # 行：返回统计

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
    ids    = [b['id']     for b in batch]                           # 行：list[str]
    smiles = [b['smiles'] for b in batch]                           # 行：list[str]
    seqs   = [b['seq']    for b in batch]                           # 行：list[str]

    g_lig_batch  = dgl.batch([b['g_lig']  for b in batch])          # 行：配体图合并
    g_prot_batch = dgl.batch([b['g_prot'] for b in batch])          # 行：蛋白图合并

    import numpy as np                                              # 行：局部导入减少启动开销
    lig3d = torch.as_tensor(np.stack([b['lig3d'] for b in batch], axis=0), dtype=torch.float32)  # 行：[B,Dl]
    poc3d = torch.as_tensor(np.stack([b['poc3d'] for b in batch], axis=0), dtype=torch.float32)  # 行：[B,Dp]
    y     = torch.as_tensor([b['y'] for b in batch], dtype=torch.float32)                        # 行：[B]

    return {
        'ids'   : ids,                                             # 行：样本 ID 列表
        'smiles': smiles,                                          # 行：SMILES 列表
        'seqs'  : seqs,                                            # 行：蛋白序列列表
        'g_lig' : g_lig_batch,                                     # 行：batched 配体图
        'g_prot': g_prot_batch,                                    # 行：batched 蛋白图
        'lig3d' : lig3d,                                           # 行：配体 3D 向量
        'poc3d' : poc3d,                                           # 行：口袋 3D 向量
        'y'     : y,                                               # 行：标签
    }

# ========== 1D 分词/编码工具 ==========
def _smiles_tokenize(s: str):
    """行：SMILES 粗分词：优先合并 'Cl'/'Br'，其余按字符"""
    i, toks = 0, []
    while i < len(s):
        if i + 1 < len(s) and s[i:i+2] in ("Cl", "Br"):            # 行：二字符卤素
            toks.append(s[i:i+2]); i += 2
        else:
            toks.append(s[i]); i += 1
    return toks

def _build_vocab_from_loader(loader: Data.DataLoader, key: str, is_smiles: bool):
    """行：从训练 loader 单次遍历构建词表（含 <PAD>/<UNK>），无需依赖 train_ds"""
    sym = {"<PAD>", "<UNK>"}                                       # 行：特殊符号
    for b in loader:                                                # 行：遍历一圈
        items = b[key]                                              # 行：list[str]
        for s in items:
            if is_smiles:
                sym.update(_smiles_tokenize(s))                     # 行：SMILES 分词
            else:
                sym.update(list(s.upper()))                         # 行：蛋白逐字符（大写）
    stoi = {t: i for i, t in enumerate(sorted(sym))}               # 行：稳定排序 → idx
    return stoi

def encode_smiles_batch(smiles_list, stoi):
    """行：批量 SMILES → 索引张量（右侧 PAD）"""
    PAD, UNK = stoi["<PAD>"], stoi["<UNK>"]                        # 行：取特殊 idx
    toks_list = [_smiles_tokenize(s) for s in smiles_list]         # 行：逐条分词
    L = max(len(t) for t in toks_list) if toks_list else 1         # 行：批内最大长度
    idx = []
    for toks in toks_list:
        row = [stoi.get(t, UNK) for t in toks]                     # 行：映射到 idx
        row += [PAD] * (L - len(row))                              # 行：右侧补 PAD
        idx.append(row)
    return torch.as_tensor(idx, dtype=torch.long)                  # 行：[B,L]

def encode_protein_batch(seq_list, stoi):
    """行：批量蛋白序列（大写）→ 索引张量（右侧 PAD）"""
    PAD, UNK = stoi["<PAD>"], stoi["<UNK>"]                        # 行：取特殊 idx
    seq_list = [s.upper() for s in seq_list]                       # 行：统一大写
    L = max(len(s) for s in seq_list) if seq_list else 1           # 行：批内最大长度
    idx = []
    for s in seq_list:
        row = [stoi.get(ch, UNK) for ch in s]                      # 行：逐字符映射
        row += [PAD] * (L - len(row))                              # 行：右侧补 PAD
        idx.append(row)
    return torch.as_tensor(idx, dtype=torch.long)                  # 行：[B,L]

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
    model.eval()                                                   # 行：评估模式
    obs, pred, loss_sum, n_batches = [], [], 0.0, 0               # 行：容器与计数
    for b in loader:                                               # 行：遍历 dataloader
        smiles, seqs = b['smiles'], b['seqs']                      # 行：文本
        g_lig, g_prot = b['g_lig'], b['g_prot']                   # 行：图（batched）
        lig3d = b['lig3d'].to(device, non_blocking=True)          # 行：3D 上设备
        poc3d = b['poc3d'].to(device, non_blocking=True)          # 行：3D 上设备
        y_true = b['y'].to(device, non_blocking=True)             # 行：标签上设备

        drug_idx = encode_smiles_batch(smiles, smiles_stoi).to(device)  # 行：SMILES → 索引
        prot_idx = encode_protein_batch(seqs,  prot_stoi).to(device)    # 行：蛋白  → 索引

        with torch.cuda.amp.autocast(enabled=(USE_AMP and device.type == "cuda")):  # 行：AMP 推理
            y_hat = model(drug_seq=drug_idx, prot_seq=prot_idx,
                          g_lig=g_lig, g_prot=g_prot,
                          lig3d=lig3d, poc3d=poc3d)                 # 行：前向
            if criterion is not None:
                loss_sum += float(criterion(y_hat, y_true))         # 行：累计 loss
                n_batches += 1                                      # 行：batch 计数

        obs.extend(y_true.view(-1).tolist())                        # 行：收集真值
        pred.extend(y_hat.view(-1).tolist())                        # 行：收集预测

    rmse    = util.get_RMSE(obs, pred)                              # 行：RMSE
    pearson = util.get_pearsonr(obs, pred)                          # 行：Pearson
    spear   = util.get_spearmanr(obs, pred)                         # 行：Spearman
    if criterion is None:
        return rmse, pearson, spear                                 # 行：仅指标
    return rmse, pearson, spear, (loss_sum / max(1, n_batches))     # 行：含平均 loss

# ========== 主程序 ==========
if __name__ == "__main__":
    # —— 基础准备 —— #
    logging.basicConfig(level=LOG_LEVEL)                            # 行：初始化日志
    util.seed_torch()                                                # 行：固定随机种子
    result_root = Path(OUT_DIR)                                      # 行：结果目录
    result_root.mkdir(parents=True, exist_ok=True)                   # 行：确保存在

    # —— 打印原始 LMDB 拆分统计（可选）—— #
    counts, ratios, total = count_si60_splits(DATA_DB_SI60)         # 行：统计 si-60
    logging.info(f"[si-60] counts(raw): {counts} | total={total}")   # 行：打印条数
    logging.info(f"[si-60] ratios(raw): "                            # 行：打印占比
                 f"train={ratios['train']:.4f}, val={ratios['val']:.4f}, test={ratios['test']:.4f}")

    # —— 加载对齐后的三模态数据 —— #
    kw_common = dict(
        out_mm="../dataset/ATOM3D/processed_mm_si60",               # 行：si-60 缓存目录
        unimol2_size="unimol2_small",                               # 行：Uni-Mol2 规格
        contact_threshold=8.0, dis_min=1.0,                         # 行：2D 构图阈值
        prot_self_loop=False, bond_bidirectional=True,              # 行：图构建开关
        prefer_model=None, force_refresh=False,                     # 行：缓存策略
        use_cuda_for_unimol=True                                    # 行：3D 抽取是否用 GPU
    )
    mm_train = LoadData_atom3d_si60_multimodal(root_base=DATA_DB_SI60, split="train", **kw_common)["train"]  # 行：train
    mm_val   = LoadData_atom3d_si60_multimodal(root_base=DATA_DB_SI60, split="val",   **kw_common)["val"]    # 行：val
    mm_test  = LoadData_atom3d_si60_multimodal(root_base=DATA_DB_SI60, split="test",  **kw_common)["test"]   # 行：test

    # —— 包装 Dataset/DataLoader —— #
    train_ds, val_ds, test_ds = MultiModalDataset(mm_train), MultiModalDataset(mm_val), MultiModalDataset(mm_test)  # 行：三段
    train_loader = Data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=multimodal_collate)   # 行：train loader
    val_loader   = Data.DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=multimodal_collate)   # 行：val loader
    test_loader  = Data.DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=multimodal_collate)   # 行：test loader

    logging.info(f"[si-60] final sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")  # 行：规模提示

    # ——（可选）Sanity Check：取一批打印形状 —— #
    if SANITY_CHECK:
        b = next(iter(train_loader))                                 # 行：拉一批
        print(">>> Batch keys:", list(b.keys()))                     # 行：字段
        print("ids(len):", len(b['ids']), "| smiles(len):", len(b['smiles']), "| seqs(len):", len(b['seqs']))  # 行：长度
        print("lig3d:", tuple(b['lig3d'].shape), "| poc3d:", tuple(b['poc3d'].shape), "| y:", tuple(b['y'].shape))  # 行：3D/标签
        print("g_lig nodes/edges:", b['g_lig'].num_nodes(), b['g_lig'].num_edges())    # 行：配体图规模
        print("g_prot nodes/edges:", b['g_prot'].num_nodes(), b['g_prot'].num_edges())  # 行：蛋白图规模
        print("[OK] multi-modal loaders ready.")                      # 行：就绪提示

    # —— 导入模型并实例化 —— #
    from model.model_multimodal import MultiModalDTA                 # 行：你的多模态模型（1D+2D+3D+CA）

    # 先用训练 loader 构词表（只扫一遍）
    logging.info("[Vocab] building from train_loader ...")           # 行：提示构词开始
    smiles_stoi = _build_vocab_from_loader(train_loader, key='smiles', is_smiles=True)  # 行：SMILES 词表
    prot_stoi   = _build_vocab_from_loader(train_loader, key='seqs',   is_smiles=False) # 行：蛋白  词表
    logging.info(f"[Vocab] drug={len(smiles_stoi)}, target={len(prot_stoi)}")          # 行：打印规模

    # 模型构建（维度与你的数据保持一致）
    model = MultiModalDTA(
        drug_vocab=len(smiles_stoi), target_vocab=len(prot_stoi),    # 行：1D 词表大小
        emb_dim_1d=128, seq_layers=2,                                # 行：1D 分支
        drug_node_dim=70, drug_edge_dim=6,                           # 行：2D 配体特征
        prot_node_dim=33, prot_edge_dim=3,                           # 行：2D 蛋白特征
        gcn_hidden=128, gcn_out=128,                                 # 行：2D 输出维
        d3_lig=512, d3_poc=512,                                      # 行：3D Uni-Mol2 维度
        d_model=256, d_attn=256, n_heads=4,                          # 行：统一维度与注意力超参
        add_interactions_3d=True                                     # 行：3D 对加入交互项
    ).to(DEVICE)                                                     # 行：放设备

    # 优化器/损失/调度/Amp
    criterion = nn.MSELoss(reduction='mean')                         # 行：回归损失
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)  # 行：AdamW
    scheduler = (optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                      patience=10, verbose=True, min_lr=1e-6)
                 if USE_SCHEDULER else None)                         # 行：Plateau 降 LR
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE.type == "cuda"))  # 行：AMP 标量

    # 日志/权重/CSV 路径
    result_dir  = result_root / "train_logs"                         # 行：指标与权重目录
    run_dir     = result_root / "runs"                               # 行：TensorBoard 日志目录
    model_best  = result_dir / "best_model.pth"                      # 行：最优权重
    ckpt_path   = result_dir / "checkpoint.pth.tar"                  # 行：断点文件
    csv_file    = result_dir / "metrics.csv"                         # 行：过程指标 CSV
    final_txt   = result_dir / "final_test_metrics.txt"              # 行：最终指标文本
    run_dir.mkdir(parents=True, exist_ok=True)                       # 行：创建目录
    result_dir.mkdir(parents=True, exist_ok=True)                    # 行：创建目录
    writer = SummaryWriter(log_dir=str(run_dir))                     # 行：初始化 TB

    # 断点恢复（可选）
    start_epoch = 0                                                  # 行：默认从 0
    if ckpt_path.exists():                                           # 行：存在断点则恢复
        start_epoch = util.load_checkpoint(str(ckpt_path), model, optimizer)  # 行：载入参数与优化器
        logging.info(f"[Resume] from epoch: {start_epoch}")          # 行：打印恢复点

    # CSV 表头（若不存在）
    if not csv_file.exists():
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                'epoch', 'train_loss', 'val_loss',
                'train_RMSE','train_Pearson','train_Spearman',
                'val_RMSE','val_Pearson','val_Spearman',
                'test_RMSE','test_Pearson','test_Spearman',
                'lr'
            ])                                                       # 行：写入表头

    # ========== 训练主循环 ==========
    best_score = float('inf')                                        # 行：初始化最好分（RMSE 越小越好）
    t0 = time.time()                                                 # 行：起始计时

    for epoch in range(start_epoch, EPOCHS):                         # 行：逐 epoch
        model.train()                                                # 行：训练模式
        optimizer.zero_grad(set_to_none=True)                        # 行：清梯度
        running_loss, n_batches, last_bidx = 0.0, 0, -1              # 行：累计器

        for last_bidx, b in enumerate(train_loader):                 # 行：遍历训练集
            # —— 取 batch & 文本编码 —— #
            smiles, seqs = b['smiles'], b['seqs']                    # 行：文本
            g_lig, g_prot = b['g_lig'], b['g_prot']                  # 行：图（batched）
            lig3d = b['lig3d'].to(DEVICE, non_blocking=True)         # 行：3D 上设备
            poc3d = b['poc3d'].to(DEVICE, non_blocking=True)         # 行：3D 上设备
            y_true = b['y'].to(DEVICE, non_blocking=True)            # 行：标签上设备

            drug_idx = encode_smiles_batch(smiles, smiles_stoi).to(DEVICE)  # 行：SMILES → 索引
            prot_idx = encode_protein_batch(seqs,  prot_stoi).to(DEVICE)    # 行：蛋白  → 索引

            # —— 前向 & 损失（AMP）—— #
            with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE.type == "cuda")):  # 行：AMP
                y_hat = model(drug_seq=drug_idx, prot_seq=prot_idx,
                              g_lig=g_lig, g_prot=g_prot,
                              lig3d=lig3d, poc3d=poc3d)               # 行：前向
                loss = criterion(y_hat, y_true)                       # 行：MSE

            # —— 非有限损失防护 —— #
            if not torch.isfinite(loss):
                logging.warning(f"Non-finite loss at epoch={epoch}, batch={last_bidx}: {float(loss)}")
                optimizer.zero_grad(set_to_none=True)                 # 行：清掉异常梯度
                continue                                              # 行：跳过该 batch

            # —— 反传（带梯度累积/AMP）—— #
            if scaler.is_enabled():
                scaler.scale(loss / ACCUM_STEPS).backward()          # 行：缩放反传
            else:
                (loss / ACCUM_STEPS).backward()                      # 行：常规模式

            # —— 累积到步：更新 —— #
            if (last_bidx + 1) % ACCUM_STEPS == 0:                   # 行：到达累积步
                if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:            # 行：梯度裁剪
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)                    # 行：裁剪前反缩放
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                if scaler.is_enabled():
                    scaler.step(optimizer); scaler.update()           # 行：AMP 更新
                else:
                    optimizer.step()                                  # 行：常规更新
                optimizer.zero_grad(set_to_none=True)                 # 行：清梯度

            running_loss += float(loss)                               # 行：累计损失
            n_batches    += 1                                         # 行：累计 batch 数

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
        train_loss = running_loss / max(1, n_batches)                # 行：均值训练损失
        writer.add_scalar('Loss/train', train_loss, epoch)           # 行：写入 TensorBoard

        # —— 评估（train/val：含 loss）—— #
        train_rmse, train_p, train_s, _   = evaluate(model, train_loader, smiles_stoi, prot_stoi, DEVICE, criterion)  # 行：train 指标
        val_rmse,   val_p,   val_s, val_loss = evaluate(model, val_loader,   smiles_stoi, prot_stoi, DEVICE, criterion)  # 行：val 指标
        writer.add_scalar('Loss/val', val_loss, epoch)               # 行：写入 val_loss
        for tag, val in [('train_RMSE',train_rmse), ('train_Pearson',train_p), ('train_Spearman',train_s),
                         ('val_RMSE',val_rmse), ('val_Pearson',val_p), ('val_Spearman',val_s)]:
            writer.add_scalar(f'Metrics/{tag}', val, epoch)          # 行：写入六指标

        # —— 测试（不计 loss）—— #
        test_rmse, test_p, test_s = evaluate(model, test_loader, smiles_stoi, prot_stoi, DEVICE, criterion=None)  # 行：test 指标
        writer.add_scalar('Metrics/test_RMSE',    test_rmse, epoch)  # 行：写入
        writer.add_scalar('Metrics/test_Pearson', test_p,    epoch)  # 行：写入
        writer.add_scalar('Metrics/test_Spearman',test_s,    epoch)  # 行：写入

        # —— 调度（看 val_RMSE 更稳）—— #
        if scheduler is not None:
            scheduler.step(val_rmse)                                  # 行：若 plateu 则降 LR
        current_lr = optimizer.param_groups[0]['lr']                  # 行：读取当前 LR
        writer.add_scalar('LR', current_lr, epoch)                    # 行：写入 LR

        # —— 写 CSV —— #
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                epoch, train_loss, val_loss,
                train_rmse, train_p, train_s,
                val_rmse,   val_p,   val_s,
                test_rmse,  test_p,  test_s,
                current_lr
            ])                                                        # 行：逐行追加

        # —— 保存“最优权重” —— #
        score_now = val_rmse if VAL_AS_BEST else test_rmse            # 行：以 val/test RMSE 判优
        if score_now < best_score:                                    # 行：更优则覆盖保存
            best_score = score_now
            torch.save(model.state_dict(), model_best)                # 行：保存最优参数
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
        )                                                             # 行：可读训练日志

        # —— 断点（可中断续训）—— #
        util.save_checkpoint(epoch + 1, model, optimizer, filename=str(ckpt_path))  # 行：保存断点

    # ========== 用“最优模型”做最终测试 ==========
    best_model = MultiModalDTA(                                      # 行：同构模型
        drug_vocab=len(smiles_stoi), target_vocab=len(prot_stoi),
        emb_dim_1d=128, seq_layers=2,
        drug_node_dim=70, drug_edge_dim=6,
        prot_node_dim=33, prot_edge_dim=3,
        gcn_hidden=128, gcn_out=128,
        d3_lig=512, d3_poc=512,
        d_model=256, d_attn=256, n_heads=4,
        add_interactions_3d=True
    ).to(DEVICE)
    best_model.load_state_dict(torch.load(model_best, map_location=DEVICE))          # 行：载入最优权重
    best_model.eval()                                                                # 行：评估模式

    with torch.no_grad():                                                            # 行：最终评估
        final_rmse, final_p, final_s = evaluate(best_model, test_loader, smiles_stoi, prot_stoi, DEVICE, criterion=None)

    logging.info(f"[FINAL] test_RMSE={final_rmse:.4f}, test_Pearson={final_p:.4f}, test_Spearman={final_s:.4f}")  # 行：打印收官指标
    with open(final_txt, 'w', encoding='utf-8') as f:                                # 行：落盘收官指标
        f.write(f"test_RMSE: {final_rmse:.4f}\n")
        f.write(f"test_Pearson: {final_p:.4f}\n")
        f.write(f"test_Spearman: {final_s:.4f}\n")

    writer.close()                                                                    # 行：关闭 TensorBoard 句柄
