import pandas as pd
import torch.utils.data as Data
import pickle
from collections import OrderedDict
import torch
import random
import re
import os
import json                               # 保存/读取本地缓存
import csv                                # 另存CSV方便查看
from pathlib import Path
import numpy as np                        # 返回/保存数组
from rdkit import Chem
from tqdm import tqdm
from typing import Optional
import atom3d.datasets as da              # 官方LMDBDataset
from typing import List
# —— 依赖 —— #
import dgl
from unimol_tools import UniMolRepr
from transformers import AutoTokenizer, AutoModel

def LoadData(path, logspance_trans=True):
    path = Path(path)

    # 构建文件路径并检查文件是否存在
    ligands_file = path / "ligands_can.txt"
    proteins_file = path / "proteins.txt"
    affinity_file = path / "Y"

    # 加载配体和蛋白质数据，并保持顺序
    with ligands_file.open() as f:
        ligands = json.load(f, object_pairs_hook=OrderedDict)
    with proteins_file.open() as f:
        proteins = json.load(f, object_pairs_hook=OrderedDict)

    # 加载亲和力矩阵数据并进行对数变换（如果需要）
    with affinity_file.open("rb") as f:
        Y = pickle.load(f, encoding='latin1')
    Y = -np.log10(Y / 1e9) if logspance_trans else Y

    return list(ligands.values()), list(proteins.values()), Y


def GetSamples(dataSet_name, drugSeqs, targetSeqs, affi_matrix):
    # 初始化用于存储药物序列、靶标序列和亲和力矩阵的列表
    drugSeqs_buff, targetSeqs_buff, affiMatrix_buff = [], [], []

    # 遍历所有药物序列
    for i, drug in enumerate(drugSeqs):
        # 遍历所有靶标序列
        for j, target in enumerate(targetSeqs):
            # 如果数据集是 'davis'，或数据集是 'kiba' 且亲和力矩阵值不是 NaN，则保存当前组合
            if dataSet_name == 'davis' or (dataSet_name == 'kiba' and not np.isnan(affi_matrix[i, j])):
                drugSeqs_buff.append(drug)
                targetSeqs_buff.append(target)
                affiMatrix_buff.append(affi_matrix[i, j])

    # 返回药物序列、靶标序列和亲和力矩阵
    return drugSeqs_buff, targetSeqs_buff, affiMatrix_buff


def LabelDT(drug_seqs, target_seqs, drugSeq_maxlen, targetSeq_maxLen):
    # 定义 targetSeq 和 drugSeq 的词汇表
    targetSeq_vocab = {
        "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7,
        "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, "O": 13,
        "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, "U": 19,
        "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25
    }

    drugSeq_vocab = {
        "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6,
        ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12,
        "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18,
        "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
        "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30,
        "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36,
        "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42,
        "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48,
        "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54,
        "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
        "t": 61, "y": 62
    }

    # 生成标签化后的药物和靶标序列，仅进行截断操作
    label_drugSeqs = [
        [drugSeq_vocab[token.split()[0]] for token in seq[:drugSeq_maxlen]]
        for seq in drug_seqs
    ]

    label_targetSeqs = [
        [targetSeq_vocab[token.split()[0]] for token in seq[:targetSeq_maxLen]]
        for seq in target_seqs
    ]

    return label_drugSeqs, label_targetSeqs


def Shuttle(drug, target, affini):
    # 生成一个随机排列的索引数组，长度与 affini 的长度相同
    index = np.random.permutation(len(affini))
    # 返回随机重排后的 drug, target, affini，使用 dtype=object 来处理不规则长度的序列
    return np.array(drug, dtype=object)[index], np.array(target, dtype=object)[index], np.array(affini)[index]


def Shuttle_2D(drug_graphs, target_graphs, affini):
    """
    打乱配体图、靶标图、亲和力标签（保持对应关系）

    参数:
      drug_graphs:   list[dgl.DGLGraph]  配体图
      target_graphs: list[dgl.DGLGraph]  靶标图
      affini:        list/np.ndarray     标签 (float)

    返回:
      (打乱后的 drug_graphs, target_graphs, affini)
    """
    n = len(affini)
    index = np.random.permutation(n)  # 随机排列索引

    # 用列表推导打乱，保持顺序一致
    d_shuf = [drug_graphs[i] for i in index]
    t_shuf = [target_graphs[i] for i in index]
    y_shuf = [affini[i] for i in index]

    return d_shuf, t_shuf, y_shuf


class DatasetIterater(Data.Dataset):
    def __init__(self, texta, textb, label):
        self.texta = texta
        self.textb = textb
        self.label = label

    def __getitem__(self, item):
        return self.texta[item], self.textb[item], self.label[item]

    def __len__(self):
        return len(self.texta)


def BatchPad(batch_data, pad=0):
    texta, textb, label = zip(*batch_data)
    max_len_a = max(len(seq) for seq in texta)
    max_len_b = max(len(seq) for seq in textb)
    texta = torch.LongTensor([seq + [pad] * (max_len_a - len(seq)) for seq in texta])
    textb = torch.LongTensor([seq + [pad] * (max_len_b - len(seq)) for seq in textb])
    label = torch.FloatTensor(label)
    return texta, textb, label


def seed_torch(seed=666):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_MSE(y_obs, y_pred):
    y_obs  = np.asarray(y_obs,  dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    assert y_obs.shape == y_pred.shape, "y_obs 和 y_pred 长度不一致"
    return float(np.mean((y_obs - y_pred) ** 2))


def get_RMSE(y_obs, y_pred):
    return float(np.sqrt(get_MSE(y_obs, y_pred)))


def get_pearsonr(y_obs, y_pred):
    """皮尔逊相关系数 r"""
    y_obs  = np.asarray(y_obs,  dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    assert y_obs.shape == y_pred.shape, "y_obs 和 y_pred 长度不一致"

    y_obs  = y_obs  - y_obs.mean()
    y_pred = y_pred - y_pred.mean()
    denom = np.sqrt((y_obs**2).sum()) * np.sqrt((y_pred**2).sum())
    return float((y_obs * y_pred).sum() / denom) if denom != 0 else 0.0


def get_spearmanr(y_obs, y_pred):
    """真正的 Spearman：对各自做秩变换后，再算 Pearson"""
    y_obs  = np.asarray(y_obs,  dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    assert y_obs.shape == y_pred.shape, "y_obs 和 y_pred 长度不一致"

    def _rank(a):
        # 排序下标（稳定排序，保证 ties 的顺序可控）
        tmp = a.argsort(kind="mergesort")
        ranks = np.empty_like(tmp, dtype=np.float64)
        ranks[tmp] = np.arange(len(a), dtype=np.float64)

        # 按值聚合求“平均秩”（等价于 scipy rankdata(method="average")）
        uniq, inv = np.unique(a, return_inverse=True)
        base_ranks = ranks.copy()
        mean_ranks = np.zeros_like(uniq, dtype=np.float64)
        for k in range(len(uniq)):
            mean_ranks[k] = base_ranks[inv == k].mean()
        return mean_ranks[inv]

    ro = _rank(y_obs)
    rp = _rank(y_pred)
    ro = ro - ro.mean()
    rp = rp - rp.mean()
    denom = np.sqrt((ro**2).sum()) * np.sqrt((rp**2).sum())
    return float((ro * rp).sum() / denom) if denom != 0 else 0.0


def get_cindex(Y, P):
    """C-index；O(N^2)，样本特别多时可能有点慢。"""
    summ = 0.0
    pair = 0
    for i in range(len(Y)):
        for j in range(i):
            if Y[i] > Y[j]:
                pair += 1
                summ += (P[i] > P[j]) + 0.5 * (P[i] == P[j])
    return float(summ / pair) if pair != 0 else 0.0


def r_squared_error(y_obs, y_pred):
    """r^2：Pearson r 的平方"""
    y_obs  = np.asarray(y_obs,  dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    y_obs  = y_obs  - y_obs.mean()
    y_pred = y_pred - y_pred.mean()

    num = (y_pred * y_obs).sum() ** 2
    den = (y_obs**2).sum() * (y_pred**2).sum()
    return float(num / den) if den != 0 else 0.0


def get_k(y_obs, y_pred):
    """无截距回归 y ≈ k * y_pred 的斜率 k"""
    y_obs  = np.asarray(y_obs,  dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    den = (y_pred * y_pred).sum()
    return float((y_obs * y_pred).sum() / den) if den != 0 else 0.0


def squared_error_zero(y_obs, y_pred):
    """r0^2：无截距回归优度（QSAR 里常见）"""
    k = get_k(y_obs, y_pred)
    y_obs  = np.asarray(y_obs,  dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_obs_mean = y_obs.mean()

    upp  = ((y_obs - k * y_pred) ** 2).sum()
    down = ((y_obs - y_obs_mean) ** 2).sum()
    return float(1 - (upp / down)) if down != 0 else 0.0


def get_rm2(ys_orig, ys_line):
    """rm2 指标：基于 r^2 和 r0^2 的综合指标"""
    r2  = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    return float(r2 * (1 - np.sqrt(np.abs((r2 * r2) - (r02 * r02)))))


def get_aupr(y_obs, y_pred, threshold=None):
    """
    AUPR（Area Under Precision-Recall curve）

    这里默认把 y_obs 按“中位数”二值化：
        y_obs >= median(y_obs) 视为正样本（高亲和）
    你要和 DeepDTA 一致，可以自己把 threshold 换成固定值（例如 KIBA/Davis 论文里的阈值）。
    """
    y_obs  = np.asarray(y_obs,  dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    assert y_obs.shape == y_pred.shape, "y_obs 和 y_pred 长度不一致"

    if threshold is None:
        threshold = np.median(y_obs)  # 默认按中位数划分正/负
    y_bin = (y_obs >= threshold).astype(np.int32)

    # 若全是 0 或 1，则 AUPR 不可定义，返回 0
    pos = y_bin.sum()
    if pos == 0 or pos == len(y_bin):
        return 0.0

    # 按预测分数从大到小排序
    order = np.argsort(-y_pred)
    y_bin_sorted = y_bin[order]

    tp = np.cumsum(y_bin_sorted).astype(np.float64)
    fp = np.cumsum(1 - y_bin_sorted).astype(np.float64)

    recall = tp / pos
    precision = tp / np.maximum(tp + fp, 1.0)

    # 用梯形法在 (recall, precision) 上积分
    aupr = np.trapz(precision, recall)
    return float(aupr)


def save_checkpoint(epoch, model, optimizer, filename):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']

def load_smiles(drug_file):
    """加载化合物（SMILES字符串）数据，并清理控制字符"""
    drug_file = Path(drug_file)

    # 检查文件是否存在
    if not drug_file.exists():
        raise FileNotFoundError(f"化合物文件 {drug_file} 不存在!")

    # 读取文件并清理无效控制字符
    with drug_file.open() as f:
        file_content = f.read()

    # 清理掉所有控制字符（ASCII 0-31），保留换行、回车、制表符等合法字符
    file_content = re.sub(r'[\x00-\x1F\x7F]', '', file_content)

    # 尝试加载 JSON 数据
    try:
        drugs = json.loads(file_content, object_pairs_hook=OrderedDict)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        raise

    # 返回字典的值部分（化合物的SMILES字符串）
    return drugs



def load_sequences(target_file):
    """加载蛋白质序列数据，并清理控制字符"""
    target_file = Path(target_file)

    # 检查文件是否存在
    if not target_file.exists():
        raise FileNotFoundError(f"蛋白质文件 {target_file} 不存在!")

    # 读取文件并清理无效控制字符
    with target_file.open() as f:
        file_content = f.read()

    # 清理掉所有控制字符（ASCII 0-31），保留换行、回车、制表符等合法字符
    file_content = re.sub(r'[\x00-\x1F\x7F]', '', file_content)

    # 尝试加载 JSON 数据
    try:
        targets = json.loads(file_content, object_pairs_hook=OrderedDict)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        raise

    # 返回字典的值部分（蛋白质的序列）
    return targets

def LoadData_f():
    df = pd.read_csv('../dataset/fdavis/affi_info.txt', sep='\t', header=None)

    smiles_list = df[1].tolist()
    seq_list = df[3].tolist()
    affinity_list = df[4].astype(float).tolist()

    y = np.array(affinity_list)

    return np.array(smiles_list), np.array(seq_list), y


Atom_Table = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
              'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
              'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
              'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
              'Pt', 'Hg', 'Pb', 'Unknown']  # Unknown表示其它元素

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def GetAtomFeatures(atom: Chem.rdchem.Atom) -> np.ndarray:
    # 1) 原子种类 one-hot（你原来的代码不动）
    symbol = atom.GetSymbol()
    symbol_feat = one_of_k_encoding(symbol, Atom_Table + ["UNK"])

    # 2) 杂化方式（不动）
    hybrid_feat = one_of_k_encoding(
        atom.GetHybridization(),
        [Chem.rdchem.HybridizationType.S,
         Chem.rdchem.HybridizationType.SP,
         Chem.rdchem.HybridizationType.SP2,
         Chem.rdchem.HybridizationType.SP3,
         Chem.rdchem.HybridizationType.SP3D]
    )

    # 3) 原子度数、氢原子数、价态
    degree_feat = one_of_k_encoding(
        min(atom.GetDegree(), 5),          # 原子度数：跟之前一样
        [0, 1, 2, 3, 4, 5]
    )

    numHs_feat = one_of_k_encoding(
        min(atom.GetTotalNumHs(), 5),      # 总氢原子数：跟之前一样
        [0, 1, 2, 3, 4, 5]
    )

    # ↓↓↓ 这里是关键修改：优先用新接口 GetValence(getExplicit=False) ↓↓↓
    try:
        # 新版 RDKit：GetValence 支持关键字参数 getExplicit
        val = atom.GetValence(getExplicit=False)
    except TypeError:
        # 老版本 RDKit 没这个参数时，退回到旧接口，避免崩溃
        val = atom.GetImplicitValence()

    valence_feat = one_of_k_encoding(
        min(int(val), 5),
        [0, 1, 2, 3, 4, 5]
    )

    # 4) 是否在环 / 芳香性（不动）
    ring_feat     = [int(atom.IsInRing())]
    aromatic_feat = [int(atom.GetIsAromatic())]

    # 5) 拼接所有特征（不动）
    features = np.array(
        symbol_feat + hybrid_feat + degree_feat + numHs_feat +
        valence_feat + ring_feat + aromatic_feat,
        dtype=np.float32
    )
    return features


def GetBondFeatures(bond: Chem.rdchem.Bond) -> np.ndarray:
    # 1) bond type one-hot
    bond_type_set = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]
    bond_type = bond.GetBondType()
    type_feat = [int(bond_type == b) for b in bond_type_set]

    # 2) 其他布尔特征
    conj_feat = [int(bond.GetIsConjugated())]
    ring_feat = [int(bond.IsInRing())]

    # 3) 拼接成 array
    features = np.array(type_feat + conj_feat + ring_feat, dtype=np.float32)

    return features


def SmileToGraph(smile, bidirectional: bool = True, allow_unsanitized: bool = True):
    from rdkit import Chem
    import torch
    import dgl

    # 行：第一次尝试——标准解析（含完整sanitize）
    mol = Chem.MolFromSmiles(smile)  # 行：默认 sanitize=True
    if mol is None and allow_unsanitized:
        # 行：第二次尝试——宽松解析（不做sanitize，且放宽严格语法）
        mol = Chem.MolFromSmiles(smile, sanitize=False)  # 行：先把拓扑读进来
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smile}")  # 行：连拓扑都读不进来，放弃

        # 行：放宽属性缓存检查（避免 valence 严格报错）
        mol.UpdatePropertyCache(strict=False)

        # 行：做“部分”标准化：跳过最易出错的 PROPERTIES/KEKULIZE（价态/共振展开）
        try:
            sanitize_ops = (Chem.SanitizeFlags.SANITIZE_ALL
                            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
                            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            Chem.SanitizeMol(mol, sanitizeOps=sanitize_ops)
        except Exception:
            # 行：兜底再做更保守的一组标注：只设芳香性/共轭/杂化/自由基信息
            Chem.SanitizeMol(
                mol,
                sanitizeOps=(Chem.SanitizeFlags.SANITIZE_FINDRADICALS
                             | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                             | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                             | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION)
            )

    if mol is None:
        # 行：不允许宽松模式 或 上述两步都失败，就按严格策略直接丢弃
        raise ValueError(f"Invalid SMILES: {smile}")

    # ====== 下面保持你原来的建图流程 ======
    num_atoms = mol.GetNumAtoms()                                  # 行：节点数
    node_feats = []
    for idx in range(num_atoms):
        atom = mol.GetAtomWithIdx(idx)
        feat = GetAtomFeatures(atom)                               # 行：你的原子特征
        node_feats.append(feat)
    node_feats = np.asarray(node_feats, dtype=np.float32)

    src, dst, edge_feats = [], [], []
    for bond in mol.GetBonds():                                    # 行：遍历真实化学键
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        f = GetBondFeatures(bond)                                  # 行：你的键特征（含芳香/共轭等）
        src.append(i); dst.append(j); edge_feats.append(f)
        if bidirectional:
            src.append(j); dst.append(i); edge_feats.append(f)

    g = dgl.graph((torch.tensor(src, dtype=torch.int64),
                   torch.tensor(dst, dtype=torch.int64)),
                  num_nodes=num_atoms)
    g.ndata['x'] = torch.from_numpy(node_feats)
    g.edata['w'] = torch.as_tensor(np.asarray(edge_feats, dtype=np.float32))
    return g



res_dict ={'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E', 'PHE':'F', 'GLY':'G', 'HIS':'H', 'ILE':'I', 'LYS':'K', 'LEU':'L',
           'MET':'M', 'ASN':'N', 'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S', 'THR':'T', 'VAL':'V', 'TRP':'W', 'TYR':'Y'}

pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}


# 预计算各数值属性的均值（用于未知残基 'X' 兜底）
_AA_KEYS = list(res_weight_table.keys())  # 20AA
_MEAN_WEIGHT = float(np.mean([res_weight_table[k] for k in _AA_KEYS]))
_MEAN_PKA    = float(np.mean([res_pka_table[k]    for k in _AA_KEYS]))
_MEAN_PKB    = float(np.mean([res_pkb_table[k]    for k in _AA_KEYS]))
_MEAN_PX     = float(np.mean([res_pkx_table[k]    for k in _AA_KEYS]))
_MEAN_PL     = float(np.mean([res_pl_table[k]     for k in _AA_KEYS]))
_MEAN_HPH2   = float(np.mean([res_hydrophobic_ph2_table[k] for k in _AA_KEYS]))
_MEAN_HPH7   = float(np.mean([res_hydrophobic_ph7_table[k] for k in _AA_KEYS]))

def residue_features(residue):
    """
    返回该残基的理化属性向量：
      - 前5维是所属类别的 one-hot（脂肪族、芳香族、极性中性、酸性带电、碱性带电）
      - 后7维是数值属性（分子量、pKa、pKb、pKx、pI、疏水性@pH2、疏水性@pH7）
    对于未知残基 'X' 或不在表中的符号，用“类别全0 + 数值属性均值”兜底，避免 KeyError。
    """
    # —— 5 个类别 one-hot ——（未知残基时，这5个全是0）
    res_property1 = [
        1 if residue in pro_res_aliphatic_table    else 0,  # 脂肪族
        1 if residue in pro_res_aromatic_table     else 0,  # 芳香族
        1 if residue in pro_res_polar_neutral_table else 0, # 极性中性
        1 if residue in pro_res_acidic_charged_table else 0,# 酸性带电
        1 if residue in pro_res_basic_charged_table  else 0 # 碱性带电
    ]

    # —— 7 个数值属性 ——（未知残基时用均值作为中性兜底）
    w   = res_weight_table.get(residue, _MEAN_WEIGHT)
    pka = res_pka_table.get(residue,    _MEAN_PKA)
    pkb = res_pkb_table.get(residue,    _MEAN_PKB)
    pkx = res_pkx_table.get(residue,    _MEAN_PX)
    pl  = res_pl_table.get(residue,     _MEAN_PL)
    h2  = res_hydrophobic_ph2_table.get(residue, _MEAN_HPH2)
    h7  = res_hydrophobic_ph7_table.get(residue, _MEAN_HPH7)

    res_property2 = [w, pka, pkb, pkx, pl, h2, h7]
    return np.array(res_property1 + res_property2, dtype=np.float32)

def seq_feature(pro_seq):
    """
    构造蛋白序列的特征矩阵：
      - one-hot（包含 'X'） + 理化属性（对 'X' 做中性兜底）
    """
    L = len(pro_seq)
    pro_hot = np.zeros((L, len(pro_res_table)), dtype=np.float32)  # one-hot: 21列（含X）
    pro_property = np.zeros((L, 12), dtype=np.float32)             # 理化属性: 12列

    for i in range(L):
        aa = pro_seq[i]
        # one-hot：未知残基 'X' 仍然被允许（pro_res_table 已含 'X'）
        pro_hot[i,] = one_of_k_encoding(aa if aa in pro_res_table else 'X', pro_res_table)
        # 理化属性：未知残基走兜底
        pro_property[i,] = residue_features(aa if aa in _AA_KEYS else 'X')

    return np.concatenate((pro_hot, pro_property), axis=1)



def TargetToGraph(contact_matrix, distance_matrix, ca_coords, seq3,
                  contact=1, dis_min=1.0, self_loop=False):
    # 1) 基本准备 ---------------------------------------------------------
    c_size = len(contact_matrix)                                   # 节点数 N
    # 把三字母序列映射成一字母或编码（你已有 res_dict/seq_feature）
    seq = [res_dict[a] if a in res_dict else 'X' for a in seq3]    # 映射未知残基到 'X'
    node_features = np.asarray(seq_feature(seq), dtype=np.float32) # [N, F]

    # 2) 生成边索引（一次性）----------------------------------------------
    # 条件：i != j 且 contact_matrix[i,j] == contact
    cm = np.asarray(contact_matrix)
    mask = (cm == contact)
    np.fill_diagonal(mask, False)                                   # 去掉自环
    src, dst = np.where(mask)                                       # 一次拿到所有 (i,j)，长度为 E
    E = src.shape[0]

    # 3) 若需要自环（可选）-----------------------------------------------
    if self_loop:
        loop_idx = np.arange(c_size, dtype=np.int64)
        src = np.concatenate([src, loop_idx])
        dst = np.concatenate([dst, loop_idx])

    # 4) 计算边特征（全部向量化）----------------------------------------
    # 4.1 余弦相似度：节点特征先做 L2 归一化，再逐边点积 -> [E]
    nf = node_features
    nf_norm = np.linalg.norm(nf, axis=1, keepdims=True) + 1e-12
    nf_unit = nf / nf_norm
    sim_ij = (nf_unit[src] * nf_unit[dst]).sum(axis=1)
    # 将 [-1,1] 映射到 [0,1]（与你原注释一致），并截断到 [0,1]
    sim_ij = np.clip(0.5 * (sim_ij + 1.0), 0.0, 1.0).astype(np.float32)

    # 4.2 距离特征：<= dis_min 用 dis_min，否则用 1/d
    dm = np.asarray(distance_matrix, dtype=np.float32)
    d_ij = dm[src, dst]
    dis_ij = np.where(d_ij <= dis_min, dis_min, 1.0 / (d_ij + 1e-12)).astype(np.float32)

    # 4.3 角度特征：使用 CA 坐标与原点的夹角余弦（∈[-1,1]）
    coords = np.asarray(ca_coords, dtype=np.float32)                # [N, 3]
    a = coords[src]                                                 # [E, 3]
    b = coords[dst]                                                 # [E, 3]
    # 归一化后点积
    a_norm = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    b_norm = np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    angle_ij = (a / a_norm * b / b_norm).sum(axis=1).astype(np.float32)  # [-1,1]

    # 4.4 组装边特征矩阵 [E(+N_loop), 3]
    edge_features = np.stack([sim_ij, dis_ij, angle_ij], axis=1) if E > 0 else np.zeros((0,3), np.float32)
    if self_loop:
        # 给自环补上特征（sim=1, dis=0, angle=1 比较合理；也可按需自定义）
        loop_feats = np.tile(np.array([1.0, 0.0, 1.0], dtype=np.float32), (c_size, 1))
        edge_features = np.vstack([edge_features, loop_feats])

    # 5) 一次性构图并挂特征 ---------------------------------------------
    G = dgl.graph((torch.from_numpy(src.astype(np.int64)),
                   torch.from_numpy(dst.astype(np.int64))),
                  num_nodes=c_size)
    G.ndata['x'] = torch.from_numpy(node_features)                     # [N, F]
    G.edata['w'] = torch.from_numpy(edge_features)                     # [E(+N_loop), 3]

    return G


# 三字母 -> 一字母（含常见修饰；未知→'X'）
AA3_TO_1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
    'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
    'THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    'MSE':'M','SEC':'C','PYL':'K','HSD':'H','HSE':'H','HSP':'H','GLX':'E','ASX':'D',
    'CSO':'C','SEP':'S','TPO':'T','PTR':'Y','MLY':'K','MLZ':'K','KCX':'K','LLP':'K',
    'FME':'M','SAR':'G','UNK':'X'
}


def atoms_to_sequence(df: pd.DataFrame, chain_id: Optional[str]=None, longest_only: bool=True) -> str:
    """行：从 atoms_pocket DataFrame 提取口袋氨基酸序列"""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return ""                                      # 行：空表直接返回空序列
    if chain_id is not None:
        df = df[df["chain"] == chain_id]               # 行：可选—只取指定链
    if df.empty:
        return ""
    # 行：只保留关键列并去重，确保每个 (chain,residue) 只出现一次
    sub = (
        df[["chain","residue","resname"]]
        .dropna(subset=["chain","residue","resname"])
        .drop_duplicates(subset=["chain","residue"])
        .copy()
    )
    # 行：规范类型与顺序（先按 chain，再按 residue 数字顺序）
    sub["chain"]   = sub["chain"].astype(str).str.strip().replace({"": "A"})
    def _res_key(x):
        try:
            return int(float(str(x).strip()))
        except:
            return 10**9
    sub["_rk"] = sub["residue"].apply(_res_key)
    chain_seqs = {}
    for ch, grp in sub.groupby("chain"):
        grp = grp.sort_values("_rk")
        seq = "".join(AA3_TO_1.get(str(r).upper().strip(), "X") for r in grp["resname"].tolist())
        if seq:
            chain_seqs[ch] = seq
    if not chain_seqs:
        return ""
    if longest_only:
        best = max(chain_seqs, key=lambda c: len(chain_seqs[c]))
        return chain_seqs[best]
    return "".join(chain_seqs[ch] for ch in sorted(chain_seqs.keys()))

def LoadData_atom3d_1d(root: str,
                    out_dir: str = "../dataset/ATOM3D/processed_raw",
                    out_name: str = "lba_pkd_pocket_raw",
                    force_refresh: bool = False):
    out_dir  = Path(out_dir)                           # 行：转 Path
    out_dir.mkdir(parents=True, exist_ok=True)         # 行：确保目录存在
    npz_path = out_dir / f"{out_name}.npz"             # 行：npz 路径
    csv_path = out_dir / f"{out_name}.csv"             # 行：csv 路径

    # —— 快速路径：已有结果直接读 ——
    if npz_path.exists() and not force_refresh:
        data = np.load(npz_path, allow_pickle=True)    # 行：读取 npz
        return data["smiles"], data["seqs"], data["pkd"]

    # —— 正常路径：读取 LMDB 并构建口袋序列 ——
    ds = da.LMDBDataset(root)                          # 行：官方Dataset（root 传 data.mdb 上一层）

    smiles_list, pocket_seq_list, y_list, ids = [], [], [], []  # 行：收集容器
    n_total = n_ok = n_empty = 0                       # 行：统计

    for sample in tqdm(ds, desc="Building pocket sequences from atoms_pocket", unit="sample"):
        n_total += 1
        smiles_list.append(sample["smiles"])           # 行：配体 SMILES
        y_list.append(float(sample["scores"]["neglog_aff"]))  # 行：标签 pKd
        ids.append(sample.get("id",""))                # 行：样本 ID

        pocket_df = sample.get("atoms_pocket", None)   # 行：口袋原子表（DataFrame）
        seq = atoms_to_sequence(pocket_df, chain_id=None, longest_only=True)  # 行：生成口袋序列
        if seq:
            n_ok += 1
            pocket_seq_list.append(seq)
        else:
            n_empty += 1
            pocket_seq_list.append("")                 # 行：留空；训练前可过滤

    # —— 转数组 & 落盘 ——
    smiles_array = np.array(smiles_list)
    seq_array    = np.array(pocket_seq_list)
    y_array      = np.array(y_list, dtype=np.float32)

    np.savez(npz_path, smiles=smiles_array, seqs=seq_array, pkd=y_array)  # 行：保存 npz（训练快速加载）
    with open(csv_path, "w", newline="", encoding="utf-8") as f:          # 行：保存 csv（人工检查）
        w = csv.writer(f)
        w.writerow(["id", "smiles", "pocket_seq", "neglog_aff(pKd)"])
        for i, smi, seq, pkd in zip(ids, smiles_array, seq_array, y_array):
            w.writerow([i, smi, seq, float(pkd)])

    print(f"[ATOM3D-LBA|POCKET] total={n_total} | pocket_seq_ok={n_ok} | empty={n_empty}")
    print(f"  - npz: {npz_path}")
    print(f"  - csv: {csv_path}")

    return smiles_array, seq_array, y_array

def _extract_seq3_and_ca(df_atoms, prefer_model=None):
    """
    从 ATOM3D 的 atoms_protein/atoms_pocket 表中，提取三字母序列 seq3 与每个残基的代表坐标。
    - 首选 CA；若无 CA：依次回退到 CB、C4'、P、C1'、N、C、O；
    - 若这些都没有，则用该残基“重原子(非H)质心”（最后兜底用全部原子质心）。
    返回: seq3(list[str]), coords(np.ndarray[N,3]), residue_keys(list[tuple])
    """

    # —— 空表兜底 ——
    if not isinstance(df_atoms, pd.DataFrame) or df_atoms.empty:
        return [], np.zeros((0, 3), dtype=np.float32), []

    df = df_atoms.copy()

    # —— 规范列并清洗 ——
    #   有些数据会有多 model，这里优选样本中最常见的 model（或使用传入 prefer_model）
    if 'model' in df.columns and df['model'].nunique() > 1:
        if prefer_model is None:
            prefer_model = df['model'].value_counts().idxmax()
        df = df[df['model'] == prefer_model].copy()

    # 名称去空格，避免 ' CA ' 之类
    if 'name' in df.columns:
        df['name'] = df['name'].astype(str).str.strip()
    else:
        df['name'] = ''  # 少数情况缺列时给空串

    # 基本列缺失直接丢（坐标、链、残基号、残基名）
    keep_cols = [c for c in ['chain', 'residue', 'resname', 'x', 'y', 'z'] if c in df.columns]
    df = df.dropna(subset=keep_cols)

    # 链名规范化
    if 'chain' in df.columns:
        df['chain'] = df['chain'].astype(str).str.strip().replace({"": "A"})
    else:
        df['chain'] = 'A'

    # 残基排序 key（把 '12A'、' 34 ' 等都尽力转成数字，不行就放到很后面）
    def _res_key(x):
        try:
            return int(float(str(x).strip()))
        except:
            return 10**9
    df['_rk'] = df['residue'].apply(_res_key) if 'residue' in df.columns else 10**9

    # 兼容 insertion_code
    if 'insertion_code' not in df.columns:
        df['insertion_code'] = ''

    # —— 按 (chain, residue, insertion_code) 聚合为“一个残基” ——
    grp = df.groupby(['chain', 'residue', 'insertion_code'], sort=False)

    seq3, coords, keys = [], [], []

    # 代表原子的优先级：氨基酸优先 CA / CB；若是核酸或个别情况可用 C4' / P / C1'；再退到主链 N/C/O
    prefer_names = ['CA', 'CB', "C4'", 'P', "C1'", 'N', 'C', 'O']

    for (ch, resi, ins), g in grp:
        g = g.sort_values('_rk')  # 稳定顺序（虽然每组 _rk 相同）
        resname = str(g['resname'].iloc[0]).upper().strip() if 'resname' in g.columns else 'UNK'

        # —— 1) 优先在该残基里按“代表原子优先级”挑一个坐标 ——
        chosen = None
        name_series = g['name'].astype(str).str.strip()
        for nm in prefer_names:
            cand = g[name_series == nm]
            if len(cand) > 0:
                # 有些结构有多条等价原子（altloc），若有占有率列('occ'/'occupancy')，取占有率最高
                occ_col = 'occ' if 'occ' in cand.columns else ('occupancy' if 'occupancy' in cand.columns else None)
                if occ_col is not None:
                    row = cand.iloc[cand[occ_col].astype(float).fillna(0.0).values.argmax()]
                else:
                    row = cand.iloc[0]
                chosen = row[['x', 'y', 'z']].to_numpy(dtype=np.float32)
                break

        # —— 2) 若以上都没有，则用“重原子(非H)质心” ——
        if chosen is None:
            if 'element' in g.columns:
                heavy = g[g['element'].astype(str).str.upper() != 'H']
                base = heavy if len(heavy) > 0 else g
            else:
                base = g
            chosen = base[['x', 'y', 'z']].to_numpy(dtype=np.float32).mean(axis=0)

        # 收集
        seq3.append(resname)
        coords.append(chosen)
        keys.append((str(ch), resi, str(ins)))

    if len(coords) == 0:
        return [], np.zeros((0, 3), dtype=np.float32), []

    coords = np.vstack(coords).astype(np.float32)
    return seq3, coords, keys



def _pairwise_distance(coords: np.ndarray) -> np.ndarray:
    """纯 numpy 计算两两 L2 距离矩阵（避免 scipy 依赖）"""
    if coords.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    diff = coords[:, None, :] - coords[None, :, :]          # [N,1,3]-[1,N,3] -> [N,N,3]
    return np.sqrt((diff ** 2).sum(axis=2)).astype(np.float32)  # [N,N]


def _build_contact_from_atoms(df_atoms, threshold: float = 8.0, prefer_model=None):
    """
    用残基代表坐标（优先 CA，回退到 CB/主链/重原子质心）构建：
      contact_matrix: dist<=threshold 的二值矩阵
      distance_matrix: 所有两两欧氏距离
      ca_coords: 代表坐标数组
      seq3: 三字母残基序列
    """
    seq3, ca_coords, _ = _extract_seq3_and_ca(df_atoms, prefer_model=prefer_model)  # 这里调用鲁棒版
    if len(ca_coords) == 0:
        # 返回空矩阵，上游会跳过该样本（遵循“三模态都成功才保留”的规则）
        return (np.zeros((0, 0), dtype=np.int8),
                np.zeros((0, 0), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                seq3)

    # 两两距离
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2)).astype(np.float32)

    # 接触阈值化
    contact = (dist <= float(threshold)).astype(np.int8)
    np.fill_diagonal(contact, 0)

    return contact, dist, ca_coords, seq3


def LoadData_atom3d_2d(
    root: str,
    out_dir: str = "../dataset/ATOM3D/processed_raw",       # 输出目录（图缓存+标签+索引）
    out_name: str = "lba_pkd_pocket_raw_2d",                      # 基础文件名
    use_pocket: bool = True,                                # True=用 atoms_pocket；False=用 atoms_protein
    contact_threshold: float = 8.0,                         # 接触判定阈值（Å）
    dis_min: float = 1.0,                                   # 你的 TargetToGraph 中的 dis_min
    prot_self_loop: bool = False,                           # TargetToGraph 是否加自环
    bond_bidirectional: bool = True,                        # SmileToGraph 是否双向边
    force_refresh: bool = False,                            # 强制重建缓存
    prefer_model: int = None                                # 多 model 时优先选的 model
):
    out_dir = Path(out_dir)                                                     # 行：转 Path
    out_dir.mkdir(parents=True, exist_ok=True)                                  # 行：确保目录存在

    # 行：DGL 图缓存路径（分别存 ligand/protein 图）
    g_lig_path  = out_dir / f"{out_name}_lig.dgl.bin"
    g_prot_path = out_dir / f"{out_name}_prot.dgl.bin"

    # 行：标签/元数据（npz）与可视化/检查（csv）
    metanpz    = out_dir / f"{out_name}.npz"
    meta_csv    = out_dir / f"{out_name}.csv"

    # —— 快速路径：已有缓存直接读 —— #
    if g_lig_path.exists() and g_prot_path.exists() and metanpz.exists() and (not force_refresh):
        # 读图（DGL 的批量加载接口）
        lig_graphs, _  = dgl.load_graphs(str(g_lig_path))                       # 行：读取配体图列表
        prot_graphs, _ = dgl.load_graphs(str(g_prot_path))                      # 行：读取蛋白图列表
        # 读标签/字符串
        meta = np.load(metanpz, allow_pickle=True)                             # 行：读取元数据 npz
        y = meta["pkd"].astype(np.float32)                                      # 行：标签 pKd
        return lig_graphs, prot_graphs, y                                       # 行：与 1D 返回风格类似（三元组）

    # —— 正常路径：从 LMDB 构建 —— #
    ds = da.LMDBDataset(root)                                                   # 行：加载官方 LMDB
    source = 'atoms_pocket' if use_pocket else 'atoms_protein'                  # 行：选择口袋/全蛋白

    lig_graphs: List[dgl.DGLGraph]  = []                                        # 行：配体图容器
    prot_graphs: List[dgl.DGLGraph] = []                                        # 行：蛋白图容器
    smiles_list, y_list, ids = [], [], []                                       # 行：元数据容器

    n_total = n_ok = n_fail = 0                                                 # 行：计数器
    for sample in tqdm(ds, desc=f"Building 2D graphs from {source}", unit="sample"):
        n_total += 1

        try:
            # —— 标签与基本字段 —— #
            smi = sample["smiles"]                                              # 行：配体 SMILES
            y   = float(sample["scores"]["neglog_aff"])                         # 行：标签 pKd
            sid = sample.get("id", "")                                          # 行：样本 ID

            # —— 配体图：SMILES -> 原子图 —— #
            lig_g = SmileToGraph(smi, bidirectional=bond_bidirectional)  # 行：构建配体图

            # —— 蛋白/口袋图：atoms_* -> CA接触图 —— #
            df_atoms = sample[source]                                           # 行：DataFrame（20列）
            cm, dm, ca, seq3 = _build_contact_from_atoms(                       # 行：构建四个输入
                df_atoms, threshold=contact_threshold, prefer_model=prefer_model
            )
            prot_g = TargetToGraph(                                             # 行：残基图（节点：残基；边：接触）
                contact_matrix=cm,
                distance_matrix=dm,
                ca_coords=ca,
                seq3=seq3,
                contact=1,
                dis_min=dis_min,
                self_loop=prot_self_loop
            )

            # —— 收集 —— #
            lig_graphs.append(lig_g)                                            # 行：收集配体图
            prot_graphs.append(prot_g)                                          # 行：收集蛋白图
            smiles_list.append(smi)                                             # 行：收集 SMILES
            y_list.append(y)                                                    # 行：收集标签
            ids.append(sid)                                                     # 行：收集 ID
            n_ok += 1                                                           # 行：计数成功

        except Exception as e:
            # 某些样本可能异常（缺字段/坐标异常/非法SMILES），这里跳过并记日志
            n_fail += 1
            # 你也可以用 logging.warning，这里简单打印
            print(f"[WARN] sample skipped due to error: {e}")

    # —— 落盘缓存（图 + 元数据） —— #
    dgl.save_graphs(str(g_lig_path), lig_graphs)                                # 行：保存配体图列表
    dgl.save_graphs(str(g_prot_path), prot_graphs)                              # 行：保存蛋白图列表

    smiles_arr = np.array(smiles_list, dtype=object)                            # 行：转为 ndarray（可变长字符串）
    y_arr      = np.array(y_list,    dtype=np.float32)                          # 行：pKd 数组
    ids_arr    = np.array(ids,       dtype=object)                              # 行：ID 数组

    np.savez(metanpz, smiles=smiles_arr, pkd=y_arr, ids=ids_arr)               # 行：保存元数据 npz
    with open(meta_csv, "w", newline="", encoding="utf-8") as f:                # 行：写 CSV 便于查看
        w = csv.writer(f)
        w.writerow(["id", "smiles", "lig_nodes", "prot_nodes", "pkd"])
        for sid, smi, lg, pg, yy in zip(ids_arr, smiles_arr, lig_graphs, prot_graphs, y_arr):
            w.writerow([sid, smi, lg.num_nodes(), pg.num_nodes(), float(yy)])

    print(f"[ATOM3D-LBA|GRAPH2D] total={n_total} | ok={n_ok} | fail={n_fail}")  # 行：统计信息
    print(f"  - ligand graphs: {g_lig_path}")                                   # 行：路径提示
    print(f"  - protein graphs: {g_prot_path}")
    print(f"  - meta npz: {metanpz}")
    print(f"  - meta csv: {meta_csv}")

    return lig_graphs, prot_graphs, y_arr

def Gcollate(samples):
    import dgl

    dgraphs, tgraphs, labels = map(list, zip(*samples))

    # --- 兜底：把空 target 图改成“1 节点 0 边”的占位图 ---
    def _ensure_non_empty(g, x_dim=33, w_dim=3):
        if g.num_nodes() == 0:
            g2 = dgl.graph(([], []), num_nodes=1)                 # 1 节点 0 边
            g2.ndata['x'] = torch.zeros(1, x_dim, dtype=torch.float32)  # 节点特征全 0
            g2.edata['w'] = torch.zeros(0, w_dim, dtype=torch.float32)  # 无边特征
            return g2
        # 没有边特征时补个空的
        if 'w' not in g.edata:
            g.edata['w'] = torch.zeros(0, w_dim, dtype=torch.float32)
        return g

    tgraphs = [_ensure_non_empty(g) for g in tgraphs]

    # --- 药物图 batch ---
    bdg = dgl.batch(dgraphs)
    bdg.ndata['x'] = torch.as_tensor(bdg.ndata['x'], dtype=torch.float32)
    bdg.edata['w'] = torch.as_tensor(bdg.edata['w'], dtype=torch.float32)

    # --- 靶标图 batch ---
    btg = dgl.batch(tgraphs)
    btg.ndata['x'] = torch.as_tensor(btg.ndata['x'], dtype=torch.float32)
    if 'w' in btg.edata:
        btg.edata['w'] = torch.as_tensor(btg.edata['w'], dtype=torch.float32)
    else:
        btg.edata['w'] = torch.zeros(0, 3, dtype=torch.float32)

    # --- 标签改成 [B,1] ---
    y = torch.as_tensor(labels, dtype=torch.float32).view(-1, 1)

    return bdg, btg, y


def LoadData_atom3d_3d(root: str,
                       out_dir: str = "../dataset/ATOM3D/processed_3d",
                       out_name: str = "lba_pkd_3d_unimol2",
                       model_size: str = "unimol2_small",
                       force_refresh: bool = False):
    """
    使用 Uni-Mol2 提取 ATOM3D-LBA 数据集的 3D 分子与口袋 embedding。

    参数:
      root: str              LMDB 数据集路径（传 data.mdb 上一层）
      out_dir: str           输出目录
      out_name: str          输出文件名前缀
      model_size: str        Uni-Mol2 模型尺寸（unimol2_small/base/large）
      force_refresh: bool    是否强制重新生成
    返回:
      lig_embeds, pocket_embeds, y_array
    """
    from unimol_tools import UniMolRepr
    from tqdm import tqdm
    from pathlib import Path
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{out_name}.npz"
    csv_path = out_dir / f"{out_name}.csv"

    # --- 如果已有结果并且不强制刷新 ---
    if npz_path.exists() and not force_refresh:
        data = np.load(npz_path, allow_pickle=True)
        return data["lig_embeds"], data["pocket_embeds"], data["pkd"]

    # --- 初始化 Uni-Mol2 模型 ---
    model = UniMolRepr(model="unimol2", model_size=model_size, use_cuda=True)
    model.model.eval()

    # --- 加载 ATOM3D LMDB 数据 ---
    ds = da.LMDBDataset(root)
    print(f"✅ 读取 ATOM3D 数据集，共 {len(ds)} 条样本")

    lig_embeds, pocket_embeds, y_list, ids = [], [], [], []

    for i, sample in enumerate(tqdm(ds, desc="Extracting 3D embeddings", unit="sample")):
        try:
            lig_df = sample["atoms_ligand"]
            pocket_df = sample["atoms_pocket"]

            lig_atoms = lig_df["element"].tolist()
            lig_coords = lig_df[["x", "y", "z"]].values.tolist()
            pocket_atoms = pocket_df["element"].tolist()
            pocket_coords = pocket_df[["x", "y", "z"]].values.tolist()

            # --- 过滤空样本 ---
            if len(lig_atoms) == 0 or len(pocket_atoms) == 0:
                continue

            # --- Uni-Mol2 提取表示 ---
            lig_emb = model.get_repr({"atoms": lig_atoms, "coordinates": lig_coords}, return_tensor=True)
            pocket_emb = model.get_repr({"atoms": pocket_atoms, "coordinates": pocket_coords}, return_tensor=True)

            lig_embeds.append(lig_emb.cpu().numpy())
            pocket_embeds.append(pocket_emb.cpu().numpy())
            y_list.append(float(sample["scores"]["neglog_aff"]))
            ids.append(sample.get("id", f"s_{i}"))

        except Exception as e:
            print(f"[WARN] sample {i} failed: {e}")
            continue

    lig_embeds = np.vstack(lig_embeds)
    pocket_embeds = np.vstack(pocket_embeds)
    y_array = np.array(y_list, dtype=np.float32)

    # --- 保存 npz 与 csv ---
    np.savez(npz_path, lig_embeds=lig_embeds, pocket_embeds=pocket_embeds, pkd=y_array, ids=np.array(ids))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "pKd", "lig_dim", "pocket_dim"])
        for sid, yv in zip(ids, y_array):
            w.writerow([sid, yv, lig_embeds.shape[1], pocket_embeds.shape[1]])

    print(f"[ATOM3D-LBA|3D] processed={len(y_array)} samples")
    print(f"  - npz: {npz_path}")
    print(f"  - csv: {csv_path}")

    return lig_embeds, pocket_embeds, y_array


def LoadData_atom3d_3d_si30(
    root_base: str,                             # 传到 data 这一层：.../split-by-sequence-identity-30/data
    split: str = "train",                       # "train" | "val" | "test" | "all"
    out_dir: str = "../dataset/ATOM3D/processed_3d",
    out_name: str = "lba_pkd_3d_unimol2_si30",  # 输出文件前缀；会自动附加 _{split}
    model_size: str = "unimol2_small",
    force_refresh: bool = False
):
    """
    使用 Uni-Mol2 提取 ATOM3D-LBA 的 30% 序列同一性拆分（sequence-identity-30）版本的
    3D 分子(配体)与口袋 embedding。

    参数:
      root_base  : str  指向 .../split-by-sequence-identity-30/data
      split      : str  "train" / "val" / "test" / "all"
      out_dir    : str  输出目录
      out_name   : str  输出文件名前缀（会自动加 _{split}）
      model_size : str  "unimol2_small/base/large"
      force_refresh:bool 是否强制重新生成

    返回:
      lig_embeds, pocket_embeds, y_array, ids
      （若 split="all"，返回三者拼接后的整体；否则返回单个拆分）
    """
    # ---- 标准库 & 第三方依赖导入（和 raw 版一致）----
    from unimol_tools import UniMolRepr                 # Uni-Mol2 表示提取器
    from tqdm import tqdm                               # 进度条
    from pathlib import Path
    import csv

    # ---- 将 root_base 处理成 Path，并规范输出目录 ----
    root_base = Path(root_base)                         # e.g. .../split-by-sequence-identity-30/data
    out_dir = Path(out_dir)                             # 输出目录根
    out_dir.mkdir(parents=True, exist_ok=True)          # 确保目录存在

    # ---- 辅助函数：对单个 split（train/val/test）做提取 ----
    def _process_one_split(split_name: str):
        """
        针对某个具体 split（train/val/test）从其 data.mdb 提取 embedding，
        并保存到 out_dir/out_name_{split}.npz / .csv
        """
        lmdb_root = root_base / split_name              # e.g. .../data/train
        # 输出文件：带上 split 后缀，互不覆盖
        npz_path = out_dir / f"{out_name}_{split_name}.npz"
        csv_path = out_dir / f"{out_name}_{split_name}.csv"

        # ---- 如果已有结果且不强制刷新，直接读取缓存 ----
        if npz_path.exists() and not force_refresh:
            data = np.load(npz_path, allow_pickle=True)
            return (data["lig_embeds"], data["pocket_embeds"], data["pkd"], data["ids"])

        # ---- 初始化 Uni-Mol2（eval 推理模式，用作特征提取器）----
        model = UniMolRepr(model="unimol2", model_size=model_size, use_cuda=True)
        model.model.eval()

        # ---- 加载该 split 的 LMDB 数据集 ----
        ds = da.LMDBDataset(str(lmdb_root))             # 传到包含 data.mdb 的目录
        print(f"✅ [si-30|{split_name}] 读取 LMDB 样本数: {len(ds)}")

        lig_list, poc_list, y_list, id_list = [], [], [], []

        # ---- 遍历样本，提取 ligand / pocket 的 3D 表示 ----
        for i, sample in enumerate(tqdm(ds, desc=f"[si-30|{split_name}] Extracting", unit="sample")):
            try:
                lig_df = sample["atoms_ligand"]         # 配体原子表（pandas DataFrame）
                pocket_df = sample["atoms_pocket"]      # 口袋原子表

                lig_atoms = lig_df["element"].tolist()                  # 原子种类列表
                lig_coords = lig_df[["x", "y", "z"]].values.tolist()    # 原子坐标列表
                poc_atoms = pocket_df["element"].tolist()
                poc_coords = pocket_df[["x", "y", "z"]].values.tolist()

                # 过滤空样本，避免报错/污染
                if len(lig_atoms) == 0 or len(poc_atoms) == 0:
                    continue

                # Uni-Mol2 前向，得到定长 embedding（如 768 维）
                lig_emb = model.get_repr({"atoms": lig_atoms, "coordinates": lig_coords}, return_tensor=True)
                poc_emb = model.get_repr({"atoms": poc_atoms, "coordinates": poc_coords}, return_tensor=True)

                # 收集到 Python 列表（转成 numpy）
                lig_list.append(lig_emb.cpu().numpy())
                poc_list.append(poc_emb.cpu().numpy())
                y_list.append(float(sample["scores"]["neglog_aff"]))     # pKd
                id_list.append(sample.get("id", f"{split_name}_{i}"))    # 样本ID，兜底用索引

            except Exception as e:
                print(f"[WARN] {split_name} sample {i} failed: {e}")
                continue

        # ---- 空集保护：若该 split 一个样本都没成功，直接给出明确报错 ----
        if len(lig_list) == 0 or len(poc_list) == 0:
            raise RuntimeError(f"[si-30|{split_name}] No valid samples extracted. "
                               f"Check LMDB at: {lmdb_root}")

        # ---- 列表堆叠成矩阵 ----
        lig_embeds = np.vstack(lig_list)                # 形如 (N, D)
        poc_embeds = np.vstack(poc_list)                # 形如 (N, D)
        y_array = np.asarray(y_list, dtype=np.float32)  # (N,)
        ids = np.asarray(id_list)                       # (N,)

        # ---- 保存缓存：npz + csv ----
        np.savez(npz_path, lig_embeds=lig_embeds, pocket_embeds=poc_embeds, pkd=y_array, ids=ids)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "pKd", "lig_dim", "pocket_dim"])
            for sid, yv in zip(ids, y_array):
                w.writerow([sid, yv, lig_embeds.shape[1], poc_embeds.shape[1]])

        print(f"[ATOM3D-LBA|si-30|{split_name}] processed={len(y_array)}")
        print(f"  - npz: {npz_path}")
        print(f"  - csv: {csv_path}")

        return lig_embeds, poc_embeds, y_array, ids

    # ---- 主逻辑：根据 split 决定处理哪个（或全部）----
    if split in ("train", "val", "test"):
        return _process_one_split(split)

    elif split == "all":
        # 依次处理 train/val/test，再拼接返回
        parts = []
        for sp in ("train", "val", "test"):
            parts.append(_process_one_split(sp))

        # 分别取出三元组并按样本维拼接
        lig_all = np.vstack([p[0] for p in parts])
        poc_all = np.vstack([p[1] for p in parts])
        y_all = np.concatenate([p[2] for p in parts], axis=0)
        ids_all = np.concatenate([p[3] for p in parts], axis=0)

        return lig_all, poc_all, y_all, ids_all

    else:
        raise ValueError(f"split 必须是 'train'/'val'/'test'/'all' 之一，收到: {split!r}")


def LoadData_atom3d_3d_si60(
    root_base: str,                             # 指到 data 这一层：.../split-by-sequence-identity-60/data
    split: str = "train",                       # "train" | "val" | "test" | "all"
    out_dir: str = "../dataset/ATOM3D/processed_3d",
    out_name: str = "lba_pkd_3d_unimol2_si60",  # 输出文件前缀；自动附加 _{split}
    model_size: str = "unimol2_small",
    force_refresh: bool = False
):
    """
    使用 Uni-Mol2 提取 ATOM3D-LBA 的 60% 序列同一性拆分（sequence-identity-60）版本的
    3D 分子(配体)与口袋 embedding。

    参数:
      root_base  : str  指向 .../split-by-sequence-identity-60/data
      split      : str  "train" / "val" / "test" / "all"
      out_dir    : str  输出目录
      out_name   : str  输出文件名前缀（会自动加 _{split}）
      model_size : str  "unimol2_small/base/large"
      force_refresh:bool 是否强制重新生成

    返回:
      lig_embeds, pocket_embeds, y_array, ids
      （若 split="all"，返回三者拼接后的整体；否则返回单个拆分）
    """
    # ---- 依赖 ----
    from unimol_tools import UniMolRepr
    from tqdm import tqdm
    from pathlib import Path
    import csv

    # ---- 路径准备 ----
    root_base = Path(root_base)               # e.g. .../split-by-sequence-identity-60/data
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _process_one_split(split_name: str):
        """
        针对某个具体 split（train/val/test）从其 data.mdb 提取 embedding，
        并保存到 out_dir/out_name_{split}.npz / .csv
        """
        lmdb_root = root_base / split_name
        npz_path = out_dir / f"{out_name}_{split_name}.npz"
        csv_path = out_dir / f"{out_name}_{split_name}.csv"

        # 缓存命中：直接读
        if npz_path.exists() and not force_refresh:
            data = np.load(npz_path, allow_pickle=True)
            return (data["lig_embeds"], data["pocket_embeds"], data["pkd"], data["ids"])

        # 初始化 Uni-Mol2（eval 推理）
        model = UniMolRepr(model="unimol2", model_size=model_size, use_cuda=True)
        model.model.eval()

        # 读取该 split 的 LMDB
        ds = da.LMDBDataset(str(lmdb_root))
        print(f"✅ [si-60|{split_name}] 读取 LMDB 样本数: {len(ds)}")

        lig_list, poc_list, y_list, id_list = [], [], [], []

        # 遍历样本并提取 3D 表示
        for i, sample in enumerate(tqdm(ds, desc=f"[si-60|{split_name}] Extracting", unit="sample")):
            try:
                lig_df = sample["atoms_ligand"]
                poc_df = sample["atoms_pocket"]

                lig_atoms = lig_df["element"].tolist()
                lig_coords = lig_df[["x", "y", "z"]].values.tolist()
                poc_atoms = poc_df["element"].tolist()
                poc_coords = poc_df[["x", "y", "z"]].values.tolist()

                if len(lig_atoms) == 0 or len(poc_atoms) == 0:
                    continue  # 跳过空样本

                lig_emb = model.get_repr({"atoms": lig_atoms, "coordinates": lig_coords}, return_tensor=True)
                poc_emb = model.get_repr({"atoms": poc_atoms, "coordinates": poc_coords}, return_tensor=True)

                lig_list.append(lig_emb.cpu().numpy())
                poc_list.append(poc_emb.cpu().numpy())
                y_list.append(float(sample["scores"]["neglog_aff"]))      # pKd
                id_list.append(sample.get("id", f"{split_name}_{i}"))

            except Exception as e:
                print(f"[WARN] {split_name} sample {i} failed: {e}")
                continue

        if len(lig_list) == 0 or len(poc_list) == 0:
            raise RuntimeError(f"[si-60|{split_name}] No valid samples extracted. Check LMDB at: {lmdb_root}")

        lig_embeds = np.vstack(lig_list)               # (N, D)
        poc_embeds = np.vstack(poc_list)               # (N, D)
        y_array    = np.asarray(y_list, dtype=np.float32)  # (N,)
        ids        = np.asarray(id_list)

        # 持久化
        np.savez(npz_path, lig_embeds=lig_embeds, pocket_embeds=poc_embeds, pkd=y_array, ids=ids)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "pKd", "lig_dim", "pocket_dim"])
            for sid, yv in zip(ids, y_array):
                w.writerow([sid, yv, lig_embeds.shape[1], poc_embeds.shape[1]])

        print(f"[ATOM3D-LBA|si-60|{split_name}] processed={len(y_array)}")
        print(f"  - npz: {npz_path}")
        print(f"  - csv: {csv_path}")

        return lig_embeds, poc_embeds, y_array, ids

    # 选择 split
    if split in ("train", "val", "test"):
        return _process_one_split(split)
    elif split == "all":
        parts = []
        for sp in ("train", "val", "test"):
            parts.append(_process_one_split(sp))
        lig_all = np.vstack([p[0] for p in parts])
        poc_all = np.vstack([p[1] for p in parts])
        y_all   = np.concatenate([p[2] for p in parts], axis=0)
        ids_all = np.concatenate([p[3] for p in parts], axis=0)
        return lig_all, poc_all, y_all, ids_all
    else:
        raise ValueError(f"split 必须是 'train'/'val'/'test'/'all' 之一，收到: {split!r}")

def LoadData_atom3d_si30_multimodal(
    root_base: str,                               # 行：指向 .../split-by-sequence-identity-30/data
    split: str = "train",                         # 行：要处理的拆分：'train'|'val'|'test'|'all'
    out_mm: str = "../dataset/ATOM3D/processed_mm_si30",  # 行：多模态缓存统一目录
    unimol2_size: str = "unimol2_small",         # 行：Uni-Mol2 模型规格（small/base/large）
    contact_threshold: float = 8.0,              # 行：2D 残基接触阈值（Å）
    dis_min: float = 1.0,                        # 行：TargetToGraph 距离下限（用于 1/d）
    prot_self_loop: bool = False,                # 行：蛋白残基图是否加自环
    bond_bidirectional: bool = True,             # 行：配体图是否双向边
    prefer_model: int = None,                    # 行：多 model（如 NMR/X-ray）时优先的 model id
    force_refresh: bool = False,                 # 行：是否忽略缓存并强制重建
    use_cuda_for_unimol: bool = True,            # 行：3D 抽取是否用 GPU
    use_pocket_for_1d2d: bool = False            # 行：关键：1D/2D 是否用口袋；默认 False = 用全蛋白
) -> dict:
    """
    目标：一次遍历 si-30 LMDB，构建严格对齐的 1D/2D/3D，任一模态失败→丢弃该样本，不写伪图/占位。
    本版：1D/2D 用全蛋白（atoms_protein），3D 仍按你的可跑通实现用口袋（atoms_pocket）。
    返回：{split: {'ids','y','smiles','seq','g_lig','g_prot','lig_3d','poc_3d'}} 或 {'all': {...}}
    """
    # —— 依赖 —— #
    from pathlib import Path                       # 行：路径安全操作
    import dgl                                     # 行：DGL 图存取
    from tqdm import tqdm                          # 行：进度条
    from unimol_tools import UniMolRepr            # 行：Uni-Mol2 表征器

    # —— 直接用你 util 里已有的工具函数 —— #
    # 这些函数你在上文 util 中均已实现：SmileToGraph / _build_contact_from_atoms / TargetToGraph / atoms_to_sequence

    # —— 原子符号→原子序号 的兜底工具（部分 unimol 版本需要） —— #
    _PERIODIC = {'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,'Na':11,'Mg':12,'Al':13,'Si':14,
                 'P':15,'S':16,'Cl':17,'Ar':18,'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,
                 'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Kr':36,'Rb':37,'Sr':38,'Y':39,'Zr':40,
                 'Nb':41,'Mo':42,'Tc':43,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,'In':49,'Sn':50,'Sb':51,'Te':52,'I':53,'Xe':54}
    def _to_atomic_numbers(sym_list):             # 行：把元素符号表转为原子序号表（极少数版本需要）
        return [int(_PERIODIC.get(s, 0)) for s in sym_list]

    # —— 规范路径 —— #
    root_base = Path(root_base)                    # 行：si-30 数据根（含 train/val/test）
    out_mm = Path(out_mm)                          # 行：缓存根目录
    out_mm.mkdir(parents=True, exist_ok=True)      # 行：若不存在则创建

    # —— 处理单拆分 —— #
    def _process_one_split(sp: str) -> dict:
        lmdb_root = root_base / sp                 # 行：该拆分 LMDB 目录（含 data.mdb）

        # 行：四类缓存（1D npz、2D 两个图、3D npz），四者齐全才算命中
        p_1dnpz = out_mm / f"si30_{sp}_1d.npz"                # 行：1D（ids/smiles/seq/pkd）
        p_2d_lig = out_mm / f"si30_{sp}_2d_lig.dgl.bin"        # 行：2D 配体图
        p_2d_pro = out_mm / f"si30_{sp}_2d_prot.dgl.bin"       # 行：2D 蛋白图（全蛋白）
        p_3dnpz = out_mm / f"si30_{sp}_3d.npz"                # 行：3D（lig/poc 向量 + pkd + ids）

        # —— 缓存直读（长度一致性校验） —— #
        if (not force_refresh) and p_1dnpz.exists() and p_2d_lig.exists() and p_2d_pro.exists() and p_3dnpz.exists():
            d1 = np.load(p_1dnpz, allow_pickle=True)          # 行：读 1D
            ids    = d1['ids']                                 # 行：样本 ID
            smiles = d1['smiles']                              # 行：SMILES
            seq    = d1['seq']                                 # 行：完整蛋白序列（来自 atoms_protein）
            y      = d1['pkd'].astype(np.float32)              # 行：标签 pKd

            g_lig, _  = dgl.load_graphs(str(p_2d_lig))        # 行：读 2D 配体图
            g_prot, _ = dgl.load_graphs(str(p_2d_pro))        # 行：读 2D 蛋白图（全蛋白）

            d3 = np.load(p_3dnpz, allow_pickle=True)          # 行：读 3D
            lig_3d = d3['lig_embeds'].astype(np.float32)       # 行：配体 3D 向量
            poc_3d = d3['pocket_embeds'].astype(np.float32)    # 行：口袋 3D 向量

            N = len(ids)                                       # 行：样本数
            assert len(smiles)==N and len(seq)==N and len(y)==N, "1D 长度不一致"       # 行：校验
            assert len(g_lig)==N and len(g_prot)==N, "2D 图数量与 1D 不一致"          # 行：校验
            assert lig_3d.shape[0]==N and poc_3d.shape[0]==N, "3D 行数与 1D 不一致"   # 行：校验

            return {                                           # 行：返回该拆分的统一包
                'ids': ids, 'y': y, 'smiles': smiles, 'seq': seq,
                'g_lig': g_lig, 'g_prot': g_prot,
                'lig_3d': lig_3d, 'poc_3d': poc_3d
            }

        # —— 正常路径：遍历 LMDB，1D/2D 用全蛋白，3D 用口袋 —— #
        ds = da.LMDBDataset(str(lmdb_root))                    # 行：加载该拆分的 LMDB
        print(f"[si-30|{sp}] LMDB samples = {len(ds)}")        # 行：信息提示

        model3d = UniMolRepr(                                  # 行：初始化 Uni-Mol2（与你可跑通版本一致）
            model="unimol2",                                   # 行：明确指定 unimol2
            model_size=unimol2_size,                           # 行：small/base/large
            use_cuda=use_cuda_for_unimol                       # 行：可用则用 CUDA
        )
        model3d.model.eval()                                   # 行：eval 模式（推理）

        source_1d2d = 'atoms_protein'                          # ★行：固定 1D/2D 用全蛋白（满足你的要求）

        # 行：收集容器（只留三模态全成功）
        ids_list, smiles_list, seq_list, y_list = [], [], [], []   # 行：1D
        g_lig_list, g_prot_list = [], []                           # 行：2D
        lig_vecs, poc_vecs = [], []                                # 行：3D

        # 行：统计器（定位失败原因）
        n_total = n_ok = n_fail = 0                                # 行：计数
        breakdown = {'missing_field':0, 'smiles':0, 'protein_ca':0, 'empty3d':0, 'unimol3d':0, 'other':0}  # 行：失败细分
        DEBUG_PRINT_N = 5                                          # 行：最多打印 5 条失败详情
        debug_shown = 0                                            # 行：已打印条数
        printed_backend = False                                    # 行：是否已打印一次后端信息

        for i, sample in enumerate(tqdm(ds, desc=f"[si-30|{sp}] tri-modal (1D/2D=protein, 3D=pocket)", unit="sample")):
            n_total += 1                                           # 行：总数+1
            try:
                # === 通用字段 ===
                smi = sample["smiles"]                              # 行：SMILES
                y   = float(sample["scores"]["neglog_aff"])         # 行：pKd
                sid = sample.get("id", f"{sp}_{i}")                 # 行：样本 ID

                # === 读取全蛋白原子表（1D/2D 来源） ===
                if source_1d2d not in sample:                       # 行：缺字段直接失败
                    breakdown['missing_field'] += 1
                    raise KeyError(f"missing_field:{source_1d2d}; have={sorted(sample.keys())}")
                df_prot = sample[source_1d2d]                       # 行：pandas DataFrame（全蛋白）

                # === 2D：配体原子图 ===
                lig_g = SmileToGraph(smi, bidirectional=bond_bidirectional)  # 行：SMILES→DGL 原子图

                # === 2D：全蛋白残基接触图（用 CA 距离阈值） ===
                cm, dm, ca, seq3 = _build_contact_from_atoms(       # 行：从全蛋白提 CA→接触矩阵/距离/序列
                    df_prot, threshold=contact_threshold, prefer_model=prefer_model
                )
                g_prot = TargetToGraph(                             # 行：构建蛋白残基图
                    contact_matrix=cm,
                    distance_matrix=dm,
                    ca_coords=ca,
                    seq3=seq3,
                    contact=1,
                    dis_min=dis_min,
                    self_loop=prot_self_loop
                )
                if g_prot.num_nodes() == 0:                         # 行：空图判失败
                    breakdown['protein_ca'] += 1
                    raise ValueError("empty protein graph (no CA)")

                # === 1D：从全蛋白导出完整一字母序列（非口袋） ===
                seq_full = atoms_to_sequence(df_prot, chain_id=None, longest_only=True) or ""  # 行：导出 AA 序列
                if not seq_full:                                    # 行：为空则失败
                    breakdown['protein_ca'] += 1
                    raise ValueError("empty target sequence from atoms_protein")

                # === 3D：仍用口袋与配体（保持你 3D 版本可跑通的设定） ===
                if "atoms_ligand" not in sample or "atoms_pocket" not in sample:               # 行：缺字段失败
                    breakdown['missing_field'] += 1
                    raise KeyError(f"missing_field:atoms_ligand/atoms_pocket; have={sorted(sample.keys())}")

                lig_df = sample["atoms_ligand"]                       # 行：配体原子表
                poc_df = sample["atoms_pocket"]                       # 行：口袋原子表
                lig_atoms = lig_df["element"].tolist()                # 行：配体元素
                lig_xyz   = lig_df[["x","y","z"]].values.tolist()     # 行：配体坐标
                poc_atoms = poc_df["element"].tolist()                # 行：口袋元素
                poc_xyz   = poc_df[["x","y","z"]].values.tolist()     # 行：口袋坐标
                if len(lig_atoms)==0 or len(poc_atoms)==0:            # 行：3D 空样本失败
                    breakdown['empty3d'] += 1
                    raise ValueError("empty ligand or pocket atoms")

                try:
                    lig_emb = model3d.get_repr({"atoms": lig_atoms, "coordinates": lig_xyz}, return_tensor=True)  # 行：配体 3D
                    poc_emb = model3d.get_repr({"atoms": poc_atoms, "coordinates": poc_xyz}, return_tensor=True)  # 行：口袋 3D
                except Exception:
                    # 行：兜底：若后端仅接受原子序号，则转换后重试
                    lig_emb = model3d.get_repr({"atoms": _to_atomic_numbers(lig_atoms), "coordinates": lig_xyz}, return_tensor=True)
                    poc_emb = model3d.get_repr({"atoms": _to_atomic_numbers(poc_atoms), "coordinates": poc_xyz}, return_tensor=True)

                if (not printed_backend):                              # 行：仅首次打印 backend/维度，确认是 Uni-Mol2
                    backend = type(model3d.model).__name__
                    print(f"[UniMol backend] {backend}")
                    print(f"[UniMol dims] ligand={int(lig_emb.shape[-1])}, pocket={int(poc_emb.shape[-1])}")
                    printed_backend = True

                # === 三模态全成功：写入收集器 ===
                ids_list.append(sid)                                   # 行：ID
                smiles_list.append(smi)                                # 行：SMILES
                seq_list.append(seq_full)                              # 行：完整蛋白序列
                y_list.append(y)                                       # 行：pKd
                g_lig_list.append(lig_g)                               # 行：2D 配体图
                g_prot_list.append(g_prot)                             # 行：2D 蛋白图（全蛋白）
                lig_vecs.append(lig_emb.cpu().numpy())                 # 行：3D 配体向量
                poc_vecs.append(poc_emb.cpu().numpy())                 # 行：3D 口袋向量
                n_ok += 1                                              # 行：成功+1

            except KeyError as e:                                      # 行：字段缺失
                n_fail += 1
                breakdown['missing_field'] += 1
                if debug_shown < DEBUG_PRINT_N:
                    print(f"[DEBUG][{sp}] KeyError at #{i} id={sample.get('id','')}: {e}")
                    debug_shown += 1
            except Exception as e:                                     # 行：其它异常
                n_fail += 1
                msg = str(e).lower()
                if "invalid smiles" in msg:
                    breakdown['smiles'] += 1
                elif "protein" in msg or "ca" in msg:
                    breakdown['protein_ca'] += 1
                elif "unimol" in msg or "repr" in msg:
                    breakdown['unimol3d'] += 1
                elif "empty ligand" in msg or "empty pocket" in msg:
                    breakdown['empty3d'] += 1
                else:
                    breakdown['other'] += 1
                if debug_shown < DEBUG_PRINT_N:
                    print(f"[DEBUG][{sp}] Exception at #{i} id={sample.get('id','')}: {e}")
                    debug_shown += 1

        # —— 至少要有一个成功样本 —— #
        if n_ok == 0:
            print(f"[FAIL][si-30|{sp}] breakdown: {breakdown}")       # 行：打印失败细分，便于排查
            raise RuntimeError(f"[si-30|{sp}] No tri-modal-valid samples. Check inputs at {lmdb_root}")

        # —— 列表→数组，并做一致性校验 —— #
        ids_arr    = np.array(ids_list,    dtype=object)               # 行：ID 数组
        smiles_arr = np.array(smiles_list, dtype=object)               # 行：SMILES 数组
        seq_arr    = np.array(seq_list,    dtype=object)               # 行：完整蛋白序列数组
        y_arr      = np.array(y_list,      dtype=np.float32)           # 行：pKd 数组
        lig_3d     = np.vstack(lig_vecs).astype(np.float32)            # 行：配体 3D 矩阵 [N,D]
        poc_3d     = np.vstack(poc_vecs).astype(np.float32)            # 行：口袋 3D 矩阵 [N,D]

        N = len(ids_arr)                                               # 行：成功样本数
        assert len(smiles_arr)==N and len(seq_arr)==N and len(y_arr)==N, "1D 长度不一致"       # 行：校验
        assert len(g_lig_list)==N and len(g_prot_list)==N, "2D 图数量与 1D 不一致"            # 行：校验
        assert lig_3d.shape[0]==N and poc_3d.shape[0]==N, "3D 行数与 1D 不一致"              # 行：校验

        print(f"[si-30|{sp}] total={n_total} | ok(tri-modal)={n_ok} | fail={n_fail} | breakdown={breakdown}")  # 行：统计输出

        # —— 落盘四份缓存（命名与之前保持一致风格） —— #
        np.savez(p_1dnpz, ids=ids_arr, smiles=smiles_arr, seq=seq_arr, pkd=y_arr)          # 行：1D npz
        dgl.save_graphs(str(p_2d_lig), g_lig_list)                                          # 行：2D 配体图
        dgl.save_graphs(str(p_2d_pro), g_prot_list)                                         # 行：2D 蛋白图（全蛋白）
        np.savez(p_3dnpz, ids=ids_arr, pkd=y_arr, lig_embeds=lig_3d, pocket_embeds=poc_3d) # 行：3D npz

        # —— 返回该拆分的统一数据包 —— #
        return {
            'ids': ids_arr, 'y': y_arr, 'smiles': smiles_arr, 'seq': seq_arr,
            'g_lig': g_lig_list, 'g_prot': g_prot_list,
            'lig_3d': lig_3d, 'poc_3d': poc_3d
        }

    # —— 顶层调度：单拆分 or 全部 —— #
    if split in ("train", "val", "test"):                            # 行：单拆分
        return {split: _process_one_split(split)}                    # 行：返回 {split: 包}
    elif split == "all":                                             # 行：三拆分合并
        parts = {sp: _process_one_split(sp) for sp in ("train","val","test")}  # 行：分别处理
        def _cat(*xs): return np.concatenate(xs, axis=0)            # 行：拼接工具
        all_pkg = {                                                  # 行：all 打包
            'ids'   : _cat(parts['train']['ids'], parts['val']['ids'], parts['test']['ids']),
            'y'     : _cat(parts['train']['y'],   parts['val']['y'],   parts['test']['y']).astype(np.float32),
            'smiles': _cat(parts['train']['smiles'], parts['val']['smiles'], parts['test']['smiles']),
            'seq'   : _cat(parts['train']['seq'],    parts['val']['seq'],    parts['test']['seq']),
            'g_lig' : parts['train']['g_lig'] + parts['val']['g_lig'] + parts['test']['g_lig'],
            'g_prot': parts['train']['g_prot']+ parts['val']['g_prot']+ parts['test']['g_prot'],
            'lig_3d': np.vstack([parts['train']['lig_3d'], parts['val']['lig_3d'], parts['test']['lig_3d']]).astype(np.float32),
            'poc_3d': np.vstack([parts['train']['poc_3d'], parts['val']['poc_3d'], parts['test']['poc_3d']]).astype(np.float32),
        }
        return {'all': all_pkg}                                      # 行：返回 {'all': 包}
    else:
        raise ValueError(f"split 必须是 'train'/'val'/'test'/'all' 之一，收到: {split!r}")  # 行：参数校验


def LoadData_atom3d_si60_multimodal(
    root_base: str,                               # 行：指向 .../split-by-sequence-identity-60/data
    split: str = "train",                         # 行：要处理的拆分：'train'|'val'|'test'|'all'
    out_mm: str = "../dataset/ATOM3D/processed_mm_si60",  # 行：多模态缓存统一目录（si60）
    unimol2_size: str = "unimol2_small",         # 行：Uni-Mol2 模型规格（small/base/large）
    contact_threshold: float = 8.0,              # 行：2D 残基接触阈值（Å）
    dis_min: float = 1.0,                        # 行：TargetToGraph 距离下限（用于 1/d）
    prot_self_loop: bool = False,                # 行：蛋白残基图是否加自环
    bond_bidirectional: bool = True,             # 行：配体图是否双向边
    prefer_model: int = None,                    # 行：多 model（如 NMR/X-ray）时优先的 model id
    force_refresh: bool = False,                 # 行：是否忽略缓存并强制重建
    use_cuda_for_unimol: bool = True,            # 行：3D 抽取是否用 GPU
    use_pocket_for_1d2d: bool = False            # 行：关键：1D/2D 是否用口袋；默认 False = 用全蛋白
) -> dict:
    """
    目标：一次遍历 si-60 LMDB，构建严格对齐的 1D/2D/3D，任一模态失败→丢弃该样本，不写伪图/占位。
    本版：默认 1D/2D 用全蛋白（atoms_protein），3D 用口袋（atoms_pocket）；可用 use_pocket_for_1d2d 改成口袋。
    返回：{split: {'ids','y','smiles','seq','g_lig','g_prot','lig_3d','poc_3d'}} 或 {'all': {...}}
    """
    # —— 依赖 —— #
    from pathlib import Path                       # 行：路径安全操作
    import numpy as np                             # 行：数值与保存 npz
    import dgl                                     # 行：DGL 图存取
    from tqdm import tqdm                          # 行：进度条
    import atom3d.datasets as da                   # 行：ATOM3D 官方 LMDBDataset
    from unimol_tools import UniMolRepr            # 行：Uni-Mol2 表征器
    # —— 你 util 里已有的工具函数（若命名不同请相应修改）—— #
    from util import SmileToGraph, _build_contact_from_atoms, TargetToGraph, atoms_to_sequence

    # —— 原子符号→原子序号 的兜底工具（部分 unimol 版本需要） —— #
    _PERIODIC = {'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,
                 'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,'K':19,'Ca':20,
                 'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,'Zn':30,
                 'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Kr':36,'Rb':37,'Sr':38,'Y':39,'Zr':40,
                 'Nb':41,'Mo':42,'Tc':43,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,'In':49,'Sn':50,
                 'Sb':51,'Te':52,'I':53,'Xe':54}
    def _to_atomic_numbers(sym_list):             # 行：把元素符号表转为原子序号表（极少数版本需要）
        return [int(_PERIODIC.get(s, 0)) for s in sym_list]

    # —— 规范路径 —— #
    root_base = Path(root_base)                    # 行：si-60 数据根（含 train/val/test）
    out_mm = Path(out_mm)                          # 行：缓存根目录（si60）
    out_mm.mkdir(parents=True, exist_ok=True)      # 行：若不存在则创建

    # —— 单拆分处理函数 —— #
    def _process_one_split(sp: str) -> dict:
        lmdb_root = root_base / sp                 # 行：该拆分 LMDB 目录（含 data.mdb）

        # 行：四类缓存（1D npz、2D 两个图、3D npz），四者齐全才算命中
        p_1dnpz = out_mm / f"si60_{sp}_1d.npz"                # 行：1D（ids/smiles/seq/pkd）
        p_2d_lig = out_mm / f"si60_{sp}_2d_lig.dgl.bin"       # 行：2D 配体图
        p_2d_pro = out_mm / f"si60_{sp}_2d_prot.dgl.bin"      # 行：2D 蛋白图（全蛋白或口袋，取决于配置）
        p_3dnpz = out_mm / f"si60_{sp}_3d.npz"                # 行：3D（lig/poc 向量 + pkd + ids）

        # —— 缓存直读（长度一致性校验） —— #
        if (not force_refresh) and p_1dnpz.exists() and p_2d_lig.exists() and p_2d_pro.exists() and p_3dnpz.exists():
            d1 = np.load(p_1dnpz, allow_pickle=True)          # 行：读 1D
            ids    = d1['ids']                                 # 行：样本 ID
            smiles = d1['smiles']                              # 行：SMILES
            seq    = d1['seq']                                 # 行：蛋白序列（全蛋白或口袋）
            y      = d1['pkd'].astype(np.float32)              # 行：标签 pKd

            g_lig, _  = dgl.load_graphs(str(p_2d_lig))        # 行：读 2D 配体图
            g_prot, _ = dgl.load_graphs(str(p_2d_pro))        # 行：读 2D 蛋白图

            d3 = np.load(p_3dnpz, allow_pickle=True)          # 行：读 3D
            lig_3d = d3['lig_embeds'].astype(np.float32)       # 行：配体 3D 向量
            poc_3d = d3['pocket_embeds'].astype(np.float32)    # 行：口袋 3D 向量

            N = len(ids)                                       # 行：样本数
            assert len(smiles)==N and len(seq)==N and len(y)==N, "1D 长度不一致"       # 行：校验
            assert len(g_lig)==N and len(g_prot)==N, "2D 图数量与 1D 不一致"          # 行：校验
            assert lig_3d.shape[0]==N and poc_3d.shape[0]==N, "3D 行数与 1D 不一致"   # 行：校验

            return {                                           # 行：返回该拆分的统一包
                'ids': ids, 'y': y, 'smiles': smiles, 'seq': seq,
                'g_lig': g_lig, 'g_prot': g_prot,
                'lig_3d': lig_3d, 'poc_3d': poc_3d
            }

        # —— 正常路径：遍历 LMDB —— #
        ds = da.LMDBDataset(str(lmdb_root))                    # 行：加载该拆分的 LMDB
        print(f"[si-60|{sp}] LMDB samples = {len(ds)}")        # 行：信息提示

        model3d = UniMolRepr(                                  # 行：初始化 Uni-Mol2（与你可跑通版本一致）
            model="unimol2",                                   # 行：明确指定 unimol2
            model_size=unimol2_size,                           # 行：small/base/large
            use_cuda=use_cuda_for_unimol                       # 行：可用则用 CUDA
        )
        model3d.model.eval()                                   # 行：eval 模式（推理）

        source_1d2d = 'atoms_pocket' if use_pocket_for_1d2d else 'atoms_protein'  # 行：1D/2D 来源选择

        # 行：收集容器（只留三模态全成功）
        ids_list, smiles_list, seq_list, y_list = [], [], [], []   # 行：1D
        g_lig_list, g_prot_list = [], []                           # 行：2D
        lig_vecs, poc_vecs = [], []                                # 行：3D

        # 行：统计器（定位失败原因）
        n_total = n_ok = n_fail = 0                                # 行：计数
        breakdown = {'missing_field':0, 'smiles':0, 'protein_ca':0, 'empty3d':0, 'unimol3d':0, 'other':0}  # 行：失败细分
        DEBUG_PRINT_N = 5                                          # 行：最多打印 5 条失败详情
        debug_shown = 0                                            # 行：已打印条数
        printed_backend = False                                    # 行：是否已打印一次后端信息

        for i, sample in enumerate(tqdm(ds, desc=f"[si-60|{sp}] tri-modal (1D/2D={source_1d2d}, 3D=pocket)", unit="sample")):
            n_total += 1                                           # 行：总数+1
            try:
                # === 通用字段 ===
                smi = sample["smiles"]                              # 行：SMILES
                y   = float(sample["scores"]["neglog_aff"])         # 行：pKd
                sid = sample.get("id", f"{sp}_{i}")                 # 行：样本 ID

                # === 读取 1D/2D 来源原子表（全蛋白或口袋） ===
                if source_1d2d not in sample:                       # 行：缺字段直接失败
                    breakdown['missing_field'] += 1
                    raise KeyError(f"missing_field:{source_1d2d}; have={sorted(sample.keys())}")
                df_for_1d2d = sample[source_1d2d]                   # 行：pandas DataFrame（全蛋白或口袋）

                # === 2D：配体原子图 ===
                lig_g = SmileToGraph(smi, bidirectional=bond_bidirectional)  # 行：SMILES→DGL 原子图

                # === 2D：残基接触图（从来源原子表计算 CA 接触） ===
                cm, dm, ca, seq3 = _build_contact_from_atoms(       # 行：从 df_for_1d2d 提 CA→接触矩阵/距离/序列
                    df_for_1d2d, threshold=contact_threshold, prefer_model=prefer_model
                )
                g_prot = TargetToGraph(                             # 行：构建蛋白残基图
                    contact_matrix=cm,
                    distance_matrix=dm,
                    ca_coords=ca,
                    seq3=seq3,
                    contact=1,
                    dis_min=dis_min,
                    self_loop=prot_self_loop
                )
                if g_prot.num_nodes() == 0:                         # 行：空图判失败
                    breakdown['protein_ca'] += 1
                    raise ValueError("empty protein graph (no CA)")

                # === 1D：由来源原子表导出一字母序列 ===
                seq_full = atoms_to_sequence(df_for_1d2d, chain_id=None, longest_only=True) or ""  # 行：导出 AA 序列
                if not seq_full:                                    # 行：为空则失败
                    breakdown['protein_ca'] += 1
                    raise ValueError("empty target sequence from atoms")

                # === 3D：固定用口袋与配体（与 si-30 版一致） ===
                if "atoms_ligand" not in sample or "atoms_pocket" not in sample:               # 行：缺字段失败
                    breakdown['missing_field'] += 1
                    raise KeyError(f"missing_field:atoms_ligand/atoms_pocket; have={sorted(sample.keys())}")

                lig_df = sample["atoms_ligand"]                       # 行：配体原子表
                poc_df = sample["atoms_pocket"]                       # 行：口袋原子表
                lig_atoms = lig_df["element"].tolist()                # 行：配体元素
                lig_xyz   = lig_df[["x","y","z"]].values.tolist()     # 行：配体坐标
                poc_atoms = poc_df["element"].tolist()                # 行：口袋元素
                poc_xyz   = poc_df[["x","y","z"]].values.tolist()     # 行：口袋坐标
                if len(lig_atoms)==0 or len(poc_atoms)==0:            # 行：3D 空样本失败
                    breakdown['empty3d'] += 1
                    raise ValueError("empty ligand or pocket atoms")

                try:
                    lig_emb = model3d.get_repr({"atoms": lig_atoms, "coordinates": lig_xyz}, return_tensor=True)  # 行：配体 3D
                    poc_emb = model3d.get_repr({"atoms": poc_atoms, "coordinates": poc_xyz}, return_tensor=True)  # 行：口袋 3D
                except Exception:
                    # 行：兜底：若后端仅接受原子序号，则转换后重试
                    lig_emb = model3d.get_repr({"atoms": _to_atomic_numbers(lig_atoms), "coordinates": lig_xyz}, return_tensor=True)
                    poc_emb = model3d.get_repr({"atoms": _to_atomic_numbers(poc_atoms), "coordinates": poc_xyz}, return_tensor=True)

                if (not printed_backend):                              # 行：仅首次打印 backend/维度，确认是 Uni-Mol2
                    backend = type(model3d.model).__name__
                    print(f"[UniMol backend] {backend}")
                    print(f"[UniMol dims] ligand={int(lig_emb.shape[-1])}, pocket={int(poc_emb.shape[-1])}")
                    printed_backend = True

                # === 三模态全成功：写入收集器 ===
                ids_list.append(sid)                                   # 行：ID
                smiles_list.append(smi)                                # 行：SMILES
                seq_list.append(seq_full)                              # 行：蛋白序列
                y_list.append(y)                                       # 行：pKd
                g_lig_list.append(lig_g)                               # 行：2D 配体图
                g_prot_list.append(g_prot)                             # 行：2D 蛋白图
                lig_vecs.append(lig_emb.cpu().numpy())                 # 行：3D 配体向量
                poc_vecs.append(poc_emb.cpu().numpy())                 # 行：3D 口袋向量
                n_ok += 1                                              # 行：成功+1

            except KeyError as e:                                      # 行：字段缺失
                n_fail += 1
                breakdown['missing_field'] += 1
                if debug_shown < DEBUG_PRINT_N:
                    print(f"[DEBUG][{sp}] KeyError at #{i} id={sample.get('id','')}: {e}")
                    debug_shown += 1
            except Exception as e:                                     # 行：其它异常
                n_fail += 1
                msg = str(e).lower()
                if "invalid smiles" in msg:
                    breakdown['smiles'] += 1
                elif "protein" in msg or "ca" in msg:
                    breakdown['protein_ca'] += 1
                elif "unimol" in msg or "repr" in msg:
                    breakdown['unimol3d'] += 1
                elif "empty ligand" in msg or "empty pocket" in msg:
                    breakdown['empty3d'] += 1
                else:
                    breakdown['other'] += 1
                if debug_shown < DEBUG_PRINT_N:
                    print(f"[DEBUG][{sp}] Exception at #{i} id={sample.get('id','')}: {e}")
                    debug_shown += 1

        # —— 至少要有一个成功样本 —— #
        if n_ok == 0:
            print(f"[FAIL][si-60|{sp}] breakdown: {breakdown}")       # 行：打印失败细分
            raise RuntimeError(f"[si-60|{sp}] No tri-modal-valid samples. Check inputs at {lmdb_root}")

        # —— 列表→数组，并做一致性校验 —— #
        ids_arr    = np.array(ids_list,    dtype=object)               # 行：ID 数组
        smiles_arr = np.array(smiles_list, dtype=object)               # 行：SMILES 数组
        seq_arr    = np.array(seq_list,    dtype=object)               # 行：蛋白序列数组
        y_arr      = np.array(y_list,      dtype=np.float32)           # 行：pKd 数组
        lig_3d     = np.vstack(lig_vecs).astype(np.float32)            # 行：配体 3D 矩阵 [N,D]
        poc_3d     = np.vstack(poc_vecs).astype(np.float32)            # 行：口袋 3D 矩阵 [N,D]

        N = len(ids_arr)                                               # 行：成功样本数
        assert len(smiles_arr)==N and len(seq_arr)==N and len(y_arr)==N, "1D 长度不一致"       # 行：校验
        assert len(g_lig_list)==N and len(g_prot_list)==N, "2D 图数量与 1D 不一致"            # 行：校验
        assert lig_3d.shape[0]==N and poc_3d.shape[0]==N, "3D 行数与 1D 不一致"              # 行：校验

        print(f"[si-60|{sp}] total={n_total} | ok(tri-modal)={n_ok} | fail={n_fail} | breakdown={breakdown}")  # 行：统计输出

        # —— 落盘四份缓存（命名与 si60 保持一致风格） —— #
        np.savez(p_1dnpz, ids=ids_arr, smiles=smiles_arr, seq=seq_arr, pkd=y_arr)          # 行：1D npz
        dgl.save_graphs(str(p_2d_lig), g_lig_list)                                          # 行：2D 配体图
        dgl.save_graphs(str(p_2d_pro), g_prot_list)                                         # 行：2D 蛋白图
        np.savez(p_3dnpz, ids=ids_arr, pkd=y_arr, lig_embeds=lig_3d, pocket_embeds=poc_3d) # 行：3D npz

        # —— 返回该拆分的统一数据包 —— #
        return {
            'ids': ids_arr, 'y': y_arr, 'smiles': smiles_arr, 'seq': seq_arr,
            'g_lig': g_lig_list, 'g_prot': g_prot_list,
            'lig_3d': lig_3d, 'poc_3d': poc_3d
        }

    # —— 顶层调度：单拆分 or 全部 —— #
    if split in ("train", "val", "test"):                            # 行：单拆分
        return {split: _process_one_split(split)}                    # 行：返回 {split: 包}
    elif split == "all":                                             # 行：三拆分合并
        parts = {sp: _process_one_split(sp) for sp in ("train","val","test")}  # 行：分别处理
        def _cat(*xs):                                               # 行：拼接工具（1D 字段）
            import numpy as _np
            return _np.concatenate(xs, axis=0)
        all_pkg = {                                                  # 行：all 打包
            'ids'   : _cat(parts['train']['ids'], parts['val']['ids'], parts['test']['ids']),
            'y'     : _cat(parts['train']['y'],   parts['val']['y'],   parts['test']['y']).astype(np.float32),
            'smiles': _cat(parts['train']['smiles'], parts['val']['smiles'], parts['test']['smiles']),
            'seq'   : _cat(parts['train']['seq'],    parts['val']['seq'],    parts['test']['seq']),
            'g_lig' : parts['train']['g_lig'] + parts['val']['g_lig'] + parts['test']['g_lig'],
            'g_prot': parts['train']['g_prot']+ parts['val']['g_prot']+ parts['test']['g_prot'],
            'lig_3d': np.vstack([parts['train']['lig_3d'], parts['val']['lig_3d'], parts['test']['lig_3d']]).astype(np.float32),
            'poc_3d': np.vstack([parts['train']['poc_3d'], parts['val']['poc_3d'], parts['test']['poc_3d']]).astype(np.float32),
        }
        return {'all': all_pkg}                                      # 行：返回 {'all': 包}
    else:
        raise ValueError(f"split 必须是 'train'/'val'/'test'/'all' 之一，收到: {split!r}")  # 行：参数校验



def LoadData_atom3d_si30_multimodal_lm(
    root_base: str,                               # 行：指向 .../split-by-sequence-identity-30/data
    split: str = "train",                         # 行：'train'|'val'|'test'|'all'
    out_mm: str = "../dataset/ATOM3D/processed_mm_si30_lm",  # 行：新的多模态+LM 缓存目录
    unimol2_size: str = "unimol2_small",         # 行：Uni-Mol2 模型规格
    # 下面这些 2D 相关参数保留占位（不再使用），避免外部调用报错
    contact_threshold: float = 8.0,
    dis_min: float = 1.0,
    prot_self_loop: bool = False,
    bond_bidirectional: bool = True,
    prefer_model: int = None,
    force_refresh: bool = False,                 # 行：是否忽略缓存并强制重建
    use_cuda_for_unimol: bool = True,            # 行：Uni-Mol2 是否用 GPU
    # --- LM 模型相关 ---
    esm2_model_name: str = "facebook/esm2_t33_650M_UR50D",    # 行：ESM-2 checkpoint
    chemberta_model_name: str = "DeepChem/ChemBERTa-77M-MTR", # 行：ChemBERTa-2 checkpoint
    lm_batch_size: int = 32,                                   # 行：LM 前向的 batch_size
    use_safetensors: bool = True                                # 行：是否优先使用 safetensors（避免 torch 版本限制）
) -> dict:
    """
    精简版 LoadData_atom3d_si30_multimodal：
      - 1D：离线用 ESM-2 / ChemBERTa-2 抽取：
          drug_lm  : ChemBERTa-2(SMILES) 向量 [N, D_d]
          prot_lm  : ESM-2(蛋白序列) 向量 [N, D_p]
      - 3D：用 Uni-Mol2 抽取配体 / 口袋向量：
          lig_3d   : Uni-Mol2(ligand) 向量 [N, D_3d]
          poc_3d   : Uni-Mol2(pocket) 向量 [N, D_3d]
      - 2D：已移除；返回结果中的 g_lig, g_prot 恒为 None，便于与旧代码接口兼容。

    缓存：
      si30_{split}_1d_lm.npz  （1D LM 特征 + 标签）
      si30_{split}_3d.npz     （3D Uni-Mol2 特征）

    返回：
        {split: {'ids','y','smiles','seq',
                 'drug_lm','prot_lm',
                 'g_lig','g_prot',      # 恒为 None
                 'lig_3d','poc_3d'}}
    """
    # —— 依赖 —— #
    from pathlib import Path                                  # 行：路径处理
    import numpy as np                                        # 行：数组/保存
    import torch                                              # 行：torch（用于 LM 推理 / 设备判断）
    from tqdm import tqdm                                     # 行：进度条显示
    import atom3d.datasets as da                              # 行：ATOM3D LMDB 读
    from unimol_tools import UniMolRepr                       # 行：Uni-Mol2 表征器
    from transformers import AutoTokenizer, AutoModel         # 行：HF tokenizer/model（Auto）
    # util 里的工具函数：这里只需要 atoms_to_sequence
    from util import atoms_to_sequence

    # —— 原子符号→原子序号 的兜底工具（部分 unimol 版本需要） —— #
    _PERIODIC = {
        'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,'Na':11,'Mg':12,'Al':13,'Si':14,
        'P':15,'S':16,'Cl':17,'Ar':18,'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,
        'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Kr':36,'Rb':37,'Sr':38,'Y':39,'Zr':40,
        'Nb':41,'Mo':42,'Tc':43,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,'In':49,'Sn':50,'Sb':51,'Te':52,'I':53,'Xe':54
    }
    def _to_atomic_numbers(sym_list):
        return [int(_PERIODIC.get(s, 0)) for s in sym_list]

    # —— LM 编码小工具 —— #
    def _encode_text_list(text_list, tokenizer, model, device, batch_size=32):
        """输入 list[str]，输出 [N, D] numpy.float32 向量（首 token 表示）"""
        all_vecs = []
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(text_list), batch_size), desc="[LM] encoding", unit="batch"):
                batch = text_list[i:i+batch_size]
                enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                try:
                    enc = enc.to(device)
                except Exception:
                    for k, v in enc.items():
                        if isinstance(v, torch.Tensor):
                            enc[k] = v.to(device)
                out = model(**enc)
                if hasattr(out, "last_hidden_state"):
                    hs = out.last_hidden_state
                elif isinstance(out, (tuple, list)) and len(out) > 0 and hasattr(out[0], "last_hidden_state"):
                    hs = out[0].last_hidden_state
                elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                    hs = out[0]
                else:
                    raise RuntimeError("Unsupported model output structure for LM encoding.")
                CLS = hs[:, 0, :].cpu().numpy().astype(np.float32)
                all_vecs.append(CLS)
        if len(all_vecs) == 0:
            return np.zeros((0, model.config.hidden_size), dtype=np.float32)
        return np.concatenate(all_vecs, axis=0)

    # —— HF 模型加载辅助 —— #
    def _hf_load_model(name: str, device: torch.device, use_safetensors: bool = True):
        try:
            model = AutoModel.from_pretrained(name, use_safetensors=use_safetensors)
            model.to(device)
            return model
        except TypeError as e:
            try:
                model = AutoModel.from_pretrained(name)
                model.to(device)
                return model
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load model '{name}' (tried with/without use_safetensors). "
                    f"Err1={e} Err2={e2}"
                ) from e2
        except Exception as e:
            msg = str(e).lower()
            if ("torch" in msg and "2.6" in msg) or ("vulnerability" in msg) or ("cve" in msg):
                raise RuntimeError(
                    f"Loading HF model '{name}' failed due to transformers' torch-version safety check: {e}\n"
                    "Possible solutions:\n"
                    "  1) 保持现有 torch，优先使用 safetensors，并确保模型仓库提供 .safetensors；\n"
                    "  2) 升级 torch 到 >=2.6；\n"
                    "  3) 在受信环境使用旧版 transformers（不推荐）。"
                ) from e
            try:
                model = AutoModel.from_pretrained(name)
                model.to(device)
                return model
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load model '{name}'. Tried safetensors and non-safetensors. "
                    f"Err1={e} Err2={e2}"
                ) from e2

    # —— 规范路径 —— #
    root_base = Path(root_base)
    out_mm = Path(out_mm)
    out_mm.mkdir(parents=True, exist_ok=True)

    # —— 处理单拆分 —— #
    def _process_one_split(sp: str) -> dict:
        lmdb_root = root_base / sp

        # 仅保留 1D + 3D 缓存
        p_1dnpz = out_mm / f"si30_{sp}_1d_lm.npz"
        p_3dnpz = out_mm / f"si30_{sp}_3d.npz"

        # —— 缓存直读 —— #
        if (not force_refresh) and p_1dnpz.exists() and p_3dnpz.exists():
            d1 = np.load(p_1dnpz, allow_pickle=True)
            ids    = d1['ids']
            smiles = d1['smiles']
            seq    = d1['seq']
            y      = d1['pkd'].astype(np.float32)
            drug_lm = d1['drug_lm'].astype(np.float32)
            prot_lm = d1['prot_lm'].astype(np.float32)

            d3 = np.load(p_3dnpz, allow_pickle=True)
            lig_3d = d3['lig_embeds'].astype(np.float32)
            poc_3d = d3['pocket_embeds'].astype(np.float32)

            N = len(ids)
            assert len(smiles) == N and len(seq) == N and len(y) == N, "1D 长度不一致"
            assert lig_3d.shape[0] == N and poc_3d.shape[0] == N, "3D 行数与 1D 不一致"
            assert drug_lm.shape[0] == N and prot_lm.shape[0] == N, "LM 行数与 1D 不一致"

            return {
                'ids': ids, 'y': y,
                'smiles': smiles, 'seq': seq,
                'drug_lm': drug_lm, 'prot_lm': prot_lm,
                'g_lig': None, 'g_prot': None,   # 2D 已移除
                'lig_3d': lig_3d, 'poc_3d': poc_3d
            }

        # —— 正常路径：遍历 LMDB，构建 1D + 3D —— #
        ds = da.LMDBDataset(str(lmdb_root))
        print(f"[si-30|{sp}] LMDB samples = {len(ds)}")

        model3d = UniMolRepr(
            model="unimol2",
            model_size=unimol2_size,
            use_cuda=use_cuda_for_unimol
        )
        model3d.model.eval()

        source_atoms = 'atoms_protein'

        ids_list, smiles_list, seq_list, y_list = [], [], [], []
        lig_vecs, poc_vecs = [], []

        n_total = n_ok = n_fail = 0
        breakdown = {'missing_field':0, 'smiles':0, 'protein_seq':0, 'empty3d':0, 'unimol3d':0, 'other':0}
        DEBUG_PRINT_N = 5
        debug_shown = 0
        printed_backend = False

        for i, sample in enumerate(
            tqdm(ds, desc=f"[si-30|{sp}] bi-modal (1D=LM, 3D=pocket)", unit="sample")
        ):
            n_total += 1
            try:
                smi = sample["smiles"]
                y   = float(sample["scores"]["neglog_aff"])
                sid = sample.get("id", f"{sp}_{i}")

                # --- 1D：从全蛋白原子导出完整氨基酸序列 ---
                if source_atoms not in sample:
                    breakdown['missing_field'] += 1
                    raise KeyError(f"missing_field:{source_atoms}; have={sorted(sample.keys())}")
                df_prot = sample[source_atoms]
                seq_full = atoms_to_sequence(df_prot, chain_id=None, longest_only=True) or ""
                if not seq_full:
                    breakdown['protein_seq'] += 1
                    raise ValueError("empty target sequence from atoms_protein")

                # --- 3D：口袋 + 配体 ---
                if "atoms_ligand" not in sample or "atoms_pocket" not in sample:
                    breakdown['missing_field'] += 1
                    raise KeyError(f"missing_field:atoms_ligand/atoms_pocket; have={sorted(sample.keys())}")

                lig_df = sample["atoms_ligand"]
                poc_df = sample["atoms_pocket"]
                lig_atoms = lig_df["element"].tolist()
                lig_xyz   = lig_df[["x","y","z"]].values.tolist()
                poc_atoms = poc_df["element"].tolist()
                poc_xyz   = poc_df[["x","y","z"]].values.tolist()
                if len(lig_atoms) == 0 or len(poc_atoms) == 0:
                    breakdown['empty3d'] += 1
                    raise ValueError("empty ligand or pocket atoms")

                try:
                    lig_emb = model3d.get_repr({"atoms": lig_atoms, "coordinates": lig_xyz}, return_tensor=True)
                    poc_emb = model3d.get_repr({"atoms": poc_atoms, "coordinates": poc_xyz}, return_tensor=True)
                except Exception:
                    lig_emb = model3d.get_repr(
                        {"atoms": _to_atomic_numbers(lig_atoms), "coordinates": lig_xyz},
                        return_tensor=True
                    )
                    poc_emb = model3d.get_repr(
                        {"atoms": _to_atomic_numbers(poc_atoms), "coordinates": poc_xyz},
                        return_tensor=True
                    )

                if (not printed_backend):
                    backend = type(model3d.model).__name__
                    print(f"[UniMol backend] {backend}")
                    print(f"[UniMol dims] ligand={int(lig_emb.shape[-1])}, pocket={int(poc_emb.shape[-1])}")
                    printed_backend = True

                ids_list.append(sid)
                smiles_list.append(smi)
                seq_list.append(seq_full)
                y_list.append(y)
                lig_vecs.append(lig_emb.cpu().numpy())
                poc_vecs.append(poc_emb.cpu().numpy())
                n_ok += 1

            except KeyError as e:
                n_fail += 1
                breakdown['missing_field'] += 1
                if debug_shown < DEBUG_PRINT_N:
                    print(f"[DEBUG][{sp}] KeyError at #{i} id={sample.get('id','')}: {e}")
                    debug_shown += 1
            except Exception as e:
                n_fail += 1
                msg = str(e).lower()
                if "invalid smiles" in msg:
                    breakdown['smiles'] += 1
                elif "sequence" in msg:
                    breakdown['protein_seq'] += 1
                elif "unimol" in msg or "repr" in msg:
                    breakdown['unimol3d'] += 1
                elif "empty ligand" in msg or "empty pocket" in msg:
                    breakdown['empty3d'] += 1
                else:
                    breakdown['other'] += 1
                if debug_shown < DEBUG_PRINT_N:
                    print(f"[DEBUG][{sp}] Exception at #{i} id={sample.get('id','')}: {e}")
                    debug_shown += 1

        if n_ok == 0:
            print(f"[FAIL][si-30|{sp}] breakdown: {breakdown}")
            raise RuntimeError(f"[si-30|{sp}] No valid samples. Check inputs at {lmdb_root}")

        ids_arr    = np.array(ids_list,    dtype=object)
        smiles_arr = np.array(smiles_list, dtype=object)
        seq_arr    = np.array(seq_list,    dtype=object)
        y_arr      = np.array(y_list,      dtype=np.float32)
        lig_3d     = np.vstack(lig_vecs).astype(np.float32)
        poc_3d     = np.vstack(poc_vecs).astype(np.float32)

        N = len(ids_arr)
        assert len(smiles_arr) == N and len(seq_arr) == N and len(y_arr) == N, "1D 长度不一致"
        assert lig_3d.shape[0] == N and poc_3d.shape[0] == N, "3D 行数与 1D 不一致"

        print(f"[si-30|{sp}] total={n_total} | ok={n_ok} | fail={n_fail} | breakdown={breakdown}")

        # —— LM 抽取 1D 向量 —— #
        lm_device = torch.device("cuda" if (use_cuda_for_unimol and torch.cuda.is_available()) else "cpu")
        print(f"[LM] using device: {lm_device}")

        print(f"[LM] loading ChemBERTa model: {chemberta_model_name} (use_safetensors={use_safetensors})")
        chem_tok = AutoTokenizer.from_pretrained(chemberta_model_name)
        chem_model = _hf_load_model(chemberta_model_name, lm_device, use_safetensors)
        drug_lm = _encode_text_list(smiles_arr.tolist(), chem_tok, chem_model, lm_device, batch_size=lm_batch_size)
        try:
            del chem_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        print(f"[LM] loading ESM model: {esm2_model_name} (use_safetensors={use_safetensors})")
        esm_tok = AutoTokenizer.from_pretrained(esm2_model_name)
        esm_model = _hf_load_model(esm2_model_name, lm_device, use_safetensors)
        prot_lm = _encode_text_list(seq_arr.tolist(), esm_tok, esm_model, lm_device, batch_size=lm_batch_size)
        try:
            del esm_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        assert drug_lm.shape[0] == N and prot_lm.shape[0] == N, "LM 编码行数与样本数不一致"

        # —— 缓存到磁盘 —— #
        np.savez(
            p_1dnpz,
            ids=ids_arr, smiles=smiles_arr, seq=seq_arr,
            pkd=y_arr, drug_lm=drug_lm, prot_lm=prot_lm
        )
        np.savez(
            p_3dnpz,
            ids=ids_arr, pkd=y_arr,
            lig_embeds=lig_3d, pocket_embeds=poc_3d
        )

        # —— 返回统一数据包（g_lig/g_prot 为 None） —— #
        return {
            'ids': ids_arr, 'y': y_arr,
            'smiles': smiles_arr, 'seq': seq_arr,
            'drug_lm': drug_lm, 'prot_lm': prot_lm,
            'g_lig': None, 'g_prot': None,
            'lig_3d': lig_3d, 'poc_3d': poc_3d
        }

    # —— 顶层调度 —— #
    if split in ("train", "val", "test"):
        return {split: _process_one_split(split)}
    elif split == "all":
        parts = {sp: _process_one_split(sp) for sp in ("train","val","test")}
        def _cat(*xs): return np.concatenate(xs, axis=0)
        all_pkg = {
            'ids'    : _cat(parts['train']['ids'],    parts['val']['ids'],    parts['test']['ids']),
            'y'      : _cat(parts['train']['y'],      parts['val']['y'],      parts['test']['y']).astype(np.float32),
            'smiles' : _cat(parts['train']['smiles'], parts['val']['smiles'], parts['test']['smiles']),
            'seq'    : _cat(parts['train']['seq'],    parts['val']['seq'],    parts['test']['seq']),
            'drug_lm': np.vstack([
                parts['train']['drug_lm'],
                parts['val']['drug_lm'],
                parts['test']['drug_lm']
            ]).astype(np.float32),
            'prot_lm': np.vstack([
                parts['train']['prot_lm'],
                parts['val']['prot_lm'],
                parts['test']['prot_lm']
            ]).astype(np.float32),
            'g_lig' : None,
            'g_prot': None,
            'lig_3d': np.vstack([
                parts['train']['lig_3d'],
                parts['val']['lig_3d'],
                parts['test']['lig_3d']
            ]).astype(np.float32),
            'poc_3d': np.vstack([
                parts['train']['poc_3d'],
                parts['val']['poc_3d'],
                parts['test']['poc_3d']
            ]).astype(np.float32),
        }
        return {'all': all_pkg}
    else:
        raise ValueError(f"split 必须是 'train'/'val'/'test'/'all' 之一，收到: {split!r}")



def LoadData_atom3d_si60_multimodal_lm(
    root_base: str,                               # 行：指向 .../split-by-sequence-identity-60/data
    split: str = "train",                         # 行：'train'|'val'|'test'|'all'
    out_mm: str = "../dataset/ATOM3D/processed_mm_si60_lm",  # 行：1D+3D+LM 缓存目录（si-60）
    unimol2_size: str = "unimol2_small",         # 行：Uni-Mol2 模型规格
    contact_threshold: float = 8.0,              # 行：保留接口参数（未使用，仅为兼容）
    dis_min: float = 1.0,                        # 行：保留接口参数（未使用，仅为兼容）
    prot_self_loop: bool = False,                # 行：保留接口参数（未使用，仅为兼容）
    bond_bidirectional: bool = True,             # 行：保留接口参数（未使用，仅为兼容）
    prefer_model: int = None,                    # 行：保留接口参数（未使用，仅为兼容）
    force_refresh: bool = False,                 # 行：是否忽略缓存并强制重建
    use_cuda_for_unimol: bool = True,            # 行：Uni-Mol2 是否用 GPU
    # --- LM 模型相关 ---
    esm2_model_name: str = "facebook/esm2_t33_650M_UR50D",    # 行：ESM-2 checkpoint
    chemberta_model_name: str = "DeepChem/ChemBERTa-77M-MTR", # 行：ChemBERTa-2 checkpoint
    lm_batch_size: int = 32,                                  # 行：LM 前向时的 batch_size
    use_safetensors: bool = True                              # 行：是否优先 safetensors
) -> dict:
    """
    基于 si-60 拆分的多模态+LM 数据加载（1D + 3D 版）：

      - 1D：离线用 ESM-2 / ChemBERTa-2 抽取：
          drug_lm  : ChemBERTa-2(SMILES) 向量 [N, D_d]
          prot_lm  : ESM-2(蛋白序列) 向量 [N, D_p]

      - 3D：配体 + 口袋 Uni-Mol2 向量

      - 2D：不再构建图结构，统一返回 None（保持接口兼容性）

    缓存文件（注意前缀 si60_）：
        si60_{split}_1d_lm.npz     # 1D+LM
        si60_{split}_3d.npz        # 3D

    返回：
        {split: {
           'ids','y','smiles','seq',
           'drug_lm','prot_lm',
           'g_lig','g_prot',       # 恒为 None
           'lig_3d','poc_3d'
        }}
    """

    # —— 依赖：都放在外层导入，避免内部再 import 造成作用域问题 —— #
    from pathlib import Path                      # 行：路径处理
    import numpy as np                            # 行：数值与数组
    import torch                                  # 行：张量 & 设备
    from tqdm import tqdm                         # 行：进度条
    import atom3d.datasets as da                  # 行：ATOM3D 提供的 LMDBDataset
    from unimol_tools import UniMolRepr           # 行：Uni-Mol2 封装
    from transformers import AutoTokenizer, AutoModel  # 行：HuggingFace LM
    from util import atoms_to_sequence            # 行：从原子表恢复氨基酸序列

    # —— 原子符号→原子序号 的兜底工具（部分 unimol 版本需要） —— #
    _PERIODIC = {
        'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,'Na':11,'Mg':12,'Al':13,'Si':14,
        'P':15,'S':16,'Cl':17,'Ar':18,'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,
        'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Kr':36,'Rb':37,'Sr':38,'Y':39,'Zr':40,
        'Nb':41,'Mo':42,'Tc':43,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,'In':49,'Sn':50,'Sb':51,'Te':52,'I':53,'Xe':54
    }

    def _to_atomic_numbers(sym_list):
        """行：把 ['C','O','N',...] 转成 [6,8,7,...]，部分 UniMol 版本需要整数原子序号"""
        return [int(_PERIODIC.get(s, 0)) for s in sym_list]

    # —— LM 编码小工具：兼容不同 HF 模型输出 —— #
    def _encode_text_list(text_list, tokenizer, model, device, batch_size=32):
        """
        行：给定字符串列表 text_list，用 HF tokenizer+model 编码；
            返回 [N, D] numpy.float32（取 CLS / 第一个 token 表示）
        """
        all_vecs = []
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(text_list), batch_size),
                          desc="[LM] encoding", unit="batch"):
                batch = text_list[i:i+batch_size]
                enc = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                # 行：BatchEncoding 直接搬到 device，旧版 transformers 可能不支持 .to()
                try:
                    enc = enc.to(device)
                except Exception:
                    for k, v in enc.items():
                        if isinstance(v, torch.Tensor):
                            enc[k] = v.to(device)

                out = model(**enc)
                # 兼容多种返回结构
                if hasattr(out, "last_hidden_state"):
                    hs = out.last_hidden_state
                elif isinstance(out, (tuple, list)) and len(out) > 0 and hasattr(out[0], "last_hidden_state"):
                    hs = out[0].last_hidden_state
                elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                    hs = out[0]
                else:
                    raise RuntimeError("Unsupported model output structure for LM encoding.")
                CLS = hs[:, 0, :].cpu().numpy().astype(np.float32)  # 行：取第 0 个 token 作为句向量
                all_vecs.append(CLS)

        if len(all_vecs) == 0:
            # 行：没有数据时兜底（一般不会走到）
            return np.zeros((0, model.config.hidden_size), dtype=np.float32)
        return np.concatenate(all_vecs, axis=0)

    # —— HF 模型加载辅助（优先 safetensors，失败再回退） —— #
    def _hf_load_model(name: str, device: torch.device, use_safetensors: bool = True):
        """行：加载 HF 模型，优先 safetensors；失败则回退"""
        try:
            model = AutoModel.from_pretrained(name, use_safetensors=use_safetensors)
            model.to(device)
            return model
        except TypeError as e:
            # 行：旧版 transformers 不认 use_safetensors
            try:
                model = AutoModel.from_pretrained(name)
                model.to(device)
                return model
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load model '{name}' (tried with/without use_safetensors). "
                    f"Err1={e} Err2={e2}"
                ) from e2
        except Exception as e:
            msg = str(e).lower()
            # 行：兼容 2.6+ 的安全检查报错信息
            if ("torch" in msg and "2.6" in msg) or ("vulnerability" in msg) or ("cve" in msg):
                raise RuntimeError(
                    f"Loading HF model '{name}' failed due to transformers' torch-version safety check: {e}\n"
                    "Possible solutions:\n"
                    "  1) 保持现有 torch，优先使用 safetensors（use_safetensors=True），并确保模型 repo 提供 .safetensors；\n"
                    "  2) 升级 torch 到 >=2.6（需要匹配 CUDA）；\n"
                    "  3) 或者在受信环境用旧版 transformers（不推荐生产）。"
                ) from e
            try:
                model = AutoModel.from_pretrained(name)
                model.to(device)
                return model
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load model '{name}'. Tried safetensors and non-safetensors. "
                    f"Err1={e} Err2={e2}"
                ) from e2

    # —— 规范路径 —— #
    root_base = Path(root_base)              # 行：LMDB 根目录 Path 化
    out_mm = Path(out_mm)                    # 行：缓存目录 Path 化
    out_mm.mkdir(parents=True, exist_ok=True)  # 行：确保缓存目录存在

    # —— 单拆分处理 —— #
    def _process_one_split(sp: str) -> dict:
        """
        行：处理单个拆分（train/val/test）：
            1）从 LMDB 读取原始 atom3d 样本
            2）生成 1D（序列）+ 3D（配体/口袋）原始数据
            3）再用 LM 模型抽 1D 特征
            4）缓存到 npz
        """
        lmdb_root = root_base / sp

        # 缓存文件：注意 si60 前缀（此处只缓存 1D + 3D）
        p_1dnpz = out_mm / f"si60_{sp}_1d_lm.npz"
        p_3dnpz = out_mm / f"si60_{sp}_3d.npz"

        # —— 若已有缓存，直接读取 —— #
        if (not force_refresh) and p_1dnpz.exists() and p_3dnpz.exists():
            d1 = np.load(p_1dnpz, allow_pickle=True)
            ids     = d1['ids']
            smiles  = d1['smiles']
            seq     = d1['seq']
            y       = d1['pkd'].astype(np.float32)
            drug_lm = d1['drug_lm'].astype(np.float32)
            prot_lm = d1['prot_lm'].astype(np.float32)

            d3 = np.load(p_3dnpz, allow_pickle=True)
            lig_3d = d3['lig_embeds'].astype(np.float32)
            poc_3d = d3['pocket_embeds'].astype(np.float32)

            N = len(ids)
            assert len(smiles) == N and len(seq) == N and len(y) == N, "1D 长度不一致"
            assert lig_3d.shape[0] == N and poc_3d.shape[0] == N, "3D 行数与 1D 不一致"
            assert drug_lm.shape[0] == N and prot_lm.shape[0] == N, "LM 行数与 1D 不一致"

            return {
                'ids': ids,
                'y': y,
                'smiles': smiles,
                'seq': seq,
                'drug_lm': drug_lm,
                'prot_lm': prot_lm,
                'g_lig': None,           # 行：不再提供 2D 图
                'g_prot': None,
                'lig_3d': lig_3d,
                'poc_3d': poc_3d
            }

        # —— 正常路径：遍历 LMDB，构建 1D/3D 原始数据 —— #
        ds = da.LMDBDataset(str(lmdb_root))     # 行：读取 LMDB 数据集
        print(f"[si-60|{sp}] LMDB samples = {len(ds)}")

        # Uni-Mol2（3D encoder）
        model3d = UniMolRepr(
            model="unimol2",                   # 行：使用 Uni-Mol2
            model_size=unimol2_size,          # 行：small/base 等
            use_cuda=use_cuda_for_unimol      # 行：是否用 GPU
        )
        model3d.model.eval()                  # 行：切 eval 模式

        source_seq = 'atoms_protein'          # 行：用于从原子结构恢复全蛋白序列

        # 收集列表
        ids_list, smiles_list, seq_list, y_list = [], [], [], []
        lig_vecs, poc_vecs = [], []

        n_total = n_ok = n_fail = 0
        breakdown = {'missing_field':0, 'protein_seq':0,
                     'empty3d':0, 'unimol3d':0, 'other':0}
        DEBUG_PRINT_N = 5
        debug_shown = 0
        printed_backend = False

        for i, sample in enumerate(
            tqdm(ds, desc=f"[si-60|{sp}] 1D(LM) + 3D(UniMol2)", unit="sample")
        ):
            n_total += 1
            try:
                smi = sample["smiles"]                         # 行：小分子 SMILES
                y   = float(sample["scores"]["neglog_aff"])    # 行：neglog_aff -> pKd
                sid = sample.get("id", f"{sp}_{i}")            # 行：样本 id，如果没有就自制一个

                # ===== 1D 序列：来自全蛋白原子表 =====
                if source_seq not in sample:
                    breakdown['missing_field'] += 1
                    raise KeyError(f"missing_field:{source_seq}; have={sorted(sample.keys())}")
                df_prot = sample[source_seq]                   # 行：pandas DataFrame，包含蛋白原子信息

                # 从 atoms_protein 恢复氨基酸序列（最长链）
                seq_full = atoms_to_sequence(df_prot, chain_id=None, longest_only=True) or ""
                if not seq_full:
                    breakdown['protein_seq'] += 1
                    raise ValueError("empty target sequence from atoms_protein")

                # ===== 3D：配体 + 口袋 =====
                if "atoms_ligand" not in sample or "atoms_pocket" not in sample:
                    breakdown['missing_field'] += 1
                    raise KeyError(f"missing_field:atoms_ligand/atoms_pocket; have={sorted(sample.keys())}")

                lig_df = sample["atoms_ligand"]
                poc_df = sample["atoms_pocket"]
                lig_atoms = lig_df["element"].tolist()
                lig_xyz   = lig_df[["x","y","z"]].values.tolist()
                poc_atoms = poc_df["element"].tolist()
                poc_xyz   = poc_df[["x","y","z"]].values.tolist()
                if len(lig_atoms) == 0 or len(poc_atoms) == 0:
                    breakdown['empty3d'] += 1
                    raise ValueError("empty ligand or pocket atoms")

                # UniMolRepr：先尝试直接用元素符号，失败则用原子序号
                try:
                    lig_emb = model3d.get_repr({"atoms": lig_atoms, "coordinates": lig_xyz}, return_tensor=True)
                    poc_emb = model3d.get_repr({"atoms": poc_atoms, "coordinates": poc_xyz}, return_tensor=True)
                except Exception:
                    lig_emb = model3d.get_repr(
                        {"atoms": _to_atomic_numbers(lig_atoms), "coordinates": lig_xyz},
                        return_tensor=True
                    )
                    poc_emb = model3d.get_repr(
                        {"atoms": _to_atomic_numbers(poc_atoms), "coordinates": poc_xyz},
                        return_tensor=True
                    )

                if (not printed_backend):
                    backend = type(model3d.model).__name__
                    print(f"[UniMol backend] {backend}")
                    print(f"[UniMol dims] ligand={int(lig_emb.shape[-1])}, pocket={int(poc_emb.shape[-1])}")
                    printed_backend = True

                # 样本通过：加入列表
                ids_list.append(sid)
                smiles_list.append(smi)
                seq_list.append(seq_full)
                y_list.append(y)
                lig_vecs.append(lig_emb.cpu().numpy())
                poc_vecs.append(poc_emb.cpu().numpy())
                n_ok += 1

            except KeyError as e:
                n_fail += 1
                breakdown['missing_field'] += 1
                if debug_shown < DEBUG_PRINT_N:
                    print(f"[DEBUG][{sp}] KeyError at #{i} id={sample.get('id','')}: {e}")
                    debug_shown += 1
            except Exception as e:
                n_fail += 1
                msg = str(e).lower()
                if "sequence" in msg or "target" in msg:
                    breakdown['protein_seq'] += 1
                elif "unimol" in msg or "repr" in msg:
                    breakdown['unimol3d'] += 1
                elif "empty ligand" in msg or "empty pocket" in msg:
                    breakdown['empty3d'] += 1
                else:
                    breakdown['other'] += 1
                if debug_shown < DEBUG_PRINT_N:
                    print(f"[DEBUG][{sp}] Exception at #{i} id={sample.get('id','')}: {e}")
                    debug_shown += 1

        if n_ok == 0:
            print(f"[FAIL][si-60|{sp}] breakdown: {breakdown}")
            raise RuntimeError(f"[si-60|{sp}] No valid samples (1D+3D). Check inputs at {lmdb_root}")

        # 列表 -> numpy
        ids_arr    = np.array(ids_list,    dtype=object)
        smiles_arr = np.array(smiles_list, dtype=object)
        seq_arr    = np.array(seq_list,    dtype=object)
        y_arr      = np.array(y_list,      dtype=np.float32)
        lig_3d     = np.vstack(lig_vecs).astype(np.float32)
        poc_3d     = np.vstack(poc_vecs).astype(np.float32)

        N = len(ids_arr)
        assert len(smiles_arr) == N and len(seq_arr) == N and len(y_arr) == N, "1D 长度不一致"
        assert lig_3d.shape[0] == N and poc_3d.shape[0] == N, "3D 行数与 1D 不一致"

        print(f"[si-60|{sp}] total={n_total} | ok(1D+3D)={n_ok} | fail={n_fail} | breakdown={breakdown}")

        # —— 用 LM 模型抽取 1D 向量 —— #
        lm_device = torch.device(
            "cuda" if (use_cuda_for_unimol and torch.cuda.is_available()) else "cpu"
        )
        print(f"[LM] using device: {lm_device}")

        # ChemBERTa-2 for SMILES
        print(f"[LM] loading ChemBERTa model: {chemberta_model_name} (use_safetensors={use_safetensors})")
        chem_tok = AutoTokenizer.from_pretrained(chemberta_model_name)
        chem_model = _hf_load_model(chemberta_model_name, lm_device, use_safetensors)
        drug_lm = _encode_text_list(
            smiles_arr.tolist(), chem_tok, chem_model, lm_device, batch_size=lm_batch_size
        )

        # 释放 ChemBERTa 占用显存
        try:
            del chem_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        # ESM-2 for protein sequence
        print(f"[LM] loading ESM model: {esm2_model_name} (use_safetensors={use_safetensors})")
        esm_tok = AutoTokenizer.from_pretrained(esm2_model_name)
        esm_model = _hf_load_model(esm2_model_name, lm_device, use_safetensors)
        prot_lm = _encode_text_list(
            seq_arr.tolist(), esm_tok, esm_model, lm_device, batch_size=lm_batch_size
        )

        # 释放 ESM
        try:
            del esm_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        assert drug_lm.shape[0] == N and prot_lm.shape[0] == N, "LM 编码行数与样本数不一致"

        # —— 落盘缓存（si60_*，只存 1D+3D+LM）—— #
        np.savez(
            p_1dnpz,
            ids=ids_arr,
            smiles=smiles_arr,
            seq=seq_arr,
            pkd=y_arr,
            drug_lm=drug_lm,
            prot_lm=prot_lm
        )
        np.savez(
            p_3dnpz,
            ids=ids_arr,
            pkd=y_arr,
            lig_embeds=lig_3d,
            pocket_embeds=poc_3d
        )

        return {
            'ids': ids_arr,
            'y': y_arr,
            'smiles': smiles_arr,
            'seq': seq_arr,
            'drug_lm': drug_lm,
            'prot_lm': prot_lm,
            'g_lig': None,           # 行：2D 图恒为 None
            'g_prot': None,
            'lig_3d': lig_3d,
            'poc_3d': poc_3d
        }

    # —— 顶层调度：单拆分 or all —— #
    if split in ("train", "val", "test"):
        # 行：返回 {split: pkg} 的形式，以兼容你现有的调用方式
        return {split: _process_one_split(split)}

    elif split == "all":
        # 行：把 train/val/test 全部拼在一起，返回一个 'all'
        parts = {sp: _process_one_split(sp) for sp in ("train", "val", "test")}

        def _cat(*xs): return np.concatenate(xs, axis=0)

        all_pkg = {
            'ids'   : _cat(parts['train']['ids'],   parts['val']['ids'],   parts['test']['ids']),
            'y'     : _cat(parts['train']['y'],     parts['val']['y'],     parts['test']['y']).astype(np.float32),
            'smiles': _cat(parts['train']['smiles'],parts['val']['smiles'],parts['test']['smiles']),
            'seq'   : _cat(parts['train']['seq'],   parts['val']['seq'],   parts['test']['seq']),
            'drug_lm': np.vstack([
                parts['train']['drug_lm'],
                parts['val']['drug_lm'],
                parts['test']['drug_lm']
            ]).astype(np.float32),
            'prot_lm': np.vstack([
                parts['train']['prot_lm'],
                parts['val']['prot_lm'],
                parts['test']['prot_lm']
            ]).astype(np.float32),
            'g_lig' : None,         # 行：all 版本同样不返回 2D 图
            'g_prot': None,
            'lig_3d': np.vstack([
                parts['train']['lig_3d'],
                parts['val']['lig_3d'],
                parts['test']['lig_3d']
            ]).astype(np.float32),
            'poc_3d': np.vstack([
                parts['train']['poc_3d'],
                parts['val']['poc_3d'],
                parts['test']['poc_3d']
            ]).astype(np.float32),
        }
        return {'all': all_pkg}

    else:
        # 行：参数错误直接抛异常
        raise ValueError(f"split 必须是 'train'/'val'/'test'/'all' 之一，收到: {split!r}")

def LoadData_davis_lm_1d(
    data_dir: str,                               # 行：指向 Davis 数据集目录（含 ligands_can.txt / proteins.txt / Y）
    split: str = "all",                          # 行：'train'|'val'|'test'|'all'
    out_1d: str = "../dataset/davis/processed_lm_1d",  # 行：1D LM 缓存目录（不存 2D/3D）
    logspace_trans: bool = True,                 # 行：是否对 Y 做 -log10(Kd/1e9) 变换（Davis 常用 pKd）
    # --- LM 模型相关 ---
    esm2_model_name: str = "facebook/esm2_t33_650M_UR50D",    # 行：ESM-2 checkpoint（蛋白）
    chemberta_model_name: str = "DeepChem/ChemBERTa-77M-MTR", # 行：ChemBERTa checkpoint（药物）
    lm_batch_size: int = 32,                     # 行：LM 前向 batch_size
    use_safetensors: bool = True,                # 行：是否优先使用 safetensors
    # --- 划分相关 ---
    split_seed: int = 2023,                      # 行：划分 train/val/test 的随机种子
    train_ratio: float = 0.8,                    # 行：训练集比例
    val_ratio: float = 0.1                       # 行：验证集比例（剩下的是 test）
) -> dict:
    """
    Davis + LM 数据加载（仅 1D，不生成 2D/3D）：

    - 原始文件：
        ligands_can.txt : {lig_id: SMILES}
        proteins.txt    : {prot_id: 序列}
        Y               : [n_lig, n_prot] Kd 矩阵（含 NaN）

    - 生成模态（全是 1D 向量）：
        drug_lm : ChemBERTa(SMILES) 向量 [N, D_d]
        prot_lm : ESM-2(序列) 向量     [N, D_p]

    - 标签：
        若 logspace_trans=True：pkd = -log10(Kd/1e9)

    - 缓存（只存 1D + 标签）：
        davis_{split}_1d_lm.npz

    返回：
        {split: {'ids','y','smiles','seq','drug_lm','prot_lm'}}
    """
    # —— 依赖 —— #
    from pathlib import Path                                  # 行：路径处理
    import numpy as np                                        # 行：数组运算
    import json                                               # 行：读取 ligands / proteins
    import pickle                                             # 行：读取 Y 矩阵
    from collections import OrderedDict                       # 行：保持字典顺序
    import torch                                              # 行：LM 前向 / 设备
    from tqdm import tqdm                                     # 行：进度条
    from transformers import AutoTokenizer, AutoModel         # 行：HF 模型

    # —— LM 编码小工具：兼容不同 HF 模型输出 —— #
    def _encode_text_list(text_list, tokenizer, model, device, batch_size=32):
        """行：输入 list[str]，输出 [N, D] numpy.float32 向量（用 CLS/首 token 表示）"""
        all_vecs = []                                          # 行：收集每个 batch 的 CLS 向量
        model.eval()                                           # 行：eval 模式，关闭 dropout 等
        with torch.no_grad():                                  # 行：关闭梯度，加速并省显存
            for i in tqdm(range(0, len(text_list), batch_size),
                          desc="[LM|DAVIS] encoding", unit="batch"):
                batch = text_list[i:i+batch_size]              # 行：取一个 batch 文本
                enc = tokenizer(                               # 行：分词并 padding/截断
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                # 行：把 token 张量搬到 device（GPU/CPU）
                try:
                    enc = enc.to(device)                       # 行：新版 transformers 支持整体 .to
                except Exception:
                    for k, v in enc.items():                   # 行：兜底：逐个 tensor 搬运
                        if isinstance(v, torch.Tensor):
                            enc[k] = v.to(device)

                out = model(**enc)                             # 行：模型前向推理
                # 行：优先取 last_hidden_state（[B, L, D]）
                if hasattr(out, "last_hidden_state"):
                    hs = out.last_hidden_state
                elif isinstance(out, (tuple, list)) and len(out) > 0 and hasattr(out[0], "last_hidden_state"):
                    hs = out[0].last_hidden_state
                elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                    hs = out[0]
                else:
                    raise RuntimeError("Unsupported model output structure for LM encoding.")

                CLS = hs[:, 0, :].cpu().numpy().astype(np.float32)  # 行：取第 0 个 token 当句向量
                all_vecs.append(CLS)                                # 行：收集该 batch 的结果

        if len(all_vecs) == 0:
            # 行：极端情况兜底（基本用不到）
            return np.zeros((0, model.config.hidden_size), dtype=np.float32)

        return np.concatenate(all_vecs, axis=0)                # 行：拼成 [N, D]

    # —— HF 模型加载辅助（优先 safetensors，失败再回退） —— #
    def _hf_load_model(name: str, device: torch.device, use_safetensors: bool = True):
        """行：优先 safetensors；失败时回退，并给出 torch 版本相关的友好报错"""
        try:
            model = AutoModel.from_pretrained(name, use_safetensors=use_safetensors)  # 行：优先 safetensors
            model.to(device)                                                         # 行：搬到 device
            return model
        except TypeError as e:
            # 行：旧版 transformers 不认 use_safetensors 参数
            try:
                model = AutoModel.from_pretrained(name)                               # 行：不带 safetensors 参数重试
                model.to(device)
                return model
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load model '{name}' (tried with/without use_safetensors). "
                    f"Err1={e} Err2={e2}"
                ) from e2
        except Exception as e:
            msg = str(e).lower()
            # 行：一些 transformers 会对 torch 版本做安全检查（提示 2.6 / CVE 等）
            if ("torch" in msg and "2.6" in msg) or ("vulnerability" in msg) or ("cve" in msg):
                raise RuntimeError(
                    f"Loading HF model '{name}' failed due to transformers' torch-version safety check: {e}\n"
                    "Possible solutions:\n"
                    "  1) 保持现有 torch，优先使用 safetensors（use_safetensors=True），并确保模型 repo 提供 .safetensors；\n"
                    "  2) 升级 torch 到 >=2.6（需要匹配 CUDA）；\n"
                    "  3) 或者在受信环境用旧版 transformers（不推荐生产）。"
                ) from e
            # 行：不是 torch 版本问题，再试一次普通加载
            try:
                model = AutoModel.from_pretrained(name)
                model.to(device)
                return model
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load model '{name}'. Tried safetensors and non-safetensors. "
                    f"Err1={e} Err2={e2}"
                ) from e2

    # —— 规范路径 —— #
    data_dir = Path(data_dir)                                  # 行：原始 Davis 路径
    out_1d = Path(out_1d)                                      # 行：缓存路径（只放 1D）
    out_1d.mkdir(parents=True, exist_ok=True)                  # 行：没有则创建
    split_idx_path = out_1d / "davis_splits.npz"               # 行：划分索引文件（保存 train/val/test 下标）

    # —— 构造 per-split 缓存文件名（只含 1D） —— #
    def _cache_path(sp: str):
        return out_1d / f"davis_{sp}_1d_lm.npz"                # 行：davis_train_1d_lm.npz 之类

    # —— 读取原始 Davis 数据 —— #
    def _load_raw_davis():
        lig_path = data_dir / "ligands_can.txt"                # 行：配体文件
        pro_path = data_dir / "proteins.txt"                   # 行：蛋白文件
        y_path   = data_dir / "Y"                              # 行：Kd 矩阵（pickle）

        # 行：读取 ligands / proteins（保持顺序）
        with lig_path.open() as f:
            lig_dict = json.load(f, object_pairs_hook=OrderedDict)
        with pro_path.open() as f:
            pro_dict = json.load(f, object_pairs_hook=OrderedDict)

        lig_list = list(lig_dict.values())                     # 行：[n_lig] SMILES
        pro_list = list(pro_dict.values())                     # 行：[n_pro] 序列

        with y_path.open("rb") as f:
            Y = pickle.load(f, encoding="latin1")              # 行：原始 Kd 矩阵
        Y = np.array(Y, dtype=np.float64)                      # 行：转 float64，便于 NaN 检查

        # 行：可选：-log10(Kd/1e9) → pKd
        if logspace_trans:
            Y = -np.log10(Y / 1e9)

        return lig_list, pro_list, Y

    # —— 构建/加载 train/val/test 索引 —— #
    def _get_splits_indices(N: int):
        if split_idx_path.exists():                            # 行：如果已经有划分文件就直接读
            data = np.load(split_idx_path, allow_pickle=True)
            train_idx = data["train_idx"]
            val_idx   = data["val_idx"]
            test_idx  = data["test_idx"]
            return train_idx, val_idx, test_idx

        # 行：否则新建划分并保存
        idx_all = np.arange(N)                                 # 行：所有样本下标 [0..N-1]
        rng = np.random.RandomState(split_seed)                # 行：固定随机种子，保证可复现
        rng.shuffle(idx_all)                                   # 行：打乱顺序
        n_train = int(N * train_ratio)                         # 行：训练集样本数
        n_val   = int(N * val_ratio)                           # 行：验证集样本数
        train_idx = idx_all[:n_train]                          # 行：train 切片
        val_idx   = idx_all[n_train:n_train + n_val]           # 行：val 切片
        test_idx  = idx_all[n_train + n_val:]                  # 行：剩余全部作为 test

        np.savez(split_idx_path,                               # 行：把索引保存到 npz，后续直接复用
                 train_idx=train_idx,
                 val_idx=val_idx,
                 test_idx=test_idx)
        return train_idx, val_idx, test_idx

    # —— 一次性构建所有样本的 1D，然后切分+落盘 —— #
    def _build_and_cache_all():
        # 1. 读原始 Davis
        lig_list, pro_list, Y = _load_raw_davis()              # 行：读取 SMILES/序列/Kd(或pKd)
        n_lig = len(lig_list)                                   # 行：配体数量
        n_pro = len(pro_list)                                   # 行：蛋白数量

        # 2. 拉平成 pair 列表（忽略 NaN）
        pair_lig_idx = []                                      # 行：每个样本对应的 ligand 索引
        pair_pro_idx = []                                      # 行：每个样本对应的 protein 索引
        y_list       = []                                      # 行：每个样本的标签
        ids_list     = []                                      # 行：每个样本的 ID（字符串）

        for i in range(n_lig):                                 # 行：遍历所有 ligand
            for j in range(n_pro):                             # 行：遍历所有 protein
                y_ij = Y[i, j]                                 # 行：取 (i,j) 这个 pair 的标签
                if np.isnan(y_ij):                             # 行：Davis 有 NaN，代表没测，跳过
                    continue
                pair_lig_idx.append(i)                         # 行：记录 ligand 索引
                pair_pro_idx.append(j)                         # 行：记录 protein 索引
                y_list.append(float(y_ij))                     # 行：记录标签（float）
                ids_list.append(f"L{i}_P{j}")                  # 行：自定义 pair ID

        N = len(y_list)                                        # 行：有效样本总数
        if N == 0:
            raise RuntimeError("Davis 数据中有效 (非 NaN) 的样本数为 0，请检查原始 Y 矩阵。")

        pair_lig_idx = np.array(pair_lig_idx, dtype=np.int64)  # 行：list → numpy，便于索引
        pair_pro_idx = np.array(pair_pro_idx, dtype=np.int64)
        y_arr        = np.array(y_list,      dtype=np.float32) # 行：标签统一 float32
        ids_arr      = np.array(ids_list,    dtype=object)     # 行：字符串数组用 object dtype

        # 行：pair 级对应的 SMILES / 序列
        smiles_arr = np.array([lig_list[i] for i in pair_lig_idx], dtype=object)
        seq_arr    = np.array([pro_list[j] for j in pair_pro_idx], dtype=object)

        # 3. 对 unique ligand / protein 做 LM 编码（省算力：每个 ligand/protein 只算一次）
        lm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 行：优先用 GPU
        print(f"[LM|DAVIS] using device: {lm_device}")

        # 3.1 Drug LM（ChemBERTa）
        print(f"[LM|DAVIS] loading ChemBERTa model: {chemberta_model_name} (use_safetensors={use_safetensors})")
        chem_tok   = AutoTokenizer.from_pretrained(chemberta_model_name)         # 行：加载 tokenizer
        chem_model = _hf_load_model(chemberta_model_name, lm_device, use_safetensors)  # 行：加载模型
        print(f"[LM|DAVIS] encoding ligands (unique={len(lig_list)})")
        lig_lm_all = _encode_text_list(lig_list, chem_tok, chem_model, lm_device, batch_size=lm_batch_size)
        # 行：释放 Chem 模型，省显存
        try:
            del chem_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        # 3.2 Protein LM（ESM-2）
        print(f"[LM|DAVIS] loading ESM model: {esm2_model_name} (use_safetensors={use_safetensors})")
        esm_tok   = AutoTokenizer.from_pretrained(esm2_model_name)               # 行：加载 tokenizer
        esm_model = _hf_load_model(esm2_model_name, lm_device, use_safetensors)  # 行：加载模型
        print(f"[LM|DAVIS] encoding proteins (unique={len(pro_list)})")
        pro_lm_all = _encode_text_list(pro_list, esm_tok, esm_model, lm_device, batch_size=lm_batch_size)
        # 行：释放 ESM 模型，省显存
        try:
            del esm_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        # 行：根据 pair 索引抽取 LM 向量 → pair 级 [N, D]
        drug_lm = lig_lm_all[pair_lig_idx]                     # 行：[N, D_d]
        prot_lm = pro_lm_all[pair_pro_idx]                     # 行：[N, D_p]

        # 4. 生成 train/val/test 索引
        train_idx, val_idx, test_idx = _get_splits_indices(N)  # 行：划分样本下标

        # 5. 按索引切分并写入缓存（只写 1D）
        def _save_split(sp: str, idx_array: np.ndarray):
            p_1dnpz = _cache_path(sp)                          # 行：该 split 的缓存路径
            idx_array = np.asarray(idx_array, dtype=np.int64)  # 行：确保是 int64

            np.savez(                                          # 行：把该 split 的内容写入 npz
                p_1dnpz,
                ids=ids_arr[idx_array],                        # 行：样本 ID
                smiles=smiles_arr[idx_array],                  # 行：SMILES
                seq=seq_arr[idx_array],                        # 行：蛋白序列
                pkd=y_arr[idx_array],                          # 行：Davis 标签（命名 pkd）
                drug_lm=drug_lm[idx_array],                    # 行：药物 LM 特征
                prot_lm=prot_lm[idx_array],                    # 行：蛋白 LM 特征
            )

        print("[CACHE|DAVIS] saving train/val/test splits (1D LM only)...")
        _save_split("train", train_idx)
        _save_split("val",   val_idx)
        _save_split("test",  test_idx)
        print("[CACHE|DAVIS] cached done (1D only, no 2D/3D).")

    # —— 判断是否需要重新构建缓存 —— #
    need_build = False
    if split in ("train", "val", "test"):
        p_1d = _cache_path(split)                              # 行：该 split 的 npz 路径
        if not p_1d.exists():                                  # 行：不存在就要重新构建
            need_build = True
    elif split == "all":
        for sp in ("train", "val", "test"):                    # 行：任一 split 缺失就重建一次
            p_1d = _cache_path(sp)
            if not p_1d.exists():
                need_build = True
                break
    else:
        raise ValueError(f"split 必须是 'train'/'val'/'test'/'all' 之一，收到: {split!r}")

    if need_build:
        _build_and_cache_all()                                 # 行：构建并缓存 train/val/test 三份

    # —— 读取单个拆分（只加载 1D） —— #
    def _load_split(sp: str) -> dict:
        p_1dnpz = _cache_path(sp)                              # 行：npz 路径
        d1 = np.load(p_1dnpz, allow_pickle=True)               # 行：读取 npz
        ids    = d1["ids"]                                     # 行：样本 ID
        smiles = d1["smiles"]                                  # 行：SMILES
        seq    = d1["seq"]                                     # 行：蛋白序列
        y      = d1["pkd"].astype(np.float32)                  # 行：Davis 标签（pkd）→ float32
        drug_lm = d1["drug_lm"].astype(np.float32)             # 行：药物 LM 向量
        prot_lm = d1["prot_lm"].astype(np.float32)             # 行：蛋白 LM 向量

        N = len(ids)                                           # 行：样本数
        assert len(smiles) == N and len(seq) == N and len(y) == N, "1D 长度不一致"
        assert drug_lm.shape[0] == N and prot_lm.shape[0] == N, "LM 行数与 1D 不一致"

        return {
            "ids": ids,                                        # 行：array(object)
            "y": y,                                            # 行：训练脚本统一从 pkg['y'] 取标签
            "smiles": smiles,                                  # 行：array(str)
            "seq": seq,                                        # 行：array(str)
            "drug_lm": drug_lm,                                # 行：np.ndarray[N, D_d]
            "prot_lm": prot_lm,                                # 行：np.ndarray[N, D_p]
        }

    # —— 顶层返回 —— #
    if split in ("train", "val", "test"):
        return {split: _load_split(split)}                     # 行：只返回对应 split
    else:  # "all"
        parts = {sp: _load_split(sp) for sp in ("train", "val", "test")}  # 行：读三份
        def _cat(*xs): return np.concatenate(xs, axis=0)         # 行：拼接工具函数

        all_pkg = {
            "ids":    _cat(parts["train"]["ids"],    parts["val"]["ids"],    parts["test"]["ids"]),
            "y":      _cat(parts["train"]["y"],      parts["val"]["y"],      parts["test"]["y"]).astype(np.float32),
            "smiles": _cat(parts["train"]["smiles"], parts["val"]["smiles"], parts["test"]["smiles"]),
            "seq":    _cat(parts["train"]["seq"],    parts["val"]["seq"],    parts["test"]["seq"]),
            "drug_lm": np.vstack([
                parts["train"]["drug_lm"],
                parts["val"]["drug_lm"],
                parts["test"]["drug_lm"],
            ]).astype(np.float32),
            "prot_lm": np.vstack([
                parts["train"]["prot_lm"],
                parts["val"]["prot_lm"],
                parts["test"]["prot_lm"],
            ]).astype(np.float32),
        }
        return {"all": all_pkg}


def LoadData_kiba_lm_1d(
    data_dir: str,                               # 行：指向 KIBA 数据集目录（含 ligands_can.txt / proteins.txt / Y）
    split: str = "all",                          # 行：'train'|'val'|'test'|'all'
    out_1d: str = "../dataset/kiba/processed_lm_1d",  # 行：1D LM 缓存目录（不存 2D/3D）
    logspace_trans: bool = False,                # 行：是否对 Y 做 log 变换（KIBA 一般不用，这里默认 False）
    # --- LM 模型相关 ---
    esm2_model_name: str = "facebook/esm2_t33_650M_UR50D",    # 行：ESM-2 checkpoint（蛋白）
    chemberta_model_name: str = "DeepChem/ChemBERTa-77M-MTR", # 行：ChemBERTa-2 checkpoint（药物）
    lm_batch_size: int = 2,                     # 行：LM 前向 batch_size
    use_safetensors: bool = True,                # 行：是否优先使用 safetensors
    # --- 划分相关 ---
    split_seed: int = 2023,                      # 行：划分 train/val/test 的随机种子
    train_ratio: float = 0.8,                    # 行：训练集比例
    val_ratio: float = 0.1                       # 行：验证集比例（剩下的是 test）
) -> dict:
    """
    KIBA + LM 数据加载（仅 1D，不生成 2D/3D）：

    - 原始文件：
        ligands_can.txt : {lig_id: SMILES}
        proteins.txt    : {prot_id: 序列}
        Y               : [n_lig, n_prot] KIBA 分数矩阵（一般无 NaN）

    - 生成模态（全是 1D 向量）：
        drug_lm : ChemBERTa-2(SMILES) 向量 [N, D_d]
        prot_lm : ESM-2(序列) 向量     [N, D_p]

    - 缓存（只存 1D + 标签）：
        kiba_{split}_1d_lm.npz

    返回：
        {split: {'ids','y','smiles','seq','drug_lm','prot_lm'}}
    """
    # —— 依赖 —— #
    from pathlib import Path                                  # 行：路径处理
    import numpy as np                                        # 行：数组运算
    import json                                               # 行：读取 ligands / proteins
    import pickle                                             # 行：读取 Y 矩阵
    from collections import OrderedDict                       # 行：保持字典顺序
    import torch                                              # 行：LM 前向 / 设备
    from tqdm import tqdm                                     # 行：进度条
    from transformers import AutoTokenizer, AutoModel         # 行：HF 模型

    # —— LM 编码小工具：兼容不同 HF 模型输出 —— #
    def _encode_text_list(text_list, tokenizer, model, device, batch_size=32):
        """行：输入 list[str]，输出 [N, D] numpy.float32 向量（用 CLS/首 token 表示）"""
        all_vecs = []                                          # 行：收集每个 batch 的 CLS 向量
        model.eval()                                           # 行：eval 模式，关闭 dropout 等
        with torch.no_grad():                                  # 行：关闭梯度，加速并省显存
            for i in tqdm(range(0, len(text_list), batch_size),
                          desc="[LM|KIBA] encoding", unit="batch"):
                batch = text_list[i:i+batch_size]              # 行：取一个 batch 文本
                enc = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                # enc 是 BatchEncoding，尝试整体 .to(device)
                try:
                    enc = enc.to(device)
                except Exception:
                    # 行：某些老版本不支持整体 .to，兜底逐个移 tensor
                    for k, v in enc.items():
                        if isinstance(v, torch.Tensor):
                            enc[k] = v.to(device)

                out = model(**enc)                             # 行：前向
                # 行：优先取 last_hidden_state
                if hasattr(out, "last_hidden_state"):
                    hs = out.last_hidden_state                 # [B, L, D]
                elif isinstance(out, (tuple, list)) and len(out) > 0 and hasattr(out[0], "last_hidden_state"):
                    hs = out[0].last_hidden_state
                elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                    hs = out[0]
                else:
                    raise RuntimeError("Unsupported model output structure for LM encoding.")
                CLS = hs[:, 0, :].cpu().numpy().astype(np.float32)  # 行：取第 0 个 token 作为句向量
                all_vecs.append(CLS)
        if len(all_vecs) == 0:
            # 行：极端情况兜底（基本用不到）
            return np.zeros((0, model.config.hidden_size), dtype=np.float32)
        return np.concatenate(all_vecs, axis=0)                # 行：拼成 [N, D]

    # —— HF 模型加载辅助（优先 safetensors，失败再回退） —— #
    def _hf_load_model(name: str, device: torch.device, use_safetensors: bool = True):
        """行：优先 safetensors；失败时回退，并给出 torch 版本相关的友好报错"""
        try:
            model = AutoModel.from_pretrained(name, use_safetensors=use_safetensors)
            model.to(device)
            return model
        except TypeError as e:
            # 行：旧版 transformers 不认 use_safetensors
            try:
                model = AutoModel.from_pretrained(name)
                model.to(device)
                return model
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load model '{name}' (tried with/without use_safetensors). "
                    f"Err1={e} Err2={e2}"
                ) from e2
        except Exception as e:
            msg = str(e).lower()
            if ("torch" in msg and "2.6" in msg) or ("vulnerability" in msg) or ("cve" in msg):
                raise RuntimeError(
                    f"Loading HF model '{name}' failed due to transformers' torch-version safety check: {e}\n"
                    "Possible solutions:\n"
                    "  1) 保持现有 torch，优先使用 safetensors（use_safetensors=True），并确保模型 repo 提供 .safetensors；\n"
                    "  2) 升级 torch 到 >=2.6（需要匹配 CUDA）；\n"
                    "  3) 或者在受信环境用旧版 transformers（不推荐生产）。"
                ) from e
            # 行：不是 torch 版本问题，再试一次普通加载
            try:
                model = AutoModel.from_pretrained(name)
                model.to(device)
                return model
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load model '{name}'. Tried safetensors and non-safetensors. "
                    f"Err1={e} Err2={e2}"
                ) from e2

    # —— 规范路径 —— #
    data_dir = Path(data_dir)                                  # 行：原始 KIBA 路径
    out_1d = Path(out_1d)                                      # 行：缓存路径（只放 1D）
    out_1d.mkdir(parents=True, exist_ok=True)                  # 行：没有则创建
    split_idx_path = out_1d / "kiba_splits.npz"                # 行：划分索引文件（保存 train/val/test 下标）

    # —— 构造 per-split 缓存文件名（只含 1D） —— #
    def _cache_path(sp: str):
        return out_1d / f"kiba_{sp}_1d_lm.npz"                 # 行：kiba_train_1d_lm.npz 之类

    # —— 读取原始 KIBA 数据 —— #
    def _load_raw_kiba():
        lig_path = data_dir / "ligands_can.txt"                # 行：配体文件
        pro_path = data_dir / "proteins.txt"                   # 行：蛋白文件
        y_path   = data_dir / "Y"                              # 行：KIBA 分数矩阵

        # 行：读取 ligands / proteins（保持顺序）
        with lig_path.open() as f:
            lig_dict = json.load(f, object_pairs_hook=OrderedDict)
        with pro_path.open() as f:
            pro_dict = json.load(f, object_pairs_hook=OrderedDict)

        lig_list = list(lig_dict.values())                     # 行：[n_lig] SMILES
        pro_list = list(pro_dict.values())                     # 行：[n_pro] 序列

        with y_path.open("rb") as f:
            Y = pickle.load(f, encoding="latin1")              # 行：原始 KIBA 矩阵
        Y = np.array(Y, dtype=np.float64)                      # 行：转成 float64，方便 NaN 检查

        # 行：可选：做 log 变换（一般 KIBA 不用）
        if logspace_trans:
            Y = np.log10(Y + 1e-8)                             # 行：举例：log10 变换 + 稍微防 0

        return lig_list, pro_list, Y

    # —— 构建/加载 train/val/test 索引 —— #
    def _get_splits_indices(N: int):
        if split_idx_path.exists():
            data = np.load(split_idx_path, allow_pickle=True)
            train_idx = data["train_idx"]
            val_idx   = data["val_idx"]
            test_idx  = data["test_idx"]
            return train_idx, val_idx, test_idx
        # 行：不存在则新建并保存
        idx_all = np.arange(N)
        rng = np.random.RandomState(split_seed)
        rng.shuffle(idx_all)
        n_train = int(N * train_ratio)
        n_val   = int(N * val_ratio)
        n_test  = N - n_train - n_val
        train_idx = idx_all[:n_train]
        val_idx   = idx_all[n_train:n_train + n_val]
        test_idx  = idx_all[n_train + n_val:]
        np.savez(split_idx_path,
                 train_idx=train_idx,
                 val_idx=val_idx,
                 test_idx=test_idx)
        return train_idx, val_idx, test_idx

    # —— 一次性构建所有样本的 1D，然后切分+落盘 —— #
    def _build_and_cache_all():
        # 1. 读原始 KIBA
        lig_list, pro_list, Y = _load_raw_kiba()
        n_lig = len(lig_list)
        n_pro = len(pro_list)

        # 2. 拉平成 pair 列表（如果有 NaN 就跳过；大多数 KIBA 是全有标签）
        pair_lig_idx = []                                     # 行：样本对应的配体索引
        pair_pro_idx = []                                     # 行：样本对应的蛋白索引
        y_list       = []                                     # 行：样本标注
        ids_list     = []                                     # 行：样本 ID

        for i in range(n_lig):
            for j in range(n_pro):
                y_ij = Y[i, j]
                if np.isnan(y_ij):                            # 行：容错：若有 NaN 就跳过
                    continue
                pair_lig_idx.append(i)
                pair_pro_idx.append(j)
                y_list.append(float(y_ij))
                ids_list.append(f"L{i}_P{j}")                 # 行：自定义一个 pair ID

        N = len(y_list)
        if N == 0:
            raise RuntimeError("KIBA 数据中有效 (非 NaN) 的样本数为 0，请检查原始 Y 矩阵。")

        pair_lig_idx = np.array(pair_lig_idx, dtype=np.int64)
        pair_pro_idx = np.array(pair_pro_idx, dtype=np.int64)
        y_arr        = np.array(y_list,      dtype=np.float32)
        ids_arr      = np.array(ids_list,    dtype=object)

        # 对应的 SMILES / 序列
        smiles_arr = np.array([lig_list[i] for i in pair_lig_idx], dtype=object)
        seq_arr    = np.array([pro_list[j] for j in pair_pro_idx], dtype=object)

        # 3. 对 unique ligand / protein 做 LM 编码，节省算力
        lm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LM|KIBA] using device: {lm_device}")

        # 3.1 Drug LM（ChemBERTa）
        print(f"[LM|KIBA] loading ChemBERTa model: {chemberta_model_name} (use_safetensors={use_safetensors})")
        chem_tok   = AutoTokenizer.from_pretrained(chemberta_model_name)
        chem_model = _hf_load_model(chemberta_model_name, lm_device, use_safetensors)
        print(f"[LM|KIBA] encoding ligands (unique={len(lig_list)})")
        lig_lm_all = _encode_text_list(lig_list, chem_tok, chem_model, lm_device, batch_size=lm_batch_size)
        # 行：释放 Chem 模型，省显存
        try:
            del chem_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        # 3.2 Protein LM（ESM-2）
        print(f"[LM|KIBA] loading ESM model: {esm2_model_name} (use_safetensors={use_safetensors})")
        esm_tok   = AutoTokenizer.from_pretrained(esm2_model_name)
        esm_model = _hf_load_model(esm2_model_name, lm_device, use_safetensors)
        print(f"[LM|KIBA] encoding proteins (unique={len(pro_list)})")
        pro_lm_all = _encode_text_list(pro_list, esm_tok, esm_model, lm_device, batch_size=lm_batch_size)
        # 行：释放 ESM
        try:
            del esm_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        # 行：根据 pair 索引抽取 LM 向量
        drug_lm = lig_lm_all[pair_lig_idx]                    # [N, D_d]
        prot_lm = pro_lm_all[pair_pro_idx]                    # [N, D_p]

        # 4. 生成 train/val/test 索引
        train_idx, val_idx, test_idx = _get_splits_indices(N)

        # 5. 按索引切分并写入缓存（只写 1D）
        def _save_split(sp: str, idx_array: np.ndarray):
            p_1dnpz = _cache_path(sp)
            idx_array = np.asarray(idx_array, dtype=np.int64)

            np.savez(
                p_1dnpz,
                ids=ids_arr[idx_array],           # 行：样本 ID
                smiles=smiles_arr[idx_array],     # 行：SMILES
                seq=seq_arr[idx_array],           # 行：蛋白序列
                kiba=y_arr[idx_array],            # 行：KIBA 标签
                drug_lm=drug_lm[idx_array],       # 行：药物 LM 特征
                prot_lm=prot_lm[idx_array],       # 行：蛋白 LM 特征
            )

        print("[CACHE|KIBA] saving train/val/test splits (1D LM only)...")
        _save_split("train", train_idx)
        _save_split("val",   val_idx)
        _save_split("test",  test_idx)
        print("[CACHE|KIBA] cached done (1D only, no 2D/3D).")

    # —— 判断是否需要重新构建缓存 —— #
    need_build = False
    if split in ("train", "val", "test"):
        p_1d = _cache_path(split)
        if not p_1d.exists():
            need_build = True
    elif split == "all":
        for sp in ("train", "val", "test"):
            p_1d = _cache_path(sp)
            if not p_1d.exists():
                need_build = True
                break
    else:
        raise ValueError(f"split 必须是 'train'/'val'/'test'/'all' 之一，收到: {split!r}")

    if need_build:
        _build_and_cache_all()

    # —— 读取单个拆分（只加载 1D） —— #
    def _load_split(sp: str) -> dict:
        p_1dnpz = _cache_path(sp)
        d1 = np.load(p_1dnpz, allow_pickle=True)
        ids    = d1["ids"]
        smiles = d1["smiles"]
        seq    = d1["seq"]
        y      = d1["kiba"].astype(np.float32)      # 行：KIBA 分数 → float32
        drug_lm = d1["drug_lm"].astype(np.float32)
        prot_lm = d1["prot_lm"].astype(np.float32)

        N = len(ids)
        assert len(smiles) == N and len(seq) == N and len(y) == N, "1D 长度不一致"
        assert drug_lm.shape[0] == N and prot_lm.shape[0] == N, "LM 行数与 1D 不一致"

        return {
            "ids": ids,              # 行：array(object)
            "y": y,                  # 行：np.float32，训练脚本统一用 pkg['y']
            "smiles": smiles,        # 行：array(str)
            "seq": seq,              # 行：array(str)
            "drug_lm": drug_lm,      # 行：np.ndarray[N, D_d]
            "prot_lm": prot_lm,      # 行：np.ndarray[N, D_p]
        }

    # —— 顶层返回 —— #
    if split in ("train", "val", "test"):
        return {split: _load_split(split)}
    else:  # "all"
        parts = {sp: _load_split(sp) for sp in ("train", "val", "test")}
        def _cat(*xs): return np.concatenate(xs, axis=0)
        all_pkg = {
            "ids":    _cat(parts["train"]["ids"],    parts["val"]["ids"],    parts["test"]["ids"]),
            "y":      _cat(parts["train"]["y"],      parts["val"]["y"],      parts["test"]["y"]).astype(np.float32),
            "smiles": _cat(parts["train"]["smiles"], parts["val"]["smiles"], parts["test"]["smiles"]),
            "seq":    _cat(parts["train"]["seq"],    parts["val"]["seq"],    parts["test"]["seq"]),
            "drug_lm": np.vstack([
                parts["train"]["drug_lm"],
                parts["val"]["drug_lm"],
                parts["test"]["drug_lm"],
            ]).astype(np.float32),
            "prot_lm": np.vstack([
                parts["train"]["prot_lm"],
                parts["val"]["prot_lm"],
                parts["test"]["prot_lm"],
            ]).astype(np.float32),
        }
        return {"all": all_pkg}


def LoadData_bindingdb_lm_1d(
    data_dir: str,                               # 行：指向 BindingDB 数据集目录（含 bindingdb_train.csv / bindingdb_test.csv）
    split: str = "all",                          # 行：'train'|'val'|'test'|'all'
    out_1d: str = "../dataset/bindingdb/processed_lm_1d",  # 行：1D LM 缓存目录（不存 2D/3D）
    logspace_trans: bool = False,                # 行：是否对 affinity 做 log 变换（默认 False，直接用 CSV 里的 affinity）
    # --- LM 模型相关 ---
    esm2_model_name: str = "facebook/esm2_t33_650M_UR50D",    # 行：ESM-2 checkpoint（蛋白）
    chemberta_model_name: str = "DeepChem/ChemBERTa-77M-MTR", # 行：ChemBERTa-2 checkpoint（药物）
    lm_batch_size: int = 32,                     # 行：LM 前向 batch_size
    use_safetensors: bool = True,                # 行：是否优先使用 safetensors
    # --- 划分相关（只在原 train 部分里划分 train/val，test 固定为 bindingdb_test.csv）---
    split_seed: int = 2023,                      # 行：划分 train/val 的随机种子
    train_ratio: float = 0.8,                    # 行：原 train 部分中 train 所占比例（会与 val_ratio 归一化）
    val_ratio: float = 0.2                       # 行：原 train 部分中 val 所占比例（会与 train_ratio 归一化）
) -> dict:
    """
    BindingDB + LM 数据加载（仅 1D，不生成 2D/3D）：

    - 原始文件（CSV）：
        bindingdb_train.csv :
            compound_iso_smiles : 配体 SMILES
            target_sequence     : 蛋白序列
            affinity            : 亲和力标签（例如 pKd/pKi 等）
        bindingdb_test.csv :
            同上结构，作为固定测试集

    - 生成模态（全是 1D 向量）：
        drug_lm : ChemBERTa-2(SMILES) 向量 [N, D_d]
        prot_lm : ESM-2(序列) 向量     [N, D_p]

    - 划分策略：
        * 先把 train/test 两个 CSV 纵向拼接，得到总样本数 N = N_train + N_test
        * train_idx / val_idx 只在“原 train 部分（前 N_train 个样本）”里随机划分
        * test_idx = 原 bindingdb_test.csv 那一部分（后 N_test 个样本），不参与随机划分

    - 缓存（只存 1D + 标签）：
        bindingdb_train_1d_lm.npz
        bindingdb_val_1d_lm.npz
        bindingdb_test_1d_lm.npz

    返回：
        {split: {'ids','y','smiles','seq','drug_lm','prot_lm'}}
    """
    # —— 依赖 —— #
    from pathlib import Path                                  # 行：路径处理
    import numpy as np                                        # 行：数组运算
    import pandas as pd                                       # 行：读取 CSV
    from collections import OrderedDict                       # 行：保持去重后的顺序
    import torch                                              # 行：LM 前向 / 设备
    from tqdm import tqdm                                     # 行：进度条
    from transformers import AutoTokenizer, AutoModel         # 行：HF 模型

    # —— LM 编码小工具：兼容不同 HF 模型输出 —— #
    def _encode_text_list(text_list, tokenizer, model, device, batch_size=32):
        """行：输入 list[str]，输出 [N, D] 的 CLS 向量（float32）"""
        import torch  # 行：局部导入，保证有 torch
        all_vecs = []  # 行：保存每个 batch 的 CLS 向量
        model.eval()  # 行：eval 模式，关闭 dropout 等

        with torch.no_grad():  # 行：不需要梯度
            for i in tqdm(
                    range(0, len(text_list), batch_size),
                    desc="[LM|BindingDB] encoding", unit="batch"
            ):
                batch = text_list[i:i + batch_size]  # 行：这一小批的序列/SMILES
                enc = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=1022,  # 行：显式限制长度，避免极端超长（可选）
                    return_tensors="pt"
                )
                # 行：把输入搬到对应设备
                try:
                    enc = enc.to(device)
                except Exception:
                    for k, v in enc.items():
                        if isinstance(v, torch.Tensor):
                            enc[k] = v.to(device)

                # ★ 行：如果在 GPU 上，用 autocast 让中间张量用 fp16，节省显存
                if device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        out = model(**enc)
                else:
                    out = model(**enc)

                # 行：拿 last_hidden_state 的 CLS 作为句向量
                if hasattr(out, "last_hidden_state"):
                    hs = out.last_hidden_state  # [B, L, D]
                elif isinstance(out, (tuple, list)) and len(out) > 0 and hasattr(out[0], "last_hidden_state"):
                    hs = out[0].last_hidden_state
                elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                    hs = out[0]
                else:
                    raise RuntimeError("Unsupported model output structure for LM encoding.")
                CLS = hs[:, 0, :].cpu().numpy().astype("float32")
                all_vecs.append(CLS)

        if len(all_vecs) == 0:
            return np.zeros((0, model.config.hidden_size), dtype="float32")
        return np.concatenate(all_vecs, axis=0).astype("float32")

    # —— HF 模型加载辅助（优先 safetensors，失败再回退） —— #
    def _hf_load_model(name: str, device: torch.device, use_safetensors: bool = True):
        """行：优先 safetensors；失败时回退，并给出 torch 版本相关的友好报错"""
        try:
            model = AutoModel.from_pretrained(name, use_safetensors=use_safetensors)
            model.to(device)
            return model
        except TypeError as e:
            # 行：旧版 transformers 不认 use_safetensors
            try:
                model = AutoModel.from_pretrained(name)
                model.to(device)
                return model
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load model '{name}' (tried with/without use_safetensors). "
                    f"Err1={e} Err2={e2}"
                ) from e2
        except Exception as e:
            msg = str(e).lower()
            if ("torch" in msg and "2.6" in msg) or ("vulnerability" in msg) or ("cve" in msg):
                raise RuntimeError(
                    f"Loading HF model '{name}' failed due to transformers' torch-version safety check: {e}\n"
                    "Possible solutions:\n"
                    "  1) 保持现有 torch，优先使用 safetensors（use_safetensors=True），并确保模型 repo 提供 .safetensors；\n"
                    "  2) 升级 torch 到 >=2.6（需要匹配 CUDA）；\n"
                    "  3) 或者在受信环境用旧版 transformers（不推荐生产）。"
                ) from e
            # 行：不是 torch 版本问题，再试一次普通加载
            try:
                model = AutoModel.from_pretrained(name)
                model.to(device)
                return model
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load model '{name}'. Tried safetensors and non-safetensors. "
                    f"Err1={e} Err2={e2}"
                ) from e2

    # —— 规范路径 —— #
    data_dir = Path(data_dir)                                  # 行：BindingDB 原始路径
    out_1d = Path(out_1d)                                      # 行：缓存路径（只放 1D）
    out_1d.mkdir(parents=True, exist_ok=True)                  # 行：没有则创建
    split_idx_path = out_1d / "bindingdb_splits.npz"           # 行：划分索引文件（保存 train/val/test 下标）

    # —— 构造 per-split 缓存文件名（只含 1D） —— #
    def _cache_path(sp: str):
        return out_1d / f"bindingdb_{sp}_1d_lm.npz"           # 行：bindingdb_train_1d_lm.npz 之类

    # —— 读取原始 BindingDB 数据（train/test 两个 CSV） —— #
    def _load_raw_bindingdb():
        train_path = data_dir / "bindingdb_train.csv"          # 行：训练 CSV
        test_path  = data_dir / "bindingdb_test.csv"           # 行：测试 CSV

        if not train_path.exists():
            raise FileNotFoundError(f"未找到文件: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"未找到文件: {test_path}")

        train_df = pd.read_csv(train_path)                     # 行：读入训练数据
        test_df  = pd.read_csv(test_path)                      # 行：读入测试数据

        # 行：检查必须的列是否存在
        required_cols = ["compound_iso_smiles", "target_sequence", "affinity"]
        for col in required_cols:
            if col not in train_df.columns:
                raise ValueError(f"train CSV 缺少列: {col}")
            if col not in test_df.columns:
                raise ValueError(f"test CSV 缺少列: {col}")

        # 行：从 DataFrame 提取 SMILES / 序列 / 标签
        smiles_train = train_df["compound_iso_smiles"].astype(str).tolist()
        seq_train    = train_df["target_sequence"].astype(str).tolist()
        y_train      = train_df["affinity"].astype(float).values

        smiles_test = test_df["compound_iso_smiles"].astype(str).tolist()
        seq_test    = test_df["target_sequence"].astype(str).tolist()
        y_test      = test_df["affinity"].astype(float).values

        # 行：合并 train + test
        smiles_all = smiles_train + smiles_test               # 行：总长度 N = N_train + N_test
        seq_all    = seq_train + seq_test
        y_all      = np.concatenate([y_train, y_test], axis=0)

        # 行：可选：对 affinity 做 log 变换（视具体含义而定，这里默认不用）
        if logspace_trans:
            # 例如：做 log10(affinity + 1e-8)，你可以按需要改成 pKd/pKi 风格
            y_all = np.log10(y_all + 1e-8)

        N_train = len(smiles_train)                           # 行：原 train 部分样本数
        N_test  = len(smiles_test)                            # 行：原 test 部分样本数

        return smiles_all, seq_all, y_all, N_train, N_test

    # —— 构建/加载 train/val/test 索引 —— #
    def _get_splits_indices(N_total: int, N_train: int):
        """
        行：N_total = 总样本数（train+test）
            N_train = 原 train 部分样本数（0..N_train-1）
            策略：仅在 0..N_train-1 内随机划分 train/val；test 固定为 N_train..N_total-1
        """
        if split_idx_path.exists():
            data = np.load(split_idx_path, allow_pickle=True)
            train_idx = data["train_idx"]
            val_idx   = data["val_idx"]
            test_idx  = data["test_idx"]
            return train_idx, val_idx, test_idx

        # 行：只在原 train 部分做随机划分
        idx_train_pool = np.arange(N_train, dtype=np.int64)   # 行：原 train 部分索引 [0..N_train-1]
        rng = np.random.RandomState(split_seed)               # 行：固定随机种子
        rng.shuffle(idx_train_pool)                           # 行：打乱顺序

        # 行：把 train_ratio / val_ratio 归一化后在原 train 部分中分配
        total_r = train_ratio + val_ratio
        if total_r <= 0:
            # 行：兜底：没有合理比例时，默认 8:2
            t_ratio_eff, v_ratio_eff = 0.8, 0.2
        else:
            t_ratio_eff = train_ratio / total_r               # 行：有效 train 比例
            v_ratio_eff = val_ratio / total_r                 # 行：有效 val 比例（其实就是 1 - t_ratio_eff）

        n_train_eff = int(N_train * t_ratio_eff)              # 行：训练样本数（向下取整）
        # 行：保证 train/val 至少各有 1 个样本
        if n_train_eff <= 0:
            n_train_eff = 1
        if n_train_eff >= N_train:
            n_train_eff = N_train - 1

        train_idx = idx_train_pool[:n_train_eff]              # 行：划分后的 train 索引（global index）
        val_idx   = idx_train_pool[n_train_eff:]              # 行：划分后的 val 索引（global index）

        # 行：test 部分索引 = [N_train .. N_total-1]
        test_idx  = np.arange(N_train, N_total, dtype=np.int64)

        # 行：保存划分结果，方便下次直接加载
        np.savez(split_idx_path,
                 train_idx=train_idx,
                 val_idx=val_idx,
                 test_idx=test_idx)
        return train_idx, val_idx, test_idx

    # —— 一次性构建所有样本的 1D，然后切分+落盘 —— #
    def _build_and_cache_all():
        # 1. 读原始 BindingDB
        smiles_all, seq_all, y_all, N_train, N_test = _load_raw_bindingdb()
        N_total = len(y_all)                                  # 行：总样本数 = N_train + N_test
        assert N_total == N_train + N_test, "样本数不一致，请检查 BindingDB CSV"

        # 2. 去重后构建 unique ligand / protein 列表，用于 LM 编码
        lig_list = list(OrderedDict.fromkeys(smiles_all).keys())  # 行：unique SMILES，保持首次出现顺序
        pro_list = list(OrderedDict.fromkeys(seq_all).keys())     # 行：unique 序列，保持顺序

        # 行：构建 SMILES / 序列 → 索引 的映射
        lig2idx = {smi: i for i, smi in enumerate(lig_list)}
        pro2idx = {seq: i for i, seq in enumerate(pro_list)}

        # 行：样本级配体 / 蛋白索引
        pair_lig_idx = np.array([lig2idx[s] for s in smiles_all], dtype=np.int64)
        pair_pro_idx = np.array([pro2idx[s] for s in seq_all],    dtype=np.int64)

        # 构造样本 ID：train_0.. / test_0..
        ids_train = [f"tr_{i}" for i in range(N_train)]       # 行：原 train 部分 ID
        ids_test  = [f"te_{i}" for i in range(N_test)]        # 行：原 test 部分 ID
        ids_arr   = np.array(ids_train + ids_test, dtype=object)

        # 3. 对 unique ligand / protein 做 LM 编码，节省算力
        lm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LM|BindingDB] using device: {lm_device}")

        # 3.1 Drug LM（ChemBERTa）
        print(f"[LM|BindingDB] loading ChemBERTa model: {chemberta_model_name} (use_safetensors={use_safetensors})")
        chem_tok   = AutoTokenizer.from_pretrained(chemberta_model_name)
        chem_model = _hf_load_model(chemberta_model_name, lm_device, use_safetensors)
        print(f"[LM|BindingDB] encoding ligands (unique={len(lig_list)})")
        lig_lm_all = _encode_text_list(lig_list, chem_tok, chem_model, lm_device, batch_size=lm_batch_size)
        # 行：释放 Chem 模型，省显存
        try:
            del chem_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        # 3.2 Protein LM（ESM-2）
        print(f"[LM|BindingDB] loading ESM model: {esm2_model_name} (use_safetensors={use_safetensors})")
        esm_tok   = AutoTokenizer.from_pretrained(esm2_model_name)
        esm_model = _hf_load_model(esm2_model_name, lm_device, use_safetensors)
        print(f"[LM|BindingDB] encoding proteins (unique={len(pro_list)})")
        pro_lm_all = _encode_text_list(pro_list, esm_tok, esm_model, lm_device, batch_size=lm_batch_size)
        # 行：释放 ESM
        try:
            del esm_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        # 行：根据样本级索引抽取 LM 向量
        drug_lm = lig_lm_all[pair_lig_idx]                    # [N_total, D_d]
        prot_lm = pro_lm_all[pair_pro_idx]                    # [N_total, D_p]

        # 4. 生成 train/val/test 索引（test 固定为原 CSV 的 test 部分）
        train_idx, val_idx, test_idx = _get_splits_indices(N_total, N_train)

        # 5. 按索引切分并写入缓存（只写 1D）
        def _save_split(sp: str, idx_array: np.ndarray):
            p_1dnpz = _cache_path(sp)                         # 行：对应 split 的 npz 路径
            idx_array = np.asarray(idx_array, dtype=np.int64)

            np.savez(
                p_1dnpz,
                ids=ids_arr[idx_array],           # 行：样本 ID
                smiles=np.array(smiles_all, dtype=object)[idx_array],  # 行：SMILES
                seq=np.array(seq_all, dtype=object)[idx_array],        # 行：蛋白序列
                affinity=y_all[idx_array],         # 行：affinity 标签（或 log 后）
                drug_lm=drug_lm[idx_array],       # 行：药物 LM 特征
                prot_lm=prot_lm[idx_array],       # 行：蛋白 LM 特征
            )

        print("[CACHE|BindingDB] saving train/val/test splits (1D LM only)...")
        _save_split("train", train_idx)
        _save_split("val",   val_idx)
        _save_split("test",  test_idx)
        print("[CACHE|BindingDB] cached done (1D only, no 2D/3D).")

    # —— 判断是否需要重新构建缓存 —— #
    need_build = False
    if split in ("train", "val", "test"):
        p_1d = _cache_path(split)
        if not p_1d.exists():
            need_build = True
    elif split == "all":
        for sp in ("train", "val", "test"):
            p_1d = _cache_path(sp)
            if not p_1d.exists():
                need_build = True
                break
    else:
        raise ValueError(f"split 必须是 'train'/'val'/'test'/'all' 之一，收到: {split!r}")

    if need_build:
        _build_and_cache_all()

    # —— 读取单个拆分（只加载 1D） —— #
    def _load_split(sp: str) -> dict:
        p_1dnpz = _cache_path(sp)                             # 行：某个 split 对应的 npz
        d1 = np.load(p_1dnpz, allow_pickle=True)
        ids    = d1["ids"]                                    # 行：array(object)
        smiles = d1["smiles"]                                 # 行：array(str)
        seq    = d1["seq"]                                    # 行：array(str)
        y      = d1["affinity"].astype(np.float32)            # 行：affinity → float32
        drug_lm = d1["drug_lm"].astype(np.float32)            # 行：ChemBERTa 向量
        prot_lm = d1["prot_lm"].astype(np.float32)            # 行：ESM-2 向量

        N = len(ids)
        assert len(smiles) == N and len(seq) == N and len(y) == N, "1D 长度不一致"
        assert drug_lm.shape[0] == N and prot_lm.shape[0] == N, "LM 行数与 1D 不一致"

        return {
            "ids": ids,              # 行：array(object)
            "y": y,                  # 行：np.float32，训练脚本统一用 pkg['y']
            "smiles": smiles,        # 行：array(str)
            "seq": seq,              # 行：array(str)
            "drug_lm": drug_lm,      # 行：np.ndarray[N, D_d]
            "prot_lm": prot_lm,      # 行：np.ndarray[N, D_p]
        }

    # —— 顶层返回 —— #
    if split in ("train", "val", "test"):
        return {split: _load_split(split)}                    # 行：单个划分
    else:  # "all"
        parts = {sp: _load_split(sp) for sp in ("train", "val", "test")}  # 行：三个划分都读出来
        def _cat(*xs): return np.concatenate(xs, axis=0)      # 行：拼接工具
        all_pkg = {
            "ids":    _cat(parts["train"]["ids"],    parts["val"]["ids"],    parts["test"]["ids"]),
            "y":      _cat(parts["train"]["y"],      parts["val"]["y"],      parts["test"]["y"]).astype(np.float32),
            "smiles": _cat(parts["train"]["smiles"], parts["val"]["smiles"], parts["test"]["smiles"]),
            "seq":    _cat(parts["train"]["seq"],    parts["val"]["seq"],    parts["test"]["seq"]),
            "drug_lm": np.vstack([
                parts["train"]["drug_lm"],
                parts["val"]["drug_lm"],
                parts["test"]["drug_lm"],
            ]).astype(np.float32),
            "prot_lm": np.vstack([
                parts["train"]["prot_lm"],
                parts["val"]["prot_lm"],
                parts["test"]["prot_lm"],
            ]).astype(np.float32),
        }
        return {"all": all_pkg}


def LoadData_fdavis_lm_1d(
    data_dir: str,                               # 行：指向 fdavis 数据集目录（含 affi_info.txt）
    split: str = "all",                          # 行：'train'|'val'|'test'|'all'
    out_1d: str = "../dataset/fdavis/processed_lm_1d",  # 行：1D LM 缓存目录（不存 2D/3D）
    logspace_trans: bool = False,                # 行：是否对 affinity 做 log 变换（默认 False，直接用文件里的值）
    # --- LM 模型相关 ---
    esm2_model_name: str = "facebook/esm2_t33_650M_UR50D",    # 行：ESM-2 checkpoint（蛋白）
    chemberta_model_name: str = "DeepChem/ChemBERTa-77M-MTR", # 行：ChemBERTa-2 checkpoint（药物）
    lm_batch_size: int = 32,                     # 行：LM 前向 batch_size
    use_safetensors: bool = True,                # 行：是否优先使用 safetensors
    # --- 划分相关（在整个 fdavis 中随机划分 train/val/test）---
    split_seed: int = 2023,                      # 行：划分 train/val/test 的随机种子
    train_ratio: float = 0.8,                    # 行：训练集比例
    val_ratio: float = 0.1                       # 行：验证集比例（剩下的是 test）
) -> dict:
    """
    Filtered Davis + LM 数据加载（仅 1D，不生成 2D/3D）：

    - 原始文件：affi_info.txt（制表符分隔，列意义）：
        0 : 样本 ID（或其它）
        1 : 配体 SMILES
        2 : 目标 ID（可忽略）
        3 : 蛋白序列
        4 : 亲和力标签（例如 pKd）

    - 生成模态（全是 1D 向量）：
        drug_lm : ChemBERTa-2(SMILES) 向量 [N, D_d]
        prot_lm : ESM-2(序列) 向量     [N, D_p]

    - 划分策略（只有一份文件，没有官方 train/test）：
        * 对全部 N 个样本按 train_ratio / val_ratio / (1 - train_ratio - val_ratio) 随机划分
        * 用 split_seed 固定随机数，保证复现

    - 缓存（只存 1D + 标签）：
        fdavis_train_1d_lm.npz
        fdavis_val_1d_lm.npz
        fdavis_test_1d_lm.npz

    返回：
        {split: {'ids','y','smiles','seq','drug_lm','prot_lm'}}
    """
    # —— 依赖 —— #
    from pathlib import Path                                  # 行：路径处理
    import numpy as np                                        # 行：数组运算
    import pandas as pd                                       # 行：读取 affi_info.txt
    from collections import OrderedDict                       # 行：做去重同时保持顺序
    import torch                                              # 行：LM 前向 / 设备
    from tqdm import tqdm                                     # 行：进度条
    from transformers import AutoTokenizer, AutoModel         # 行：HF 模型

    # —— LM 编码小工具：兼容不同 HF 模型输出 —— #
    def _encode_text_list(text_list, tokenizer, model, device, batch_size=32):
        """行：输入 list[str]，输出 [N, D] 的 CLS 向量（float32）"""
        all_vecs = []                                         # 行：保存每个 batch 的 CLS 向量
        model.eval()                                          # 行：eval 模式，关闭 dropout 等

        with torch.no_grad():                                 # 行：不需要梯度
            for i in tqdm(
                range(0, len(text_list), batch_size),
                desc="[LM|FDavis] encoding", unit="batch"
            ):
                batch = text_list[i:i + batch_size]           # 行：取一小批文本（SMILES / 序列）
                enc = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=1022,                          # 行：显式限制最大长度，防止极端超长
                    return_tensors="pt"
                )
                # 行：把输入搬到对应设备
                try:
                    enc = enc.to(device)
                except Exception:
                    for k, v in enc.items():
                        if isinstance(v, torch.Tensor):
                            enc[k] = v.to(device)

                # 行：GPU 上用 autocast 节省显存，CPU 上正常前向
                if device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        out = model(**enc)
                else:
                    out = model(**enc)

                # 行：优先从 last_hidden_state 取 [CLS] 向量
                if hasattr(out, "last_hidden_state"):
                    hs = out.last_hidden_state                # [B, L, D]
                elif isinstance(out, (tuple, list)) and len(out) > 0 and hasattr(out[0], "last_hidden_state"):
                    hs = out[0].last_hidden_state
                elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                    hs = out[0]
                else:
                    raise RuntimeError("Unsupported model output structure for LM encoding.")
                CLS = hs[:, 0, :].cpu().numpy().astype("float32")  # 行：第 0 个 token 当句向量
                all_vecs.append(CLS)

        if len(all_vecs) == 0:                                # 行：极端兜底（几乎用不到）
            return np.zeros((0, model.config.hidden_size), dtype="float32")
        return np.concatenate(all_vecs, axis=0).astype("float32")   # 行：拼成 [N, D]

    # —— HF 模型加载辅助（优先 safetensors，失败再回退） —— #
    def _hf_load_model(name: str, device: torch.device, use_safetensors: bool = True):
        """行：优先 safetensors；失败时回退，并给出 torch 版本相关的友好报错"""
        try:
            model = AutoModel.from_pretrained(name, use_safetensors=use_safetensors)  # 行：尝试 safetensors
            model.to(device)                                     # 行：模型搬到设备
            return model
        except TypeError as e:
            # 行：旧版 transformers 不认 use_safetensors 参数
            try:
                model = AutoModel.from_pretrained(name)          # 行：不用 use_safetensors 再试
                model.to(device)
                return model
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load model '{name}' (tried with/without use_safetensors). "
                    f"Err1={e} Err2={e2}"
                ) from e2
        except Exception as e:
            msg = str(e).lower()
            if ("torch" in msg and "2.6" in msg) or ("vulnerability" in msg) or ("cve" in msg):
                # 行：典型的 torch 版本安全检查报错（和 CVE 相关）
                raise RuntimeError(
                    f"Loading HF model '{name}' failed due to transformers' torch-version safety check: {e}\n"
                    "Possible solutions:\n"
                    "  1) 保持现有 torch，优先使用 safetensors（use_safetensors=True），并确保模型 repo 提供 .safetensors；\n"
                    "  2) 升级 torch 到 >=2.6（需要匹配 CUDA）；\n"
                    "  3) 或在受信环境下使用旧版 transformers（不推荐生产）。"
                ) from e
            # 行：不是 torch 版本问题，再试一次普通加载
            try:
                model = AutoModel.from_pretrained(name)
                model.to(device)
                return model
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load model '{name}'. Tried safetensors and non-safetensors. "
                    f"Err1={e} Err2={e2}"
                ) from e2

    # —— 规范路径 —— #
    data_dir = Path(data_dir)                                  # 行：fdavis 原始目录
    out_1d = Path(out_1d)                                      # 行：缓存目录（只放 1D）
    out_1d.mkdir(parents=True, exist_ok=True)                  # 行：若不存在则创建
    split_idx_path = out_1d / "fdavis_splits.npz"              # 行：保存 train/val/test 划分索引

    # —— 构造 per-split 缓存文件名（只含 1D） —— #
    def _cache_path(sp: str):
        return out_1d / f"fdavis_{sp}_1d_lm.npz"               # 行：比如 fdavis_train_1d_lm.npz

    # —— 读取原始 FDavis 数据 —— #
    def _load_raw_fdavis():
        affi_path = data_dir / "affi_info.txt"                 # 行：fdavis 主文件
        if not affi_path.exists():
            raise FileNotFoundError(f"未找到文件: {affi_path}")

        # 行：按制表符分隔，无表头
        df = pd.read_csv(affi_path, sep="\t", header=None)

        # 行：按照你之前的 LoadData_f 约定取列：1=SMILES, 3=seq, 4=affinity
        smiles_list   = df[1].astype(str).tolist()             # 行：配体 SMILES 列
        seq_list      = df[3].astype(str).tolist()             # 行：蛋白序列列
        affinity_list = df[4].astype(float).values             # 行：亲和力标签

        y_all = affinity_list.astype(np.float64)               # 行：先用 float64，后面再转 float32

        # 行：如需 log 变换，可打开 logspace_trans
        if logspace_trans:
            y_all = np.log10(y_all + 1e-8)                     # 行：举例：log10 变换 + 防 0

        return smiles_list, seq_list, y_all                    # 行：返回全部样本的 SMILES/seq/标签

    # —— 构建/加载 train/val/test 索引 —— #
    def _get_splits_indices(N: int):
        """
        行：N = 总样本数；
            若已有 fdavis_splits.npz，则直接加载；
            否则按 train_ratio/val_ratio 随机划分，剩余为 test。
        """
        if split_idx_path.exists():
            data = np.load(split_idx_path, allow_pickle=True)  # 行：读已有划分
            train_idx = data["train_idx"]
            val_idx   = data["val_idx"]
            test_idx  = data["test_idx"]
            return train_idx, val_idx, test_idx

        idx_all = np.arange(N, dtype=np.int64)                 # 行：所有样本索引 [0..N-1]
        rng = np.random.RandomState(split_seed)                # 行：固定随机种子
        rng.shuffle(idx_all)                                   # 行：就地打乱

        n_train = int(N * train_ratio)                         # 行：训练集样本数
        n_val   = int(N * val_ratio)                           # 行：验证集样本数
        # 行：确保 train/val/test 均至少 1 个（极端情况兜底）
        if n_train <= 0:
            n_train = 1
        if n_val <= 0:
            n_val = 1
        if n_train + n_val >= N:
            n_val = max(1, N - n_train - 1)

        train_idx = idx_all[:n_train]                          # 行：前 n_train 个为 train
        val_idx   = idx_all[n_train:n_train + n_val]           # 行：接下来的 n_val 个为 val
        test_idx  = idx_all[n_train + n_val:]                  # 行：剩余的为 test

        # 行：保存划分，方便下次直接复用
        np.savez(
            split_idx_path,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx
        )
        return train_idx, val_idx, test_idx

    # —— 一次性构建所有样本的 1D，然后切分+落盘 —— #
    def _build_and_cache_all():
        # 1. 读原始 fdavis
        smiles_all, seq_all, y_all = _load_raw_fdavis()        # 行：全部样本
        N = len(y_all)                                         # 行：总样本数
        assert len(smiles_all) == N and len(seq_all) == N, "fdavis 长度不一致"

        # 2. 去重后构建 unique ligand / protein 列表
        lig_list = list(OrderedDict.fromkeys(smiles_all).keys())  # 行：unique SMILES
        pro_list = list(OrderedDict.fromkeys(seq_all).keys())     # 行：unique 序列

        # 行：构建 SMILES / 序列 → 索引 的映射
        lig2idx = {smi: i for i, smi in enumerate(lig_list)}
        pro2idx = {seq: i for i, seq in enumerate(pro_list)}

        # 行：样本级配体 / 蛋白索引（global）
        pair_lig_idx = np.array([lig2idx[s] for s in smiles_all], dtype=np.int64)
        pair_pro_idx = np.array([pro2idx[s] for s in seq_all],    dtype=np.int64)

        # 行：样本 ID（简单用行号）
        ids_arr = np.array([f"fd_{i}" for i in range(N)], dtype=object)

        # 3. 对 unique ligand / protein 做 LM 编码，节省算力
        lm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 行：LM 工作设备
        print(f"[LM|FDavis] using device: {lm_device}")

        # 3.1 Drug LM（ChemBERTa）
        print(f"[LM|FDavis] loading ChemBERTa model: {chemberta_model_name} (use_safetensors={use_safetensors})")
        chem_tok   = AutoTokenizer.from_pretrained(chemberta_model_name)          # 行：配体 tokenizer
        chem_model = _hf_load_model(chemberta_model_name, lm_device, use_safetensors)  # 行：配体模型
        print(f"[LM|FDavis] encoding ligands (unique={len(lig_list)})")
        lig_lm_all = _encode_text_list(lig_list, chem_tok, chem_model, lm_device, batch_size=lm_batch_size)
        # 行：释放 ChemBERTa，节省显存
        try:
            del chem_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        # 3.2 Protein LM（ESM-2）
        print(f"[LM|FDavis] loading ESM model: {esm2_model_name} (use_safetensors={use_safetensors})")
        esm_tok   = AutoTokenizer.from_pretrained(esm2_model_name)                # 行：蛋白 tokenizer
        esm_model = _hf_load_model(esm2_model_name, lm_device, use_safetensors)   # 行：蛋白模型
        print(f"[LM|FDavis] encoding proteins (unique={len(pro_list)})")
        pro_lm_all = _encode_text_list(pro_list, esm_tok, esm_model, lm_device, batch_size=lm_batch_size)
        # 行：释放 ESM，节省显存
        try:
            del esm_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        # 行：根据样本索引抽取 LM 向量
        drug_lm = lig_lm_all[pair_lig_idx]                    # [N, D_d]
        prot_lm = pro_lm_all[pair_pro_idx]                    # [N, D_p]

        # 4. 生成 train/val/test 索引
        train_idx, val_idx, test_idx = _get_splits_indices(N)

        # 5. 按索引切分并写入缓存（只写 1D）
        smiles_all_arr = np.array(smiles_all, dtype=object)   # 行：转成 array，方便索引
        seq_all_arr    = np.array(seq_all,    dtype=object)

        def _save_split(sp: str, idx_array: np.ndarray):
            p_1dnpz = _cache_path(sp)                         # 行：当前 split 的 npz 路径
            idx_array = np.asarray(idx_array, dtype=np.int64)

            np.savez(
                p_1dnpz,
                ids=ids_arr[idx_array],                       # 行：样本 ID
                smiles=smiles_all_arr[idx_array],             # 行：SMILES
                seq=seq_all_arr[idx_array],                   # 行：蛋白序列
                affinity=y_all[idx_array],                    # 行：亲和力标签（或 log 后）
                drug_lm=drug_lm[idx_array],                   # 行：药物 LM 特征
                prot_lm=prot_lm[idx_array],                   # 行：蛋白 LM 特征
            )

        print("[CACHE|FDavis] saving train/val/test splits (1D LM only)...")
        _save_split("train", train_idx)
        _save_split("val",   val_idx)
        _save_split("test",  test_idx)
        print("[CACHE|FDavis] cached done (1D only, no 2D/3D).")

    # —— 判断是否需要重新构建缓存 —— #
    need_build = False                                        # 行：标记是否要重新跑 LM 编码
    if split in ("train", "val", "test"):
        p_1d = _cache_path(split)                             # 行：当前 split 的缓存路径
        if not p_1d.exists():
            need_build = True
    elif split == "all":
        for sp in ("train", "val", "test"):
            p_1d = _cache_path(sp)
            if not p_1d.exists():                             # 行：只要有一个缺失就需要重新构建
                need_build = True
                break
    else:
        raise ValueError(f"split 必须是 'train'/'val'/'test'/'all' 之一，收到: {split!r}")

    if need_build:
        _build_and_cache_all()                                # 行：第一次或缓存缺失时构建全部缓存

    # —— 读取单个拆分（只加载 1D） —— #
    def _load_split(sp: str) -> dict:
        p_1dnpz = _cache_path(sp)                             # 行：某个 split 对应的 npz
        d1 = np.load(p_1dnpz, allow_pickle=True)              # 行：读入 npz

        ids    = d1["ids"]                                    # 行：array(object)
        smiles = d1["smiles"]                                 # 行：array(str)
        seq    = d1["seq"]                                    # 行：array(str)
        y      = d1["affinity"].astype(np.float32)            # 行：亲和力 → float32
        drug_lm = d1["drug_lm"].astype(np.float32)            # 行：ChemBERTa 向量
        prot_lm = d1["prot_lm"].astype(np.float32)            # 行：ESM-2 向量

        N = len(ids)                                          # 行：样本数
        assert len(smiles) == N and len(seq) == N and len(y) == N, "1D 长度不一致"
        assert drug_lm.shape[0] == N and prot_lm.shape[0] == N, "LM 行数与 1D 不一致"

        return {
            "ids": ids,              # 行：array(object)
            "y": y,                  # 行：np.float32，训练脚本统一用 pkg['y']
            "smiles": smiles,        # 行：array(str)
            "seq": seq,              # 行：array(str)
            "drug_lm": drug_lm,      # 行：np.ndarray[N, D_d]
            "prot_lm": prot_lm,      # 行：np.ndarray[N, D_p]
        }

    # —— 顶层返回 —— #
    if split in ("train", "val", "test"):
        return {split: _load_split(split)}                    # 行：只返回一个划分
    else:  # "all"
        parts = {sp: _load_split(sp) for sp in ("train", "val", "test")}  # 行：三个划分都读出来

        def _cat(*xs): return np.concatenate(xs, axis=0)      # 行：拼接工具

        all_pkg = {
            "ids":    _cat(parts["train"]["ids"],    parts["val"]["ids"],    parts["test"]["ids"]),
            "y":      _cat(parts["train"]["y"],      parts["val"]["y"],      parts["test"]["y"]).astype(np.float32),
            "smiles": _cat(parts["train"]["smiles"], parts["val"]["smiles"], parts["test"]["smiles"]),
            "seq":    _cat(parts["train"]["seq"],    parts["val"]["seq"],    parts["test"]["seq"]),
            "drug_lm": np.vstack([                   # 行：三块 LM 特征在第 0 维拼接
                parts["train"]["drug_lm"],
                parts["val"]["drug_lm"],
                parts["test"]["drug_lm"],
            ]).astype(np.float32),
            "prot_lm": np.vstack([
                parts["train"]["prot_lm"],
                parts["val"]["prot_lm"],
                parts["test"]["prot_lm"],
            ]).astype(np.float32),
        }
        return {"all": all_pkg}                              # 行：all 模式下把整体打包返回


def LoadData_metz_lm_1d(
    data_dir: str,                                # Metz 数据集目录（包含 drug_info.txt, targ_info.txt, affi_info.txt）
    split: str = "all",                           # 'train'|'val'|'test'|'all'
    out_1d: str = "../dataset/metz/processed_lm_1d",  # 1D LM 缓存目录
    logspace_trans: bool = False,                 # 是否对亲和力做 log10 变换
    esm2_model_name: str = "facebook/esm2_t33_650M_UR50D",  # ESM-2 蛋白 LM
    chemberta_model_name: str = "DeepChem/ChemBERTa-77M-MTR", # ChemBERTa-2 药物 LM
    lm_batch_size: int = 32,
    use_safetensors: bool = True,
    split_seed: int = 2023,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> dict:
    """
    加载 Metz 数据集 + LM 编码，仅 1D 特征，不生成 2D/3D。
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch
    from tqdm import tqdm
    from collections import OrderedDict
    from transformers import AutoTokenizer, AutoModel

    data_dir = Path(data_dir)
    out_1d = Path(out_1d)
    out_1d.mkdir(parents=True, exist_ok=True)
    split_idx_path = out_1d / "metz_splits.npz"

    # ----------------- LM 编码辅助 ----------------- #
    def _encode_text_list(text_list, tokenizer, model, device, batch_size=32):
        all_vecs = []
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(text_list), batch_size), desc="[LM|Metz] encoding", unit="batch"):
                batch = text_list[i:i + batch_size]
                enc = tokenizer(batch, padding=True, truncation=True, max_length=1022, return_tensors="pt")
                try:
                    enc = {k:v.to(device) for k,v in enc.items()}
                except Exception:
                    pass
                if device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        out = model(**enc)
                else:
                    out = model(**enc)
                if hasattr(out, "last_hidden_state"):
                    hs = out.last_hidden_state
                elif isinstance(out, (tuple,list)) and len(out)>0 and hasattr(out[0], "last_hidden_state"):
                    hs = out[0].last_hidden_state
                elif isinstance(out, (tuple,list)) and len(out)>0 and torch.is_tensor(out[0]):
                    hs = out[0]
                else:
                    raise RuntimeError("Unsupported model output structure")
                CLS = hs[:,0,:].cpu().numpy().astype("float32")
                all_vecs.append(CLS)
        if len(all_vecs) == 0:
            return np.zeros((0, model.config.hidden_size), dtype="float32")
        return np.concatenate(all_vecs, axis=0).astype("float32")

    def _hf_load_model(name, device, use_safetensors=True):
        try:
            model = AutoModel.from_pretrained(name, use_safetensors=use_safetensors)
            model.to(device)
            return model
        except Exception:
            model = AutoModel.from_pretrained(name)
            model.to(device)
            return model

    # ----------------- 读取原始文件 ----------------- #
    drug_df = pd.read_csv(data_dir / "drug_info.txt", sep="\t", header=None)
    targ_df = pd.read_csv(data_dir / "targ_info.txt", sep="\t", header=None)
    affi_df = pd.read_csv(data_dir / "affi_info.txt", sep="\t", header=None)

    drug_dict = dict(zip(drug_df[0], drug_df[2]))   # drug_id -> SMILES
    prot_dict = dict(zip(targ_df[0], targ_df[2]))   # targ_id -> sequence

    smiles_list, seq_list, y_list = [], [], []
    for _, row in affi_df.iterrows():
        drug_id, targ_id, affi = row[0], row[1], row[2]
        if drug_id in drug_dict and targ_id in prot_dict:
            smiles_list.append(drug_dict[drug_id])
            seq_list.append(prot_dict[targ_id])
            y_list.append(float(affi))
    y_all = np.array(y_list, dtype=np.float64)
    if logspace_trans:
        y_all = np.log10(y_all + 1e-8)

    # ----------------- 划分索引 ----------------- #
    N = len(y_all)
    if split_idx_path.exists():
        data = np.load(split_idx_path, allow_pickle=True)
        train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
    else:
        idx_all = np.arange(N)
        rng = np.random.RandomState(split_seed)
        rng.shuffle(idx_all)
        n_train = int(N * train_ratio)
        n_val = int(N * val_ratio)
        if n_train <= 0: n_train = 1
        if n_val <= 0: n_val = 1
        if n_train + n_val >= N: n_val = max(1, N-n_train-1)
        train_idx = idx_all[:n_train]
        val_idx = idx_all[n_train:n_train+n_val]
        test_idx = idx_all[n_train+n_val:]
        np.savez(split_idx_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    # ----------------- 构建 unique LM ----------------- #
    lig_list = list(OrderedDict.fromkeys(smiles_list).keys())
    pro_list = list(OrderedDict.fromkeys(seq_list).keys())
    lig2idx = {smi:i for i,smi in enumerate(lig_list)}
    pro2idx = {seq:i for i,seq in enumerate(pro_list)}
    pair_lig_idx = np.array([lig2idx[s] for s in smiles_list], dtype=np.int64)
    pair_pro_idx = np.array([pro2idx[s] for s in seq_list], dtype=np.int64)
    ids_arr = np.array([f"metz_{i}" for i in range(N)], dtype=object)

    lm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LM|Metz] using device: {lm_device}")

    chem_tok = AutoTokenizer.from_pretrained(chemberta_model_name)
    chem_model = _hf_load_model(chemberta_model_name, lm_device, use_safetensors)
    lig_lm_all = _encode_text_list(lig_list, chem_tok, chem_model, lm_device, batch_size=lm_batch_size)
    del chem_model; torch.cuda.empty_cache()

    esm_tok = AutoTokenizer.from_pretrained(esm2_model_name)
    esm_model = _hf_load_model(esm2_model_name, lm_device, use_safetensors)
    pro_lm_all = _encode_text_list(pro_list, esm_tok, esm_model, lm_device, batch_size=lm_batch_size)
    del esm_model; torch.cuda.empty_cache()

    drug_lm = lig_lm_all[pair_lig_idx]
    prot_lm = pro_lm_all[pair_pro_idx]

    # ----------------- 切分保存 ----------------- #
    def _cache_path(sp):
        return out_1d / f"metz_{sp}_1d_lm.npz"

    def _save_split(sp, idx_array):
        np.savez(
            _cache_path(sp),
            ids=ids_arr[idx_array],
            smiles=np.array(smiles_list, dtype=object)[idx_array],
            seq=np.array(seq_list, dtype=object)[idx_array],
            affinity=y_all[idx_array],
            drug_lm=drug_lm[idx_array],
            prot_lm=prot_lm[idx_array],
        )

    for sp, idx in zip(["train","val","test"], [train_idx, val_idx, test_idx]):
        _save_split(sp, idx)
    print("[CACHE|Metz] cached 1D LM features done.")

    # ----------------- 读取拆分 ----------------- #
    def _load_split(sp):
        d1 = np.load(_cache_path(sp), allow_pickle=True)
        return {
            "ids": d1["ids"],
            "y": d1["affinity"].astype(np.float32),
            "smiles": d1["smiles"],
            "seq": d1["seq"],
            "drug_lm": d1["drug_lm"].astype(np.float32),
            "prot_lm": d1["prot_lm"].astype(np.float32),
        }

    if split in ("train","val","test"):
        return {split: _load_split(split)}
    else:
        parts = {sp:_load_split(sp) for sp in ("train","val","test")}
        all_pkg = {
            "ids": np.concatenate([parts["train"]["ids"], parts["val"]["ids"], parts["test"]["ids"]]),
            "y": np.concatenate([parts["train"]["y"], parts["val"]["y"], parts["test"]["y"]]).astype(np.float32),
            "smiles": np.concatenate([parts["train"]["smiles"], parts["val"]["smiles"], parts["test"]["smiles"]]),
            "seq": np.concatenate([parts["train"]["seq"], parts["val"]["seq"], parts["test"]["seq"]]),
            "drug_lm": np.vstack([parts["train"]["drug_lm"], parts["val"]["drug_lm"], parts["test"]["drug_lm"]]).astype(np.float32),
            "prot_lm": np.vstack([parts["train"]["prot_lm"], parts["val"]["prot_lm"], parts["test"]["prot_lm"]]).astype(np.float32),
        }
        return {"all": all_pkg}


def LoadData_toxcast_lm_1d(
    data_dir: str,                               # 行：指向 ToxCast 数据集目录（含 data_train.csv / data_test.csv）
    split: str = "all",                          # 行：'train'|'val'|'test'|'all'
    out_1d: str = "../dataset/toxcast/processed_lm_1d",  # 行：1D LM 缓存目录（不存 2D/3D）
    logspace_trans: bool = True,                # 行：是否对 label 做 log 变换（默认 False）
    # --- LM 模型相关 ---
    esm2_model_name: str = "facebook/esm2_t33_650M_UR50D",    # 行：ESM-2 checkpoint（蛋白）
    chemberta_model_name: str = "DeepChem/ChemBERTa-77M-MTR", # 行：ChemBERTa-2 checkpoint（药物）
    lm_batch_size: int = 32,                     # 行：LM 前向 batch_size
    use_safetensors: bool = True,                # 行：是否优先使用 safetensors
    # --- 划分相关（只在原 train 部分里划分 train/val，test 固定为 data_test.csv）---
    split_seed: int = 2023,                      # 行：划分 train/val 的随机种子
    train_ratio: float = 0.8,                    # 行：原 train 部分中 train 所占比例（会与 val_ratio 归一化）
    val_ratio: float = 0.2                       # 行：原 train 部分中 val 所占比例（会与 train_ratio 归一化）
) -> dict:
    """
    ToxCast + LM 数据加载（仅 1D，不生成 2D/3D）：

    - 原始文件（CSV）：
        data_train.csv :
            ... 其他列 ...
            sequence : 蛋白序列
            smiles   : 配体 SMILES
            label    : 数值标签（当前文件里大量为 1，少部分 >1）
        data_test.csv :
            同上结构，作为固定测试集

    - 生成模态（全是 1D 向量）：
        drug_lm : ChemBERTa-2(SMILES) 向量 [N, D_d]
        prot_lm : ESM-2(序列) 向量     [N, D_p]

    - 划分策略：
        * 先把 train/test 两个 CSV 纵向拼接，得到总样本数 N = N_train + N_test
        * train_idx / val_idx 只在“原 train 部分（前 N_train 个样本）”里随机划分
        * test_idx = 原 data_test.csv 那一部分（后 N_test 个样本），不参与随机划分

    - 缓存（只存 1D + 标签）：
        toxcast_train_1d_lm.npz
        toxcast_val_1d_lm.npz
        toxcast_test_1d_lm.npz

    返回：
        {split: {'ids','y','smiles','seq','drug_lm','prot_lm'}}
    """
    # —— 依赖 —— #
    from pathlib import Path                                  # 行：路径处理
    import numpy as np                                        # 行：数组运算
    import pandas as pd                                       # 行：读取 CSV
    from collections import OrderedDict                       # 行：保持去重后的顺序
    import torch                                              # 行：LM 前向 / 设备
    from tqdm import tqdm                                     # 行：进度条
    from transformers import AutoTokenizer, AutoModel         # 行：HF 模型

    # ====================== LM 编码小工具 ====================== #
    def _encode_text_list(text_list, tokenizer, model, device, batch_size=32):
        """行：输入 list[str]，输出 [N, D] 的 CLS 向量（float32）"""
        all_vecs = []                              # 行：保存每个 batch 的 CLS 向量
        model.eval()                               # 行：eval 模式，关闭 dropout 等

        with torch.no_grad():                      # 行：不需要梯度
            for i in tqdm(
                range(0, len(text_list), batch_size),
                desc="[LM|ToxCast] encoding", unit="batch"
            ):
                batch = text_list[i:i + batch_size]    # 行：这一小批的序列/SMILES
                enc = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=1022,                   # 行：显式限制长度，避免极端超长
                    return_tensors="pt"
                )
                # 行：把输入搬到对应设备
                try:
                    enc = enc.to(device)
                except Exception:
                    for k, v in enc.items():
                        if isinstance(v, torch.Tensor):
                            enc[k] = v.to(device)

                # 行：GPU 上用 autocast 省显存
                if device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        out = model(**enc)
                else:
                    out = model(**enc)

                # 行：统一拿 last_hidden_state[:,0,:] 作为 CLS 向量
                if hasattr(out, "last_hidden_state"):
                    hs = out.last_hidden_state            # [B, L, D]
                elif isinstance(out, (tuple, list)) and len(out) > 0 and hasattr(out[0], "last_hidden_state"):
                    hs = out[0].last_hidden_state
                elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                    hs = out[0]
                else:
                    raise RuntimeError("Unsupported model output structure for LM encoding.")

                CLS = hs[:, 0, :].cpu().numpy().astype("float32")  # 行：CLS token 向量
                all_vecs.append(CLS)

        if len(all_vecs) == 0:
            # 行：极端兜底（理论上不会走到）
            return np.zeros((0, model.config.hidden_size), dtype="float32")
        return np.concatenate(all_vecs, axis=0).astype("float32")  # 行：拼成 [N, D]

    # ====================== HF 模型加载辅助 ====================== #
    def _hf_load_model(name: str, device: torch.device, use_safetensors: bool = True):
        """行：优先 safetensors；失败时回退，并给出 torch 版本相关的友好报错"""
        try:
            model = AutoModel.from_pretrained(name, use_safetensors=use_safetensors)
            model.to(device)
            return model
        except TypeError as e:
            # 行：旧版 transformers 不认 use_safetensors
            try:
                model = AutoModel.from_pretrained(name)
                model.to(device)
                return model
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load model '{name}' (tried with/without use_safetensors). "
                    f"Err1={e} Err2={e2}"
                ) from e2
        except Exception as e:
            msg = str(e).lower()
            if ("torch" in msg and "2.6" in msg) or ("vulnerability" in msg) or ("cve" in msg):
                raise RuntimeError(
                    f"Loading HF model '{name}' failed due to transformers' torch-version safety check: {e}\n"
                    "Possible solutions:\n"
                    "  1) 保持现有 torch，优先使用 safetensors（use_safetensors=True），并确保模型 repo 提供 .safetensors；\n"
                    "  2) 升级 torch 到 >=2.6（需要匹配 CUDA）；\n"
                    "  3) 或者在受信环境用旧版 transformers（不推荐生产）。"
                ) from e
            # 行：不是 torch 版本问题，再试一次普通加载
            try:
                model = AutoModel.from_pretrained(name)
                model.to(device)
                return model
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load model '{name}'. Tried safetensors and non-safetensors. "
                    f"Err1={e} Err2={e2}"
                ) from e2

    # ====================== 路径和缓存文件 ====================== #
    data_dir = Path(data_dir)                                  # 行：ToxCast 原始路径
    out_1d = Path(out_1d)                                      # 行：缓存路径（只放 1D）
    out_1d.mkdir(parents=True, exist_ok=True)                  # 行：没有则创建
    split_idx_path = out_1d / "toxcast_splits.npz"             # 行：划分索引文件（保存 train/val/test 下标）

    def _cache_path(sp: str):
        """行：构造 per-split 缓存文件名（只含 1D）"""
        return out_1d / f"toxcast_{sp}_1d_lm.npz"              # 行：toxcast_train_1d_lm.npz 之类

    # ====================== 读取原始 ToxCast ====================== #
    def _load_raw_toxcast():
        train_path = data_dir / "data_train.csv"               # 行：训练 CSV
        test_path  = data_dir / "data_test.csv"                # 行：测试 CSV

        if not train_path.exists():
            raise FileNotFoundError(f"未找到文件: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"未找到文件: {test_path}")

        train_df = pd.read_csv(train_path)                     # 行：读入训练数据
        test_df  = pd.read_csv(test_path)                      # 行：读入测试数据

        # 行：需要的列名（其他列先忽略）
        required_cols = ["smiles", "sequence", "label"]
        for col in required_cols:
            if col not in train_df.columns:
                raise ValueError(f"data_train.csv 缺少列: {col}")
            if col not in test_df.columns:
                raise ValueError(f"data_test.csv 缺少列: {col}")

        # 行：从 DataFrame 提取 SMILES / 序列 / 标签
        smiles_train = train_df["smiles"].astype(str).tolist()
        seq_train    = train_df["sequence"].astype(str).tolist()
        y_train      = train_df["label"].astype(float).values

        smiles_test = test_df["smiles"].astype(str).tolist()
        seq_test    = test_df["sequence"].astype(str).tolist()
        y_test      = test_df["label"].astype(float).values

        # 行：合并 train + test，后续统一做 LM 编码
        smiles_all = smiles_train + smiles_test                # 行：总长度 N = N_train + N_test
        seq_all    = seq_train + seq_test
        y_all      = np.concatenate([y_train, y_test], axis=0)

        # 行：可选：对 label 做 log 变换（默认不用）
        if logspace_trans:
            y_all = np.log10(y_all + 1e-8)

        N_train = len(smiles_train)                            # 行：原 train 部分样本数
        N_test  = len(smiles_test)                             # 行：原 test 部分样本数

        return smiles_all, seq_all, y_all, N_train, N_test

    # ====================== 构建/加载 train/val/test 索引 ====================== #
    def _get_splits_indices(N_total: int, N_train: int):
        """
        行：N_total = 总样本数（train+test）
            N_train = 原 train 部分样本数（0..N_train-1）
            策略：仅在 0..N_train-1 内随机划分 train/val；test 固定为 N_train..N_total-1
        """
        if split_idx_path.exists():
            data = np.load(split_idx_path, allow_pickle=True)
            train_idx = data["train_idx"]
            val_idx   = data["val_idx"]
            test_idx  = data["test_idx"]
            return train_idx, val_idx, test_idx

        # 行：只在原 train 部分做随机划分
        idx_train_pool = np.arange(N_train, dtype=np.int64)    # 行：原 train 部分索引 [0..N_train-1]
        rng = np.random.RandomState(split_seed)                # 行：固定随机种子
        rng.shuffle(idx_train_pool)                            # 行：打乱顺序

        # 行：把 train_ratio / val_ratio 归一化后在原 train 部分中分配
        total_r = train_ratio + val_ratio
        if total_r <= 0:
            t_ratio_eff, v_ratio_eff = 0.8, 0.2                # 行：兜底默认 8:2
        else:
            t_ratio_eff = train_ratio / total_r
            v_ratio_eff = val_ratio / total_r

        n_train_eff = int(N_train * t_ratio_eff)               # 行：训练样本数（向下取整）
        # 行：保证 train/val 至少各有 1 个样本
        if n_train_eff <= 0:
            n_train_eff = 1
        if n_train_eff >= N_train:
            n_train_eff = N_train - 1

        train_idx = idx_train_pool[:n_train_eff]               # 行：划分后的 train 索引（全局）
        val_idx   = idx_train_pool[n_train_eff:]               # 行：划分后的 val 索引（全局）

        # 行：test 部分索引 = [N_train .. N_total-1]
        test_idx  = np.arange(N_train, N_total, dtype=np.int64)

        # 行：保存划分结果，方便下次直接加载
        np.savez(split_idx_path,
                 train_idx=train_idx,
                 val_idx=val_idx,
                 test_idx=test_idx)
        return train_idx, val_idx, test_idx

    # ====================== 一次性构建所有样本的 1D，然后切分+落盘 ====================== #
    def _build_and_cache_all():
        # 1. 读原始 ToxCast
        smiles_all, seq_all, y_all, N_train, N_test = _load_raw_toxcast()
        N_total = len(y_all)                                  # 行：总样本数 = N_train + N_test
        assert N_total == N_train + N_test, "样本数不一致，请检查 ToxCast CSV"

        # 2. 去重后构建 unique ligand / protein 列表，用于 LM 编码
        lig_list = list(OrderedDict.fromkeys(smiles_all).keys())  # 行：unique SMILES，保持首次出现顺序
        pro_list = list(OrderedDict.fromkeys(seq_all).keys())     # 行：unique 序列，保持顺序

        # 行：构建 SMILES / 序列 → 索引 的映射
        lig2idx = {smi: i for i, smi in enumerate(lig_list)}
        pro2idx = {seq: i for i, seq in enumerate(pro_list)}

        # 行：样本级配体 / 蛋白索引
        pair_lig_idx = np.array([lig2idx[s] for s in smiles_all], dtype=np.int64)
        pair_pro_idx = np.array([pro2idx[s] for s in seq_all],    dtype=np.int64)

        # 构造样本 ID：train_0.. / test_0..
        ids_train = [f"tr_{i}" for i in range(N_train)]       # 行：原 train 部分 ID
        ids_test  = [f"te_{i}" for i in range(N_test)]        # 行：原 test 部分 ID
        ids_arr   = np.array(ids_train + ids_test, dtype=object)

        # 3. 对 unique ligand / protein 做 LM 编码，节省算力
        lm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LM|ToxCast] using device: {lm_device}")

        # 3.1 Drug LM（ChemBERTa）
        print(f"[LM|ToxCast] loading ChemBERTa model: {chemberta_model_name} (use_safetensors={use_safetensors})")
        chem_tok   = AutoTokenizer.from_pretrained(chemberta_model_name)
        chem_model = _hf_load_model(chemberta_model_name, lm_device, use_safetensors)
        print(f"[LM|ToxCast] encoding ligands (unique={len(lig_list)})")
        lig_lm_all = _encode_text_list(lig_list, chem_tok, chem_model, lm_device, batch_size=lm_batch_size)
        # 行：释放 Chem 模型，省显存
        try:
            del chem_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        # 3.2 Protein LM（ESM-2）
        print(f"[LM|ToxCast] loading ESM model: {esm2_model_name} (use_safetensors={use_safetensors})")
        esm_tok   = AutoTokenizer.from_pretrained(esm2_model_name)
        esm_model = _hf_load_model(esm2_model_name, lm_device, use_safetensors)
        print(f"[LM|ToxCast] encoding proteins (unique={len(pro_list)})")
        pro_lm_all = _encode_text_list(pro_list, esm_tok, esm_model, lm_device, batch_size=lm_batch_size)
        # 行：释放 ESM
        try:
            del esm_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        # 行：根据样本级索引抽取 LM 向量
        drug_lm = lig_lm_all[pair_lig_idx]                    # [N_total, D_d]
        prot_lm = pro_lm_all[pair_pro_idx]                    # [N_total, D_p]

        # 4. 生成 train/val/test 索引（test 固定为原 CSV 的 test 部分）
        train_idx, val_idx, test_idx = _get_splits_indices(N_total, N_train)

        # 5. 按索引切分并写入缓存（只写 1D）
        def _save_split(sp: str, idx_array: np.ndarray):
            p_1dnpz = _cache_path(sp)                         # 行：对应 split 的 npz 路径
            idx_array = np.asarray(idx_array, dtype=np.int64)

            np.savez(
                p_1dnpz,
                ids=ids_arr[idx_array],           # 行：样本 ID
                smiles=np.array(smiles_all, dtype=object)[idx_array],  # 行：SMILES
                seq=np.array(seq_all, dtype=object)[idx_array],        # 行：蛋白序列
                label=y_all[idx_array],           # 行：原始标签（或 log 后）
                drug_lm=drug_lm[idx_array],       # 行：药物 LM 特征
                prot_lm=prot_lm[idx_array],       # 行：蛋白 LM 特征
            )

        print("[CACHE|ToxCast] saving train/val/test splits (1D LM only)...")
        _save_split("train", train_idx)
        _save_split("val",   val_idx)
        _save_split("test",  test_idx)
        print("[CACHE|ToxCast] cached done (1D only, no 2D/3D).")

    # ====================== 判断是否需要重新构建缓存 ====================== #
    need_build = False
    if split in ("train", "val", "test"):
        p_1d = _cache_path(split)
        if not p_1d.exists():
            need_build = True
    elif split == "all":
        for sp in ("train", "val", "test"):
            p_1d = _cache_path(sp)
            if not p_1d.exists():
                need_build = True
                break
    else:
        raise ValueError(f"split 必须是 'train'/'val'/'test'/'all' 之一，收到: {split!r}")

    if need_build:
        _build_and_cache_all()

    # ====================== 读取单个拆分（只加载 1D） ====================== #
    def _load_split(sp: str) -> dict:
        p_1dnpz = _cache_path(sp)                             # 行：某个 split 对应的 npz
        d1 = np.load(p_1dnpz, allow_pickle=True)
        ids    = d1["ids"]                                    # 行：array(object)
        smiles = d1["smiles"]                                 # 行：array(str)
        seq    = d1["seq"]                                    # 行：array(str)
        y      = d1["label"].astype(np.float32)               # 行：label → float32
        drug_lm = d1["drug_lm"].astype(np.float32)            # 行：ChemBERTa 向量
        prot_lm = d1["prot_lm"].astype(np.float32)            # 行：ESM-2 向量

        N = len(ids)
        assert len(smiles) == N and len(seq) == N and len(y) == N, "1D 长度不一致"
        assert drug_lm.shape[0] == N and prot_lm.shape[0] == N, "LM 行数与 1D 不一致"

        return {
            "ids": ids,              # 行：array(object)
            "y": y,                  # 行：np.float32，训练脚本统一用 pkg['y']
            "smiles": smiles,        # 行：array(str)
            "seq": seq,              # 行：array(str)
            "drug_lm": drug_lm,      # 行：np.ndarray[N, D_d]
            "prot_lm": prot_lm,      # 行：np.ndarray[N, D_p]
        }

    # ====================== 顶层返回 ====================== #
    if split in ("train", "val", "test"):
        return {split: _load_split(split)}                    # 行：单个划分
    else:  # "all"
        parts = {sp: _load_split(sp) for sp in ("train", "val", "test")}  # 行：三个划分都读出来
        def _cat(*xs): return np.concatenate(xs, axis=0)      # 行：拼接工具
        all_pkg = {
            "ids":    _cat(parts["train"]["ids"],    parts["val"]["ids"],    parts["test"]["ids"]),
            "y":      _cat(parts["train"]["y"],      parts["val"]["y"],      parts["test"]["y"]).astype(np.float32),
            "smiles": _cat(parts["train"]["smiles"], parts["val"]["smiles"], parts["test"]["smiles"]),
            "seq":    _cat(parts["train"]["seq"],    parts["val"]["seq"],    parts["test"]["seq"]),
            "drug_lm": np.vstack([
                parts["train"]["drug_lm"],
                parts["val"]["drug_lm"],
                parts["test"]["drug_lm"],
            ]).astype(np.float32),
            "prot_lm": np.vstack([
                parts["train"]["prot_lm"],
                parts["val"]["prot_lm"],
                parts["test"]["prot_lm"],
            ]).astype(np.float32),
        }
        return {"all": all_pkg}


def LoadData_il6_aai_lm_1d(
    data_dir: str,                               # 行：指向 IL-6 抗原-抗体数据集目录（含 il6_aai_dataset.csv）
    csv_name: str = "il6_aai_dataset.csv",       # 行：CSV 文件名
    split: str = "all",                          # 行：'train'|'val'|'test'|'all'
    out_1d: str = "../dataset/AVIDa-hIL6/processed_lm_1d",  # 行：1D LM 缓存目录（不存 2D/3D）
    logspace_trans: bool = False,                # 行：是否对 label 做 log 变换（当前是 0/1 分类，一般不用）
    # --- LM 模型相关（VHH & 抗原都用 ESM）--- #
    esm_ab_model_name: str = "facebook/esm2_t33_650M_UR50D",  # 行：VHH 侧 ESM-2 模型
    esm_ag_model_name: str = "facebook/esm2_t33_650M_UR50D",  # 行：抗原侧 ESM-2 模型（可同可不同）
    lm_batch_size: int = 32,                     # 行：LM 前向 batch_size
    use_safetensors: bool = True,                # 行：是否优先使用 safetensors
    # --- 划分相关（直接在整张表上划 train/val/test）--- #
    split_seed: int = 2023,                      # 行：划分 train/val/test 的随机种子
    train_ratio: float = 0.8,                    # 行：整体样本中 train 所占比例
    val_ratio: float = 0.1                       # 行：整体样本中 val 所占比例（test = 剩余）
) -> dict:
    """
    IL-6 抗原-抗体 (AVIDa-hIL6) + LM 数据加载（仅 1D，不生成 2D/3D）：

    原始 CSV (il6_aai_dataset.csv) 需要包含列：
        VHH_sequence : 抗体/VHH 序列
        Ag_sequence  : 抗原序列
        Ag_label     : 抗原名字（例如 IL-6_WTs）
        label        : 0/1 标签

    生成模态（全是 1D 向量）：
        ab_lm : ESM-2(VHH) 向量 [N, D_ab]
        ag_lm : ESM-2(Ag)  向量 [N, D_ag]

    划分策略：
        * 在整张表上随机划分 train/val/test
        * 划分索引保存在 out_1d/il6_aai_splits.npz，保证可复现

    缓存（只存 1D + 标签 + 序列）：
        il6_aai_train_1d_lm.npz
        il6_aai_val_1d_lm.npz
        il6_aai_test_1d_lm.npz

    返回：
        {split: {'ids','y','vhh_seq','ag_seq','ag_label','ab_lm','ag_lm'}}
    """
    # —— 依赖 —— #
    from pathlib import Path                                  # 行：路径处理
    import numpy as np                                        # 行：数组运算
    import pandas as pd                                       # 行：读取 CSV
    from collections import OrderedDict                       # 行：保持去重后的顺序
    import torch                                              # 行：LM 前向 / 设备
    from tqdm import tqdm                                     # 行：进度条
    from transformers import AutoTokenizer, AutoModel         # 行：HF 模型

    # ====================== LM 编码小工具 ====================== #
    def _encode_text_list(text_list, tokenizer, model, device, batch_size=32):
        """行：输入 list[str]，输出 [N, D] 的 CLS 向量（float32）"""
        all_vecs = []                              # 行：保存每个 batch 的 CLS 向量
        model.eval()                               # 行：eval 模式，关闭 dropout 等

        with torch.no_grad():                      # 行：不需要梯度
            for i in tqdm(
                range(0, len(text_list), batch_size),
                desc="[LM|IL6-AAI] encoding", unit="batch"
            ):
                batch = text_list[i:i + batch_size]    # 行：这一小批的序列
                enc = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=1022,                   # 行：显式限制长度，避免极端超长
                    return_tensors="pt"
                )
                # 行：把输入搬到对应设备
                try:
                    enc = enc.to(device)
                except Exception:
                    for k, v in enc.items():
                        if isinstance(v, torch.Tensor):
                            enc[k] = v.to(device)

                # 行：GPU 上用 autocast 省显存
                if device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        out = model(**enc)
                else:
                    out = model(**enc)

                # 行：统一拿 last_hidden_state[:,0,:] 作为 CLS 向量
                if hasattr(out, "last_hidden_state"):
                    hs = out.last_hidden_state            # [B, L, D]
                elif isinstance(out, (tuple, list)) and len(out) > 0 and hasattr(out[0], "last_hidden_state"):
                    hs = out[0].last_hidden_state
                elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                    hs = out[0]
                else:
                    raise RuntimeError("Unsupported model output structure for LM encoding.")

                CLS = hs[:, 0, :].cpu().numpy().astype("float32")  # 行：CLS token 向量
                all_vecs.append(CLS)

        if len(all_vecs) == 0:
            # 行：极端兜底（理论上不会走到）
            return np.zeros((0, model.config.hidden_size), dtype="float32")
        return np.concatenate(all_vecs, axis=0).astype("float32")  # 行：拼成 [N, D]

    # ====================== HF 模型加载辅助 ====================== #
    def _hf_load_model(name: str, device: torch.device, use_safetensors: bool = True):
        """行：优先 safetensors；失败时回退，并给出 torch 版本相关的友好报错"""
        try:
            model = AutoModel.from_pretrained(name, use_safetensors=use_safetensors)
            model.to(device)
            return model
        except TypeError as e:
            # 行：旧版 transformers 不认 use_safetensors
            try:
                model = AutoModel.from_pretrained(name)
                model.to(device)
                return model
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load model '{name}' (tried with/without use_safetensors). "
                    f"Err1={e} Err2={e2}"
                ) from e2
        except Exception as e:
            msg = str(e).lower()
            if ("torch" in msg and "2.6" in msg) or ("vulnerability" in msg) or ("cve" in msg):
                raise RuntimeError(
                    f"Loading HF model '{name}' failed due to transformers' torch-version safety check: {e}\n"
                    "Possible solutions:\n"
                    "  1) 保持现有 torch，优先使用 safetensors（use_safetensors=True），并确保模型 repo 提供 .safetensors；\n"
                    "  2) 升级 torch 到 >=2.6（需要匹配 CUDA）；\n"
                    "  3) 或者在受信环境用旧版 transformers（不推荐生产）。"
                ) from e
            # 行：不是 torch 版本问题，再试一次普通加载
            try:
                model = AutoModel.from_pretrained(name)
                model.to(device)
                return model
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load model '{name}'. Tried safetensors and non-safetensors. "
                    f"Err1={e} Err2={e2}"
                ) from e2

    # ====================== 路径和缓存文件 ====================== #
    data_dir = Path(data_dir)                                  # 行：IL6 原始目录
    out_1d = Path(out_1d)                                      # 行：缓存目录（只放 1D）
    out_1d.mkdir(parents=True, exist_ok=True)                  # 行：没有则创建
    split_idx_path = out_1d / "il6_aai_splits.npz"             # 行：划分索引文件（train/val/test 下标）

    def _cache_path(sp: str):
        """行：构造 per-split 缓存文件名（只含 1D）"""
        return out_1d / f"il6_aai_{sp}_1d_lm.npz"              # 行：il6_aai_train_1d_lm.npz 等

    # ====================== 读取原始 IL6-AAI ====================== #
    def _load_raw_il6():
        csv_path = data_dir / csv_name                         # 行：完整 CSV 路径
        if not csv_path.exists():
            raise FileNotFoundError(f"未找到文件: {csv_path}")

        df = pd.read_csv(csv_path)                             # 行：读入 CSV
        df.columns = df.columns.str.strip()                    # 行：列名去掉首尾空格

        # 行：需要的列名
        required_cols = ["VHH_sequence", "Ag_sequence", "Ag_label", "label"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"{csv_name} 缺少列: {col}，实际列: {list(df.columns)}")

        # 行：去掉关键列里有缺失值的行
        df = df.dropna(subset=required_cols).reset_index(drop=True)

        # 行：提取 VHH / Ag 序列与标签
        vhh_all   = df["VHH_sequence"].astype(str).tolist()    # 行：VHH 序列 list[str]
        ag_all    = df["Ag_sequence"].astype(str).tolist()     # 行：抗原序列 list[str]
        ag_label  = df["Ag_label"].astype(str).tolist()        # 行：抗原名字 list[str]
        y_all     = df["label"].astype(float).values           # 行：标签 array[float]

        if logspace_trans:
            # 行：如果真要对 label 做 log10，例如做某种回归，可以打开（当前 0/1 分类一般不需要）
            y_all = np.log10(y_all + 1e-8)

        N_total = len(y_all)                                   # 行：总样本数
        if N_total == 0:
            raise ValueError("il6_aai_dataset.csv 为空，请检查数据。")

        return vhh_all, ag_all, ag_label, y_all

    # ====================== 构建/加载 train/val/test 索引 ====================== #
    def _get_splits_indices(N_total: int):
        """
        行：在 0..N_total-1 范围内随机划分 train/val/test。
        """
        if split_idx_path.exists():
            data = np.load(split_idx_path, allow_pickle=True)
            train_idx = data["train_idx"]
            val_idx   = data["val_idx"]
            test_idx  = data["test_idx"]
            return train_idx, val_idx, test_idx

        # 行：整体样本索引 0..N_total-1
        idx_all = np.arange(N_total, dtype=np.int64)
        rng = np.random.RandomState(split_seed)                # 行：固定随机种子
        rng.shuffle(idx_all)                                   # 行：打乱

        # 行：按比例计算 train / val 样本数
        n_train = int(N_total * train_ratio)
        n_val   = int(N_total * val_ratio)

        # 行：兜底：保证 train / val / test 都有样本
        if n_train <= 0:
            n_train = 1
        if n_val <= 0:
            n_val = 1
        if n_train + n_val >= N_total:
            n_val = max(1, N_total - n_train - 1)

        train_idx = idx_all[:n_train]                          # 行：训练集索引
        val_idx   = idx_all[n_train:n_train + n_val]           # 行：验证集索引
        test_idx  = idx_all[n_train + n_val:]                  # 行：测试集索引（剩余）

        # 行：保存划分结果
        np.savez(split_idx_path,
                 train_idx=train_idx,
                 val_idx=val_idx,
                 test_idx=test_idx)
        return train_idx, val_idx, test_idx

    # ====================== 一次性构建所有样本的 1D，然后切分+落盘 ====================== #
    def _build_and_cache_all():
        # 1. 读原始 IL6-AAI
        vhh_all, ag_all, ag_label_all, y_all = _load_raw_il6()
        N_total = len(y_all)                                  # 行：总样本数

        # 2. 去重后构建 unique VHH / Ag 列表，用于 LM 编码
        ab_list = list(OrderedDict.fromkeys(vhh_all).keys())  # 行：unique VHH，保持首次出现顺序
        ag_list = list(OrderedDict.fromkeys(ag_all).keys())   # 行：unique Ag，保持顺序

        # 行：构建 VHH / Ag → 索引 的映射
        ab2idx = {seq: i for i, seq in enumerate(ab_list)}
        ag2idx = {seq: i for i, seq in enumerate(ag_list)}

        # 行：样本级 VHH / Ag 索引
        pair_ab_idx = np.array([ab2idx[s] for s in vhh_all], dtype=np.int64)
        pair_ag_idx = np.array([ag2idx[s] for s in ag_all],   dtype=np.int64)

        # 样本 ID：il6_0..il6_(N-1)
        ids_arr = np.array([f"il6_{i}" for i in range(N_total)], dtype=object)

        # 3. 对 unique VHH / Ag 做 LM 编码
        lm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LM|IL6-AAI] using device: {lm_device}")

        # 3.1 VHH LM（ESM-2）
        print(f"[LM|IL6-AAI] loading ESM model for VHH: {esm_ab_model_name} (use_safetensors={use_safetensors})")
        ab_tok   = AutoTokenizer.from_pretrained(esm_ab_model_name)
        ab_model = _hf_load_model(esm_ab_model_name, lm_device, use_safetensors)
        print(f"[LM|IL6-AAI] encoding VHH (unique={len(ab_list)})")
        ab_lm_all = _encode_text_list(ab_list, ab_tok, ab_model, lm_device, batch_size=lm_batch_size)
        # 行：释放 VHH 模型
        try:
            del ab_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        # 3.2 抗原 LM（ESM-2）
        print(f"[LM|IL6-AAI] loading ESM model for antigen: {esm_ag_model_name} (use_safetensors={use_safetensors})")
        ag_tok   = AutoTokenizer.from_pretrained(esm_ag_model_name)
        ag_model = _hf_load_model(esm_ag_model_name, lm_device, use_safetensors)
        print(f"[LM|IL6-AAI] encoding antigens (unique={len(ag_list)})")
        ag_lm_all = _encode_text_list(ag_list, ag_tok, ag_model, lm_device, batch_size=lm_batch_size)
        # 行：释放 Ag 模型
        try:
            del ag_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        # 行：根据样本级索引抽取 LM 向量
        ab_lm = ab_lm_all[pair_ab_idx]                        # [N_total, D_ab]
        ag_lm = ag_lm_all[pair_ag_idx]                        # [N_total, D_ag]

        # 4. 生成 train/val/test 索引
        train_idx, val_idx, test_idx = _get_splits_indices(N_total)

        # 5. 按索引切分并写入缓存（只写 1D）
        def _save_split(sp: str, idx_array: np.ndarray):
            p_1dnpz = _cache_path(sp)                         # 行：对应 split 的 npz 路径
            idx_array = np.asarray(idx_array, dtype=np.int64)

            np.savez(
                p_1dnpz,
                ids=ids_arr[idx_array],                       # 行：样本 ID
                vhh_seq=np.array(vhh_all, dtype=object)[idx_array],     # 行：VHH 序列
                ag_seq=np.array(ag_all, dtype=object)[idx_array],       # 行：抗原序列
                ag_label=np.array(ag_label_all, dtype=object)[idx_array], # 行：抗原名字
                label=y_all[idx_array],                       # 行：标签（或 log 后）
                ab_lm=ab_lm[idx_array],                       # 行：VHH LM 特征
                ag_lm=ag_lm[idx_array],                       # 行：Ag LM 特征
            )

        print("[CACHE|IL6-AAI] saving train/val/test splits (1D LM only)...")
        _save_split("train", train_idx)
        _save_split("val",   val_idx)
        _save_split("test",  test_idx)
        print("[CACHE|IL6-AAI] cached done (1D only, no 2D/3D).")

    # ====================== 判断是否需要重新构建缓存 ====================== #
    need_build = False
    if split in ("train", "val", "test"):
        p_1d = _cache_path(split)
        if not p_1d.exists():
            need_build = True
    elif split == "all":
        for sp in ("train", "val", "test"):
            p_1d = _cache_path(sp)
            if not p_1d.exists():
                need_build = True
                break
    else:
        raise ValueError(f"split 必须是 'train'/'val'/'test'/'all' 之一，收到: {split!r}")

    if need_build:
        _build_and_cache_all()

    # ====================== 读取单个拆分（只加载 1D） ====================== #
    def _load_split(sp: str) -> dict:
        p_1dnpz = _cache_path(sp)                             # 行：某个 split 对应的 npz
        d1 = np.load(p_1dnpz, allow_pickle=True)
        ids      = d1["ids"]                                  # 行：array(object)
        vhh_seq  = d1["vhh_seq"]                              # 行：array(str)
        ag_seq   = d1["ag_seq"]                               # 行：array(str)
        ag_label = d1["ag_label"]                             # 行：array(str)
        y        = d1["label"].astype(np.float32)             # 行：label → float32
        ab_lm    = d1["ab_lm"].astype(np.float32)             # 行：VHH ESM 向量
        ag_lm    = d1["ag_lm"].astype(np.float32)             # 行：Ag  ESM 向量

        N = len(ids)
        assert len(vhh_seq) == N and len(ag_seq) == N and len(ag_label) == N and len(y) == N, "1D 长度不一致"
        assert ab_lm.shape[0] == N and ag_lm.shape[0] == N, "LM 行数与 1D 不一致"

        return {
            "ids": ids,              # 行：array(object)
            "y": y,                  # 行：np.float32，训练脚本统一用 pkg['y']
            "vhh_seq": vhh_seq,      # 行：array(str)
            "ag_seq": ag_seq,        # 行：array(str)
            "ag_label": ag_label,    # 行：array(str)
            "ab_lm": ab_lm,          # 行：np.ndarray[N, D_ab]
            "ag_lm": ag_lm,          # 行：np.ndarray[N, D_ag]
        }

    # ====================== 顶层返回 ====================== #
    if split in ("train", "val", "test"):
        return {split: _load_split(split)}                    # 行：单个划分
    else:  # "all"
        parts = {sp: _load_split(sp) for sp in ("train", "val", "test")}  # 行：三个划分都读出来
        def _cat(*xs): return np.concatenate(xs, axis=0)      # 行：拼接工具
        all_pkg = {
            "ids":      _cat(parts["train"]["ids"],      parts["val"]["ids"],      parts["test"]["ids"]),
            "y":        _cat(parts["train"]["y"],        parts["val"]["y"],        parts["test"]["y"]).astype(np.float32),
            "vhh_seq":  _cat(parts["train"]["vhh_seq"],  parts["val"]["vhh_seq"],  parts["test"]["vhh_seq"]),
            "ag_seq":   _cat(parts["train"]["ag_seq"],   parts["val"]["ag_seq"],   parts["test"]["ag_seq"]),
            "ag_label": _cat(parts["train"]["ag_label"], parts["val"]["ag_label"], parts["test"]["ag_label"]),
            "ab_lm": np.vstack([
                parts["train"]["ab_lm"],
                parts["val"]["ab_lm"],
                parts["test"]["ab_lm"],
            ]).astype(np.float32),
            "ag_lm": np.vstack([
                parts["train"]["ag_lm"],
                parts["val"]["ag_lm"],
                parts["test"]["ag_lm"],
            ]).astype(np.float32),
        }
        return {"all": all_pkg}

def LoadData_bd2017_lm_1d(
    data_dir: str,                                   # 行：指向 BD2017 目录（含 ba_data.txt 与 MHC_pseudo.dat）
    ba_name: str = "ba_data.txt",                     # 行：BA 主数据文件名（pep, label, allele）
    pseudo_name: str = "MHC_pseudo.dat",              # 行：allele → pseudo-seq 映射表文件名
    split: str = "all",                               # 行：'train'|'val'|'test'|'all'
    out_1d: str = "../dataset/BD2017/processed_lm_1d", # 行：1D LM 缓存目录（只存 1D）
    logspace_trans: bool = False,                     # 行：是否对连续标签做 log 变换（一般不需要）
    # --- LM 模型相关（pep 与 mhc 都用 ESM）--- #
    esm_pep_model_name: str = "facebook/esm2_t33_650M_UR50D",  # 行：pep 侧 ESM-2 模型
    esm_mhc_model_name: str = "facebook/esm2_t33_650M_UR50D",  # 行：mhc pseudo-seq 侧 ESM-2 模型
    lm_batch_size: int = 32,                          # 行：LM 前向 batch_size（看显存）
    use_safetensors: bool = True,                     # 行：是否优先使用 safetensors
    # --- 划分相关（整表随机划分 train/val/test）--- #
    split_seed: int = 2023,                           # 行：划分随机种子（保证可复现）
    train_ratio: float = 0.8,                         # 行：train 比例
    val_ratio: float = 0.1                            # 行：val 比例（test = 剩余）
) -> dict:
    """
    BD2017 (MHC-I BA) + LM 数据加载（仅 1D，不生成 2D/3D）：

    原始数据（常见格式）：
      1) ba_data.txt：每行 3 列（tab/空白分隔均可）
            peptide   label   allele
         label 可能是：
           - IC50(nM)（如 50 / 500 / 50000）
           - 或已转换到 [0,1] 的 “logic/affinity” 分数（论文常用）

      2) MHC_pseudo.dat：allele → pseudo-seq 映射（通常 34 aa）
            allele  pseudo_seq

    输出/缓存内容（每个 split 一个 npz）：
      - ids      : 样本ID（字符串）
      - pep_seq  : peptide 序列（字符串）
      - mhc_seq  : MHC pseudo 序列（字符串）
      - allele   : allele 名称（字符串）
      - pep_len  : peptide 长度（int）
      - y        : 连续标签（float32，通常在[0,1]）
      - y_bin    : 二分类标签（int64，按论文 binder 阈值：IC50 < 500nM）
      - pep_lm   : ESM-2(peptide) CLS 向量（float32，[N, D]）
      - mhc_lm   : ESM-2(mhc_pseudo_seq) CLS 向量（float32，[N, D]）
      - ic50     : （可选）若原始第二列像 IC50，则保存 IC50(nM)；否则为 NaN

    返回：
      {split: {'ids','y','y_bin','pep_seq','pep_len','mhc_seq','allele','pep_lm','mhc_lm','ic50'}}
      或 {'all': ...}（把 train/val/test 拼起来）
    """

    # ====================== 依赖（与项目其它 loader 风格保持一致）====================== #
    from pathlib import Path                                  # 行：路径处理
    import math                                               # 行：log/阈值换算
    import numpy as np                                        # 行：数组处理
    import pandas as pd                                       # 行：表格处理
    from collections import OrderedDict                       # 行：保持顺序去重
    import torch                                              # 行：LM 前向/设备
    from tqdm import tqdm                                     # 行：进度条
    from transformers import AutoTokenizer, AutoModel         # 行：HF 模型/分词器

    # ====================== 常量：论文 BD2017 的转换与阈值 ====================== #
    IC50_MAX = 50000.0                                        # 行：论文里归一化用的最大 IC50
    IC50_BINDER = 500.0                                       # 行：binder 阈值：IC50 < 500nM
    CUTOFF_Y = 1.0 - (math.log(IC50_BINDER) / math.log(IC50_MAX))  # 行：把 500nM 映射到 y 空间的阈值

    # ====================== LM 编码：list[str] -> [N, D] CLS 向量 ====================== #
    def _encode_text_list(text_list, tokenizer, model, device, batch_size=32):
        """行：输入 list[str]，输出 [N, D] 的 CLS 向量（float32）"""
        all_vecs = []                                         # 行：保存每个 batch 的 CLS
        model.eval()                                          # 行：评估模式（关 dropout）

        with torch.no_grad():                                 # 行：不计算梯度
            for i in tqdm(range(0, len(text_list), batch_size),
                          desc="[LM|BD2017] encoding", unit="batch"):
                batch = text_list[i:i + batch_size]           # 行：取当前 batch 序列

                enc = tokenizer(                               # 行：分词 + padding
                    batch,
                    padding=True,                              # 行：batch 内补齐
                    truncation=True,                           # 行：截断超长
                    max_length=1022,                           # 行：ESM-2 常用上限
                    return_tensors="pt"                        # 行：返回 PyTorch Tensor
                )

                # 行：把 tokenizer 输出整体搬到 device（兼容 dict/tensor）
                try:
                    enc = enc.to(device)
                except Exception:
                    for k, v in enc.items():
                        if isinstance(v, torch.Tensor):
                            enc[k] = v.to(device)

                # 行：cuda 上用 autocast 省显存；cpu 不开
                if device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        out = model(**enc)
                else:
                    out = model(**enc)

                # 行：取 last_hidden_state（不同 transformers 输出结构做兼容）
                if hasattr(out, "last_hidden_state"):
                    hs = out.last_hidden_state                # 行：[B, L, D]
                elif isinstance(out, (tuple, list)) and len(out) > 0 and hasattr(out[0], "last_hidden_state"):
                    hs = out[0].last_hidden_state
                elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                    hs = out[0]
                else:
                    raise RuntimeError("Unsupported model output structure for LM encoding.")

                cls = hs[:, 0, :].detach().cpu().numpy().astype("float32")  # 行：取 CLS（token=0）
                all_vecs.append(cls)                           # 行：收集 batch CLS

        if len(all_vecs) == 0:                                 # 行：输入为空时返回空矩阵
            d = getattr(model.config, "hidden_size", 0)
            return np.zeros((0, d), dtype="float32")

        return np.concatenate(all_vecs, axis=0).astype("float32")  # 行：拼成 [N, D]

    # ====================== HF 模型加载：优先 safetensors，失败回退 ====================== #
    def _hf_load_model(name: str, device: torch.device, use_safetensors: bool = True):
        """行：加载 HF 模型到 device，尽量用 safetensors，失败则回退普通加载"""
        try:
            m = AutoModel.from_pretrained(name, use_safetensors=use_safetensors)  # 行：优先 safetensors
            m.to(device)
            return m
        except TypeError:
            # 行：旧 transformers 可能不支持 use_safetensors 参数
            m = AutoModel.from_pretrained(name)
            m.to(device)
            return m
        except Exception:
            # 行：再兜底一次普通加载
            m = AutoModel.from_pretrained(name)
            m.to(device)
            return m

    # ====================== 路径与缓存文件 ====================== #
    data_dir = Path(data_dir)                                  # 行：原始目录 Path 化
    out_1d = Path(out_1d)                                      # 行：缓存目录 Path 化
    out_1d.mkdir(parents=True, exist_ok=True)                  # 行：创建缓存目录

    split_idx_path = out_1d / "bd2017_splits.npz"              # 行：保存 train/val/test 索引

    def _cache_path(sp: str):
        """行：构造某个 split 的缓存文件名"""
        return out_1d / f"bd2017_{sp}_1d_lm.npz"

    # ====================== 读取 MHC pseudo 映射表 ====================== #
    def _load_pseudo_map(pseudo_path: Path) -> dict:
        """行：解析 MHC_pseudo.dat，返回 dict: allele -> pseudo_seq"""
        if not pseudo_path.exists():
            raise FileNotFoundError(f"未找到 pseudo 文件: {pseudo_path}")

        allele2seq = {}                                       # 行：allele->pseudo_seq
        with pseudo_path.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if (not ln) or ln.startswith("#"):
                    continue
                parts = ln.split()                             # 行：按任意空白分隔
                if len(parts) < 2:
                    continue
                allele = parts[0].strip()
                pseudo_seq = parts[1].strip()
                if allele and pseudo_seq:
                    allele2seq[allele] = pseudo_seq

        if len(allele2seq) == 0:
            raise ValueError(f"{pseudo_path} 解析后为空，请检查格式。")
        return allele2seq

    # ====================== 读取 ba_data.txt 并生成 y/y_bin/pep_len ====================== #
    def _load_raw_ba():
        """
        行：读 ba_data.txt，返回：
          pep_all(list[str]),
          mhc_all(list[str]),
          allele_all(list[str]),
          y_all(np.ndarray float32, 连续),
          y_bin_all(np.ndarray int64, 二值),
          pep_len_all(np.ndarray int32),
          ic50_all(np.ndarray float64 或 None)
        """
        ba_path = data_dir / ba_name
        pseudo_path = data_dir / pseudo_name

        if not ba_path.exists():
            raise FileNotFoundError(f"未找到 BA 文件: {ba_path}")

        allele2seq = _load_pseudo_map(pseudo_path)

        rows = []                                              # 行：保存 (pep, raw_label, allele)
        with ba_path.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if (not ln) or ln.startswith("#"):
                    continue
                parts = ln.split()
                if len(parts) < 3:
                    continue
                pep = parts[0].strip()
                raw = parts[1].strip()
                allele = parts[2].strip()
                rows.append((pep, raw, allele))

        if len(rows) == 0:
            raise ValueError(f"{ba_path} 为空或解析失败，请检查分隔符/内容。")

        df = pd.DataFrame(rows, columns=["pep", "raw", "allele"])
        df["pep"] = df["pep"].astype(str)
        df["allele"] = df["allele"].astype(str)
        df["raw"] = df["raw"].astype(float)

        # 行：补 mhc pseudo-seq
        mhc_seq_list, miss = [], 0
        for a in df["allele"].tolist():
            if a in allele2seq:
                mhc_seq_list.append(allele2seq[a])
            else:
                mhc_seq_list.append("")
                miss += 1
        df["mhc_seq"] = mhc_seq_list

        # 行：丢弃没有 pseudo-seq 的样本
        before = len(df)
        df = df[df["mhc_seq"].astype(str).str.len() > 0].reset_index(drop=True)
        after = len(df)
        if after == 0:
            raise ValueError(f"所有样本都无法在 {pseudo_name} 中找到 allele→pseudo-seq 映射。")
        if miss > 0:
            print(f"[WARN|BD2017] {miss} alleles not found in {pseudo_name}, dropped {before - after} samples.")

        raw_vals = df["raw"].values.astype(np.float64)          # 行：第二列数值

        # 行：判断 raw 更像 IC50(nM) 还是已变换到 [0,1]
        looks_like_ic50 = (np.nanmax(raw_vals) > 1.5)

        if looks_like_ic50:
            ic50_all = raw_vals                                 # 行：raw 就是 IC50(nM)
            # 行：按论文公式 y = 1 - log(IC50)/log(50000)
            y_all = 1.0 - (np.log(ic50_all + 1e-12) / np.log(IC50_MAX))
            # 行：二值标签按论文阈值：IC50<500 为 binder
            y_bin_all = (ic50_all < IC50_BINDER).astype(np.int64)
        else:
            ic50_all = None                                     # 行：没有可靠 IC50
            y_all = raw_vals                                    # 行：raw 已是 [0,1] 连续标签
            # 行：等价阈值：y >= CUTOFF_Y 视为 binder
            y_bin_all = (y_all >= CUTOFF_Y).astype(np.int64)

        # 行：把连续 y 截断到 [0,1]，防止极少数异常
        y_all = np.clip(y_all, 0.0, 1.0).astype(np.float32)

        # 行：可选：log 变换入口（一般不建议对 [0,1] 做 log）
        if logspace_trans:
            y_all = np.log10(y_all + 1e-8).astype(np.float32)

        pep_all = df["pep"].astype(str).tolist()
        mhc_all = df["mhc_seq"].astype(str).tolist()
        allele_all = df["allele"].astype(str).tolist()
        pep_len_all = np.array([len(s) for s in pep_all], dtype=np.int32)

        return pep_all, mhc_all, allele_all, y_all, y_bin_all, pep_len_all, ic50_all

    # ====================== 构建/加载 train/val/test 索引 ====================== #
    def _get_splits_indices(N_total: int):
        """行：随机划分 train/val/test，并保存索引保证可复现"""
        if split_idx_path.exists():
            d = np.load(split_idx_path, allow_pickle=True)
            return d["train_idx"], d["val_idx"], d["test_idx"]

        idx_all = np.arange(N_total, dtype=np.int64)
        rng = np.random.RandomState(split_seed)
        rng.shuffle(idx_all)

        n_train = int(N_total * train_ratio)
        n_val = int(N_total * val_ratio)

        # 行：兜底，保证每个 split 至少 1 个样本
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        if n_train + n_val >= N_total:
            n_val = max(1, N_total - n_train - 1)

        train_idx = idx_all[:n_train]
        val_idx = idx_all[n_train:n_train + n_val]
        test_idx = idx_all[n_train + n_val:]

        np.savez(split_idx_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
        return train_idx, val_idx, test_idx

    # ====================== 构建所有样本 LM 并缓存到 npz ====================== #
    def _build_and_cache_all():
        # 1) 读原始 BA + pseudo-seq + 标签
        pep_all, mhc_all, allele_all, y_all, y_bin_all, pep_len_all, ic50_all = _load_raw_ba()
        N_total = len(y_all)
        if N_total == 0:
            raise ValueError("BD2017 数据为空，请检查 ba_data.txt。")

        # 2) unique 序列去重（减少 LM 编码量）
        pep_list = list(OrderedDict.fromkeys(pep_all).keys())
        mhc_list = list(OrderedDict.fromkeys(mhc_all).keys())

        # 3) 序列 -> unique 索引
        pep2idx = {s: i for i, s in enumerate(pep_list)}
        mhc2idx = {s: i for i, s in enumerate(mhc_list)}

        # 4) 样本级索引：每个样本对应 unique pep/mhc 的编号
        pair_pep_idx = np.array([pep2idx[s] for s in pep_all], dtype=np.int64)
        pair_mhc_idx = np.array([mhc2idx[s] for s in mhc_all], dtype=np.int64)

        # 5) 样本 IDs
        ids_arr = np.array([f"bd2017_{i}" for i in range(N_total)], dtype=object)

        # 6) 编码 unique pep / mhc
        lm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LM|BD2017] using device: {lm_device}")

        # 6.1 pep ESM
        print(f"[LM|BD2017] loading ESM for peptide: {esm_pep_model_name} (use_safetensors={use_safetensors})")
        pep_tok = AutoTokenizer.from_pretrained(esm_pep_model_name)
        pep_model = _hf_load_model(esm_pep_model_name, lm_device, use_safetensors)
        print(f"[LM|BD2017] encoding peptides (unique={len(pep_list)})")
        pep_lm_unique = _encode_text_list(pep_list, pep_tok, pep_model, lm_device, batch_size=lm_batch_size)

        # 行：释放 pep 模型节省显存
        try:
            del pep_model
            if lm_device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

        # 6.2 mhc ESM
        print(f"[LM|BD2017] loading ESM for MHC pseudo-seq: {esm_mhc_model_name} (use_safetensors={use_safetensors})")
        mhc_tok = AutoTokenizer.from_pretrained(esm_mhc_model_name)
        mhc_model = _hf_load_model(esm_mhc_model_name, lm_device, use_safetensors)
        print(f"[LM|BD2017] encoding MHC pseudo-seqs (unique={len(mhc_list)})")
        mhc_lm_unique = _encode_text_list(mhc_list, mhc_tok, mhc_model, lm_device, batch_size=lm_batch_size)

        # 行：释放 mhc 模型
        try:
            del mhc_model
            if lm_device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

        # 7) 回填到样本级
        pep_lm = pep_lm_unique[pair_pep_idx].astype(np.float32)   # 行：[N, D_pep]
        mhc_lm = mhc_lm_unique[pair_mhc_idx].astype(np.float32)   # 行：[N, D_mhc]

        # 8) 划分索引
        train_idx, val_idx, test_idx = _get_splits_indices(N_total)

        # 9) 落盘
        pep_all_arr = np.array(pep_all, dtype=object)
        mhc_all_arr = np.array(mhc_all, dtype=object)
        allele_arr = np.array(allele_all, dtype=object)
        pep_len_arr = np.array(pep_len_all, dtype=np.int32)
        y_arr = np.array(y_all, dtype=np.float32)
        ybin_arr = np.array(y_bin_all, dtype=np.int64)

        # 行：ic50 若不存在，用 NaN 占位（保持字段一致，训练/评估更好写）
        if ic50_all is None:
            ic50_arr = np.full((N_total,), np.nan, dtype=np.float32)
        else:
            ic50_arr = np.array(ic50_all, dtype=np.float32)

        def _save_split(sp: str, idx_array: np.ndarray):
            idx_array = np.asarray(idx_array, dtype=np.int64)
            p = _cache_path(sp)

            np.savez(
                p,
                ids=ids_arr[idx_array],                        # 行：样本ID
                pep_seq=pep_all_arr[idx_array],                # 行：pep 序列
                mhc_seq=mhc_all_arr[idx_array],                # 行：mhc pseudo 序列
                allele=allele_arr[idx_array],                  # 行：allele
                pep_len=pep_len_arr[idx_array],                # 行：pep 长度
                y=y_arr[idx_array],                            # 行：连续标签（回归用）
                y_bin=ybin_arr[idx_array],                     # 行：二值标签（算AUC/AUPRC/PPV/Sens/F1用）
                ic50=ic50_arr[idx_array],                      # 行：IC50（若原始是IC50则真实，否则NaN）
                pep_lm=pep_lm[idx_array],                      # 行：pep LM
                mhc_lm=mhc_lm[idx_array],                      # 行：mhc LM
            )

        print("[CACHE|BD2017] saving train/val/test splits (1D LM only)...")
        _save_split("train", train_idx)
        _save_split("val", val_idx)
        _save_split("test", test_idx)
        print("[CACHE|BD2017] cached done (1D only, no 2D/3D).")

    # ====================== 判断是否需要构建缓存 ====================== #
    need_build = False
    if split in ("train", "val", "test"):
        if not _cache_path(split).exists():
            need_build = True
    elif split == "all":
        for sp in ("train", "val", "test"):
            if not _cache_path(sp).exists():
                need_build = True
                break
    else:
        raise ValueError(f"split 必须是 'train'/'val'/'test'/'all' 之一，收到: {split!r}")

    if need_build:
        _build_and_cache_all()

    # ====================== 读取某个 split 的 npz ====================== #
    def _load_split(sp: str) -> dict:
        p = _cache_path(sp)
        d = np.load(p, allow_pickle=True)

        ids = d["ids"]
        pep_seq = d["pep_seq"]
        mhc_seq = d["mhc_seq"]
        allele = d["allele"]
        pep_len = d["pep_len"].astype(np.int32)

        y = d["y"].astype(np.float32)
        y_bin = d["y_bin"].astype(np.int64)
        ic50 = d["ic50"].astype(np.float32)

        pep_lm = d["pep_lm"].astype(np.float32)
        mhc_lm = d["mhc_lm"].astype(np.float32)

        N = len(ids)
        assert len(pep_seq) == N and len(mhc_seq) == N and len(allele) == N, "字符串字段长度不一致"
        assert len(pep_len) == N and len(y) == N and len(y_bin) == N and len(ic50) == N, "数值字段长度不一致"
        assert pep_lm.shape[0] == N and mhc_lm.shape[0] == N, "LM 行数与样本数不一致"

        return {
            "ids": ids,               # 行：array(object) / array(str)
            "pep_seq": pep_seq,       # 行：peptide 序列
            "mhc_seq": mhc_seq,       # 行：MHC pseudo 序列
            "allele": allele,         # 行：allele 名
            "pep_len": pep_len,       # 行：peptide 长度（用于按长度分桶算表格）
            "y": y,                   # 行：连续标签（PCC 用这个）
            "y_bin": y_bin,           # 行：二值标签（AUC/AUPRC/PPV/Sens/F1 用这个）
            "ic50": ic50,             # 行：IC50（原始像IC50就是真值，否则NaN）
            "pep_lm": pep_lm,         # 行：pep ESM CLS
            "mhc_lm": mhc_lm,         # 行：mhc ESM CLS
        }

    # ====================== 顶层返回 ====================== #
    if split in ("train", "val", "test"):
        return {split: _load_split(split)}
    else:
        parts = {sp: _load_split(sp) for sp in ("train", "val", "test")}

        def _cat(*xs):
            return np.concatenate(xs, axis=0)

        all_pkg = {
            "ids": _cat(parts["train"]["ids"], parts["val"]["ids"], parts["test"]["ids"]),
            "pep_seq": _cat(parts["train"]["pep_seq"], parts["val"]["pep_seq"], parts["test"]["pep_seq"]),
            "mhc_seq": _cat(parts["train"]["mhc_seq"], parts["val"]["mhc_seq"], parts["test"]["mhc_seq"]),
            "allele": _cat(parts["train"]["allele"], parts["val"]["allele"], parts["test"]["allele"]),
            "pep_len": _cat(parts["train"]["pep_len"], parts["val"]["pep_len"], parts["test"]["pep_len"]).astype(np.int32),
            "y": _cat(parts["train"]["y"], parts["val"]["y"], parts["test"]["y"]).astype(np.float32),
            "y_bin": _cat(parts["train"]["y_bin"], parts["val"]["y_bin"], parts["test"]["y_bin"]).astype(np.int64),
            "ic50": _cat(parts["train"]["ic50"], parts["val"]["ic50"], parts["test"]["ic50"]).astype(np.float32),
            "pep_lm": np.vstack([parts["train"]["pep_lm"], parts["val"]["pep_lm"], parts["test"]["pep_lm"]]).astype(np.float32),
            "mhc_lm": np.vstack([parts["train"]["mhc_lm"], parts["val"]["mhc_lm"], parts["test"]["mhc_lm"]]).astype(np.float32),
        }
        return {"all": all_pkg}


def LoadData_iedb2016_lm_1d(
    data_dir: str,                                      # 行：IEDB2016 目录（含 data.csv）
    csv_name: str = "data.csv",                          # 行：文件名
    split: str = "all",                                  # 行：'train'|'val'|'test'|'all'
    out_1d: str = "../dataset/IEDB2016/processed_lm_1d",  # 行：缓存目录
    logspace_trans: bool = False,                        # 行：可选 log 变换（一般不建议）
    esm_pep_model_name: str = "facebook/esm2_t33_650M_UR50D",  # 行：pep ESM
    esm_mhc_model_name: str = "facebook/esm2_t33_650M_UR50D",  # 行：mhc ESM
    lm_batch_size: int = 32,                             # 行：编码 batch
    use_safetensors: bool = True,                        # 行：优先 safetensors
    split_seed: int = 2023,                              # 行：划分随机种子
    train_ratio: float = 0.8,                            # 行：train 比例
    val_ratio: float = 0.1                               # 行：val 比例
) -> dict:
    """
    IEDB2016（你这个格式）：

      CSV 列：pep, logic, allele, mhc
        - pep   : peptide 序列
        - logic : 连续值（可能是 [0,1] 或 IC50/其它）
        - allele: 名称
        - mhc   : 这里直接给了 MHC 序列（无需 pseudo 映射）

    输出字段（每个 split）：
      - ids, pep_seq, mhc_seq, allele
      - pep_len, mhc_len
      - y（连续，用于 PCC）
      - y_bin（二值，用于 AUC/AUPRC/PPV/Sens/F1）
      - ic50（若像 IC50 则存，否则 NaN）
      - pep_lm, mhc_lm（ESM CLS）
    """
    from pathlib import Path
    import math
    import numpy as np
    import pandas as pd
    from collections import OrderedDict
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModel

    IC50_MAX = 50000.0
    IC50_BINDER = 500.0
    CUTOFF_Y = 1.0 - (math.log(IC50_BINDER) / math.log(IC50_MAX))

    def _encode_text_list(text_list, tokenizer, model, device, batch_size=32):
        all_vecs = []
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(text_list), batch_size),
                          desc="[LM|IEDB2016] encoding", unit="batch"):
                batch = text_list[i:i + batch_size]
                enc = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=1022,
                    return_tensors="pt"
                )
                try:
                    enc = enc.to(device)
                except Exception:
                    for k, v in enc.items():
                        if isinstance(v, torch.Tensor):
                            enc[k] = v.to(device)

                if device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        out = model(**enc)
                else:
                    out = model(**enc)

                if hasattr(out, "last_hidden_state"):
                    hs = out.last_hidden_state
                elif isinstance(out, (tuple, list)) and len(out) > 0 and hasattr(out[0], "last_hidden_state"):
                    hs = out[0].last_hidden_state
                elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                    hs = out[0]
                else:
                    raise RuntimeError("Unsupported HF output structure when encoding.")

                cls = hs[:, 0, :].detach().cpu().numpy().astype("float32")
                all_vecs.append(cls)

        if len(all_vecs) == 0:
            d = getattr(model.config, "hidden_size", 0)
            return np.zeros((0, d), dtype="float32")

        return np.concatenate(all_vecs, axis=0).astype("float32")

    def _hf_load_model(name: str, device: torch.device, use_safetensors: bool = True):
        try:
            m = AutoModel.from_pretrained(name, use_safetensors=use_safetensors)
            m.to(device)
            return m
        except TypeError:
            m = AutoModel.from_pretrained(name)
            m.to(device)
            return m
        except Exception:
            m = AutoModel.from_pretrained(name)
            m.to(device)
            return m

    data_dir = Path(data_dir)
    out_1d = Path(out_1d)
    out_1d.mkdir(parents=True, exist_ok=True)

    split_idx_path = out_1d / "iedb2016_splits.npz"

    def _cache_path(sp: str):
        return out_1d / f"iedb2016_{sp}_1d_lm.npz"

    def _load_raw():
        csv_path = data_dir / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"未找到 IEDB2016 CSV: {csv_path}")

        df = pd.read_csv(csv_path)  # 行：按逗号读
        # 行：确保列存在
        for col in ["pep", "logic", "allele", "mhc"]:
            if col not in df.columns:
                raise KeyError(f"IEDB2016 CSV 缺少列: {col}, 实际列={list(df.columns)}")

        df["pep"] = df["pep"].astype(str)
        df["mhc"] = df["mhc"].astype(str)
        df["allele"] = df["allele"].astype(str)
        df["logic"] = df["logic"].astype(float)

        raw_vals = df["logic"].values.astype(np.float64)

        looks_like_ic50 = (np.nanmax(raw_vals) > 1.5)

        if looks_like_ic50:
            ic50 = raw_vals
            y = 1.0 - (np.log(ic50 + 1e-12) / np.log(IC50_MAX))
            y_bin = (ic50 < IC50_BINDER).astype(np.int64)
            ic50_arr = ic50.astype(np.float32)
        else:
            y = raw_vals
            y_bin = (y >= CUTOFF_Y).astype(np.int64)
            ic50_arr = np.full((len(y),), np.nan, dtype=np.float32)

        y = np.clip(y, 0.0, 1.0).astype(np.float32)

        if logspace_trans:
            y = np.log10(y + 1e-8).astype(np.float32)

        pep_all = df["pep"].tolist()
        mhc_all = df["mhc"].tolist()
        allele_all = df["allele"].tolist()

        pep_len = np.array([len(s) for s in pep_all], dtype=np.int32)
        mhc_len = np.array([len(s) for s in mhc_all], dtype=np.int32)

        return pep_all, mhc_all, allele_all, y, y_bin, pep_len, mhc_len, ic50_arr

    def _get_splits_indices(N_total: int):
        if split_idx_path.exists():
            d = np.load(split_idx_path, allow_pickle=True)
            return d["train_idx"], d["val_idx"], d["test_idx"]

        idx_all = np.arange(N_total, dtype=np.int64)
        rng = np.random.RandomState(split_seed)
        rng.shuffle(idx_all)

        n_train = max(1, int(N_total * train_ratio))
        n_val = max(1, int(N_total * val_ratio))
        if n_train + n_val >= N_total:
            n_val = max(1, N_total - n_train - 1)

        train_idx = idx_all[:n_train]
        val_idx = idx_all[n_train:n_train + n_val]
        test_idx = idx_all[n_train + n_val:]

        np.savez(split_idx_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
        return train_idx, val_idx, test_idx

    def _build_and_cache_all():
        pep_all, mhc_all, allele_all, y_all, ybin_all, pep_len_all, mhc_len_all, ic50_all = _load_raw()
        N_total = len(y_all)
        if N_total == 0:
            raise ValueError("IEDB2016 数据为空。")

        pep_list = list(OrderedDict.fromkeys(pep_all).keys())
        mhc_list = list(OrderedDict.fromkeys(mhc_all).keys())

        pep2idx = {s: i for i, s in enumerate(pep_list)}
        mhc2idx = {s: i for i, s in enumerate(mhc_list)}

        pair_pep_idx = np.array([pep2idx[s] for s in pep_all], dtype=np.int64)
        pair_mhc_idx = np.array([mhc2idx[s] for s in mhc_all], dtype=np.int64)

        ids_arr = np.array([f"iedb2016_{i}" for i in range(N_total)], dtype=object)

        lm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LM|IEDB2016] using device: {lm_device}")

        print(f"[LM|IEDB2016] load pep ESM: {esm_pep_model_name}")
        pep_tok = AutoTokenizer.from_pretrained(esm_pep_model_name)
        pep_model = _hf_load_model(esm_pep_model_name, lm_device, use_safetensors)
        print(f"[LM|IEDB2016] encoding pep unique={len(pep_list)}")
        pep_lm_u = _encode_text_list(pep_list, pep_tok, pep_model, lm_device, batch_size=lm_batch_size)
        try:
            del pep_model
            if lm_device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

        print(f"[LM|IEDB2016] load mhc ESM: {esm_mhc_model_name}")
        mhc_tok = AutoTokenizer.from_pretrained(esm_mhc_model_name)
        mhc_model = _hf_load_model(esm_mhc_model_name, lm_device, use_safetensors)
        print(f"[LM|IEDB2016] encoding mhc unique={len(mhc_list)}")
        mhc_lm_u = _encode_text_list(mhc_list, mhc_tok, mhc_model, lm_device, batch_size=lm_batch_size)
        try:
            del mhc_model
            if lm_device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

        pep_lm = pep_lm_u[pair_pep_idx].astype(np.float32)
        mhc_lm = mhc_lm_u[pair_mhc_idx].astype(np.float32)

        train_idx, val_idx, test_idx = _get_splits_indices(N_total)

        pep_all_arr = np.array(pep_all, dtype=object)
        mhc_all_arr = np.array(mhc_all, dtype=object)
        allele_arr = np.array(allele_all, dtype=object)

        pep_len_arr = np.array(pep_len_all, dtype=np.int32)
        mhc_len_arr = np.array(mhc_len_all, dtype=np.int32)
        y_arr = np.array(y_all, dtype=np.float32)
        ybin_arr = np.array(ybin_all, dtype=np.int64)
        ic50_arr = np.array(ic50_all, dtype=np.float32)

        def _save_split(sp: str, idx_array: np.ndarray):
            idx_array = np.asarray(idx_array, dtype=np.int64)
            p = _cache_path(sp)
            np.savez(
                p,
                ids=ids_arr[idx_array],
                pep_seq=pep_all_arr[idx_array],
                mhc_seq=mhc_all_arr[idx_array],
                allele=allele_arr[idx_array],
                pep_len=pep_len_arr[idx_array],
                mhc_len=mhc_len_arr[idx_array],
                y=y_arr[idx_array],
                y_bin=ybin_arr[idx_array],
                ic50=ic50_arr[idx_array],
                pep_lm=pep_lm[idx_array],
                mhc_lm=mhc_lm[idx_array],
            )

        print("[CACHE|IEDB2016] saving train/val/test ...")
        _save_split("train", train_idx)
        _save_split("val", val_idx)
        _save_split("test", test_idx)
        print("[CACHE|IEDB2016] done.")

    need_build = False
    if split in ("train", "val", "test"):
        if not _cache_path(split).exists():
            need_build = True
    elif split == "all":
        for sp in ("train", "val", "test"):
            if not _cache_path(sp).exists():
                need_build = True
                break
    else:
        raise ValueError(f"split 必须是 'train'/'val'/'test'/'all'，收到: {split!r}")

    if need_build:
        _build_and_cache_all()

    def _load_split(sp: str) -> dict:
        p = _cache_path(sp)
        d = np.load(p, allow_pickle=True)

        ids = d["ids"]
        pep_seq = d["pep_seq"]
        mhc_seq = d["mhc_seq"]
        allele = d["allele"]

        pep_len = d["pep_len"].astype(np.int32) if "pep_len" in d else np.array([len(str(s)) for s in pep_seq], dtype=np.int32)
        mhc_len = d["mhc_len"].astype(np.int32) if "mhc_len" in d else np.array([len(str(s)) for s in mhc_seq], dtype=np.int32)

        y = d["y"].astype(np.float32)
        y_bin = d["y_bin"].astype(np.int64) if "y_bin" in d else (y >= CUTOFF_Y).astype(np.int64)
        ic50 = d["ic50"].astype(np.float32) if "ic50" in d else np.full((len(y),), np.nan, dtype=np.float32)

        pep_lm = d["pep_lm"].astype(np.float32)
        mhc_lm = d["mhc_lm"].astype(np.float32)

        return {
            "ids": ids,
            "pep_seq": pep_seq,
            "mhc_seq": mhc_seq,
            "allele": allele,
            "pep_len": pep_len,
            "mhc_len": mhc_len,
            "y": y,
            "y_bin": y_bin,
            "ic50": ic50,
            "pep_lm": pep_lm,
            "mhc_lm": mhc_lm,
        }

    if split in ("train", "val", "test"):
        return {split: _load_split(split)}
    else:
        parts = {sp: _load_split(sp) for sp in ("train", "val", "test")}

        def _cat(*xs):
            return np.concatenate(xs, axis=0)

        all_pkg = {
            "ids": _cat(parts["train"]["ids"], parts["val"]["ids"], parts["test"]["ids"]),
            "pep_seq": _cat(parts["train"]["pep_seq"], parts["val"]["pep_seq"], parts["test"]["pep_seq"]),
            "mhc_seq": _cat(parts["train"]["mhc_seq"], parts["val"]["mhc_seq"], parts["test"]["mhc_seq"]),
            "allele": _cat(parts["train"]["allele"], parts["val"]["allele"], parts["test"]["allele"]),
            "pep_len": _cat(parts["train"]["pep_len"], parts["val"]["pep_len"], parts["test"]["pep_len"]).astype(np.int32),
            "mhc_len": _cat(parts["train"]["mhc_len"], parts["val"]["mhc_len"], parts["test"]["mhc_len"]).astype(np.int32),
            "y": _cat(parts["train"]["y"], parts["val"]["y"], parts["test"]["y"]).astype(np.float32),
            "y_bin": _cat(parts["train"]["y_bin"], parts["val"]["y_bin"], parts["test"]["y_bin"]).astype(np.int64),
            "ic50": _cat(parts["train"]["ic50"], parts["val"]["ic50"], parts["test"]["ic50"]).astype(np.float32),
            "pep_lm": np.vstack([parts["train"]["pep_lm"], parts["val"]["pep_lm"], parts["test"]["pep_lm"]]).astype(np.float32),
            "mhc_lm": np.vstack([parts["train"]["mhc_lm"], parts["val"]["mhc_lm"], parts["test"]["mhc_lm"]]).astype(np.float32),
        }
        return {"all": all_pkg}
