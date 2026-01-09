import math

import numpy as np                       # 数值计算
import dgl                               # DGL 图框架
import torch
import torch.nn as nn                    # 神经网络模块
import torch.nn.functional as F          # 常用函数（激活、loss 等）
import dgl.nn.pytorch as dglnn           # DGL 的图层
from typing import Optional, Tuple


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()                                                      # 行：父类初始化
        # 行：第一层 GraphConv，输出通道设为 4*hidden，并允许 0 入度（更稳）
        self.layer1 = dglnn.GraphConv(in_dim, hidden_dim * 4, bias=False, allow_zero_in_degree=True)
        # 行：与第一层并联的线性捷径分支（把原始特征映射到同维度后拼接）
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 4, bias=False),                     # 行：线性升维
            nn.LayerNorm(hidden_dim * 4),                                      # 行：层归一化
            nn.ReLU(inplace=True),                                             # 行：激活
        )
        # 行：第二、三层 GraphConv
        self.layer2 = dglnn.GraphConv(hidden_dim * 8, hidden_dim * 4, bias=False, allow_zero_in_degree=True)
        self.layer3 = dglnn.GraphConv(hidden_dim * 4, out_dim,        bias=False, allow_zero_in_degree=True)

    @staticmethod
    def _to_scalar_edge_weight(
        w: Optional[torch.Tensor], g: dgl.DGLGraph, device
    ) -> Optional[torch.Tensor]:
        """行：把 [E,3]/[E] 的边特征压成 GraphConv 可用的一维标量权重并放到 device 上。"""
        if w is None or w.numel() == 0:                                        # 行：无边或无特征→返回 None
            return None
        if w.dim() == 2:                                                       # 行：[E,3] → 取通道均值到 [E]
            w = w.mean(dim=-1)
        else:                                                                  # 行：其它形状→展平成 [E]
            w = w.view(-1)
        if w.shape[0] > g.num_edges():                                         # 行：防御式截断，确保长度不超过边数
            w = w[:g.num_edges()]
        return w.to(device)                                                    # 行：放到目标设备

    @staticmethod
    def _add_self_loop_and_patch_weight(
        g: dgl.DGLGraph, w: Optional[torch.Tensor], device
    ) -> Tuple[dgl.DGLGraph, Optional[torch.Tensor]]:
        """行：给图补自环；若提供边权则为新增自环补上权重 1.0。"""
        g_sl = dgl.add_self_loop(g)                                            # 行：为缺失入度的点补自环（幂等）
        if w is None:                                                          # 行：没有边权就直接返回
            return g_sl, None
        E_orig = g.num_edges()                                                 # 行：原始边数
        loops_added = g_sl.num_edges() - E_orig                                # 行：新增自环数量
        if loops_added > 0:                                                    # 行：若确实新增了自环
            w_loop = torch.ones(loops_added, dtype=w.dtype, device=device)     # 行：为自环创建权重=1.0
            w = torch.cat([w, w_loop], dim=0)                                  # 行：把自环权重接到原权重后
        return g_sl, w                                                         # 行：返回新图与新权重

    def forward(self, graph: dgl.DGLGraph, x: torch.Tensor, w: Optional[torch.Tensor]):
        device = next(self.parameters()).device                                # 行：以模型参数所在设备为准

        graph = graph.to(device)                                               # 行：把图搬到同一设备（ndata/edata 随之迁移）
        x = x.to(device)                                                       # 行：节点特征到同一设备
        w_scalar = self._to_scalar_edge_weight(w, graph, device)               # 行：[E,3]/[E] → [E] 并放到设备
        graph, w_scalar = self._add_self_loop_and_patch_weight(graph, w_scalar, device)  # 行：补自环与自环权重

        x1 = self.layer1(graph, x, edge_weight=w_scalar)                       # 行：GCN 第1层（带边权）
        x1 = F.relu(x1, inplace=True)                                          # 行：激活
        f1 = self.fc1(x)                                                       # 行：线性捷径分支
        x1f1 = torch.cat((x1, f1), dim=1)                                      # 行：拼成 [N, 8*hidden]

        x2 = self.layer2(graph, x1f1, edge_weight=w_scalar)                     # 行：GCN 第2层
        x2 = F.relu(x2, inplace=True)                                          # 行：激活

        x3 = self.layer3(graph, x2, edge_weight=w_scalar)                       # 行：GCN 第3层
        x3 = F.relu(x3, inplace=True)                                          # 行：激活

        with graph.local_scope():                                              # 行：局部作用域，避免污染原图
            graph.ndata['x'] = x3                                              # 行：挂节点表征
            readout = dgl.sum_nodes(graph, 'x')                                # 行：批图按图求和读出 → [B, out_dim]
            return readout                                                     # 行：返回图级表征


class GraphConvolution(nn.Module):
    def __init__(self, in_features_v, out_features_v, in_features_e, out_features_e, bias=True, node_layer=True):
        super().__init__()
        self.in_features_e = in_features_e
        self.out_features_e = out_features_e
        self.in_features_v = in_features_v
        self.out_features_v = out_features_v

        if node_layer:
            self.node_layer = True
            self.weight = nn.Parameter(torch.FloatTensor(in_features_v, out_features_v))
            self.p = nn.Parameter(torch.from_numpy(np.random.normal(size=(1, in_features_e))).float())
            if bias:
                self.bias = nn.Parameter(torch.FloatTensor(out_features_v))
            else:
                self.register_parameter('bias', None)
        else:
            self.node_layer = False
            self.weight = nn.Parameter(torch.FloatTensor(in_features_e, out_features_e))
            self.p = nn.Parameter(torch.from_numpy(np.random.normal(size=(1, in_features_v))).float())
            if bias:
                self.bias = nn.Parameter(torch.FloatTensor(out_features_e))
            else:
                self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, H_v, H_e, adj_e, adj_v, T):
        device = H_v.device
        if self.node_layer:
            # 从边特征生成调制矩阵 M1，作用在节点邻接上
            multiplier1 = torch.spmm(T, torch.diag((H_e @ self.p.t()).t()[0])) @ T.to_dense().t()
            mask1 = torch.eye(multiplier1.shape[0]).to(device)
            M1 = mask1 * torch.ones(multiplier1.shape[0], device=device) + (1. - mask1) * multiplier1
            adjusted_A = torch.mul(M1, adj_v.to_dense())                   # 加权后的节点邻接
            output = torch.mm(adjusted_A, torch.mm(H_v, self.weight))         # 信息传播 + 线性变换
            if self.bias is not None:
                output = output + self.bias
            return output, H_e                                          # 仅更新节点
        else:
            # 从节点特征生成调制矩阵 M3，作用在边邻接上
            multiplier2 = torch.spmm(T.t(), torch.diag((H_v @ self.p.t()).t()[0])) @ T.to_dense()
            mask2 = torch.eye(multiplier2.shape[0]).to(device)
            M3 = mask2 * torch.ones(multiplier2.shape[0], device=device) + (1. - mask2) * multiplier2
            adjusted_A = torch.mul(M3, adj_e.to_dense())
            normalized_adjusted_A = adjusted_A / adjusted_A.max(0, keepdim=True)[0]  # 归一化
            output = torch.mm(normalized_adjusted_A, torch.mm(H_e, self.weight))           # 更新边特征
            if self.bias is not None:
                output = output + self.bias
            return H_v, output                                          # 仅更新边

class CensNet(nn.Module):
    def __init__(self, nfeat_v, nfeat_e, nhid, nclass, dropout):
        super().__init__()
        # 第1轮：更新节点，外加节点的线性支路
        self.gc1 = GraphConvolution(nfeat_v, nhid, nfeat_e, nfeat_e, node_layer=True)
        self.fc1 = nn.Sequential(
            nn.Linear(nfeat_v, nhid, bias=False),
            nn.LayerNorm(nhid),
            nn.ReLU(inplace=True),
        )
        # 第2轮：更新边，外加边特征的线性支路
        self.gc2 = GraphConvolution(nhid*2, nhid*2, nfeat_e, nfeat_e, node_layer=False)
        self.fc2 = nn.Sequential(
            nn.Linear(nfeat_e, nfeat_e, bias=False),
            nn.LayerNorm(nfeat_e),
            nn.ReLU(inplace=True),
        )
        # 第3~5轮：节点->边->节点
        self.gc3 = GraphConvolution(nhid*2, nhid, nfeat_e*2, nfeat_e*2, node_layer=True)
        self.gc4 = GraphConvolution(nhid, nhid, nfeat_e*2, nfeat_e, node_layer=False)
        self.gc5 = GraphConvolution(nhid, nclass, nfeat_e, nfeat_e, node_layer=True)
        self.dropout = dropout

    def forward(self, X, Z, adj_e, adj_v, T):
        # 1) 节点更新 + 节点线性分支
        X1, Z1 = F.relu(self.gc1(X, Z, adj_e, adj_v, T)[0]), F.relu(self.gc1(X, Z, adj_e, adj_v, T)[1])
        X1 = F.dropout(X1, self.dropout, training=self.training)
        Z1 = F.dropout(Z1, self.dropout, training=self.training)
        F1 = self.fc1(X)
        X1F1 = torch.cat((X1, F1), 1)                                      # 节点拼接

        # 2) 边更新 + 边线性分支
        X2, Z2 = F.relu(self.gc2(X1F1, Z1, adj_e, adj_v, T)[0]), F.relu(self.gc2(X1F1, Z1, adj_e, adj_v, T)[1])
        X2 = F.dropout(X2, self.dropout, training=self.training)
        Z2 = F.dropout(Z2, self.dropout, training=self.training)
        F2 = self.fc2(Z)
        Z2F2 = torch.cat((Z2, F2), 1)                                      # 边拼接

        # 3) 节点->边->节点
        X3, Z3 = F.relu(self.gc3(X2, Z2F2, adj_e, adj_v, T)[0]), F.relu(self.gc3(X2, Z2F2, adj_e, adj_v, T)[1])
        X3 = F.dropout(X3, self.dropout, training=self.training)
        Z3 = F.dropout(Z3, self.dropout, training=self.training)

        X4, Z4 = F.relu(self.gc4(X3, Z3, adj_e, adj_v, T)[0]), F.relu(self.gc4(X3, Z3, adj_e, adj_v, T)[1])
        X4 = F.dropout(X4, self.dropout, training=self.training)
        Z4 = F.dropout(Z4, self.dropout, training=self.training)

        X5, Z5 = self.gc5(X4, Z4, adj_e, adj_v, T)                      # 最后一层只取节点输出
        return X5                                                       # [Nv, nclass] 或图级读出前的节点表示

# === 1) 特征行归一化（L2） ===
def _normalize_features(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    nrm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp_min(eps)
    return x / nrm

# === 2) 稠密 -> 稀疏 COO（torch） ===
def _dense_to_coo(M: torch.Tensor) -> torch.Tensor:
    idx = M.nonzero(as_tuple=False).t()               # [2, nnz]
    vals = M[idx[0], idx[1]].to(M.dtype)
    return torch.sparse_coo_tensor(idx, vals, M.shape, device=M.device)

# === 3) 构造图的邻接(节点×节点) —— 用 edges() 通用实现 ===
def _graph_adj_coo(g: dgl.DGLGraph) -> torch.Tensor:
    N = g.num_nodes()
    src, dst = g.edges()                              # 1D LongTensor
    if src.numel() == 0:
        idx = torch.zeros(2, 0, dtype=torch.long, device=g.device if hasattr(g, "device") else None)
        vals = torch.zeros(0, dtype=torch.float32, device=idx.device)
    else:
        idx = torch.stack([src, dst], dim=0)             # [2, E]
        vals = torch.ones(src.shape[0], dtype=torch.float32, device=src.device)
    return torch.sparse_coo_tensor(idx, vals, (N, N), device=vals.device)

# === 4) 线图的邻接(边×边) —— 先生成线图，再走同样的 edges() 路径 ===
def _linegraph_adj_coo(g: dgl.DGLGraph) -> torch.Tensor:
    lg = dgl.line_graph(g, backtracking=False)
    return _graph_adj_coo(lg)

# === 5) 节点-边关联矩阵(节点×边) —— 通用实现：每条边在其两个端点处放 1 ===
def _incidence_coo(g: dgl.DGLGraph) -> torch.Tensor:
    N = g.num_nodes()
    E = g.num_edges()
    device = g.device if hasattr(g, "device") else None
    if E == 0:
        idx = torch.zeros(2, 0, dtype=torch.long, device=device)
        vals = torch.zeros(0, dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(idx, vals, (N, E), device=device)
    src, dst = g.edges()
    rows = torch.cat([src, dst], dim=0)                  # [2E]
    cols = torch.cat([torch.arange(E, device=src.device),   # [E]
                   torch.arange(E, device=src.device)], dim=0)  # [2E]
    idx = torch.stack([rows, cols], dim=0)               # [2, 2E]
    vals = torch.ones(2 * E, dtype=torch.float32, device=src.device)
    return torch.sparse_coo_tensor(idx, vals, (N, E), device=src.device)

# === 6) 稀疏 COO 加自环 I ===
def _add_self_loops(A: torch.Tensor) -> torch.Tensor:
    N = A.size(0)
    device = A.device
    eye_idx = torch.arange(N, device=device)
    I = torch.sparse_coo_tensor(
        torch.stack([eye_idx, eye_idx], dim=0),
        torch.ones(N, dtype=A.dtype, device=device),
        (N, N), device=device
    )
    return (A + I).coalesce()

# === 7) 稀疏 COO 对称归一化 D^{-1/2} A D^{-1/2} ===
def _sym_normalize_coo(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    A = A.coalesce()
    N = A.size(0)
    ones = torch.ones(N, 1, device=A.device, dtype=A.dtype)
    d = torch.sparse.mm(A, ones).squeeze(1)              # 度向量
    d_inv_sqrt = (d + eps).pow(-0.5)
    idx = A.indices()
    vals = A.values() * d_inv_sqrt[idx[0]] * d_inv_sqrt[idx[1]]
    return torch.sparse_coo_tensor(idx, vals, A.shape, device=A.device)

# === 8) 主函数：返回 CensNet 需要的五件套 ===
def batch_normalize(batch_graph: dgl.DGLGraph):
    """
    返回顺序（与 CensNet.forward 对齐）：
      nor_features   : [Nv, Fv]  dense
      nor_e_features : [Ne, Fe]  dense
      nor_e_adj      : [Ne, Ne]  sparse COO
      nor_adj        : [Nv, Nv]  sparse COO
      sparse_t_mat   : [Nv, Ne]  sparse COO
    """
    # 取原始特征
    features   = batch_graph.ndata['x'].float()       # [Nv, Fv]
    e_features = batch_graph.edata['w'].float()       # [Ne, Fe]

    # 构造三大稀疏矩阵
    A  = _graph_adj_coo(batch_graph)                  # [Nv, Nv]
    Ae = _linegraph_adj_coo(batch_graph)              # [Ne, Ne]
    T  = _incidence_coo(batch_graph)                  # [Nv, Ne]

    # 特征行归一化
    nor_features   = _normalize_features(features)
    nor_e_features = _normalize_features(e_features)

    # 邻接：加自环 + 对称归一化
    nor_adj   = _sym_normalize_coo(_add_self_loops(A))
    nor_e_adj = _sym_normalize_coo(_add_self_loops(Ae))

    return nor_features, nor_e_features, nor_e_adj, nor_adj, T

class GPCNDTA(nn.Module):
    """只用 2D 图（配体+靶标），**单样本**也能前向；不需要序列/药效团/bs。"""
    DEFAULT_CFG = {
        "drug_node_dim":   70,   # 配体节点特征维度（你当前 70）
        "drug_edge_dim":    6,   # 配体边特征维度（你当前 6）
        "target_node_dim": 33,   # 靶标节点特征维度（你当前 33）
        "target_edge_dim":  3,   # 靶标边特征维度（你当前 3）
        "node_hidden_dim": 128,  # 隐层
        "node_out_dim":    128,  # 图编码后维度
        "dropout_ratio":   0.1,
        "regression_value_dim": 1
    }

    def __init__(self, **overrides):
        super().__init__()
        cfg = dict(self.DEFAULT_CFG); cfg.update(overrides or {})
        self.cfg = cfg

        # 读配置
        self.din  = cfg["drug_node_dim"]
        self.die  = cfg["drug_edge_dim"]
        self.tin  = cfg["target_node_dim"]
        self.tie  = cfg["target_edge_dim"]
        self.hid  = cfg["node_hidden_dim"]
        self.out  = cfg["node_out_dim"]
        self.drop = cfg["dropout_ratio"]
        self.rdim = cfg["regression_value_dim"]

        # 分支：配体图编码（节点级 -> 读出）
        self.dCensNet = CensNet(self.din, self.die, self.hid, self.out, self.drop)

        # 分支：靶标图编码（节点级或图级，视你的 GCN 实现而定）
        self.tgcn = GCN(self.tin, self.hid, self.out)

        # 融合 + 回归
        self.fuse = nn.Sequential(
            nn.Linear(self.out * 2, self.hid * 4, bias=False),
            nn.LayerNorm(self.hid * 4),
            nn.Dropout(self.drop),
            nn.ReLU(inplace=True),
            nn.Linear(self.hid * 4, self.hid * 2, bias=False),
            nn.ReLU(inplace=True),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(self.hid * 2, self.hid * 4, bias=False),
            nn.LayerNorm(self.hid * 4),
            nn.Dropout(self.drop),
            nn.ReLU(inplace=True),
            nn.Linear(self.hid * 4, self.rdim, bias=False)
        )

    @staticmethod
    def _readout_mean(g: dgl.DGLGraph, node_emb: torch.Tensor) -> torch.Tensor:
        """单图读出：把每个节点表征做 mean pooling -> [1, D]"""
        # 把节点表征挂到图上
        g = g.local_var()
        g.ndata['_h'] = node_emb
        # 平均读出（也可换 sum/max）
        hg = dgl.mean_nodes(g, '_h')  # [1, D]
        return hg

    def _ensure_non_empty_target(self, tg: dgl.DGLGraph) -> dgl.DGLGraph:
        """若靶标图是空的，兜底成一个1节点0边的占位图，x/w 全 0"""
        if tg.num_nodes() > 0:
            return tg
        # 构造占位图
        tg_fallback = dgl.graph(([], []), num_nodes=1)
        tg_fallback.ndata['x'] = torch.zeros(1, self.tin)  # [1, target_node_dim]
        tg_fallback.edata['w'] = torch.zeros(0, self.tie)  # [0, target_edge_dim]
        return tg_fallback

    def forward(self,
                drug_g: dgl.DGLGraph,     # 单个配体图
                target_g: dgl.DGLGraph    # 单个靶标图
                ):
        # ===== 1) 配体图：归一化 -> CensNet -> 单图读出 =====
        d_x, d_w = drug_g.ndata['x'], drug_g.edata['w']                   # 取节点/边特征
        # 自检维度
        if d_x.size(-1) != self.din:
            raise ValueError(f"[drug_node_dim] expect {self.din}, got {d_x.size(-1)}")
        if d_w.size(-1) != self.die:
            raise ValueError(f"[drug_edge_dim] expect {self.die}, got {d_w.size(-1)}")

        # 归一化得到 CensNet 需要的五件套
        nor_fx, nor_ew, nor_e_adj, nor_adj, t_mat = batch_normalize(drug_g)
        # 过 CensNet 得每个节点表征
        d_nodes = self.dCensNet(nor_fx, nor_ew, nor_e_adj, nor_adj, t_mat)   # [Nd, out]
        # 单图读出
        d_vec = self._readout_mean(drug_g, d_nodes)                          # [1, out]

        # ===== 2) 靶标图：兜底空图 -> GCN -> （如返回节点级就自己读出）=====
        target_g = self._ensure_non_empty_target(target_g)
        tx, tw = target_g.ndata['x'], target_g.edata['w']
        if tx.size(-1) != self.tin:
            raise ValueError(f"[target_node_dim] expect {self.tin}, got {tx.size(-1)}")
        if tw.numel() > 0 and tw.size(-1) != self.tie:
            raise ValueError(f"[target_edge_dim] expect {self.tie}, got {tw.size(-1)}")

        Gtar = self.tgcn(target_g, tx, tw)  # 通常返回 [B, out]

        # 若返回的是节点级（第一维==图里总节点数），再做一次 mean 读出；
        # 否则直接当作图级向量用。
        if Gtar.size(0) == target_g.num_nodes():
            t_vec = self._readout_mean(target_g, Gtar)  # [B, out]
        else:
            t_vec = Gtar  # [B, out]

        # 假设这里已经得到 d_vec, t_vec，形状期望都是 [B, out]
        B = d_vec.size(0)
        if t_vec.size(0) != B:
            if t_vec.size(0) == 1:
                # 把 [1, out] 扩成 [B, out]（同一图用于整个 batch；仅用于兜底）
                t_vec = t_vec.expand(B, -1)
            else:
                raise ValueError(f"Batch mismatch: drug={B}, target={t_vec.size(0)}")


        feat = torch.cat([d_vec, t_vec], dim=-1)

        h = self.fuse(feat)                               # [1, 2*hid]
        y = self.reg_head(h)                              # [1, rdim]
        return y
