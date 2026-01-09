# -*- coding: utf-8 -*-
# =========================================================
# MultiModalPPReg_LM
# 目标：Protein–Protein 回归（PP regression）
#
# 设计：
#   - 主体：沿用 MultiModalPPI_LM（prot1/prot2 的 1D+2D+3D 多模态骨架）
#   - Cross-Attn：沿用 DTA 版（带 gate 的 CrossAttentionFuse）
#   - 头部：沿用 DTA 版（reg_head_ca / reg_head_no_ca）
#
# 输入：
#   1D: prot1_lm [B, D1], prot2_lm [B, D2]
#   2D: g_prot1, g_prot2 (DGLGraph) with ndata['x'], edata['w']
#   3D: prot13d [B, D3_1], prot23d [B, D3_2]
#
# 输出：
#   y_pred: [B] 连续值（回归）
# =========================================================

from typing import Optional

import torch
import torch.nn as nn
import dgl


# ======【2D】带边权的 GCN（EW-GCN），用于两条蛋白链的图表示 ======
class GraphConvEW(nn.Module):
    """三层 GraphConv（DGL），带边权；显式分段读出，保证输出 [B, out_dim]。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()                                      # 行：父类初始化
        self.gcn1 = dgl.nn.GraphConv(                           # 行：第1层 GCN
            in_dim, hidden_dim, bias=False, allow_zero_in_degree=True
        )
        self.gcn2 = dgl.nn.GraphConv(                           # 行：第2层 GCN
            hidden_dim, hidden_dim, bias=False, allow_zero_in_degree=True
        )
        self.gcn3 = dgl.nn.GraphConv(                           # 行：第3层 GCN
            hidden_dim, out_dim, bias=False, allow_zero_in_degree=True
        )
        self.ln1 = nn.LayerNorm(hidden_dim)                     # 行：LayerNorm 稳定训练
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU(inplace=True)                        # 行：非线性激活
        self.dropout = nn.Dropout(0.1)                          # 行：dropout 防过拟合

    @staticmethod
    def _edge_feat_to_scalar(
        w: Optional[torch.Tensor], g: dgl.DGLGraph
    ) -> Optional[torch.Tensor]:
        """行：把 [E,Fe]/[E] 边特征压成 [E] 标量，供 GraphConv 的 edge_weight 使用。"""
        if (w is None) or (w.numel() == 0):                     # 行：无边特征
            return None
        if w.dim() == 2:                                        # 行：[E,Fe] → 平均到 [E]
            w = w.mean(dim=-1)
        else:
            w = w.view(-1)                                      # 行：保证为 1D
        if w.shape[0] > g.num_edges():                          # 行：若长度比边数多，截断
            w = w[: g.num_edges()]
        return w                                                # 行：返回 [E] 标量边权

    @staticmethod
    def _segmented_sum_by_graph(
        g: dgl.DGLGraph, node_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        行：显式按子图分段求和，返回 [B, D]。
        这样不会因为 dgl.sum_nodes 的 batch 语义差异导致错位。
        """
        counts_fn = getattr(g, "batch_num_nodes", None)          # 行：尝试取 batch_num_nodes
        if callable(counts_fn):
            counts = counts_fn()                                # 行：每个子图节点数
            if not torch.is_tensor(counts):
                counts = torch.as_tensor(counts, device=node_feat.device)
            else:
                counts = counts.to(node_feat.device)
            if counts.dim() == 0:
                counts = counts.view(1)

            offsets = torch.zeros(
                (counts.numel() + 1,), dtype=torch.long, device=node_feat.device
            )                                                   # 行：prefix-sum 索引
            offsets[1:] = torch.cumsum(counts, dim=0)

            outs = []
            for i in range(counts.numel()):
                seg = node_feat[offsets[i] : offsets[i + 1]]    # 行：该子图节点特征切片
                outs.append(seg.sum(dim=0, keepdim=True))        # 行：sum readout
            return torch.cat(outs, dim=0)                        # 行：[B,D]

        # 行：兜底：拆成单图逐个 sum
        graphs = dgl.unbatch(g)
        outs, start = [], 0
        for gi in graphs:
            n = gi.num_nodes()
            outs.append(node_feat[start : start + n].sum(dim=0, keepdim=True))
            start += n
        return torch.cat(outs, dim=0)

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor, w: Optional[torch.Tensor]) -> torch.Tensor:
        """行：输入批图 g、节点特征 x、边特征 w；输出每图向量 [B,out_dim]。"""
        device = next(self.parameters()).device                 # 行：当前模块所在 device

        w_scalar = self._edge_feat_to_scalar(w, g)              # 行：边权压到 [E]

        g = g.to(device)                                        # 行：图搬到 device
        x = x.to(device)                                        # 行：节点特征搬到 device
        if w_scalar is not None:
            w_scalar = w_scalar.to(device)                      # 行：边权搬到 device

        h = self.gcn1(g, x, edge_weight=w_scalar)               # 行：第1层 GCN
        h = self.ln1(self.act(h))                               # 行：ReLU + LN
        h = self.dropout(h)                                     # 行：Dropout

        h = self.gcn2(g, h, edge_weight=w_scalar)               # 行：第2层 GCN
        h = self.ln2(self.act(h))                               # 行：ReLU + LN
        h = self.dropout(h)                                     # 行：Dropout

        h = self.gcn3(g, h, edge_weight=w_scalar)               # 行：第3层 GCN

        readout = self._segmented_sum_by_graph(g, h)            # 行：图级 readout [B,out_dim]
        return readout


# ======【3D】蛋白对 3D 向量编码器（拼接 + 交互 → d_model） ======
class Pair3DEncoder(nn.Module):
    """行：把 prot13d / prot23d 两路 3D 向量融合成一个 [B,d_model]。"""

    def __init__(
        self,
        d_in_prot1: int = 512,                                  # 行：蛋白1 3D 向量维度
        d_in_prot2: int = 512,                                  # 行：蛋白2 3D 向量维度
        d_model: int = 256,                                     # 行：统一输出维度
        add_interactions: bool = True,                          # 行：是否加入交互项
    ):
        super().__init__()
        self.add_interactions = add_interactions                # 行：保存配置
        self.norm_1 = nn.LayerNorm(d_in_prot1)                  # 行：输入归一化
        self.norm_2 = nn.LayerNorm(d_in_prot2)
        self.proj_1 = nn.Linear(d_in_prot1, d_model, bias=False)# 行：投影到 d_model
        self.proj_2 = nn.Linear(d_in_prot2, d_model, bias=False)
        in_dim = d_model * (4 if add_interactions else 2)       # 行：拼接后的维度
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model, bias=False),             # 行：压回 d_model
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, prot13d: torch.Tensor, prot23d: torch.Tensor) -> torch.Tensor:
        """行：输入 [B,D1],[B,D2]；输出 [B,d_model]。"""
        p1 = self.norm_1(prot13d.float())                       # 行：归一化
        p2 = self.norm_2(prot23d.float())
        p1 = self.proj_1(p1)                                    # 行：投影
        p2 = self.proj_2(p2)

        if self.add_interactions:
            feat = torch.cat([p1, p2, p1 * p2, torch.abs(p1 - p2)], dim=-1)  # 行：交互拼接
        else:
            feat = torch.cat([p1, p2], dim=-1)                  # 行：仅拼接
        return self.mlp(feat)                                   # 行：MLP 输出 [B,d_model]


# ====== Cross-Attention（DTA 版：带 gate） ======
class CrossAttentionFuseGated(nn.Module):
    """行：多头交叉注意力 + 门控 + FFN，两层残差（Transformer Encoder 风格）。"""

    def __init__(
        self,
        dim_q: int,                                             # 行：Q 输入维度（2*d_model）
        dim_kv: int,                                            # 行：KV 输入维度（d_model）
        d_attn: int = 256,                                      # 行：注意力内部维度
        n_heads: int = 4,                                       # 行：多头数
        dropout: float = 0.1,                                   # 行：dropout
    ):
        super().__init__()
        self.q_proj = nn.Linear(dim_q, d_attn, bias=False)      # 行：Q 投影
        self.k_proj = nn.Linear(dim_kv, d_attn, bias=False)     # 行：K 投影
        self.v_proj = nn.Linear(dim_kv, d_attn, bias=False)     # 行：V 投影

        self.mha = nn.MultiheadAttention(                       # 行：标准多头注意力
            embed_dim=d_attn, num_heads=n_heads, dropout=dropout, batch_first=True
        )

        self.gate_linear = nn.Linear(dim_q, d_attn, bias=True)  # 行：由原始 q_vec 生成门控
        self.gate_act = nn.Sigmoid()                            # 行：门控压到 [0,1]

        self.ln1 = nn.LayerNorm(d_attn)                         # 行：残差1后的 LN
        self.ff = nn.Sequential(                                # 行：前馈网络
            nn.Linear(d_attn, 2 * d_attn, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_attn, d_attn, bias=False),
        )
        self.ln2 = nn.LayerNorm(d_attn)                         # 行：残差2后的 LN

    def forward(self, q_vec: torch.Tensor, kv_vec: torch.Tensor) -> torch.Tensor:
        """
        q_vec:  [B, Dq]（1D+2D 拼接向量）
        kv_vec: [B, Dk]（3D 对向量）
        return: [B, d_attn]
        """
        q = self.q_proj(q_vec).unsqueeze(1)                     # 行：[B,1,D]
        k = self.k_proj(kv_vec).unsqueeze(1)
        v = self.v_proj(kv_vec).unsqueeze(1)

        attn_out, _ = self.mha(q, k, v, need_weights=False)     # 行：交叉注意力输出 [B,1,D]

        gate_score = self.gate_act(self.gate_linear(q_vec))     # 行：[B,D] 门控分数
        gate_score = gate_score.unsqueeze(1)                    # 行：[B,1,D] 便于逐元素相乘
        attn_out = attn_out * gate_score                        # 行：门控注意力

        y = self.ln1(attn_out + q)                              # 行：残差1 + LN
        y2 = self.ff(y)                                         # 行：FFN
        y = self.ln2(y + y2)                                    # 行：残差2 + LN
        return y.squeeze(1)                                     # 行：[B,D]


# ====== 总模型：PP 回归（PPI 骨架 + DTA 的 gated cross-attn & heads） ======
class MultiModalPPA_LM(nn.Module):
    """
    多模态 Protein–Protein 回归模型（LM 版 + 可选模态）：

      - 1D：ESM 等 LM 特征
            prot1_lm: [B, D_prot1_lm] 或 None
            prot2_lm: [B, D_prot2_lm] 或 None
        → 得到 z_1d ∈ R^{B×d_model}

      - 2D：EW-GCN（蛋白图1 / 蛋白图2）
        → 得到 z_2d ∈ R^{B×d_model}

      - 3D：蛋白 3D 对（prot13d / prot23d）
        → 得到 z_3d ∈ R^{B×d_model}

      - 路径选择：
        * 若 has_3d 且 (has_1d 或 has_2d)：Cross-Attn(Gated) → reg_head_ca → y
        * 否则：concat(z1,z2,z3) → fuse_modalities_no_ca → reg_head_no_ca → y
    """

    def __init__(
        self,
        # 1D LM 特征维度
        d_prot1_lm: int,                                       # 行：蛋白1 LM 向量维度
        d_prot2_lm: int,                                       # 行：蛋白2 LM 向量维度
        # 2D 图输入维（节点/边）
        prot1_node_dim: int = 33,                              # 行：蛋白1 图节点特征维
        prot1_edge_dim: int = 3,                               # 行：蛋白1 图边特征维（仅说明）
        prot2_node_dim: int = 33,                              # 行：蛋白2 图节点特征维
        prot2_edge_dim: int = 3,                               # 行：蛋白2 图边特征维（仅说明）
        gcn_hidden: int = 128,                                 # 行：GCN 隐层
        gcn_out: int = 128,                                    # 行：GCN 输出
        # 3D 输入维
        d3_prot1: int = 512,                                   # 行：蛋白1 3D 向量维
        d3_prot2: int = 512,                                   # 行：蛋白2 3D 向量维
        # 融合与注意力维
        d_model: int = 256,                                    # 行：统一特征维度
        d_attn: int = 256,                                     # 行：Cross-Attn 内部维度
        n_heads: int = 4,                                      # 行：多头数
        add_interactions_3d: bool = True,                      # 行：3D 对是否加交互项
    ):
        super().__init__()                                     # 行：父类初始化

        # ---- 1D：LM 特征投影（prot1/prot2）----
        self.proj_prot1_lm = nn.Sequential(
            nn.LayerNorm(d_prot1_lm),                           # 行：先 LN 稳定输入分布
            nn.Linear(d_prot1_lm, d_model, bias=False),         # 行：投影到 d_model
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.proj_prot2_lm = nn.Sequential(
            nn.LayerNorm(d_prot2_lm),
            nn.Linear(d_prot2_lm, d_model, bias=False),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.fuse1d = nn.Sequential(
            nn.Linear(2 * d_model, d_model, bias=False),        # 行：拼接后压回 d_model
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # ---- 2D：两路 EW-GCN ----
        self.enc2d_prot1 = GraphConvEW(
            in_dim=prot1_node_dim, hidden_dim=gcn_hidden, out_dim=gcn_out
        )                                                       # 行：蛋白1 图编码器
        self.enc2d_prot2 = GraphConvEW(
            in_dim=prot2_node_dim, hidden_dim=gcn_hidden, out_dim=gcn_out
        )                                                       # 行：蛋白2 图编码器
        self.fuse2d = nn.Sequential(
            nn.Linear(2 * gcn_out, d_model, bias=False),         # 行：拼接两个图向量后压到 d_model
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # ---- 3D：蛋白对编码器 ----
        self.enc3d_pair = Pair3DEncoder(
            d_in_prot1=d3_prot1,
            d_in_prot2=d3_prot2,
            d_model=d_model,
            add_interactions=add_interactions_3d,
        )                                                       # 行：3D 对 → d_model

        # ---- Cross-Attn（用 DTA 的 gated 版）----
        self.cross = CrossAttentionFuseGated(
            dim_q=2 * d_model, dim_kv=d_model, d_attn=d_attn, n_heads=n_heads
        )                                                       # 行：Q=cat(z1d,z2d), KV=z3d

        # ---- 头部（用 DTA 的头）----
        self.reg_head_ca = nn.Sequential(
            nn.Linear(d_attn, 2 * d_model, bias=False),          # 行：CrossAttn 输出扩维
            nn.LayerNorm(2 * d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * d_model, 1, bias=True),                # 行：输出标量
        )

        self.fuse_modalities_no_ca = nn.Sequential(
            nn.Linear(3 * d_model, d_model, bias=False),         # 行：无 CrossAttn 时三模态拼接→压缩
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.reg_head_no_ca = nn.Sequential(
            nn.Linear(d_model, 2 * d_model, bias=False),         # 行：无 CrossAttn 时的回归头
            nn.LayerNorm(2 * d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * d_model, 1, bias=True),
        )

    def forward(
        self,
        prot1_lm: Optional[torch.Tensor] = None,                # 行：蛋白1 LM [B,D] 或 None
        prot2_lm: Optional[torch.Tensor] = None,                # 行：蛋白2 LM [B,D] 或 None
        g_prot1: Optional[dgl.DGLGraph] = None,                 # 行：蛋白1 图
        g_prot2: Optional[dgl.DGLGraph] = None,                 # 行：蛋白2 图
        prot13d: Optional[torch.Tensor] = None,                 # 行：蛋白1 3D 向量
        prot23d: Optional[torch.Tensor] = None,                 # 行：蛋白2 3D 向量
    ) -> torch.Tensor:
        device = next(self.parameters()).device                 # 行：当前模型 device

        # 行：判断输入模态是否存在
        has_1d = (prot1_lm is not None) and (prot2_lm is not None)
        has_2d = (g_prot1 is not None) and (g_prot2 is not None)
        has_3d = (prot13d is not None) and (prot23d is not None)

        if not (has_1d or has_2d or has_3d):                    # 行：至少要有一种模态
            raise ValueError("至少需要提供 1D/2D/3D 中的一种模态输入。")

        z1d = z2d = z3d = None                                  # 行：三模态统一向量

        # === 1) 1D：LM 分支 ===
        if has_1d:
            prot1_lm = prot1_lm.to(device)                      # 行：搬到 device
            prot2_lm = prot2_lm.to(device)
            p1 = self.proj_prot1_lm(prot1_lm)                   # 行：映射到 d_model
            p2 = self.proj_prot2_lm(prot2_lm)
            z1d = self.fuse1d(torch.cat([p1, p2], dim=-1))      # 行：拼接后压缩到 [B,d_model]

        # === 2) 2D：图分支 ===
        if has_2d:
            v1 = self.enc2d_prot1(
                g_prot1, g_prot1.ndata["x"].float(), g_prot1.edata["w"].float()
            )                                                   # 行：蛋白1 图 → [B,gcn_out]
            v2 = self.enc2d_prot2(
                g_prot2, g_prot2.ndata["x"].float(), g_prot2.edata["w"].float()
            )                                                   # 行：蛋白2 图 → [B,gcn_out]
            z2d = self.fuse2d(torch.cat([v1, v2], dim=-1))      # 行：2D 融合 → [B,d_model]

        # === 3) 3D：对向量分支 ===
        if has_3d:
            prot13d = torch.nan_to_num(
                prot13d.to(device), nan=0.0, posinf=1e6, neginf=-1e6
            )                                                   # 行：NaN/Inf 清理
            prot23d = torch.nan_to_num(
                prot23d.to(device), nan=0.0, posinf=1e6, neginf=-1e6
            )
            z3d = self.enc3d_pair(prot13d, prot23d)             # 行：3D 对编码 → [B,d_model]

        # 行：若多模态并存，检查 batch 是否一致
        B = None
        for z in (z1d, z2d, z3d):
            if z is not None:
                B = z.size(0) if B is None else B
                assert z.size(0) == B, f"batch size 不一致：期望 {B}, 但有 {z.size(0)}"

        # === 分支 A：有 3D 且存在 1D 或 2D → gated Cross-Attn ===
        if has_3d and (z1d is not None or z2d is not None):
            if z1d is not None and z2d is not None:
                q_vec = torch.cat([z1d, z2d], dim=-1)           # 行：Q=[z1||z2]
            elif z1d is not None:
                q_vec = torch.cat([z1d, torch.zeros_like(z1d)], dim=-1)  # 行：补 0
            else:
                q_vec = torch.cat([torch.zeros_like(z2d), z2d], dim=-1)

            ca = self.cross(q_vec, z3d)                         # 行：交叉注意力融合 [B,d_attn]
            y = self.reg_head_ca(ca).squeeze(-1)                # 行：回归头输出 [B]
            return y

        # === 分支 B：无 3D 或只有 3D → 直接融合 ===
        base_vec = z1d if z1d is not None else (z2d if z2d is not None else z3d)
        assert base_vec is not None

        zero = torch.zeros_like(base_vec)                       # 行：构造 0 向量占位
        z1 = z1d if z1d is not None else zero
        z2 = z2d if z2d is not None else zero
        z3 = z3d if z3d is not None else zero

        fused = self.fuse_modalities_no_ca(torch.cat([z1, z2, z3], dim=-1))  # 行：[B,d_model]
        y = self.reg_head_no_ca(fused).squeeze(-1)              # 行：输出 [B]
        return y
