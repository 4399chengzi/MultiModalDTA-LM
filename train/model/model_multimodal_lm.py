# -*- coding: utf-8 -*-
# =========================================================
# 多模态 DTA 模型（1D: 预训练 LM 特征 + 2D 图 + 3D Uni-Mol2 + Cross-Attention）
#
# 设计要点：
#   - 1D 分支：直接输入 ChemBERTa-2 / ESM-2 等预训练大模型的向量：
#         drug_lm: [B, D_drug_lm]
#         prot_lm: [B, D_prot_lm]
#     投影到统一维度 d_model 后融合成 1D 表示 z_1d ∈ R^{B×d_model}
#
#   - 2D 分支：GraphConvEW（带边权），对配体图/蛋白图做三层 GCN + 图级读出
#     得到 2D 表示 z_2d ∈ R^{B×d_model}
#
#   - 3D 分支：Pair3DEncoder，把 Uni-Mol2 的 [lig3d, poc3d] 编码到 z_3d ∈ R^{B×d_model}
#
#   - Cross-Attention：
#       * 若存在 3D 且 (存在 1D 或 2D)：
#           - Q = concat(1D, 2D) → 若缺一支则用 0 向量补齐，维度恒为 2*d_model
#           - K,V = z_3d
#           - 输出 ca ∈ R^{B×d_attn}，走 reg_head_ca 做回归
#       * 否则（无 3D，或只有 3D 没有 1D/2D）：
#           - 把 (z_1d, z_2d, z_3d) 拼成 [z1||z2||z3]，缺失模态用 0 向量补齐
#           - 经 fuse_modalities_no_ca 压到 d_model，再走 reg_head_no_ca 做回归
#
#   - 因此：同一个模型可以支持：
#       * 1D-only
#       * 2D-only
#       * 3D-only
#       * 1D+2D
#       * 1D+3D
#       * 2D+3D
#       * 1D+2D+3D
# =========================================================

from typing import Optional

import torch
import torch.nn as nn
import dgl


# ======【2D】带边权的 GCN（EW-GCN），用于药物图与蛋白图统一编码 ======
class GraphConvEW(nn.Module):
    """三层 GraphConv（DGL），带边权；显式分段读出，保证输出 [B, out_dim]。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.gcn1 = dgl.nn.GraphConv(
            in_dim, hidden_dim, bias=False, allow_zero_in_degree=True
        )
        self.gcn2 = dgl.nn.GraphConv(
            hidden_dim, hidden_dim, bias=False, allow_zero_in_degree=True
        )
        self.gcn3 = dgl.nn.GraphConv(
            hidden_dim, out_dim, bias=False, allow_zero_in_degree=True
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)

    @staticmethod
    def _edge_feat_to_scalar(
        w: Optional[torch.Tensor], g: dgl.DGLGraph
    ) -> Optional[torch.Tensor]:
        """把 [E,Fe]/[E] 边特征压成 [E] 标量，供 GraphConv 的 edge_weight 使用。"""
        if (w is None) or (w.numel() == 0):
            return None
        if w.dim() == 2:  # [E,Fe] → 通道均值
            w = w.mean(dim=-1)
        else:
            w = w.view(-1)
        if w.shape[0] > g.num_edges():
            w = w[: g.num_edges()]
        return w

    @staticmethod
    def _segmented_sum_by_graph(
        g: dgl.DGLGraph, node_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        显式按子图分段求和（不依赖 dgl.sum_nodes 的批语义）。
        返回 [B, D]，B 为批内子图数（即 batch size）。
        """
        counts_fn = getattr(g, "batch_num_nodes", None)
        if callable(counts_fn):
            counts = counts_fn()
            if not torch.is_tensor(counts):
                counts = torch.as_tensor(counts, device=node_feat.device)
            else:
                counts = counts.to(node_feat.device)
            if counts.dim() == 0:  # 单图情况
                counts = counts.view(1)
            offsets = torch.zeros(
                (counts.numel() + 1,), dtype=torch.long, device=node_feat.device
            )
            offsets[1:] = torch.cumsum(counts, dim=0)
            outs = []
            for i in range(counts.numel()):
                seg = node_feat[offsets[i] : offsets[i + 1]]
                outs.append(seg.sum(dim=0, keepdim=True))  # [1, D]
            return torch.cat(outs, dim=0)  # [B, D]

        # 兜底：用 unbatch（略慢但通用）
        graphs = dgl.unbatch(g)
        outs = []
        start = 0
        for gi in graphs:
            n = gi.num_nodes()
            outs.append(node_feat[start : start + n].sum(dim=0, keepdim=True))
            start += n
        return torch.cat(outs, dim=0)  # [B, D]

    def forward(
        self, g: dgl.DGLGraph, x: torch.Tensor, w: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """输入批图 g，节点特征 x、边特征 w；输出每图读出向量 [B, out_dim]。"""
        device = next(self.parameters()).device

        # 边权降到标量
        w_scalar = self._edge_feat_to_scalar(w, g)

        # 统一搬到同一设备
        g = g.to(device)
        x = x.to(device)
        if w_scalar is not None:
            w_scalar = w_scalar.to(device)

        # 三层 GraphConv
        h = self.gcn1(g, x, edge_weight=w_scalar)
        h = self.ln1(self.act(h))
        h = self.dropout(h)

        h = self.gcn2(g, h, edge_weight=w_scalar)
        h = self.ln2(self.act(h))
        h = self.dropout(h)

        h = self.gcn3(g, h, edge_weight=w_scalar)

        # 图级读出
        readout = self._segmented_sum_by_graph(g, h)  # [B, out_dim]
        return readout


# ======【3D】Uni-Mol2 向量对编码器（拼接 + 可选交互 → 降维到 d_model） ======
class Pair3DEncoder(nn.Module):
    """把 lig3d/poc3d 两路向量拼成一个“3D 对”特征，并降维到统一 d_model。"""

    def __init__(
        self,
        d_in_lig: int = 512,
        d_in_poc: int = 512,
        d_model: int = 256,
        add_interactions: bool = True,
    ):
        super().__init__()
        self.add_interactions = add_interactions
        self.norm_l = nn.LayerNorm(d_in_lig)
        self.norm_p = nn.LayerNorm(d_in_poc)
        self.proj_l = nn.Linear(d_in_lig, d_model, bias=False)
        self.proj_p = nn.Linear(d_in_poc, d_model, bias=False)
        in_dim = d_model * (4 if add_interactions else 2)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model, bias=False),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, lig3d: torch.Tensor, poc3d: torch.Tensor) -> torch.Tensor:
        """输入 [B,Dl] 与 [B,Dp]；输出 [B,d_model] 的 3D 对表示"""
        l = self.norm_l(lig3d.float())
        p = self.norm_p(poc3d.float())
        l = self.proj_l(l)
        p = self.proj_p(p)
        if self.add_interactions:
            feat = torch.cat([l, p, l * p, torch.abs(l - p)], dim=-1)  # [B, 4*d_model]
        else:
            feat = torch.cat([l, p], dim=-1)  # [B, 2*d_model]
        out = self.mlp(feat)  # [B, d_model]
        return out


# ====== Cross-Attention（把 1D+2D 的融合向量作为 Q，3D 对向量作为 K/V） ======
# ====== Cross-Attention（把 1D+2D 的融合向量作为 Q，3D 对向量作为 K/V） ======
class CrossAttentionFuse(nn.Module):
    """多头交叉注意力 + 门控机制 + 前馈，两层残差（Transformer Encoder 风格）。"""

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        d_attn: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # 行：把 1D+2D 融合向量投影到注意力空间，作为 Q
        self.q_proj = nn.Linear(dim_q, d_attn, bias=False)
        # 行：把 3D 向量投影到注意力空间，作为 K/V
        self.k_proj = nn.Linear(dim_kv, d_attn, bias=False)
        self.v_proj = nn.Linear(dim_kv, d_attn, bias=False)

        # 行：标准多头注意力（Scaled Dot-Product Attention）
        self.mha = nn.MultiheadAttention(
            embed_dim=d_attn, num_heads=n_heads, dropout=dropout, batch_first=True
        )

        # ====== 新增：门控层（Gated Attention） ======
        # 行：使用原始的 q_vec 生成门控分数，再用 sigmoid 压到 [0,1]
        self.gate_linear = nn.Linear(dim_q, d_attn, bias=True)
        self.gate_act = nn.Sigmoid()

        # 行：第一个残差 + LayerNorm
        self.ln1 = nn.LayerNorm(d_attn)

        # 行：前馈网络（Position-wise FFN）
        self.ff = nn.Sequential(
            nn.Linear(d_attn, 2 * d_attn, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_attn, d_attn, bias=False),
        )

        # 行：第二个残差 + LayerNorm
        self.ln2 = nn.LayerNorm(d_attn)

    def forward(self, q_vec: torch.Tensor, kv_vec: torch.Tensor) -> torch.Tensor:
        """
        q_vec:  [B, Dq]（1D+2D 拼接后向量）
        kv_vec: [B, Dk]（3D 对向量）
        return: [B, d_attn]（融合后的表示）
        """
        # 行：投影到注意力空间，并加上长度维度 L=1
        q = self.q_proj(q_vec).unsqueeze(1)      # [B, 1, D]
        k = self.k_proj(kv_vec).unsqueeze(1)     # [B, 1, D]
        v = self.v_proj(kv_vec).unsqueeze(1)     # [B, 1, D]

        # 行：标准多头交叉注意力
        attn_out, _ = self.mha(q, k, v, need_weights=False)  # [B, 1, D]

        # ====== 关键创新：在 SDPA 输出后加入门控 ======
        # 行：由原始 q_vec 计算门控分数，形状 [B, D]
        gate_score = self.gate_act(self.gate_linear(q_vec))  # [B, D]
        # 行：扩展一个长度维度，方便与 attn_out 做逐元素相乘
        gate_score = gate_score.unsqueeze(1)                 # [B, 1, D]
        # 行：Gated Attention：Y' = Y ⊙ σ(X W_θ)
        attn_out = attn_out * gate_score                     # [B, 1, D]

        # ====== 后续仍然是 Transformer Encoder 风格 ======
        # 行：第一层残差 + LayerNorm
        y = self.ln1(attn_out + q)                           # [B, 1, D]

        # 行：前馈网络
        y2 = self.ff(y)                                      # [B, 1, D]

        # 行：第二层残差 + LayerNorm
        y = self.ln2(y + y2)                                 # [B, 1, D]

        # 行：去掉长度维度，返回 [B, D]
        return y.squeeze(1)



# ====== 总模型：1D(LM) + 2D(GCN) + 3D(Uni-Mol2) + Cross-Attn → 预测 pKd ======
class MultiModalDTA_LM(nn.Module):
    """
    多模态 DTA 模型（LM 版 + 可选模态）：

      - 1D：预训练 LM 特征（如 ChemBERTa-2 / ESM-2）
            drug_lm: [B, D_drug_lm] 或 None
            prot_lm: [B, D_prot_lm] 或 None
        → 若两者都不为 None，则得到 z_1d ∈ R^{B×d_model}

      - 2D：EW-GCN（药物图/蛋白图），若 g_lig/g_prot 不为 None，则得到 z_2d ∈ R^{B×d_model}

      - 3D：Uni-Mol2（药物/口袋），若 lig3d/poc3d 不为 None，则得到 z_3d ∈ R^{B×d_model}

      - 路径选择：
        * 若 has_3d 且 (has_1d 或 has_2d)：
            - Q = concat(z_1d, z_2d)，缺失模态用 0 向量补齐，维度恒为 2*d_model
            - KV = z_3d
            - CrossAttention → ca ∈ R^{B×d_attn}
            - reg_head_ca(ca) → 标量预测

        * 否则（无 3D，或只有 3D）：
            - z1 = z_1d 或 0；z2 = z_2d 或 0；z3 = z_3d 或 0
            - fuse_modalities_no_ca([z1||z2||z3]) → fused ∈ R^{B×d_model}
            - reg_head_no_ca(fused) → 标量预测
    """

    def __init__(
        self,
        # 1D LM 特征维度
        d_drug_lm: int,
        d_prot_lm: int,
        # 2D 输入维（与你数据对齐：药物节点70/边6；蛋白节点33/边3）
        drug_node_dim: int = 70,
        drug_edge_dim: int = 6,  # 目前仅用于说明，不直接参与计算
        prot_node_dim: int = 33,
        prot_edge_dim: int = 3,  # 同上
        gcn_hidden: int = 128,
        gcn_out: int = 128,
        # 3D 输入维（Uni-Mol2）
        d3_lig: int = 512,
        d3_poc: int = 512,
        # 统一融合维与注意力
        d_model: int = 256,
        d_attn: int = 256,
        n_heads: int = 4,
        add_interactions_3d: bool = True,
    ):
        super().__init__()

        # ---- 1D：预训练 LM 特征投影 ----
        self.proj_drug_lm = nn.Sequential(
            nn.LayerNorm(d_drug_lm),
            nn.Linear(d_drug_lm, d_model, bias=False),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.proj_prot_lm = nn.Sequential(
            nn.LayerNorm(d_prot_lm),
            nn.Linear(d_prot_lm, d_model, bias=False),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.fuse1d = nn.Sequential(
            nn.Linear(2 * d_model, d_model, bias=False),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # ---- 2D：两路 EW-GCN（药物图/蛋白图），图级读出向量 ----
        self.enc2d_drug = GraphConvEW(
            in_dim=drug_node_dim, hidden_dim=gcn_hidden, out_dim=gcn_out
        )
        self.enc2d_prot = GraphConvEW(
            in_dim=prot_node_dim, hidden_dim=gcn_hidden, out_dim=gcn_out
        )
        self.fuse2d = nn.Sequential(
            nn.Linear(2 * gcn_out, d_model, bias=False),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # ---- 3D：pair 编码（药物/口袋 → d_model） ----
        self.enc3d_pair = Pair3DEncoder(
            d_in_lig=d3_lig,
            d_in_poc=d3_poc,
            d_model=d_model,
            add_interactions=add_interactions_3d,
        )

        # ---- Cross-Attention：Q=cat(1D, 2D)，KV=3D ----
        self.cross = CrossAttentionFuse(
            dim_q=2 * d_model, dim_kv=d_model, d_attn=d_attn, n_heads=n_heads
        )

        # ---- 回归头（带 Cross-Attn 情况） ----
        self.reg_head_ca = nn.Sequential(
            nn.Linear(d_attn, 2 * d_model, bias=False),
            nn.LayerNorm(2 * d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * d_model, 1, bias=True),
        )

        # ---- 模态融合 + 回归头（无 Cross-Attn 情况：无 3D 或只有 3D） ----
        # 先把 (z1,z2,z3) 拼成 [z1||z2||z3]（缺失模态用 0），维度 3*d_model
        self.fuse_modalities_no_ca = nn.Sequential(
            nn.Linear(3 * d_model, d_model, bias=False),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.reg_head_no_ca = nn.Sequential(
            nn.Linear(d_model, 2 * d_model, bias=False),
            nn.LayerNorm(2 * d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * d_model, 1, bias=True),
        )

    def forward(
        self,
        drug_lm: Optional[torch.Tensor] = None,  # [B, D_drug_lm] 或 None
        prot_lm: Optional[torch.Tensor] = None,  # [B, D_prot_lm] 或 None
        g_lig: Optional[dgl.DGLGraph] = None,
        g_prot: Optional[dgl.DGLGraph] = None,
        lig3d: Optional[torch.Tensor] = None,  # [B, Dl] 或 None
        poc3d: Optional[torch.Tensor] = None,  # [B, Dp] 或 None
    ) -> torch.Tensor:
        device = next(self.parameters()).device

        # 判定哪些模态存在
        has_1d = (drug_lm is not None) and (prot_lm is not None)
        has_2d = (g_lig is not None) and (g_prot is not None)
        has_3d = (lig3d is not None) and (poc3d is not None)

        if not (has_1d or has_2d or has_3d):
            raise ValueError("至少需要提供 1D/2D/3D 中的一种模态输入。")

        z1d = z2d = z3d = None  # 各模态的 d_model 级向量

        # === 1) 1D：LM 特征分支 ===
        if has_1d:
            drug_lm = drug_lm.to(device)
            prot_lm = prot_lm.to(device)
            drug_1d = self.proj_drug_lm(drug_lm)  # [B, d_model]
            prot_1d = self.proj_prot_lm(prot_lm)  # [B, d_model]
            pair_1d = torch.cat([drug_1d, prot_1d], dim=-1)  # [B, 2*d_model]
            z1d = self.fuse1d(pair_1d)  # [B, d_model]

        # === 2) 2D：两路图编码并拼接 ===
        if has_2d:
            lig_vec = self.enc2d_drug(
                g_lig, g_lig.ndata["x"].float(), g_lig.edata["w"].float()
            )  # [B, gcn_out]
            prot_vec = self.enc2d_prot(
                g_prot, g_prot.ndata["x"].float(), g_prot.edata["w"].float()
            )  # [B, gcn_out]
            pair_2d = torch.cat([lig_vec, prot_vec], dim=-1)  # [B, 2*gcn_out]
            z2d = self.fuse2d(pair_2d)  # [B, d_model]

        # === 3) 3D：药物/口袋 → pair 编码 ===
        if has_3d:
            lig3d = torch.nan_to_num(
                lig3d.to(device), nan=0.0, posinf=1e6, neginf=-1e6
            )
            poc3d = torch.nan_to_num(
                poc3d.to(device), nan=0.0, posinf=1e6, neginf=-1e6
            )
            z3d = self.enc3d_pair(lig3d, poc3d)  # [B, d_model]

        # === 统一 batch 大小检查（如果多模态同时出现） ===
        B = None
        for z in (z1d, z2d, z3d):
            if z is not None:
                if B is None:
                    B = z.size(0)
                else:
                    assert (
                        z.size(0) == B
                    ), f"batch size 不一致：期望 {B}, 但有 {z.size(0)}"

        # === 分支 A：有 3D 且存在 1D 或 2D → Cross-Attn 路径 ===
        if has_3d and (z1d is not None or z2d is not None):
            # 构造 Q=[z1d||z2d]，缺失模态用 0 向量补齐，维度恒为 2*d_model
            if z1d is not None and z2d is not None:
                q_vec = torch.cat([z1d, z2d], dim=-1)  # [B, 2*d_model]
            elif z1d is not None:
                zero = torch.zeros_like(z1d)
                q_vec = torch.cat([z1d, zero], dim=-1)
            else:  # 只有 2D
                zero = torch.zeros_like(z2d)
                q_vec = torch.cat([zero, z2d], dim=-1)

            ca = self.cross(q_vec, z3d)  # [B, d_attn]
            y = self.reg_head_ca(ca).squeeze(-1)  # [B]
            return y

        # === 分支 B：无 3D，或只有 3D → 直接 MLP 路径 ===
        # 这里把 z1/z2/z3 拼成 [z1||z2||z3]，缺失模态用 0 向量补齐
        #   * 无 3D：z3=0
        #   * 只有 3D：z1=z2=0
        base_vec = z1d if z1d is not None else (z2d if z2d is not None else z3d)
        assert base_vec is not None  # 前面已经保证至少有一个模态

        zero = torch.zeros_like(base_vec)
        z1 = z1d if z1d is not None else zero
        z2 = z2d if z2d is not None else zero
        z3 = z3d if z3d is not None else zero

        fused = self.fuse_modalities_no_ca(torch.cat([z1, z2, z3], dim=-1))  # [B,d_model]
        y = self.reg_head_no_ca(fused).squeeze(-1)  # [B]
        return y
