# -*- coding: utf-8 -*-                                   # 行：声明源码文件使用 UTF-8 编码，防止中文注释乱码
# =========================================================
# 多模态 PPI / 抗原-抗体 模型（1D: 预训练 LM 特征 + 2D 图 + 3D 向量 + Cross-Attention）
#
# 设计与 MultiModalDTA_LM 完全同构，只是把：
#   - drug / prot → prot1 / prot2（可以理解为：链 A / 链 B，或 VHH / 抗原）
#
# 支持模态：
#   - 1D：预训练 LM 特征（ESM-2 等）
#       prot1_lm: [B, D_prot1_lm] 或 None
#       prot2_lm: [B, D_prot2_lm] 或 None
#     → 若两者都不为 None，则得到 z_1d ∈ R^{B×d_model}
#
#   - 2D：EW-GCN 图分支（蛋白图1 / 蛋白图2）
#       g_prot1, g_prot2 不为 None 则得到 z_2d ∈ R^{B×d_model}
#
#   - 3D：Pair3DEncoder（蛋白1 / 蛋白2 三维向量）
#       prot13d, prot23d 不为 None 则得到 z_3d ∈ R^{B×d_model}
#
#   - 路径选择逻辑与 DTA 版完全一致：
#       * 若 has_3d 且 (has_1d 或 has_2d)：
#             Q = concat(z_1d, z_2d)（缺失模态用 0 补齐，维度恒为 2*d_model）
#             K,V = z_3d
#             CrossAttention → ca ∈ R^{B×d_attn}
#             reg_head_ca(ca) → 标量（可用于回归或 BCEWithLogits）
#
#       * 否则（无 3D，或只有 3D）：
#             z1 = z_1d 或 0；z2 = z_2d 或 0；z3 = z_3d 或 0
#             [z1||z2||z3] → fuse_modalities_no_ca → reg_head_no_ca → 标量
#
#   - 任务类型：
#       * 若是回归（如 binding score），用 MSELoss 即可；
#       * 若是分类（如 binder / non-binder），可以用 BCEWithLogitsLoss；
#         直接对输出 logit 做 sigmoid 即可得到概率。
# =========================================================

from typing import Optional                         # 行：用于类型注解 Optional

import torch                                        # 行：导入 PyTorch 主包
import torch.nn as nn                               # 行：导入神经网络模块
import dgl                                          # 行：导入 DGL（用于图神经网络）


# ======【2D】带边权的 GCN（EW-GCN），用于两条蛋白链的图表示 ======
class GraphConvEW(nn.Module):                       # 行：定义一个带边权的 GCN 模块
    """三层 GraphConv（DGL），带边权；显式分段读出，保证输出 [B, out_dim]。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()                          # 行：调用父类构造函数
        self.gcn1 = dgl.nn.GraphConv(               # 行：第一层 GraphConv
            in_dim, hidden_dim, bias=False, allow_zero_in_degree=True
        )
        self.gcn2 = dgl.nn.GraphConv(               # 行：第二层 GraphConv
            hidden_dim, hidden_dim, bias=False, allow_zero_in_degree=True
        )
        self.gcn3 = dgl.nn.GraphConv(               # 行：第三层 GraphConv
            hidden_dim, out_dim, bias=False, allow_zero_in_degree=True
        )
        self.ln1 = nn.LayerNorm(hidden_dim)         # 行：第一层后接 LayerNorm
        self.ln2 = nn.LayerNorm(hidden_dim)         # 行：第二层后接 LayerNorm
        self.act = nn.ReLU(inplace=True)            # 行：激活函数 ReLU
        self.dropout = nn.Dropout(0.1)              # 行：Dropout 防止过拟合

    @staticmethod
    def _edge_feat_to_scalar(
        w: Optional[torch.Tensor], g: dgl.DGLGraph
    ) -> Optional[torch.Tensor]:
        """把 [E,Fe]/[E] 边特征压成 [E] 标量，供 GraphConv 的 edge_weight 使用。"""
        if (w is None) or (w.numel() == 0):         # 行：没有边特征直接返回 None
            return None
        if w.dim() == 2:                            # 行：若是 [E, Fe]，对特征维求均值 → [E]
            w = w.mean(dim=-1)
        else:
            w = w.view(-1)                          # 行：保证是一维 [E]
        if w.shape[0] > g.num_edges():              # 行：长度过长则截断到边数
            w = w[: g.num_edges()]
        return w                                    # 行：返回标量边权 [E]

    @staticmethod
    def _segmented_sum_by_graph(
        g: dgl.DGLGraph, node_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        显式按子图分段求和（不依赖 dgl.sum_nodes 的批语义）。
        返回 [B, D]，B 为批内子图数（即 batch size）。
        """
        counts_fn = getattr(g, "batch_num_nodes", None)  # 行：尝试获取每个子图节点数接口
        if callable(counts_fn):                          # 行：如果存在 batch_num_nodes
            counts = counts_fn()                         # 行：得到每个图的节点数 list / tensor
            if not torch.is_tensor(counts):              # 行：非 tensor 时转换成 tensor
                counts = torch.as_tensor(counts, device=node_feat.device)
            else:
                counts = counts.to(node_feat.device)
            if counts.dim() == 0:                        # 行：只有一个子图时变成 [1]
                counts = counts.view(1)
            offsets = torch.zeros(                       # 行：构造前缀和索引
                (counts.numel() + 1,), dtype=torch.long, device=node_feat.device
            )
            offsets[1:] = torch.cumsum(counts, dim=0)    # 行：前缀和得到每段起止位置
            outs = []                                    # 行：用于收集每个子图的读出
            for i in range(counts.numel()):              # 行：遍历每个子图
                seg = node_feat[offsets[i] : offsets[i + 1]]  # 行：取该子图的节点特征
                outs.append(seg.sum(dim=0, keepdim=True))     # 行：按节点求和 → [1, D]
            return torch.cat(outs, dim=0)                # 行：拼成 [B, D]

        # 兜底：用 unbatch（略慢但通用）
        graphs = dgl.unbatch(g)                          # 行：拆成单图 list
        outs = []                                        # 行：保存每个单图读出
        start = 0                                        # 行：起始节点索引
        for gi in graphs:                                # 行：遍历每个单图
            n = gi.num_nodes()                           # 行：该图节点数
            outs.append(node_feat[start : start + n].sum(dim=0, keepdim=True))
            start += n                                   # 行：移动到下一个图的起点
        return torch.cat(outs, dim=0)                    # 行：返回 [B, D]

    def forward(
        self, g: dgl.DGLGraph, x: torch.Tensor, w: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """输入批图 g，节点特征 x、边特征 w；输出每图读出向量 [B, out_dim]。"""
        device = next(self.parameters()).device          # 行：取得当前模块所在设备

        # 边权降到标量
        w_scalar = self._edge_feat_to_scalar(w, g)       # 行：边特征转成 [E] 的标量边权

        # 统一搬到同一设备
        g = g.to(device)                                 # 行：图搬到 device
        x = x.to(device)                                 # 行：节点特征搬到 device
        if w_scalar is not None:                         # 行：如有边权也搬到 device
            w_scalar = w_scalar.to(device)

        # 三层 GraphConv
        h = self.gcn1(g, x, edge_weight=w_scalar)        # 行：第一层 GCN
        h = self.ln1(self.act(h))                        # 行：ReLU + LayerNorm
        h = self.dropout(h)                              # 行：Dropout

        h = self.gcn2(g, h, edge_weight=w_scalar)        # 行：第二层 GCN
        h = self.ln2(self.act(h))                        # 行：ReLU + LayerNorm
        h = self.dropout(h)                              # 行：Dropout

        h = self.gcn3(g, h, edge_weight=w_scalar)        # 行：第三层 GCN（输出维 out_dim）

        # 图级读出
        readout = self._segmented_sum_by_graph(g, h)     # 行：按子图对节点求和 → [B, out_dim]
        return readout                                   # 行：返回图级向量


# ======【3D】蛋白对 3D 向量编码器（拼接 + 可选交互 → 降维到 d_model） ======
class Pair3DEncoder(nn.Module):                         # 行：定义 3D 对编码器
    """把 prot13d/prot23d 两路向量拼成一个“3D 对”特征，并降维到统一 d_model。"""

    def __init__(
        self,
        d_in_prot1: int = 512,                           # 行：蛋白1 3D 向量维度
        d_in_prot2: int = 512,                           # 行：蛋白2 3D 向量维度
        d_model: int = 256,                              # 行：输出统一维度
        add_interactions: bool = True,                   # 行：是否加入乘积/差值交互特征
    ):
        super().__init__()                               # 行：父类构造
        self.add_interactions = add_interactions         # 行：保存配置
        self.norm_1 = nn.LayerNorm(d_in_prot1)           # 行：蛋白1 输入 LayerNorm
        self.norm_2 = nn.LayerNorm(d_in_prot2)           # 行：蛋白2 输入 LayerNorm
        self.proj_1 = nn.Linear(d_in_prot1, d_model, bias=False)  # 行：蛋白1 投影到 d_model
        self.proj_2 = nn.Linear(d_in_prot2, d_model, bias=False)  # 行：蛋白2 投影到 d_model
        in_dim = d_model * (4 if add_interactions else 2)         # 行：拼接后总维度
        self.mlp = nn.Sequential(                        # 行：后续 MLP 压到 d_model
            nn.Linear(in_dim, d_model, bias=False),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, prot13d: torch.Tensor, prot23d: torch.Tensor) -> torch.Tensor:
        """输入 [B,D1] 与 [B,D2]；输出 [B,d_model] 的 3D 对表示"""
        p1 = self.norm_1(prot13d.float())                # 行：蛋白1 3D 向量归一化
        p2 = self.norm_2(prot23d.float())                # 行：蛋白2 3D 向量归一化
        p1 = self.proj_1(p1)                             # 行：蛋白1 映射到 d_model
        p2 = self.proj_2(p2)                             # 行：蛋白2 映射到 d_model
        if self.add_interactions:                        # 行：若使用交互特征
            feat = torch.cat(
                [p1, p2, p1 * p2, torch.abs(p1 - p2)], dim=-1
            )                                            # 行：拼接自身+乘积+差值 → [B,4*d_model]
        else:
            feat = torch.cat([p1, p2], dim=-1)           # 行：只拼接两个向量 → [B,2*d_model]
        out = self.mlp(feat)                             # 行：MLP 压到 d_model
        return out                                       # 行：返回 [B,d_model]


# ====== Cross-Attention（把 1D+2D 的融合向量作为 Q，3D 对向量作为 K/V） ======
class CrossAttentionFuse(nn.Module):                     # 行：定义交叉注意力融合模块
    """标准多头交叉注意力 + 前馈，两层残差（Transformer Encoder 风格）。"""

    def __init__(
        self,
        dim_q: int,                                      # 行：Q 向量维度（2*d_model）
        dim_kv: int,                                     # 行：K/V 向量维度（d_model）
        d_attn: int = 256,                               # 行：内部注意力维度
        n_heads: int = 4,                                # 行：注意力头数
        dropout: float = 0.1,                            # 行：Dropout 概率
    ):
        super().__init__()                               # 行：父类构造
        self.q_proj = nn.Linear(dim_q, d_attn, bias=False)  # 行：Q 线性投影到 d_attn
        self.k_proj = nn.Linear(dim_kv, d_attn, bias=False) # 行：K 线性投影到 d_attn
        self.v_proj = nn.Linear(dim_kv, d_attn, bias=False) # 行：V 线性投影到 d_attn
        self.mha = nn.MultiheadAttention(                # 行：PyTorch 自带多头注意力
            embed_dim=d_attn, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(d_attn)                  # 行：第一层残差后的 LayerNorm
        self.ff = nn.Sequential(                         # 行：前馈网络（FeedForward）
            nn.Linear(d_attn, 2 * d_attn, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_attn, d_attn, bias=False),
        )
        self.ln2 = nn.LayerNorm(d_attn)                  # 行：第二层残差后的 LayerNorm

    def forward(self, q_vec: torch.Tensor, kv_vec: torch.Tensor) -> torch.Tensor:
        """
        q_vec:  [B, Dq]（1D+2D 拼接后向量）
        kv_vec: [B, Dk]（3D 对向量）
        return: [B, d_attn]（融合后的表示）
        """
        q = self.q_proj(q_vec).unsqueeze(1)              # 行：Q 投影并加一个序列维 → [B,1,D]
        k = self.k_proj(kv_vec).unsqueeze(1)             # 行：K 同理 → [B,1,D]
        v = self.v_proj(kv_vec).unsqueeze(1)             # 行：V 同理 → [B,1,D]
        attn_out, _ = self.mha(q, k, v, need_weights=False)  # 行：多头注意力输出 [B,1,D]
        y = self.ln1(attn_out + q)                       # 行：残差 + LayerNorm（第1层）
        y2 = self.ff(y)                                  # 行：前馈网络
        y = self.ln2(y + y2)                             # 行：再次残差 + LayerNorm
        return y.squeeze(1)                              # 行：去掉长度为1的序列维 → [B,D]


# ====== 总模型：1D(LM) + 2D(GCN) + 3D(向量) + Cross-Attn → 预测 PPI 分数 ======
class MultiModalPPI_LM(nn.Module):                       # 行：定义多模态 PPI 模型主体
    """
    多模态 PPI / 抗原-抗体 模型（LM 版 + 可选模态）：

      - 1D：预训练 LM 特征（多为 ESM-2）
            prot1_lm: [B, D_prot1_lm] 或 None
            prot2_lm: [B, D_prot2_lm] 或 None
        → 若两者都不为 None，则得到 z_1d ∈ R^{B×d_model}

      - 2D：EW-GCN（蛋白图1/蛋白图2），若 g_prot1/g_prot2 不为 None，则得到 z_2d ∈ R^{B×d_model}

      - 3D：蛋白 3D 对（prot13d / prot23d），若都不为 None，则得到 z_3d ∈ R^{B×d_model}

      - 路径选择：
        * 若 has_3d 且 (has_1d 或 has_2d)：
            - Q = concat(z_1d, z_2d)，缺失模态用 0 向量补齐
            - KV = z_3d
            - CrossAttention → ca ∈ R^{B×d_attn}
            - reg_head_ca(ca) → 标量预测（可接 MSE 或 BCEWithLogits）

        * 否则（无 3D，或只有 3D）：
            - z1 = z_1d 或 0；z2 = z_2d 或 0；z3 = z_3d 或 0
            - [z1||z2||z3] → fuse_modalities_no_ca → reg_head_no_ca → 标量预测
    """

    def __init__(
        self,
        # 1D LM 特征维度
        d_prot1_lm: int,                                # 行：蛋白1 LM 维度
        d_prot2_lm: int,                                # 行：蛋白2 LM 维度
        # 2D 输入维（两条链的图特征维度，可设成相同）
        prot1_node_dim: int = 33,                       # 行：蛋白1 图节点维度
        prot1_edge_dim: int = 3,                        # 行：蛋白1 图边维度（目前仅用来说明）
        prot2_node_dim: int = 33,                       # 行：蛋白2 图节点维度
        prot2_edge_dim: int = 3,                        # 行：蛋白2 图边维度
        gcn_hidden: int = 128,                          # 行：GCN 中间隐藏维度
        gcn_out: int = 128,                             # 行：GCN 输出维度
        # 3D 输入维
        d3_prot1: int = 512,                            # 行：蛋白1 三维向量维度
        d3_prot2: int = 512,                            # 行：蛋白2 三维向量维度
        # 统一融合维与注意力相关
        d_model: int = 256,                             # 行：模态统一特征维度
        d_attn: int = 256,                              # 行：Cross-Attn 内部维度
        n_heads: int = 4,                               # 行：多头注意力头数
        add_interactions_3d: bool = True,               # 行：是否在 3D 对里加入交互特征
    ):
        super().__init__()                              # 行：父类构造

        # ---- 1D：预训练 LM 特征投影（prot1/prot2） ----
        self.proj_prot1_lm = nn.Sequential(             # 行：蛋白1 LM 映射到 d_model
            nn.LayerNorm(d_prot1_lm),
            nn.Linear(d_prot1_lm, d_model, bias=False),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.proj_prot2_lm = nn.Sequential(             # 行：蛋白2 LM 映射到 d_model
            nn.LayerNorm(d_prot2_lm),
            nn.Linear(d_prot2_lm, d_model, bias=False),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.fuse1d = nn.Sequential(                    # 行：把 [prot1, prot2] 拼接后再压到 d_model
            nn.Linear(2 * d_model, d_model, bias=False),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # ---- 2D：两路 EW-GCN（蛋白1/蛋白2），图级读出向量 ----
        self.enc2d_prot1 = GraphConvEW(                 # 行：蛋白1 图编码器
            in_dim=prot1_node_dim, hidden_dim=gcn_hidden, out_dim=gcn_out
        )
        self.enc2d_prot2 = GraphConvEW(                 # 行：蛋白2 图编码器
            in_dim=prot2_node_dim, hidden_dim=gcn_hidden, out_dim=gcn_out
        )
        self.fuse2d = nn.Sequential(                    # 行：把两个图向量拼接后映射到 d_model
            nn.Linear(2 * gcn_out, d_model, bias=False),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # ---- 3D：蛋白对编码器 ----
        self.enc3d_pair = Pair3DEncoder(                # 行：3D 对编码成 d_model
            d_in_prot1=d3_prot1,
            d_in_prot2=d3_prot2,
            d_model=d_model,
            add_interactions=add_interactions_3d,
        )

        # ---- Cross-Attention：Q=cat(1D, 2D)，KV=3D ----
        self.cross = CrossAttentionFuse(                # 行：交叉注意力模块
            dim_q=2 * d_model, dim_kv=d_model, d_attn=d_attn, n_heads=n_heads
        )

        # ---- 回归/分类头（有 Cross-Attn 情况） ----
        self.reg_head_ca = nn.Sequential(               # 行：针对 Cross-Attn 输出的头部
            nn.Linear(d_attn, 2 * d_model, bias=False),
            nn.LayerNorm(2 * d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * d_model, 1, bias=True),
        )

        # ---- 模态融合 + 回归头（无 Cross-Attn 情况：无 3D 或只有 3D） ----
        #   输入拼接为 [z1||z2||z3]，维度 3*d_model
        self.fuse_modalities_no_ca = nn.Sequential(     # 行：三模态直连融合层
            nn.Linear(3 * d_model, d_model, bias=False),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.reg_head_no_ca = nn.Sequential(            # 行：无 Cross-Attn 时的头部
            nn.Linear(d_model, 2 * d_model, bias=False),
            nn.LayerNorm(2 * d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * d_model, 1, bias=True),
        )

    def forward(
        self,
        prot1_lm: Optional[torch.Tensor] = None,        # 行：蛋白1 LM 特征 [B,D_prot1_lm] 或 None
        prot2_lm: Optional[torch.Tensor] = None,        # 行：蛋白2 LM 特征 [B,D_prot2_lm] 或 None
        g_prot1: Optional[dgl.DGLGraph] = None,         # 行：蛋白1 图
        g_prot2: Optional[dgl.DGLGraph] = None,         # 行：蛋白2 图
        prot13d: Optional[torch.Tensor] = None,         # 行：蛋白1 3D 向量 [B,D1] 或 None
        prot23d: Optional[torch.Tensor] = None,         # 行：蛋白2 3D 向量 [B,D2] 或 None
    ) -> torch.Tensor:
        device = next(self.parameters()).device         # 行：取得模型所在设备

        # 判定哪些模态存在
        has_1d = (prot1_lm is not None) and (prot2_lm is not None)   # 行：1D 模态是否存在
        has_2d = (g_prot1 is not None) and (g_prot2 is not None)     # 行：2D 模态是否存在
        has_3d = (prot13d is not None) and (prot23d is not None)     # 行：3D 模态是否存在

        if not (has_1d or has_2d or has_3d):            # 行：三种模态都没给就抛异常
            raise ValueError("至少需要提供 1D/2D/3D 中的一种模态输入。")

        z1d = z2d = z3d = None                          # 行：分别存储 1D/2D/3D 的 d_model 级向量

        # === 1) 1D：LM 特征分支 ===
        if has_1d:                                      # 行：若提供了 prot1_lm & prot2_lm
            prot1_lm = prot1_lm.to(device)              # 行：搬到 device
            prot2_lm = prot2_lm.to(device)
            p1 = self.proj_prot1_lm(prot1_lm)           # 行：蛋白1 LM → d_model
            p2 = self.proj_prot2_lm(prot2_lm)           # 行：蛋白2 LM → d_model
            pair_1d = torch.cat([p1, p2], dim=-1)       # 行：拼接得到 [B,2*d_model]
            z1d = self.fuse1d(pair_1d)                  # 行：fuse1d 压缩成 [B,d_model]

        # === 2) 2D：两路图编码并拼接 ===
        if has_2d:                                      # 行：若提供了两条链的图
            prot1_vec = self.enc2d_prot1(               # 行：蛋白1 图编码 → [B,gcn_out]
                g_prot1,
                g_prot1.ndata["x"].float(),
                g_prot1.edata["w"].float(),
            )
            prot2_vec = self.enc2d_prot2(               # 行：蛋白2 图编码 → [B,gcn_out]
                g_prot2,
                g_prot2.ndata["x"].float(),
                g_prot2.edata["w"].float(),
            )
            pair_2d = torch.cat([prot1_vec, prot2_vec], dim=-1)  # 行：拼接两个图向量
            z2d = self.fuse2d(pair_2d)                  # 行：2D 融合后得到 [B,d_model]

        # === 3) 3D：蛋白3D对编码 ===
        if has_3d:                                      # 行：若提供了两条链的 3D 向量
            prot13d = torch.nan_to_num(                 # 行：将 NaN/Inf 替换为合理数值
                prot13d.to(device), nan=0.0, posinf=1e6, neginf=-1e6
            )
            prot23d = torch.nan_to_num(
                prot23d.to(device), nan=0.0, posinf=1e6, neginf=-1e6
            )
            z3d = self.enc3d_pair(prot13d, prot23d)     # 行：编码成 [B,d_model]

        # === 统一 batch 大小检查（如果多模态同时出现） ===
        B = None                                        # 行：记录 batch size
        for z in (z1d, z2d, z3d):                       # 行：遍历已有模态
            if z is not None:
                if B is None:
                    B = z.size(0)                       # 行：第一份非 None 设为基准 batch
                else:
                    assert z.size(0) == B,              \
                        f"batch size 不一致：期望 {B}, 但有 {z.size(0)}"

        # === 分支 A：有 3D 且存在 1D 或 2D → Cross-Attn 路径 ===
        if has_3d and (z1d is not None or z2d is not None):
            # 构造 Q=[z1d||z2d]，缺失模态用 0 向量补齐，维度恒为 2*d_model
            if z1d is not None and z2d is not None:
                q_vec = torch.cat([z1d, z2d], dim=-1)   # 行：都有时直接拼接
            elif z1d is not None:
                zero = torch.zeros_like(z1d)            # 行：只有 1D，用 0 向量代替 z2d
                q_vec = torch.cat([z1d, zero], dim=-1)
            else:  # 只有 2D
                zero = torch.zeros_like(z2d)            # 行：只有 2D，用 0 向量代替 z1d
                q_vec = torch.cat([zero, z2d], dim=-1)

            ca = self.cross(q_vec, z3d)                 # 行：Cross-Attn 融合 → [B,d_attn]
            y = self.reg_head_ca(ca).squeeze(-1)        # 行：全连接头 → [B] 标量
            return y                                    # 行：返回输出（回归 / 分类 logits）

        # === 分支 B：无 3D，或只有 3D → 直接 MLP 路径 ===
        # 这里把 z1 / z2 / z3 拼成 [z1||z2||z3]，缺失模态用 0 向量补齐
        base_vec = z1d if z1d is not None else (z2d if z2d is not None else z3d)
        assert base_vec is not None                     # 行：一定至少存在一个模态

        zero = torch.zeros_like(base_vec)               # 行：构造全零向量
        z1 = z1d if z1d is not None else zero           # 行：z1 若缺失用 0 补
        z2 = z2d if z2d is not None else zero           # 行：z2 若缺失用 0 补
        z3 = z3d if z3d is not None else zero           # 行：z3 若缺失用 0 补

        fused = self.fuse_modalities_no_ca(             # 行：拼接后压缩到 d_model
            torch.cat([z1, z2, z3], dim=-1)
        )
        y = self.reg_head_no_ca(fused).squeeze(-1)      # 行：通过头部得到 [B] 标量
        return y                                        # 行：返回输出
