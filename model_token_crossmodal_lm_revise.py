# -*- coding: utf-8 -*-
# =========================================================
# Token-level MultiModalDTA 模型
#
# 目的：
#   解决关于 “Fake Cross-Attention” 和 “简单 sigmoid gate 缺乏创新性” 的质疑。
#
# 核心变化：
#   旧版：
#       drug_lm: [B, D]
#       prot_lm: [B, D]
#       lig3d:   [B, D]
#       poc3d:   [B, D]
#       然后用 GatedCrossModalFuse 做全局向量融合。
#
#   新版：
#       drug_lm_tokens: [B, Ld, D_drug]
#       prot_lm_tokens: [B, Lp, D_prot]
#       lig3d_tokens:   [B, La, D_lig3d]
#       poc3d_tokens:   [B, Lc, D_poc3d]
#
#       sequence_tokens = [drug_lm_tokens, prot_lm_tokens]
#       structure_tokens = [lig3d_tokens, poc3d_tokens]
#
#       使用 nn.MultiheadAttention 做真正的多 token cross-attention：
#           sequence_tokens -> attend to -> structure_tokens
#           structure_tokens -> attend to -> sequence_tokens
#
# 注意：
#   1. 这个模型不再使用 GatedCrossModalFuse。
#   2. 输入必须是 token-level embedding，不要再传 [B, D] pooled 向量。
#   3. mask 中 True 表示 padding / special token，需要在 attention 中忽略。
# =========================================================

from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn

try:
    import dgl
except ImportError:
    dgl = None


# =========================================================
# 工具函数：修复 padding mask
# =========================================================
def _fix_all_padding_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    修复全 padding 的样本。

    PyTorch MultiheadAttention 中：
        key_padding_mask=True 表示该 token 被忽略。
    如果某个样本所有 token 都是 True，会导致 attention 全部被 mask，可能产生 NaN。

    所以如果发现某个样本全是 padding，就强制保留第 0 个 token。
    """
    if mask is None:
        return None

    mask = mask.bool()

    if mask.dim() != 2:
        raise ValueError(f"mask 必须是 [B, L]，但收到 shape={tuple(mask.shape)}")

    all_pad = mask.all(dim=1)

    if all_pad.any():
        mask = mask.clone()
        mask[all_pad, 0] = False

    return mask


# =========================================================
# 2D 图分支：保留你原来的 EW-GCN，方便后续扩展
# =========================================================
class GraphConvEW(nn.Module):
    """
    三层 GraphConv，带边权。

    说明：
        你目前 2D 图模态关闭，g_lig/g_prot 通常是 None。
        这里保留这个类，是为了兼容你旧模型结构。
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()

        if dgl is None:
            raise ImportError("当前环境没有安装 dgl，但 GraphConvEW 需要 dgl。")

        self.gcn1 = dgl.nn.GraphConv(
            in_dim,
            hidden_dim,
            bias=False,
            allow_zero_in_degree=True
        )

        self.gcn2 = dgl.nn.GraphConv(
            hidden_dim,
            hidden_dim,
            bias=False,
            allow_zero_in_degree=True
        )

        self.gcn3 = dgl.nn.GraphConv(
            hidden_dim,
            out_dim,
            bias=False,
            allow_zero_in_degree=True
        )

        self.ln1 = nn.LayerNorm(hidden_dim)

        self.ln2 = nn.LayerNorm(hidden_dim)

        self.act = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(0.1)

    @staticmethod
    def _edge_feat_to_scalar(
        w: Optional[torch.Tensor],
        g: Any
    ) -> Optional[torch.Tensor]:
        """
        把边特征压成 GraphConv 可用的边权。

        输入：
            w: [E, Fe] 或 [E]
            g: DGLGraph

        输出：
            w_scalar: [E]
        """
        if (w is None) or (w.numel() == 0):
            return None

        if w.dim() == 2:
            w = w.mean(dim=-1)
        else:
            w = w.view(-1)

        if w.shape[0] > g.num_edges():
            w = w[: g.num_edges()]

        return w

    @staticmethod
    def _segmented_sum_by_graph(
        g: Any,
        node_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        按 batch 中每个子图分别求和，得到图级表示。

        输入：
            node_feat: [sum_nodes, D]

        输出：
            readout: [B, D]
        """
        counts_fn = getattr(g, "batch_num_nodes", None)

        if callable(counts_fn):
            counts = counts_fn()

            if not torch.is_tensor(counts):
                counts = torch.as_tensor(counts, device=node_feat.device)
            else:
                counts = counts.to(node_feat.device)

            if counts.dim() == 0:
                counts = counts.view(1)

            offsets = torch.zeros(
                (counts.numel() + 1,),
                dtype=torch.long,
                device=node_feat.device
            )

            offsets[1:] = torch.cumsum(counts, dim=0)

            outs = []

            for i in range(counts.numel()):
                seg = node_feat[offsets[i]: offsets[i + 1]]
                outs.append(seg.sum(dim=0, keepdim=True))

            return torch.cat(outs, dim=0)

        graphs = dgl.unbatch(g)

        outs = []

        start = 0

        for gi in graphs:
            n = gi.num_nodes()
            outs.append(node_feat[start: start + n].sum(dim=0, keepdim=True))
            start += n

        return torch.cat(outs, dim=0)

    def forward(
        self,
        g: Any,
        x: torch.Tensor,
        w: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        输入：
            g: DGL batch graph
            x: 节点特征
            w: 边特征

        输出：
            readout: [B, out_dim]
        """
        device = next(self.parameters()).device

        w_scalar = self._edge_feat_to_scalar(w, g)

        g = g.to(device)

        x = x.to(device)

        if w_scalar is not None:
            w_scalar = w_scalar.to(device)

        h = self.gcn1(g, x, edge_weight=w_scalar)

        h = self.ln1(self.act(h))

        h = self.dropout(h)

        h = self.gcn2(g, h, edge_weight=w_scalar)

        h = self.ln2(self.act(h))

        h = self.dropout(h)

        h = self.gcn3(g, h, edge_weight=w_scalar)

        readout = self._segmented_sum_by_graph(g, h)

        return readout


# =========================================================
# Token 投影层
# =========================================================
class TokenProjector(nn.Module):
    """
    把不同来源的 token embedding 投影到统一维度。

    例如：
        ChemBERTa token: [B, Ld, D_drug_lm]
        ESM-2 token:     [B, Lp, D_prot_lm]
        Uni-Mol2 token:  [B, La, D_3d]

    都投影成：
        [B, L, d_model]
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.norm = nn.LayerNorm(in_dim)

        self.proj = nn.Linear(in_dim, out_dim, bias=False)

        self.act = nn.GELU()

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        输入：
            x:    [B, L, in_dim]
            mask: [B, L]，True 表示 padding，需要忽略

        输出：
            x:    [B, L, out_dim]
        """
        if x is None:
            raise ValueError("TokenProjector 收到 None 输入。")

        if x.dim() != 3:
            raise ValueError(
                f"TokenProjector 需要 [B, L, D] token 输入，"
                f"但收到 shape={tuple(x.shape)}。"
                f"这通常说明你仍然传入了 pooled 向量 [B, D]。"
            )

        x = torch.nan_to_num(
            x.float(),
            nan=0.0,
            posinf=1e6,
            neginf=-1e6
        )

        x = self.norm(x)

        x = self.proj(x)

        x = self.act(x)

        x = self.dropout(x)

        if mask is not None:
            mask = mask.bool().to(x.device)
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)

        return x


# =========================================================
# 带 mask 的 attention pooling
# =========================================================
class MaskedAttentivePooling(nn.Module):
    """
    把 token 序列池化成一个全局向量。

    相比 mean pooling：
        attention pooling 可以学习哪些 token 更重要。

    输入：
        x:    [B, L, D]
        mask: [B, L]，True 表示 padding，需要忽略

    输出：
        out:  [B, D]
    """

    def __init__(self, d_model: int):
        super().__init__()

        self.score = nn.Linear(d_model, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        执行带 mask 的注意力池化。
        """
        if x.dim() != 3:
            raise ValueError(f"MaskedAttentivePooling 需要 [B, L, D]，收到 {tuple(x.shape)}")

        score = self.score(x).squeeze(-1)

        if mask is not None:
            mask = _fix_all_padding_mask(mask).to(x.device)
            score = score.masked_fill(mask, -1e4)

        weight = torch.softmax(score, dim=-1)

        if mask is not None:
            weight = weight.masked_fill(mask, 0.0)
            denom = weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            weight = weight / denom

        out = torch.sum(x * weight.unsqueeze(-1), dim=1)

        return out


# =========================================================
# 真正的 Cross-Attention Block
# =========================================================
class CrossAttentionBlock(nn.Module):
    """
    真正的多 token cross-attention 模块。

    关键点：
        q_tokens 不是 [B, 1, D]；
        kv_tokens 也不是 [B, 1, D]；
        而是多 token 序列。

    输入：
        q_tokens:  [B, Lq, Dq]
        kv_tokens: [B, Lk, Dkv]

    输出：
        out:       [B, Lq, d_model]
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        d_model: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} 必须能被 n_heads={n_heads} 整除。"
            )

        self.q_proj = nn.Linear(dim_q, d_model, bias=False)

        self.k_proj = nn.Linear(dim_kv, d_model, bias=False)

        self.v_proj = nn.Linear(dim_kv, d_model, bias=False)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ln1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model, bias=False)
        )

        self.ln2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q_tokens: torch.Tensor,
        kv_tokens: torch.Tensor,
        q_padding_mask: Optional[torch.Tensor] = None,
        kv_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        输入：
            q_tokens:        [B, Lq, Dq]
            kv_tokens:       [B, Lk, Dkv]
            q_padding_mask:  [B, Lq]，True 表示 q 中的 padding token
            kv_padding_mask: [B, Lk]，True 表示 kv 中的 padding token
            return_attn:     是否返回注意力权重

        输出：
            x:            [B, Lq, d_model]
            attn_weights: [B, n_heads, Lq, Lk] 或 None
        """
        if q_tokens.dim() != 3:
            raise ValueError(
                f"q_tokens 必须是 [B, Lq, Dq]，但收到 {tuple(q_tokens.shape)}"
            )

        if kv_tokens.dim() != 3:
            raise ValueError(
                f"kv_tokens 必须是 [B, Lk, Dkv]，但收到 {tuple(kv_tokens.shape)}"
            )

        q = self.q_proj(q_tokens)

        k = self.k_proj(kv_tokens)

        v = self.v_proj(kv_tokens)

        if q_padding_mask is not None:
            q_padding_mask = _fix_all_padding_mask(q_padding_mask).to(q.device)

        if kv_padding_mask is not None:
            kv_padding_mask = _fix_all_padding_mask(kv_padding_mask).to(q.device)

        attn_out, attn_weights = self.attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=kv_padding_mask,
            need_weights=return_attn,
            average_attn_weights=False
        )

        x = self.ln1(q + self.dropout(attn_out))

        if q_padding_mask is not None:
            x = x.masked_fill(q_padding_mask.unsqueeze(-1), 0.0)

        y = self.ffn(x)

        x = self.ln2(x + self.dropout(y))

        if q_padding_mask is not None:
            x = x.masked_fill(q_padding_mask.unsqueeze(-1), 0.0)

        if not return_attn:
            attn_weights = None

        return x, attn_weights


# =========================================================
# Token-level 多模态 DTA 主模型
# =========================================================
class TokenLevelMultiModalDTA(nn.Module):
    """
    真正的 token-level sequence-structure cross-attention DTA 模型。

    推荐输入：
        drug_lm_tokens: [B, Ld, D_drug_lm]
        prot_lm_tokens: [B, Lp, D_prot_lm]
        lig3d_tokens:   [B, La, D3_lig]
        poc3d_tokens:   [B, Lc, D3_poc]

    推荐 mask：
        drug_lm_mask: [B, Ld]
        prot_lm_mask: [B, Lp]
        lig3d_mask:   [B, La]
        poc3d_mask:   [B, Lc]

    mask 规则：
        True  = padding / special token，需要忽略
        False = 真实 token，需要参与 attention
    """

    def __init__(
        self,
        # 1D LM token 维度
        d_drug_lm: int,
        d_prot_lm: int,
        # 3D Uni-Mol2 atom token 维度
        d3_lig: int = 512,
        d3_poc: int = 512,
        # 可选 2D 图参数，默认保留
        drug_node_dim: int = 70,
        drug_edge_dim: int = 6,
        prot_node_dim: int = 33,
        prot_edge_dim: int = 3,
        gcn_hidden: int = 128,
        gcn_out: int = 128,
        # 统一维度
        d_model: int = 256,
        d_attn: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_2d: bool = False
    ):
        super().__init__()

        self.d_model = d_model

        self.d_attn = d_attn

        self.use_2d = use_2d

        # =========================
        # 1D token 投影
        # =========================
        self.drug_token_proj = TokenProjector(
            in_dim=d_drug_lm,
            out_dim=d_attn,
            dropout=dropout
        )

        self.prot_token_proj = TokenProjector(
            in_dim=d_prot_lm,
            out_dim=d_attn,
            dropout=dropout
        )

        # =========================
        # 3D token 投影
        # =========================
        self.lig3d_token_proj = TokenProjector(
            in_dim=d3_lig,
            out_dim=d_attn,
            dropout=dropout
        )

        self.poc3d_token_proj = TokenProjector(
            in_dim=d3_poc,
            out_dim=d_attn,
            dropout=dropout
        )

        # =========================
        # token type embedding
        # 0 = drug sequence token
        # 1 = protein sequence token
        # 2 = ligand 3D atom token
        # 3 = pocket 3D atom token
        # =========================
        self.token_type_emb = nn.Embedding(4, d_attn)

        self.type_dropout = nn.Dropout(dropout)

        # =========================
        # 真正的双向 cross-attention
        # =========================
        self.seq_to_struct_attn = CrossAttentionBlock(
            dim_q=d_attn,
            dim_kv=d_attn,
            d_model=d_attn,
            n_heads=n_heads,
            dropout=dropout
        )

        self.struct_to_seq_attn = CrossAttentionBlock(
            dim_q=d_attn,
            dim_kv=d_attn,
            d_model=d_attn,
            n_heads=n_heads,
            dropout=dropout
        )

        # =========================
        # token 序列池化
        # =========================
        self.seq_pool = MaskedAttentivePooling(d_attn)

        self.struct_pool = MaskedAttentivePooling(d_attn)

        # =========================
        # cross-attention 后的 sequence-structure pair 融合
        # =========================
        self.cross_pair_fuse = nn.Sequential(
            nn.Linear(4 * d_attn, 2 * d_attn, bias=False),
            nn.LayerNorm(2 * d_attn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_attn, d_attn, bias=False),
            nn.LayerNorm(d_attn),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # =========================
        # structure-only 仍沿用旧的单模态投影。
        # 1D-only 会走后面新增的 drug-protein cross-attention。
        # =========================
        self.single_modal_fuse = nn.Sequential(
            nn.Linear(d_attn, d_attn, bias=False),
            nn.LayerNorm(d_attn),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # =========================
        # 可选 2D 图分支
        # 你目前可以 use_2d=False，不影响训练。
        # =========================
        if use_2d:
            self.enc2d_drug = GraphConvEW(
                in_dim=drug_node_dim,
                hidden_dim=gcn_hidden,
                out_dim=gcn_out
            )

            self.enc2d_prot = GraphConvEW(
                in_dim=prot_node_dim,
                hidden_dim=gcn_hidden,
                out_dim=gcn_out
            )

            self.fuse2d = nn.Sequential(
                nn.Linear(2 * gcn_out, d_attn, bias=False),
                nn.LayerNorm(d_attn),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            self.enc2d_drug = None
            self.enc2d_prot = None
            self.fuse2d = None

        # =========================
        # 最终融合：cross-attention 表示 + 2D 表示
        # 如果没有 2D，就用 0 向量补齐。
        # =========================
        self.final_fuse = nn.Sequential(
            nn.Linear(2 * d_attn, d_attn, bias=False),
            nn.LayerNorm(d_attn),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # =========================
        # 回归头：预测 pKd / neglog_aff
        # =========================
        self.reg_head = nn.Sequential(
            nn.Linear(d_attn, 2 * d_attn, bias=False),
            nn.LayerNorm(2 * d_attn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_attn, 1, bias=True)
        )

        # =========================
        # 1D-only 专用 drug-protein 双向 cross-attention
        # 放在旧模块之后，避免改变旧模块的初始化顺序。
        # 只有 has_seq=True 且 has_struct=False 时使用。
        # =========================
        self.drug_to_prot_attn = CrossAttentionBlock(
            dim_q=d_attn,
            dim_kv=d_attn,
            d_model=d_attn,
            n_heads=n_heads,
            dropout=dropout
        )

        self.prot_to_drug_attn = CrossAttentionBlock(
            dim_q=d_attn,
            dim_kv=d_attn,
            d_model=d_attn,
            n_heads=n_heads,
            dropout=dropout
        )

        self.drug_seq_pool = MaskedAttentivePooling(d_attn)

        self.prot_seq_pool = MaskedAttentivePooling(d_attn)

        self.seq_pair_fuse = nn.Sequential(
            nn.Linear(4 * d_attn, 2 * d_attn, bias=False),
            nn.LayerNorm(2 * d_attn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_attn, d_attn, bias=False),
            nn.LayerNorm(d_attn),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def _make_zero_mask(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        如果外部没有传 mask，就自动生成全 False mask。

        False 表示真实 token，不屏蔽。
        """
        return torch.zeros(
            x.shape[:2],
            dtype=torch.bool,
            device=x.device
        )

    def _add_type_embedding(
        self,
        x: torch.Tensor,
        type_id: int
    ) -> torch.Tensor:
        """
        给不同来源的 token 加 type embedding。

        这样模型可以区分：
            drug token
            protein token
            ligand 3D token
            pocket 3D token
        """
        type_ids = torch.full(
            x.shape[:2],
            fill_value=type_id,
            dtype=torch.long,
            device=x.device
        )

        x = x + self.token_type_emb(type_ids)

        x = self.type_dropout(x)

        return x

    def _encode_2d_if_available(
        self,
        g_lig: Optional[Any],
        g_prot: Optional[Any],
        device: torch.device,
        batch_size: int
    ) -> torch.Tensor:
        """
        可选 2D 图编码。

        如果 use_2d=False 或 g_lig/g_prot 为 None，则返回 0 向量。
        """
        if (not self.use_2d) or (g_lig is None) or (g_prot is None):
            return torch.zeros(
                batch_size,
                self.d_attn,
                dtype=torch.float32,
                device=device
            )

        lig_vec = self.enc2d_drug(
            g_lig,
            g_lig.ndata["x"].float(),
            g_lig.edata["w"].float()
        )

        prot_vec = self.enc2d_prot(
            g_prot,
            g_prot.ndata["x"].float(),
            g_prot.edata["w"].float()
        )

        pair_2d = torch.cat([lig_vec, prot_vec], dim=-1)

        z2d = self.fuse2d(pair_2d)

        return z2d

    def forward(
        self,
        drug_lm_tokens: Optional[torch.Tensor] = None,
        prot_lm_tokens: Optional[torch.Tensor] = None,
        lig3d_tokens: Optional[torch.Tensor] = None,
        poc3d_tokens: Optional[torch.Tensor] = None,
        drug_lm_mask: Optional[torch.Tensor] = None,
        prot_lm_mask: Optional[torch.Tensor] = None,
        lig3d_mask: Optional[torch.Tensor] = None,
        poc3d_mask: Optional[torch.Tensor] = None,
        g_lig: Optional[Any] = None,
        g_prot: Optional[Any] = None,
        return_attn: bool = False,
        # 下面四个参数只用于捕获旧代码误传，防止 silent bug
        drug_lm: Optional[torch.Tensor] = None,
        prot_lm: Optional[torch.Tensor] = None,
        lig3d: Optional[torch.Tensor] = None,
        poc3d: Optional[torch.Tensor] = None
    ):
        """
        前向传播。

        新版推荐调用方式：
            model(
                drug_lm_tokens=...,
                prot_lm_tokens=...,
                lig3d_tokens=...,
                poc3d_tokens=...,
                drug_lm_mask=...,
                prot_lm_mask=...,
                lig3d_mask=...,
                poc3d_mask=...
            )

        不推荐继续传：
            drug_lm, prot_lm, lig3d, poc3d

        因为这些名字在旧模型中代表 pooled 向量。
        """
        device = next(self.parameters()).device

        if drug_lm is not None or prot_lm is not None or lig3d is not None or poc3d is not None:
            raise ValueError(
                "你正在使用旧版 pooled 输入名 drug_lm/prot_lm/lig3d/poc3d。"
                "为了实现真正的 token-level cross-attention，请改用 "
                "drug_lm_tokens/prot_lm_tokens/lig3d_tokens/poc3d_tokens。"
            )

        has_seq = (drug_lm_tokens is not None) and (prot_lm_tokens is not None)

        has_struct = (lig3d_tokens is not None) and (poc3d_tokens is not None)

        if not has_seq and not has_struct and not self.use_2d:
            raise ValueError(
                "至少需要提供 sequence tokens、structure tokens 或 2D 图模态中的一种。"
            )

        attn_info: Dict[str, Any] = {}

        z_cross = None

        batch_size = None

        # =====================================================
        # 1. 构建 sequence tokens
        # =====================================================
        if has_seq:
            drug_lm_tokens = drug_lm_tokens.to(device, non_blocking=True)

            prot_lm_tokens = prot_lm_tokens.to(device, non_blocking=True)

            if drug_lm_mask is not None:
                drug_lm_mask = drug_lm_mask.to(device, non_blocking=True).bool()
            else:
                drug_lm_mask = self._make_zero_mask(drug_lm_tokens)

            if prot_lm_mask is not None:
                prot_lm_mask = prot_lm_mask.to(device, non_blocking=True).bool()
            else:
                prot_lm_mask = self._make_zero_mask(prot_lm_tokens)

            drug_tokens = self.drug_token_proj(
                drug_lm_tokens,
                drug_lm_mask
            )

            prot_tokens = self.prot_token_proj(
                prot_lm_tokens,
                prot_lm_mask
            )

            drug_tokens = self._add_type_embedding(
                drug_tokens,
                type_id=0
            )

            prot_tokens = self._add_type_embedding(
                prot_tokens,
                type_id=1
            )

            seq_tokens = torch.cat(
                [drug_tokens, prot_tokens],
                dim=1
            )

            seq_mask = torch.cat(
                [drug_lm_mask, prot_lm_mask],
                dim=1
            )

            batch_size = seq_tokens.size(0)

        else:
            seq_tokens = None
            seq_mask = None

        # =====================================================
        # 2. 构建 structure tokens
        # =====================================================
        if has_struct:
            lig3d_tokens = lig3d_tokens.to(device, non_blocking=True)

            poc3d_tokens = poc3d_tokens.to(device, non_blocking=True)

            if lig3d_mask is not None:
                lig3d_mask = lig3d_mask.to(device, non_blocking=True).bool()
            else:
                lig3d_mask = self._make_zero_mask(lig3d_tokens)

            if poc3d_mask is not None:
                poc3d_mask = poc3d_mask.to(device, non_blocking=True).bool()
            else:
                poc3d_mask = self._make_zero_mask(poc3d_tokens)

            lig_tokens = self.lig3d_token_proj(
                lig3d_tokens,
                lig3d_mask
            )

            poc_tokens = self.poc3d_token_proj(
                poc3d_tokens,
                poc3d_mask
            )

            lig_tokens = self._add_type_embedding(
                lig_tokens,
                type_id=2
            )

            poc_tokens = self._add_type_embedding(
                poc_tokens,
                type_id=3
            )

            struct_tokens = torch.cat(
                [lig_tokens, poc_tokens],
                dim=1
            )

            struct_mask = torch.cat(
                [lig3d_mask, poc3d_mask],
                dim=1
            )

            if batch_size is None:
                batch_size = struct_tokens.size(0)

        else:
            struct_tokens = None
            struct_mask = None

        # =====================================================
        # 3. 真正的 token-level bidirectional cross-attention
        # =====================================================
        if has_seq and has_struct:
            if seq_tokens.size(0) != struct_tokens.size(0):
                raise ValueError(
                    f"sequence batch={seq_tokens.size(0)} 与 structure batch={struct_tokens.size(0)} 不一致。"
                )

            seq_enhanced, attn_seq_to_struct = self.seq_to_struct_attn(
                q_tokens=seq_tokens,
                kv_tokens=struct_tokens,
                q_padding_mask=seq_mask,
                kv_padding_mask=struct_mask,
                return_attn=return_attn
            )

            struct_enhanced, attn_struct_to_seq = self.struct_to_seq_attn(
                q_tokens=struct_tokens,
                kv_tokens=seq_tokens,
                q_padding_mask=struct_mask,
                kv_padding_mask=seq_mask,
                return_attn=return_attn
            )

            z_seq = self.seq_pool(
                seq_enhanced,
                seq_mask
            )

            z_struct = self.struct_pool(
                struct_enhanced,
                struct_mask
            )

            cross_feat = torch.cat(
                [
                    z_seq,
                    z_struct,
                    z_seq * z_struct,
                    torch.abs(z_seq - z_struct)
                ],
                dim=-1
            )

            z_cross = self.cross_pair_fuse(cross_feat)

            if return_attn:
                attn_info["seq_to_struct"] = attn_seq_to_struct
                attn_info["struct_to_seq"] = attn_struct_to_seq

        # =====================================================
        # 4. 单模态路径
        #    - 只有 sequence：drug ↔ protein token-level cross-attention
        #    - 只有 structure：保持旧的 single_modal_fuse
        # =====================================================
        elif has_seq:
            # 仅 1D-only 时启用 drug ↔ protein token-level cross-attention。
            drug_enhanced, attn_drug_to_prot = self.drug_to_prot_attn(
                q_tokens=drug_tokens,
                kv_tokens=prot_tokens,
                q_padding_mask=drug_lm_mask,
                kv_padding_mask=prot_lm_mask,
                return_attn=return_attn
            )

            prot_enhanced, attn_prot_to_drug = self.prot_to_drug_attn(
                q_tokens=prot_tokens,
                kv_tokens=drug_tokens,
                q_padding_mask=prot_lm_mask,
                kv_padding_mask=drug_lm_mask,
                return_attn=return_attn
            )

            z_drug = self.drug_seq_pool(
                drug_enhanced,
                drug_lm_mask
            )

            z_prot = self.prot_seq_pool(
                prot_enhanced,
                prot_lm_mask
            )

            seq_pair_feat = torch.cat(
                [
                    z_drug,
                    z_prot,
                    z_drug * z_prot,
                    torch.abs(z_drug - z_prot)
                ],
                dim=-1
            )

            z_cross = self.seq_pair_fuse(seq_pair_feat)

            if return_attn:
                attn_info["drug_to_prot"] = attn_drug_to_prot
                attn_info["prot_to_drug"] = attn_prot_to_drug

        elif has_struct:
            z_struct = self.struct_pool(
                struct_tokens,
                struct_mask
            )

            z_cross = self.single_modal_fuse(z_struct)

        # =====================================================
        # 5. 可选 2D 图分支
        # =====================================================
        z2d = self._encode_2d_if_available(
            g_lig=g_lig,
            g_prot=g_prot,
            device=device,
            batch_size=batch_size
        )

        # =====================================================
        # 6. 最终融合与回归
        # =====================================================
        final_feat = torch.cat(
            [z_cross, z2d],
            dim=-1
        )

        fused = self.final_fuse(final_feat)

        y = self.reg_head(fused).squeeze(-1)

        if return_attn:
            return y, attn_info

        return y


# =========================================================
# 兼容旧导入名
# 如果你的 train 里仍然写：
#     from model.model_token_crossmodal_lm_revise import MultiModalDTA_LM
# 也可以正常导入。
# =========================================================
MultiModalDTA_LM = TokenLevelMultiModalDTA