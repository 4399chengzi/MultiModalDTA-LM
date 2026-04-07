# -*- coding: utf-8 -*-
# =========================================================
# 多模态 DTA 模型（Ablation-Friendly Revision）
#
# 支持：
#   1) 模态消融：
#      - 1D-only
#      - 2D-only
#      - 3D-only
#      - 1D+3D
#      - 2D+3D
#      - 1D+2D+3D
#
#   2) 融合消融：
#      - gated fusion
#      - direct add fusion
#      - concat fusion
#
#   3) 3D 交互项消融：
#      - add_interactions_3d = True / False
#
#   4) gate 输入消融：
#      - full      : [q, kv, q*kv, |q-kv|]
#      - q_kv      : [q, kv]
#      - q_kv_prod : [q, kv, q*kv]
#      - q_kv_abs  : [q, kv, |q-kv|]
#
#   5) 融合后结构消融：
#      - use_fusion_ffn = True / False
#      - use_fusion_residual = True / False
# =========================================================

from typing import Optional, Dict, Any

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
        if w.dim() == 2:
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
        显式按子图分段求和。
        返回 [B, D]，B 为批内子图数。
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
                (counts.numel() + 1,), dtype=torch.long, device=node_feat.device
            )
            offsets[1:] = torch.cumsum(counts, dim=0)

            outs = []
            for i in range(counts.numel()):
                seg = node_feat[offsets[i] : offsets[i + 1]]
                outs.append(seg.sum(dim=0, keepdim=True))
            return torch.cat(outs, dim=0)

        graphs = dgl.unbatch(g)
        outs = []
        start = 0
        for gi in graphs:
            n = gi.num_nodes()
            outs.append(node_feat[start : start + n].sum(dim=0, keepdim=True))
            start += n
        return torch.cat(outs, dim=0)

    def forward(
        self, g: dgl.DGLGraph, x: torch.Tensor, w: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """输入批图 g，节点特征 x、边特征 w；输出每图读出向量 [B, out_dim]。"""
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


# ======【3D】Uni-Mol2 向量对编码器 ======
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
            feat = torch.cat([l, p, l * p, torch.abs(l - p)], dim=-1)
        else:
            feat = torch.cat([l, p], dim=-1)

        out = self.mlp(feat)
        return out


# ====== 融合模块基类 ======
class BaseFusion(nn.Module):
    """所有融合模块的基类，便于统一接口。"""

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        d_attn: int = 256,
        dropout: float = 0.1,
        use_fusion_ffn: bool = True,
        use_fusion_residual: bool = True,
    ):
        super().__init__()
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.d_attn = d_attn
        self.use_fusion_ffn = use_fusion_ffn
        self.use_fusion_residual = use_fusion_residual
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_vec: torch.Tensor, kv_vec: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ====== Gated Cross-Modal Fusion ======
class GatedCrossModalFuse(BaseFusion):
    """
    门控跨模态融合模块。
    支持 gate 输入消融：
      - full
      - q_kv
      - q_kv_prod
      - q_kv_abs
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        d_attn: int = 256,
        dropout: float = 0.1,
        gate_feature_mode: str = "full",
        use_fusion_ffn: bool = True,
        use_fusion_residual: bool = True,
    ):
        super().__init__(
            dim_q=dim_q,
            dim_kv=dim_kv,
            d_attn=d_attn,
            dropout=dropout,
            use_fusion_ffn=use_fusion_ffn,
            use_fusion_residual=use_fusion_residual,
        )

        self.gate_feature_mode = gate_feature_mode.lower()

        self.q_proj = nn.Linear(dim_q, d_attn, bias=False)
        self.kv_proj = nn.Linear(dim_kv, d_attn, bias=False)

        gate_in_dim = self._get_gate_input_dim(self.gate_feature_mode, d_attn)

        self.gate_net = nn.Sequential(
            nn.Linear(gate_in_dim, d_attn, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_attn, d_attn, bias=True),
            nn.Sigmoid(),
        )

        self.fuse_proj = nn.Linear(d_attn, d_attn, bias=False)
        self.ln1 = nn.LayerNorm(d_attn)

        self.ff = nn.Sequential(
            nn.Linear(d_attn, 2 * d_attn, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_attn, d_attn, bias=False),
        )
        self.ln2 = nn.LayerNorm(d_attn)

    @staticmethod
    def _get_gate_input_dim(mode: str, d_attn: int) -> int:
        """根据 gate 输入模式，计算输入维度。"""
        if mode == "full":
            return 4 * d_attn
        if mode == "q_kv":
            return 2 * d_attn
        if mode == "q_kv_prod":
            return 3 * d_attn
        if mode == "q_kv_abs":
            return 3 * d_attn
        raise ValueError(f"不支持的 gate_feature_mode: {mode}")

    def _build_gate_input(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """根据 gate_feature_mode 构造门控输入。"""
        if self.gate_feature_mode == "full":
            return torch.cat([q, kv, q * kv, torch.abs(q - kv)], dim=-1)
        if self.gate_feature_mode == "q_kv":
            return torch.cat([q, kv], dim=-1)
        if self.gate_feature_mode == "q_kv_prod":
            return torch.cat([q, kv, q * kv], dim=-1)
        if self.gate_feature_mode == "q_kv_abs":
            return torch.cat([q, kv, torch.abs(q - kv)], dim=-1)
        raise ValueError(f"不支持的 gate_feature_mode: {self.gate_feature_mode}")

    def forward(self, q_vec: torch.Tensor, kv_vec: torch.Tensor) -> torch.Tensor:
        """
        q_vec:  [B, Dq]
        kv_vec: [B, Dk]
        return: [B, d_attn]
        """
        q = self.q_proj(q_vec)
        kv = self.kv_proj(kv_vec)

        gate_in = self._build_gate_input(q, kv)
        gate = self.gate_net(gate_in)

        injected = gate * kv
        injected = self.fuse_proj(injected)
        injected = self.dropout(injected)

        if self.use_fusion_residual:
            y = self.ln1(q + injected)
        else:
            y = self.ln1(injected)

        if self.use_fusion_ffn:
            y2 = self.ff(y)
            y2 = self.dropout(y2)
            y = self.ln2(y + y2)

        return y


# ====== Direct Add Fusion ======
class AddCrossModalFuse(BaseFusion):
    """
    直接加和融合：
      - 不使用 gate
      - 只把 3D 投影后直接加到主干表示上
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        d_attn: int = 256,
        dropout: float = 0.1,
        use_fusion_ffn: bool = True,
        use_fusion_residual: bool = True,
    ):
        super().__init__(
            dim_q=dim_q,
            dim_kv=dim_kv,
            d_attn=d_attn,
            dropout=dropout,
            use_fusion_ffn=use_fusion_ffn,
            use_fusion_residual=use_fusion_residual,
        )

        self.q_proj = nn.Linear(dim_q, d_attn, bias=False)
        self.kv_proj = nn.Linear(dim_kv, d_attn, bias=False)
        self.fuse_proj = nn.Linear(d_attn, d_attn, bias=False)

        self.ln1 = nn.LayerNorm(d_attn)
        self.ff = nn.Sequential(
            nn.Linear(d_attn, 2 * d_attn, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_attn, d_attn, bias=False),
        )
        self.ln2 = nn.LayerNorm(d_attn)

    def forward(self, q_vec: torch.Tensor, kv_vec: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(q_vec)
        kv = self.kv_proj(kv_vec)

        injected = self.fuse_proj(kv)
        injected = self.dropout(injected)

        if self.use_fusion_residual:
            y = self.ln1(q + injected)
        else:
            y = self.ln1(injected)

        if self.use_fusion_ffn:
            y2 = self.ff(y)
            y2 = self.dropout(y2)
            y = self.ln2(y + y2)

        return y


# ====== Concat Fusion ======
class ConcatCrossModalFuse(BaseFusion):
    """
    拼接融合：
      - 先把 q 和 kv 各自投影到 d_attn
      - 再拼接成 [q || kv]
      - 用 MLP 压回 d_attn
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        d_attn: int = 256,
        dropout: float = 0.1,
        use_fusion_ffn: bool = True,
        use_fusion_residual: bool = True,
    ):
        super().__init__(
            dim_q=dim_q,
            dim_kv=dim_kv,
            d_attn=d_attn,
            dropout=dropout,
            use_fusion_ffn=use_fusion_ffn,
            use_fusion_residual=use_fusion_residual,
        )

        self.q_proj = nn.Linear(dim_q, d_attn, bias=False)
        self.kv_proj = nn.Linear(dim_kv, d_attn, bias=False)

        self.concat_mlp = nn.Sequential(
            nn.Linear(2 * d_attn, d_attn, bias=False),
            nn.LayerNorm(d_attn),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.ln1 = nn.LayerNorm(d_attn)
        self.ff = nn.Sequential(
            nn.Linear(d_attn, 2 * d_attn, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_attn, d_attn, bias=False),
        )
        self.ln2 = nn.LayerNorm(d_attn)

    def forward(self, q_vec: torch.Tensor, kv_vec: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(q_vec)
        kv = self.kv_proj(kv_vec)

        fused = self.concat_mlp(torch.cat([q, kv], dim=-1))
        fused = self.dropout(fused)

        if self.use_fusion_residual:
            y = self.ln1(q + fused)
        else:
            y = self.ln1(fused)

        if self.use_fusion_ffn:
            y2 = self.ff(y)
            y2 = self.dropout(y2)
            y = self.ln2(y + y2)

        return y


# ====== 总模型 ======
class MultiModalDTA_LM(nn.Module):
    """
    多模态 DTA 模型（支持消融版本）。

    关键新增参数：
      - ablation_mode
      - enable_1d / enable_2d / enable_3d
      - fusion_mode
      - gate_feature_mode
      - use_fusion_ffn
      - use_fusion_residual
    """

    SUPPORTED_ABLATIONS = {
        "full",
        "1d_only",
        "2d_only",
        "3d_only",
        "1d_3d_gated",
        "1d_3d_add",
        "1d_3d_concat",
        "2d_3d_gated",
        "2d_3d_add",
        "2d_3d_concat",
        "1d_2d_3d_gated",
        "1d_2d_3d_add",
        "1d_2d_3d_concat",
        "no_3d_interactions",
        "gate_q_kv_only",
        "gate_q_kv_prod",
        "gate_q_kv_abs",
        "no_fusion_ffn",
        "no_fusion_residual",
    }

    def __init__(
        self,
        d_drug_lm: int,
        d_prot_lm: int,
        drug_node_dim: int = 70,
        drug_edge_dim: int = 6,   # 为兼容旧接口，保留但当前不直接用
        prot_node_dim: int = 33,
        prot_edge_dim: int = 3,   # 为兼容旧接口，保留但当前不直接用
        gcn_hidden: int = 128,
        gcn_out: int = 128,
        d3_lig: int = 512,
        d3_poc: int = 512,
        d_model: int = 256,
        d_attn: int = 256,
        n_heads: int = 4,         # 为兼容旧接口，保留但当前不直接用
        add_interactions_3d: bool = True,
        # ====== 新增：消融控制项 ======
        ablation_mode: str = "full",
        enable_1d: bool = True,
        enable_2d: bool = True,
        enable_3d: bool = True,
        fusion_mode: str = "gated",           # gated / add / concat
        gate_feature_mode: str = "full",      # full / q_kv / q_kv_prod / q_kv_abs
        use_fusion_ffn: bool = True,
        use_fusion_residual: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 先保存原始配置
        self.d_model = d_model
        self.d_attn = d_attn
        self.ablation_mode = ablation_mode.lower()

        # 根据预设模式自动覆盖配置
        (
            enable_1d,
            enable_2d,
            enable_3d,
            fusion_mode,
            gate_feature_mode,
            add_interactions_3d,
            use_fusion_ffn,
            use_fusion_residual,
        ) = self._apply_ablation_preset(
            ablation_mode=self.ablation_mode,
            enable_1d=enable_1d,
            enable_2d=enable_2d,
            enable_3d=enable_3d,
            fusion_mode=fusion_mode,
            gate_feature_mode=gate_feature_mode,
            add_interactions_3d=add_interactions_3d,
            use_fusion_ffn=use_fusion_ffn,
            use_fusion_residual=use_fusion_residual,
        )

        self.enable_1d = enable_1d
        self.enable_2d = enable_2d
        self.enable_3d = enable_3d
        self.fusion_mode = fusion_mode.lower()
        self.gate_feature_mode = gate_feature_mode.lower()
        self.use_fusion_ffn = use_fusion_ffn
        self.use_fusion_residual = use_fusion_residual
        self.add_interactions_3d = add_interactions_3d

        # ---- 1D ----
        self.proj_drug_lm = nn.Sequential(
            nn.LayerNorm(d_drug_lm),
            nn.Linear(d_drug_lm, d_model, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj_prot_lm = nn.Sequential(
            nn.LayerNorm(d_prot_lm),
            nn.Linear(d_prot_lm, d_model, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fuse1d = nn.Sequential(
            nn.Linear(2 * d_model, d_model, bias=False),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---- 2D ----
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
            nn.Dropout(dropout),
        )

        # ---- 3D ----
        self.enc3d_pair = Pair3DEncoder(
            d_in_lig=d3_lig,
            d_in_poc=d3_poc,
            d_model=d_model,
            add_interactions=add_interactions_3d,
        )

        # ---- 选择融合模块 ----
        self.cross = self._build_fusion_module(
            fusion_mode=self.fusion_mode,
            dim_q=2 * d_model,
            dim_kv=d_model,
            d_attn=d_attn,
            dropout=dropout,
            gate_feature_mode=self.gate_feature_mode,
            use_fusion_ffn=self.use_fusion_ffn,
            use_fusion_residual=self.use_fusion_residual,
        )

        # ---- 回归头：有 3D 且有 1D/2D 的融合路径 ----
        self.reg_head_ca = nn.Sequential(
            nn.Linear(d_attn, 2 * d_model, bias=False),
            nn.LayerNorm(2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, 1, bias=True),
        )

        # ---- 回归头：无 3D 或只有 3D ----
        self.fuse_modalities_no_ca = nn.Sequential(
            nn.Linear(3 * d_model, d_model, bias=False),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.reg_head_no_ca = nn.Sequential(
            nn.Linear(d_model, 2 * d_model, bias=False),
            nn.LayerNorm(2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, 1, bias=True),
        )

    # =========================
    # 预设消融模式
    # =========================
    @classmethod
    def _apply_ablation_preset(
        cls,
        ablation_mode: str,
        enable_1d: bool,
        enable_2d: bool,
        enable_3d: bool,
        fusion_mode: str,
        gate_feature_mode: str,
        add_interactions_3d: bool,
        use_fusion_ffn: bool,
        use_fusion_residual: bool,
    ):
        mode = ablation_mode.lower()

        if mode not in cls.SUPPORTED_ABLATIONS:
            raise ValueError(
                f"不支持的 ablation_mode: {mode}，可选为 {sorted(cls.SUPPORTED_ABLATIONS)}"
            )

        if mode == "full":
            return (
                enable_1d, enable_2d, enable_3d, fusion_mode, gate_feature_mode,
                add_interactions_3d, use_fusion_ffn, use_fusion_residual
            )

        if mode == "1d_only":
            return True, False, False, fusion_mode, gate_feature_mode, add_interactions_3d, use_fusion_ffn, use_fusion_residual

        if mode == "2d_only":
            return False, True, False, fusion_mode, gate_feature_mode, add_interactions_3d, use_fusion_ffn, use_fusion_residual

        if mode == "3d_only":
            return False, False, True, fusion_mode, gate_feature_mode, add_interactions_3d, use_fusion_ffn, use_fusion_residual

        if mode == "1d_3d_gated":
            return True, False, True, "gated", gate_feature_mode, add_interactions_3d, use_fusion_ffn, use_fusion_residual

        if mode == "1d_3d_add":
            return True, False, True, "add", gate_feature_mode, add_interactions_3d, use_fusion_ffn, use_fusion_residual

        if mode == "1d_3d_concat":
            return True, False, True, "concat", gate_feature_mode, add_interactions_3d, use_fusion_ffn, use_fusion_residual

        if mode == "2d_3d_gated":
            return False, True, True, "gated", gate_feature_mode, add_interactions_3d, use_fusion_ffn, use_fusion_residual

        if mode == "2d_3d_add":
            return False, True, True, "add", gate_feature_mode, add_interactions_3d, use_fusion_ffn, use_fusion_residual

        if mode == "2d_3d_concat":
            return False, True, True, "concat", gate_feature_mode, add_interactions_3d, use_fusion_ffn, use_fusion_residual

        if mode == "1d_2d_3d_gated":
            return True, True, True, "gated", gate_feature_mode, add_interactions_3d, use_fusion_ffn, use_fusion_residual

        if mode == "1d_2d_3d_add":
            return True, True, True, "add", gate_feature_mode, add_interactions_3d, use_fusion_ffn, use_fusion_residual

        if mode == "1d_2d_3d_concat":
            return True, True, True, "concat", gate_feature_mode, add_interactions_3d, use_fusion_ffn, use_fusion_residual

        if mode == "no_3d_interactions":
            return enable_1d, enable_2d, enable_3d, fusion_mode, gate_feature_mode, False, use_fusion_ffn, use_fusion_residual

        if mode == "gate_q_kv_only":
            return enable_1d, enable_2d, enable_3d, "gated", "q_kv", add_interactions_3d, use_fusion_ffn, use_fusion_residual

        if mode == "gate_q_kv_prod":
            return enable_1d, enable_2d, enable_3d, "gated", "q_kv_prod", add_interactions_3d, use_fusion_ffn, use_fusion_residual

        if mode == "gate_q_kv_abs":
            return enable_1d, enable_2d, enable_3d, "gated", "q_kv_abs", add_interactions_3d, use_fusion_ffn, use_fusion_residual

        if mode == "no_fusion_ffn":
            return enable_1d, enable_2d, enable_3d, fusion_mode, gate_feature_mode, add_interactions_3d, False, use_fusion_residual

        if mode == "no_fusion_residual":
            return enable_1d, enable_2d, enable_3d, fusion_mode, gate_feature_mode, add_interactions_3d, use_fusion_ffn, False

        raise ValueError(f"未处理的 ablation_mode: {mode}")

    @staticmethod
    def _build_fusion_module(
        fusion_mode: str,
        dim_q: int,
        dim_kv: int,
        d_attn: int,
        dropout: float,
        gate_feature_mode: str,
        use_fusion_ffn: bool,
        use_fusion_residual: bool,
    ) -> nn.Module:
        """根据 fusion_mode 构建融合模块。"""
        fusion_mode = fusion_mode.lower()

        if fusion_mode == "gated":
            return GatedCrossModalFuse(
                dim_q=dim_q,
                dim_kv=dim_kv,
                d_attn=d_attn,
                dropout=dropout,
                gate_feature_mode=gate_feature_mode,
                use_fusion_ffn=use_fusion_ffn,
                use_fusion_residual=use_fusion_residual,
            )
        if fusion_mode == "add":
            return AddCrossModalFuse(
                dim_q=dim_q,
                dim_kv=dim_kv,
                d_attn=d_attn,
                dropout=dropout,
                use_fusion_ffn=use_fusion_ffn,
                use_fusion_residual=use_fusion_residual,
            )
        if fusion_mode == "concat":
            return ConcatCrossModalFuse(
                dim_q=dim_q,
                dim_kv=dim_kv,
                d_attn=d_attn,
                dropout=dropout,
                use_fusion_ffn=use_fusion_ffn,
                use_fusion_residual=use_fusion_residual,
            )
        raise ValueError(f"不支持的 fusion_mode: {fusion_mode}")

    # =========================
    # 配置查看
    # =========================
    def get_ablation_config(self) -> Dict[str, Any]:
        """返回当前模型的消融配置，方便训练日志打印。"""
        return {
            "ablation_mode": self.ablation_mode,
            "enable_1d": self.enable_1d,
            "enable_2d": self.enable_2d,
            "enable_3d": self.enable_3d,
            "fusion_mode": self.fusion_mode,
            "gate_feature_mode": self.gate_feature_mode,
            "add_interactions_3d": self.add_interactions_3d,
            "use_fusion_ffn": self.use_fusion_ffn,
            "use_fusion_residual": self.use_fusion_residual,
        }

    # =========================
    # 前向传播
    # =========================
    def forward(
        self,
        drug_lm: Optional[torch.Tensor] = None,
        prot_lm: Optional[torch.Tensor] = None,
        g_lig: Optional[dgl.DGLGraph] = None,
        g_prot: Optional[dgl.DGLGraph] = None,
        lig3d: Optional[torch.Tensor] = None,
        poc3d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = next(self.parameters()).device

        # 先看输入是否存在，再结合 enable_* 判断最终是否启用
        has_1d = self.enable_1d and (drug_lm is not None) and (prot_lm is not None)
        has_2d = self.enable_2d and (g_lig is not None) and (g_prot is not None)
        has_3d = self.enable_3d and (lig3d is not None) and (poc3d is not None)

        if not (has_1d or has_2d or has_3d):
            raise ValueError(
                "当前 ablation 配置下，没有可用模态。请检查 enable_1d/2d/3d 或输入是否为 None。"
            )

        z1d = z2d = z3d = None

        # === 1) 1D ===
        if has_1d:
            drug_lm = drug_lm.to(device)
            prot_lm = prot_lm.to(device)

            drug_1d = self.proj_drug_lm(drug_lm)
            prot_1d = self.proj_prot_lm(prot_lm)
            pair_1d = torch.cat([drug_1d, prot_1d], dim=-1)
            z1d = self.fuse1d(pair_1d)

        # === 2) 2D ===
        if has_2d:
            lig_vec = self.enc2d_drug(
                g_lig, g_lig.ndata["x"].float(), g_lig.edata["w"].float()
            )
            prot_vec = self.enc2d_prot(
                g_prot, g_prot.ndata["x"].float(), g_prot.edata["w"].float()
            )
            pair_2d = torch.cat([lig_vec, prot_vec], dim=-1)
            z2d = self.fuse2d(pair_2d)

        # === 3) 3D ===
        if has_3d:
            lig3d = torch.nan_to_num(
                lig3d.to(device), nan=0.0, posinf=1e6, neginf=-1e6
            )
            poc3d = torch.nan_to_num(
                poc3d.to(device), nan=0.0, posinf=1e6, neginf=-1e6
            )
            z3d = self.enc3d_pair(lig3d, poc3d)

        # === batch size 检查 ===
        B = None
        for z in (z1d, z2d, z3d):
            if z is not None:
                if B is None:
                    B = z.size(0)
                else:
                    assert z.size(0) == B, f"batch size 不一致：期望 {B}, 但有 {z.size(0)}"

        # === A 路径：有 3D 且至少有 1D/2D ===
        if has_3d and (z1d is not None or z2d is not None):
            if z1d is not None and z2d is not None:
                q_vec = torch.cat([z1d, z2d], dim=-1)
            elif z1d is not None:
                zero = torch.zeros_like(z1d)
                q_vec = torch.cat([z1d, zero], dim=-1)
            else:
                zero = torch.zeros_like(z2d)
                q_vec = torch.cat([zero, z2d], dim=-1)

            fused = self.cross(q_vec, z3d)
            y = self.reg_head_ca(fused).squeeze(-1)
            return y

        # === B 路径：无 3D，或只有 3D ===
        base_vec = z1d if z1d is not None else (z2d if z2d is not None else z3d)
        assert base_vec is not None

        zero = torch.zeros_like(base_vec)
        z1 = z1d if z1d is not None else zero
        z2 = z2d if z2d is not None else zero
        z3 = z3d if z3d is not None else zero

        fused = self.fuse_modalities_no_ca(torch.cat([z1, z2, z3], dim=-1))
        y = self.reg_head_no_ca(fused).squeeze(-1)
        return y