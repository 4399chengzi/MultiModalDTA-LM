# -*- coding: utf-8 -*-
# =========================================================
# 多模态 DTA 模型（1D + 2D + 3D with Cross-Attention）
# 依赖假设：
#   - 你已定义：PositionalEncoding, SimbaBlock（见你上文代码）
#   - DGL 已安装；batch 输入的 g_lig/g_prot 已含 ndata['x'], edata['w']
#   - 3D 输入 lig3d/poc3d 来自 Uni-Mol2（如 512 维）
# =========================================================

import math                                 # 行：数学函数（sqrt/exp 等）
import torch                                # 行：PyTorch 主库
import torch.nn as nn                       # 行：神经网络模块
import torch.nn.functional as F             # 行：常见激活/函数
import dgl                                  # 行：DGL 图框架（批图）
from einops import einsum, rearrange, repeat
import dgl.nn as dglnn  # 行：补充：用 dglnn 别名调用 GraphConv
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    行：正弦位置编码（Sinusoidal PE）
    行：与常见实现不同，此版本 **不预先固定 max_len**，而是在 forward 里
    行：根据当前输入序列长度 L 动态生成编码，彻底避免 “L > max_len” 报错。
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)  # 行：位置编码后做一次 dropout（常规）
        self.d_model = d_model               # 行：保存通道维 D（用于计算频率因子）
        self.max_len = max_len               # 行：保留该参数仅为兼容旧接口（不再强依赖）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        行：x 形状为 [L, B, D]（标准 Transformer 输入的时间维/批维/通道维）
        行：返回同形状张量，叠加上正弦/余弦位置编码后再做 dropout。
        """
        L, B, D = x.size()                   # 行：取当前序列长度 L、批大小 B、通道维 D
        device, dtype = x.device, x.dtype    # 行：确保生成的 PE 与 x 同设备/数据类型

        # —— 生成正弦/余弦基 ——
        position = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)  # 行：[L,1] 位置索引
        div_term = torch.exp(                                                   # 行：频率衰减项（按论文公式）
            torch.arange(0, D, 2, device=device, dtype=dtype) *
            (-math.log(10000.0) / D)
        )  # 行：[D/2]

        # —— 组装 PE（偶数维用 sin，奇数维用 cos）——
        pe = torch.zeros(L, D, device=device, dtype=dtype)  # 行：[L,D] 占位
        pe[:, 0::2] = torch.sin(position * div_term)        # 行：填充偶数下标维度
        pe[:, 1::2] = torch.cos(position * div_term)        # 行：填充奇数下标维度

        # —— 广播到批维并相加 ——
        x = x + pe.unsqueeze(1)                             # 行：pe->[L,1,D]，与 x 相加得到 [L,B,D]
        return self.dropout(x)                              # 行：返回加过 PE 并做过 dropout 的张量



class Mamba(nn.Module):
    def __init__(self, d_model, d_state=64, expand=2, d_conv=4, conv_bias=True, bias=False):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.d_model = d_model  # Model dimension d_model
        self.d_state = d_state  # SSM state expansion factor
        self.d_conv = d_conv  # Local convolution width
        self.expand = expand  # Block expansion factor
        self.conv_bias = conv_bias
        self.bias = bias
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=self.bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=self.conv_bias,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=self.bias)

    def forward(self, x):
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')

        x = F.silu(x)

        y = self.ssm(x)

        y = y * F.silu(res)

        output = self.out_proj(y)
        return output

    def ssm(self, x):
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        # # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # # is additionally hardware-aware (like FlashAttention).
        # x = torch.zeros((b, d_in, n), device=deltaA.device)
        # ys = []
        # for i in range(l):
        #     x = deltaA[:, i] * x + deltaB_u[:, i]
        #     y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
        #     ys.append(y)
        # y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        # Convert sequential scan to cumulative sum (parallel-friendly operation)
        # Using cumsum to emulate sequential behavior in a parallel way
        cumulative_A = torch.cumsum(deltaA, dim=1)  # (b, l, d_in, n)
        cumulative_B = torch.cumsum(deltaB_u, dim=1)  # (b, l, d_in, n)

        # Combining the results with einsum
        y = einsum(cumulative_A, C, 'b l d_in n, b l n -> b l d_in')  # (b, l, d_in)

        y = y + u * D

        return y


class EinFFT(nn.Module):
    def __init__(self, dim: int, sequence_length: Optional[int] = None):
        super().__init__()                                # 行：父类初始化
        self.dim = dim                                    # 行：通道维 D
        self.sequence_length = sequence_length            # 行：可选的“期望长度 S0”
        self.act = nn.SiLU()                              # 行：SwiSH 激活，数值更稳

        S = 1 if sequence_length is None else sequence_length  # 行：未给长度则用 1，前向时广播

        # —— 全部“实数参数”，AMP 更安全 —— #
        self.w_cr = nn.Parameter(torch.randn(S, dim) * 0.02)  # 行：复权重实部（S×D）
        self.w_ci = nn.Parameter(torch.randn(S, dim) * 0.02)  # 行：复权重虚部（S×D）
        self.w_r  = nn.Parameter(torch.randn(S, dim) * 0.02)  # 行：实部分支缩放

    @staticmethod
    def _match_len(w: torch.Tensor, S: int) -> torch.Tensor:
        """行：把参数第一维从 S0 对齐到目标 S；S0=1 广播，S0≠S 线性插值。"""
        S0, D = w.shape                                   # 行：取当前长度/通道
        if S0 == S:                                       # 行：一致→原样返回
            return w
        if S0 == 1:                                       # 行：常数核→广播
            return w.expand(S, D)
        # —— 线性插值到 S —— #
        w_bc = w.permute(1, 0).unsqueeze(0)               # 行：[S0,D]→[1,D,S0]
        w_rs = F.interpolate(w_bc, size=S, mode='linear', align_corners=False)  # 行：插值
        return w_rs.squeeze(0).permute(1, 0)              # 行：还原到 [S,D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, S, D] 实数；return: [B, S, D] 实数
        """
        B, S, D = x.shape                                 # 行：批/序列/通道
        X = torch.fft.fft(x, dim=-2)                      # 行：沿 S 做 FFT（复数）
        xr, xi = X.real, X.imag                           # 行：拆实/虚

        # 权重对齐到当前 S
        w_cr = self._match_len(self.w_cr, S)              # 行：[S,D]
        w_ci = self._match_len(self.w_ci, S)              # 行：[S,D]
        w_r  = self._match_len(self.w_r,  S)              # 行：[S,D]

        # 复乘：(xr+i*xi)*(w_cr+i*w_ci)
        y_r = xr * w_cr - xi * w_ci                       # 行：实部
        y_i = xr * w_ci + xi * w_cr                       # 行：虚部

        # 非线性
        y_r = self.act(y_r)                               # 行：实部激活
        y_i = self.act(y_i)                               # 行：虚部激活

        # 实部分支再调制
        y_r = y_r * w_r                                   # 行：逐元素缩放

        # IFFT 并取实部
        Y = torch.complex(y_r, y_i)                       # 行：组装复数
        out = torch.fft.ifft(Y, dim=-2).real              # 行：回到时域
        return out                                        # 行：输出 [B,S,D]


class SimbaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dropout: float = 0.1,
        d_state: int = 32,
        d_conv: int = 32,
    ):
        super().__init__()                                # 行：父类初始化
        self.dim = dim                                    # 行：通道维
        self.layer_norm = nn.LayerNorm(dim)               # 行：LN 稳定数值
        self.dropout = nn.Dropout(dropout)                # 行：Dropout

        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv)  # 行：Mamba 模块
        self.einfft = EinFFT(dim=dim)                     # 行：频域块（已兼容任意 S）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x                                      # 行：残差
        x = self.layer_norm(x)                            # 行：LN
        x = self.mamba(x)                                 # 行：Mamba
        x = self.dropout(x)                               # 行：Dropout
        x = x + residual                                  # 行：残差叠加

        residual = x                                      # 行：第二段残差
        x = self.layer_norm(x)                            # 行：LN
        x = self.einfft(x)                                # 行：频域变换
        x = self.dropout(x)                               # 行：Dropout
        x = x + residual                                  # 行：残差叠加
        return x                                          # 行：输出




# ======【2D】带边权的 GCN（EW-GCN），用于药物图与蛋白图统一编码 ======
class GraphConvEW(nn.Module):
    """行：三层 GraphConv（DGL），带边权；显式分段读出，保证输出 [B, out_dim]。"""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()                                                      # 行：父类初始化
        self.gcn1 = dgl.nn.GraphConv(in_dim, hidden_dim, bias=False, allow_zero_in_degree=True)   # 行：GCN-1
        self.gcn2 = dgl.nn.GraphConv(hidden_dim, hidden_dim, bias=False, allow_zero_in_degree=True) # 行：GCN-2
        self.gcn3 = dgl.nn.GraphConv(hidden_dim, out_dim, bias=False, allow_zero_in_degree=True)    # 行：GCN-3
        self.ln1  = nn.LayerNorm(hidden_dim)                                    # 行：LN-1
        self.ln2  = nn.LayerNorm(hidden_dim)                                    # 行：LN-2
        self.act  = nn.ReLU(inplace=True)                                       # 行：激活
        self.dropout = nn.Dropout(0.1)                                          # 行：Dropout

    @staticmethod
    def _edge_feat_to_scalar(w: Optional[torch.Tensor], g: dgl.DGLGraph) -> Optional[torch.Tensor]:
        """行：把 [E,Fe]/[E] 边特征压成 [E] 标量，供 GraphConv 的 edge_weight 使用。"""
        if (w is None) or (w.numel() == 0):                                     # 行：无边权
            return None
        if w.dim() == 2:                                                        # 行：[E,Fe] → 通道均值
            w = w.mean(dim=-1)
        else:
            w = w.view(-1)                                                      # 行：兜底展平
        if w.shape[0] > g.num_edges():                                          # 行：防越界
            w = w[:g.num_edges()]
        return w

    @staticmethod
    def _segmented_sum_by_graph(g: dgl.DGLGraph, node_feat: torch.Tensor) -> torch.Tensor:
        """
        行：显式按子图分段求和（不依赖 dgl.sum_nodes 的批语义）。
        返回 [B, D]，B 为批内子图数（即 batch size）。
        """
        counts_fn = getattr(g, "batch_num_nodes", None)                          # 行：优先走 batch_num_nodes
        if callable(counts_fn):
            counts = counts_fn()                                                 # 行：Tensor 或 list
            if not torch.is_tensor(counts):
                counts = torch.as_tensor(counts, device=node_feat.device)
            else:
                counts = counts.to(node_feat.device)
            if counts.dim() == 0:                                               # 行：单图情况
                counts = counts.view(1)
            offsets = torch.zeros((counts.numel() + 1,), dtype=torch.long, device=node_feat.device)
            offsets[1:] = torch.cumsum(counts, dim=0)
            outs = []
            for i in range(counts.numel()):
                seg = node_feat[offsets[i]:offsets[i+1]]                        # 行：该子图的节点切片
                outs.append(seg.sum(dim=0, keepdim=True))                       # 行：[1, D]
            return torch.cat(outs, dim=0)                                       # 行：[B, D]

        # 兜底：用 unbatch（略慢但通用）
        graphs = dgl.unbatch(g)
        outs = []
        start = 0
        for gi in graphs:
            n = gi.num_nodes()
            outs.append(node_feat[start:start+n].sum(dim=0, keepdim=True))
            start += n
        return torch.cat(outs, dim=0)                                           # 行：[B, D]

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor, w: Optional[torch.Tensor]):
        """行：输入批图 g，节点特征 x、边特征 w；输出每图读出向量 [B, out_dim]。"""
        device = next(self.parameters()).device  # 行：以模块参数所在设备为目标设备

        # —— 先把边权压成标量（在 CPU/原图上统计边数是安全的）——
        w_scalar = self._edge_feat_to_scalar(w, g)  # 行：w->[E] 或 None

        # —— 再把三样东西统一搬到同一设备（关键修复点）——
        g = g.to(device)  # ✅ 行：图迁移到目标设备
        x = x.to(device)  # ✅ 行：节点特征迁移
        if w_scalar is not None:
            w_scalar = w_scalar.to(device)  # ✅ 行：边权迁移

        # —— 三层 GraphConv ——
        h = self.gcn1(g, x, edge_weight=w_scalar)  # 行：GCN-1（设备已对齐）
        h = self.ln1(self.act(h))  # 行：LN+ReLU
        h = self.dropout(h)  # 行：Dropout

        h = self.gcn2(g, h, edge_weight=w_scalar)  # 行：GCN-2
        h = self.ln2(self.act(h))  # 行：LN+ReLU
        h = self.dropout(h)  # 行：Dropout

        h = self.gcn3(g, h, edge_weight=w_scalar)  # 行：GCN-3 → 节点级特征

        # —— 显式分段读出，保证 [B, out_dim] ——
        readout = self._segmented_sum_by_graph(g, h)  # 行：按子图求和
        return readout


# ======【1D】序列编码器（薄封装调用你已有的 SimbaBlock/Mamba/EinfFFT） ======

class SequenceEncoder1D(nn.Module):
    """行：对药物/蛋白序列做嵌入+位置编码+若干 SimbaBlock，池化得到定长向量。"""
    def __init__(self,
                 vocab_size: int,               # 行：词表大小（药物默认 62；蛋白默认 25）
                 emb_dim: int = 128,            # 行：嵌入维度 / SimbaBlock 通道维
                 num_layers: int = 2,           # 行：SimbaBlock 层数
                 max_len: int = 2048,           # 行：最大序列长度（仅用于位置编码 buffer）
                 pool: str = "sum"              # 行：池化方式：'sum' | 'mean'
                 ):
        super().__init__()                                                           # 行：父类初始化
        self.emb = nn.Embedding(vocab_size, emb_dim)                                 # 行：离散 token → 连续向量
        self.pos = PositionalEncoding(d_model=emb_dim, dropout=0.1, max_len=max_len) # 行：绝对位置编码（正余弦）

        self.blocks = nn.ModuleList([
            SimbaBlock(dim=emb_dim, dropout=0.1, d_state=8, d_conv=8)  # 行：不再传 num_classes
            for _ in range(num_layers)
        ])

        self.pool = pool                                                             # 行：记录池化策略
        self.norm = nn.LayerNorm(emb_dim)                                            # 行：最后做个层归一化，稳定数值

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        行：tokens [B,L]（已编码好的索引）
        return: [B, emb_dim]（池化后的序列向量）
        """
        x = self.emb(tokens)                                                          # 行：查表得到 [B,L,D]
        x = self.pos(x.transpose(0, 1)).transpose(0, 1)                               # 行：加位置编码，注意 PE 接口是 [L,B,D]
        for blk in self.blocks:                                                       # 行：逐层 SimbaBlock
            x = blk(x)                                                                # 行：保持形状 [B,L,D]
        if self.pool == "sum":                                                        # 行：按需池化
            x = x.sum(dim=1)                                                          # 行：sum 池化（与你原先一致）
        else:
            x = x.mean(dim=1)                                                         # 行：mean 池化（可选）
        x = self.norm(x)                                                              # 行：层归一化
        return x                                                                      # 行：返回 [B,D]


# ======【3D】Uni-Mol2 向量对编码器（拼接 + 可选交互 → 降维到 d_model） ======

class Pair3DEncoder(nn.Module):
    """行：把 lig3d/poc3d 两路向量拼成一个“3D对”特征，并降维到统一 d_model。"""
    def __init__(self, d_in_lig: int = 512, d_in_poc: int = 512,
                 d_model: int = 256, add_interactions: bool = True):
        super().__init__()                                                           # 行：父类初始化
        self.add_interactions = add_interactions                                     # 行：是否加入交互项
        self.norm_l = nn.LayerNorm(d_in_lig)                                         # 行：对输入做 LN 增稳
        self.norm_p = nn.LayerNorm(d_in_poc)
        self.proj_l = nn.Linear(d_in_lig, d_model, bias=False)                       # 行：线性投影到 d_model
        self.proj_p = nn.Linear(d_in_poc, d_model, bias=False)
        in_dim = d_model * (4 if add_interactions else 2)                            # 行：决定后续 MLP 输入维
        self.mlp = nn.Sequential(                                                    # 行：小 MLP 把拼接+交互压到 d_model
            nn.Linear(in_dim, d_model, bias=False),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, lig3d: torch.Tensor, poc3d: torch.Tensor) -> torch.Tensor:
        """行：输入 [B,Dl] 与 [B,Dp]；输出 [B,d_model] 的 3D 对表示"""
        l = self.norm_l(lig3d.float())                                               # 行：LayerNorm + 浮点
        p = self.norm_p(poc3d.float())
        l = self.proj_l(l)                                                           # 行：投影到 d_model
        p = self.proj_p(p)
        if self.add_interactions:                                                    # 行：可选交互（Hadamard + |diff|）
            feat = torch.cat([l, p, l * p, torch.abs(l - p)], dim=-1)                # 行：[B, 4*d_model]
        else:
            feat = torch.cat([l, p], dim=-1)                                         # 行：[B, 2*d_model]
        out = self.mlp(feat)                                                         # 行：降维到 [B,d_model]
        return out                                                                   # 行：返回 3D 对特征


# ====== Cross-Attention（把 1D+2D 的融合向量作为 Q，3D 对向量作为 K/V） ======

class CrossAttentionFuse(nn.Module):
    """行：标准多头交叉注意力 + 前馈，两层残差（Transformer Encoder 风格）。"""
    def __init__(self, dim_q: int, dim_kv: int, d_attn: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()                                                           # 行：父类初始化
        self.q_proj = nn.Linear(dim_q,  d_attn, bias=False)                          # 行：把 Q 投影到注意力维
        self.k_proj = nn.Linear(dim_kv, d_attn, bias=False)                          # 行：把 K 投影到注意力维
        self.v_proj = nn.Linear(dim_kv, d_attn, bias=False)                          # 行：把 V 投影到注意力维
        self.mha    = nn.MultiheadAttention(embed_dim=d_attn, num_heads=n_heads,
                                            dropout=dropout, batch_first=True)       # 行：PyTorch 自带 MHA（batch_first）
        self.ln1 = nn.LayerNorm(d_attn)                                              # 行：注意力后 LN
        self.ff  = nn.Sequential(                                                    # 行：前馈（两层 MLP）
            nn.Linear(d_attn, 2 * d_attn, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_attn, d_attn, bias=False),
        )
        self.ln2 = nn.LayerNorm(d_attn)                                              # 行：前馈后 LN

    def forward(self, q_vec: torch.Tensor, kv_vec: torch.Tensor) -> torch.Tensor:
        """
        行：q_vec [B, Dq]（1D+2D 拼接后向量）
            kv_vec [B, Dk]（3D 对向量）
        return:      [B, d_attn]（融合后的表示）
        """
        q = self.q_proj(q_vec).unsqueeze(1)                                          # 行：投影并扩成序列长度 1 → [B,1,D]
        k = self.k_proj(kv_vec).unsqueeze(1)                                         # 行：同上
        v = self.v_proj(kv_vec).unsqueeze(1)                                         # 行：同上
        attn_out, _ = self.mha(q, k, v, need_weights=False)                          # 行：多头注意力
        y = self.ln1(attn_out + q)                                                   # 行：残差 + LN
        y2 = self.ff(y)                                                              # 行：前馈
        y = self.ln2(y + y2)                                                         # 行：残差 + LN
        return y.squeeze(1)                                                          # 行：去掉 seq 维，得 [B,d_attn]


# ====== 总模型：把 1D/2D/3D 串起来并回归 pKd ======

class MultiModalDTA(nn.Module):
    """
    行：多模态 DTA 主干
      - 1D：SimbaBlock 版序列编码（药物/蛋白），拼接 -> 线性融合到 d_model
      - 2D：EW-GCN（药物图/蛋白图），拼接 -> 线性融合到 d_model
      - 3D：Uni-Mol2（药物/口袋）→ pair 编码成 d_model
      - Cross-Attn：以 cat(1D_pair, 2D_pair) 为 Q；以 3D_pair 为 K/V
      - 回归头：拼接 [CA输出, 1D_pair, 2D_pair, 3D_pair] → MLP → 标量
    """
    def __init__(self,
                 # 1D 超参（与你之前保持一致）
                 drug_vocab: int = 62, target_vocab: int = 25, emb_dim_1d: int = 128, seq_layers: int = 2,
                 seq_pool: str = "sum",
                 # 2D 输入维（与你数据对齐：药物节点70/边6；蛋白节点33/边3）
                 drug_node_dim: int = 70, drug_edge_dim: int = 6,
                 prot_node_dim: int = 33, prot_edge_dim: int = 3,
                 gcn_hidden: int = 128, gcn_out: int = 128,
                 # 3D 输入维（Uni-Mol2）
                 d3_lig: int = 512, d3_poc: int = 512,
                 # 统一融合维与注意力
                 d_model: int = 256, d_attn: int = 256, n_heads: int = 4,
                 add_interactions_3d: bool = True):
        super().__init__()                                                           # 行：父类初始化

        # ---- 1D：两个序列编码器（药物/蛋白） ----
        self.enc1d_drug = SequenceEncoder1D(vocab_size=drug_vocab,   emb_dim=emb_dim_1d,
                                            num_layers=seq_layers,   pool=seq_pool)  # 行：药物序列编码
        self.enc1d_prot = SequenceEncoder1D(vocab_size=target_vocab, emb_dim=emb_dim_1d,
                                            num_layers=seq_layers,   pool=seq_pool)  # 行：蛋白序列编码
        self.fuse1d = nn.Sequential(                                                 # 行：把 [drug1d||prot1d] 融合到 d_model
            nn.Linear(2 * emb_dim_1d, d_model, bias=False),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # ---- 2D：两路 EW-GCN（药物图/蛋白图），图级读出向量 ----
        self.enc2d_drug = GraphConvEW(in_dim=drug_node_dim, hidden_dim=gcn_hidden, out_dim=gcn_out)  # 行：药物图编码
        self.enc2d_prot = GraphConvEW(in_dim=prot_node_dim, hidden_dim=gcn_hidden, out_dim=gcn_out)  # 行：蛋白图编码
        self.fuse2d = nn.Sequential(                                                 # 行：把 [drug2d||prot2d] 融合到 d_model
            nn.Linear(2 * gcn_out, d_model, bias=False),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # ---- 3D：pair 编码（药物/口袋 → d_model） ----
        self.enc3d_pair = Pair3DEncoder(d_in_lig=d3_lig, d_in_poc=d3_poc, d_model=d_model,
                                        add_interactions=add_interactions_3d)        # 行：3D 对编码器

        # ---- Cross-Attention：Q=cat(1D_pair, 2D_pair)，KV=3D_pair ----
        self.cross = CrossAttentionFuse(dim_q=2 * d_model, dim_kv=d_model, d_attn=d_attn, n_heads=n_heads)  # 行：CA 模块

        # ---- 回归头：拼接 CA 输出与各路 pair，稳健预测 ----
        final_in = d_attn + 3 * d_model                                              # 行：拼接 [CA, 1D, 2D, 3D]
        self.reg_head = nn.Sequential(
            nn.Linear(final_in, 4 * d_model, bias=False),                            # 行：放大到 4*d_model
            nn.LayerNorm(4 * d_model),                                               # 行：层归一化
            nn.GELU(),                                                               # 行：激活
            nn.Dropout(0.1),                                                         # 行：Dropout
            nn.Linear(4 * d_model, 2 * d_model, bias=False),                         # 行：再降到 2*d_model
            nn.GELU(),                                                               # 行：激活
            nn.Dropout(0.1),                                                         # 行：Dropout
            nn.Linear(2 * d_model, 1, bias=True)                                     # 行：输出标量 pKd
        )

    def forward(self,
                drug_seq: torch.Tensor,            # 行：[B, Ld] 药物序列（token 索引）
                prot_seq: torch.Tensor,            # 行：[B, Lt] 蛋白序列（token 索引）
                g_lig: dgl.DGLGraph,               # 行：批图：配体；需含 ndata['x'], edata['w']
                g_prot: dgl.DGLGraph,              # 行：批图：蛋白；需含 ndata['x'], edata['w']
                lig3d: torch.Tensor,               # 行：[B, Dl] Uni-Mol2 配体向量
                poc3d: torch.Tensor                # 行：[B, Dp] Uni-Mol2 口袋向量
                ) -> torch.Tensor:
        # === 1) 1D：两路序列编码并拼接 ===
        drug_1d = self.enc1d_drug(drug_seq)                                           # 行：[B, D1]
        prot_1d = self.enc1d_prot(prot_seq)                                           # 行：[B, D1]
        pair_1d = torch.cat([drug_1d, prot_1d], dim=-1)                               # 行：[B, 2*D1]
        pair_1d = self.fuse1d(pair_1d)                                                # 行：→ [B, d_model]

        # === 2) 2D：两路图编码并拼接 ===
        lig_vec  = self.enc2d_drug(g_lig,  g_lig.ndata['x'].float(),  g_lig.edata['w'].float())     # 行：[B, gcn_out]
        prot_vec = self.enc2d_prot(g_prot, g_prot.ndata['x'].float(), g_prot.edata['w'].float())    # 行：[B, gcn_out]
        # —— 形状守卫：确保与 1D 的 batch 一致 ——
        B = drug_seq.size(0)  # 行：当前批大小
        assert lig_vec.size(0) == B and prot_vec.size(0) == B, \
            f"2D batch mismatch: lig={lig_vec.size(0)}, prot={prot_vec.size(0)}, expected {B}"

        pair_2d  = torch.cat([lig_vec, prot_vec], dim=-1)                                           # 行：[B, 2*gcn_out]
        pair_2d  = self.fuse2d(pair_2d)                                                             # 行：→ [B, d_model]

        # === 3) 3D：药物/口袋 → pair 编码 ===
        # === 预处理：对 3D/图特征做一次数值兜底（更稳） ===
        lig3d = torch.nan_to_num(lig3d, nan=0.0, posinf=1e6, neginf=-1e6)      # 行：3D 向量兜底
        poc3d  = torch.nan_to_num(poc3d,  nan=0.0, posinf=1e6, neginf=-1e6)
        pair_3d = self.enc3d_pair(lig3d, poc3d)                                                     # 行：[B, d_model]

        # === 4) Cross-Attention：Q=cat(1D_pair, 2D_pair)，KV=3D_pair ===
        q_12 = torch.cat([pair_1d, pair_2d], dim=-1)                                                # 行：[B, 2*d_model]
        ca   = self.cross(q_12, pair_3d)                                                            # 行：[B, d_attn]

        # === 5) 回归头：拼接 CA 与三路 pair，输出标量 ===
        feat_final = torch.cat([ca, pair_1d, pair_2d, pair_3d], dim=-1)                             # 行：[B, d_attn+3*d_model]
        y = self.reg_head(feat_final).squeeze(-1)                                                   # 行：[B]
        return y                                                                                    # 行：返回 pKd 预测
