def LoadData_atom3d_si30_multimodal_lm_token(
    root_base: str,                                              # 行：指向 .../split-by-sequence-identity-30/data
    split: str = "train",                                        # 行：'train' | 'val' | 'test' | 'all'
    out_mm: str = "../dataset/ATOM3D/processed_mm_si30_lm_token", # 行：token 版缓存目录，避免覆盖旧 pooled 缓存
    unimol2_size: str = "unimol2_small",                         # 行：Uni-Mol2 模型规格，保持你原来的写法
    # 下面这些 2D 相关参数保留占位，避免外部旧调用报错
    contact_threshold: float = 8.0,
    dis_min: float = 1.0,
    prot_self_loop: bool = False,
    bond_bidirectional: bool = True,
    prefer_model: int = None,
    force_refresh: bool = False,                                  # 行：是否忽略缓存并强制重建
    use_cuda_for_unimol: bool = True,                             # 行：Uni-Mol2 是否使用 GPU
    # --- LM 模型相关 ---
    esm2_model_name: str = "facebook/esm2_t33_650M_UR50D",         # 行：ESM-2 checkpoint
    chemberta_model_name: str = "DeepChem/ChemBERTa-77M-MTR",      # 行：ChemBERTa checkpoint
    lm_batch_size: int = 8,                                        # 行：LM token 前向 batch_size，比 pooled 版更吃显存
    chem_max_len: int = 128,                                       # 行：SMILES token 最大长度
    prot_max_len: int = 1024,                                      # 行：蛋白 token 最大长度；显存不够可先改 512
    lig_max_atoms: int = 128,                                      # 行：配体最多保留多少个原子 token
    pocket_max_atoms: int = 256,                                   # 行：口袋最多保留多少个原子 token
    use_safetensors: bool = True,                                  # 行：是否优先使用 safetensors
    mask_special_tokens: bool = True,                              # 行：是否在 attention 中屏蔽 CLS/SEP/BOS/EOS 等特殊 token
    allow_pooled_3d_fallback: bool = False                         # 行：若 Uni-Mol2 原子级表示失败，是否允许退化为 1 个 pooled token
) -> dict:
    """
    ATOM3D SI-30 token-level 多模态数据加载函数。

    与旧版 LoadData_atom3d_si30_multimodal_lm 的区别：
      1. 1D 不再保存 CLS / pooled 向量，而是保存 token-level last_hidden_state：
            drug_lm_tokens : [N, chem_max_len, D_drug_lm]
            drug_lm_mask   : [N, chem_max_len]，True 表示 padding/special token，需要在 attention 中忽略

            prot_lm_tokens : [N, prot_max_len, D_prot_lm]
            prot_lm_mask   : [N, prot_max_len]，True 表示 padding/special token，需要在 attention 中忽略

      2. 3D 不再保存单个 pooled 向量，而是尽量保存 Uni-Mol2 atom-level representations：
            lig_3d_tokens  : [N, lig_max_atoms, D_3d]
            lig_3d_mask    : [N, lig_max_atoms]

            poc_3d_tokens  : [N, pocket_max_atoms, D_3d]
            poc_3d_mask    : [N, pocket_max_atoms]

      3. 多链蛋白处理：
            1D 分支使用 ATOM3D 样本中提供的 sample["seq"] 作为全蛋白序列；
            不再使用 longest_only=True，也不再任意保留最长链；
            若 sample["seq"] 缺失或为空，则从 atoms_protein 的所有链中重建完整蛋白序列作为兜底。

    返回：
        {split: {
            'ids', 'y', 'smiles', 'seq', 'seq_chain_policy',
            'drug_lm_tokens', 'drug_lm_mask',
            'prot_lm_tokens', 'prot_lm_mask',
            'lig_3d_tokens', 'lig_3d_mask',
            'poc_3d_tokens', 'poc_3d_mask',
            'g_lig', 'g_prot'
        }}
    """

    # =========================
    # 依赖导入
    # =========================
    from pathlib import Path
    import numpy as np
    import torch
    from tqdm import tqdm
    import atom3d.datasets as da
    from unimol_tools import UniMolRepr
    from transformers import AutoTokenizer, AutoModel

    # 注意：
    # 这个函数是放在 util.py 里的，所以这里直接调用 util.py 里已有的 atoms_to_sequence。
    # 不要再写 from util import atoms_to_sequence，避免在 util.py 内部自导入造成潜在循环问题。

    # =========================
    # 原子符号到原子序号的兜底表
    # =========================
    _PERIODIC = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
        'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
        'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
        'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54
    }

    def _to_atomic_numbers(sym_list):
        """把原子符号列表转成原子序号列表，供部分 Uni-Mol 版本兜底使用。"""
        return [int(_PERIODIC.get(str(s), 0)) for s in sym_list]

    def _as_numpy(x):
        """把 torch.Tensor / list / np.ndarray 统一转成 numpy.float32。"""
        if torch.is_tensor(x):
            return x.detach().cpu().numpy().astype(np.float32)
        return np.asarray(x, dtype=np.float32)

    def _detect_chain_col(df):
        """
        自动判断 DataFrame 中哪个字段是 chain id。
        不同 ATOM3D / PDB 解析版本字段名可能不完全一致，所以这里做兼容。
        """
        candidates = [
            "chain", "chain_id", "chainID", "chain_name",
            "asym_id", "auth_asym_id", "label_asym_id"
        ]
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def _unique_nonempty_values(df, col):
        """提取某个列中的非空唯一值，并统一转成字符串。"""
        if col is None or col not in df.columns:
            return []
        vals = []
        for v in df[col].tolist():
            if v is None:
                continue
            s = str(v).strip()
            if s == "" or s.lower() == "nan":
                continue
            vals.append(s)
        return sorted(set(vals))

    # =========================
    # 三字母氨基酸转一字母氨基酸
    # =========================
    _AA3_TO_1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",

        # 常见修饰残基
        "MSE": "M", "SEC": "C", "PYL": "K",
        "HSD": "H", "HSE": "H", "HSP": "H",
        "HID": "H", "HIE": "H", "HIP": "H",
        "CYX": "C", "CME": "C", "CSO": "C",
        "SEP": "S", "TPO": "T", "PTR": "Y",
    }

    def _clean_protein_sequence(seq):
        """
        清洗蛋白序列。

        输入可能是：
            1. 字符串
            2. list / tuple
            3. dict，例如多链序列

        输出：
            只保留标准氨基酸字符的字符串。
        """
        if seq is None:
            return ""

        if isinstance(seq, dict):
            parts = []

            for k in sorted(seq.keys()):
                v = seq[k]

                if v is None:
                    continue

                parts.append(str(v))

            seq = "".join(parts)

        elif isinstance(seq, (list, tuple)):
            seq = "".join([str(x) for x in seq if x is not None])

        else:
            seq = str(seq)

        seq = seq.upper()

        valid = set("ACDEFGHIKLMNPQRSTVWY")

        seq = "".join([c for c in seq if c in valid])

        return seq

    def _atoms_df_to_full_sequence_direct(df_prot):
        """
        从 atoms_protein 的所有链中重建完整蛋白序列。

        这是 sample["seq"] 缺失时的兜底方案。
        注意：
            这里使用所有 protein chains；
            不使用 pocket chain 过滤；
            不使用 longest_only=True。
        """
        if df_prot is None or len(df_prot) == 0:
            return ""

        required_cols = ["chain", "residue", "resname"]

        for col in required_cols:
            if col not in df_prot.columns:
                print(f"[SEQ-ERROR] missing column: {col}")
                print("[SEQ-ERROR] available columns:", list(df_prot.columns))
                return ""

        tmp = df_prot.copy()

        tmp["chain"] = tmp["chain"].astype(str)

        tmp["resname"] = tmp["resname"].astype(str).str.upper()

        tmp["_order"] = np.arange(len(tmp))

        if "insertion_code" in tmp.columns:
            tmp["insertion_code"] = tmp["insertion_code"].fillna("").astype(str)

            residue_df = tmp.drop_duplicates(
                subset=["chain", "residue", "insertion_code"],
                keep="first"
            ).copy()

            residue_df = residue_df.sort_values(
                by=["chain", "residue", "insertion_code", "_order"]
            )
        else:
            residue_df = tmp.drop_duplicates(
                subset=["chain", "residue"],
                keep="first"
            ).copy()

            residue_df = residue_df.sort_values(
                by=["chain", "residue", "_order"]
            )

        seq_chars = []

        for aa3 in residue_df["resname"].tolist():
            aa1 = _AA3_TO_1.get(aa3, "")

            if aa1:
                seq_chars.append(aa1)

        return "".join(seq_chars)

    def _get_full_protein_sequence(sample, df_prot):
        """
        获取 1D 分支使用的全蛋白序列。

        优先级：
            1. 优先使用 ATOM3D 样本自带的 sample["seq"]。
            2. 如果 sample["seq"] 为空，则从 atoms_protein 的所有链重建序列。

        返回：
            seq_full: 全蛋白序列
            policy:   记录序列来源，方便写日志和回复审稿人
        """
        seq_from_sample = _clean_protein_sequence(sample.get("seq", None))

        if seq_from_sample:
            return seq_from_sample, "full_protein_sequence_from_sample_seq"

        seq_from_atoms = _atoms_df_to_full_sequence_direct(df_prot)

        if seq_from_atoms:
            return seq_from_atoms, "full_protein_sequence_rebuilt_from_atoms_protein_all_chains"

        return "", "failed_to_extract_full_protein_sequence"

    def _hf_load_model(name: str, device: torch.device, use_safetensors: bool = True):
        """加载 HuggingFace AutoModel，并兼容 use_safetensors 参数。"""
        try:
            model = AutoModel.from_pretrained(name, use_safetensors=use_safetensors)
            model.to(device)
            return model
        except TypeError:
            model = AutoModel.from_pretrained(name)
            model.to(device)
            return model
        except Exception as e:
            msg = str(e).lower()

            if ("torch" in msg and "2.6" in msg) or ("vulnerability" in msg) or ("cve" in msg):
                raise RuntimeError(
                    f"Loading HF model '{name}' failed due to transformers / torch safety check:\n{e}\n"
                    "Possible solutions:\n"
                    "  1) 优先使用 safetensors；\n"
                    "  2) 升级 torch 到 >=2.6；\n"
                    "  3) 使用兼容版本 transformers。"
                ) from e

            try:
                model = AutoModel.from_pretrained(name)
                model.to(device)
                return model
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load HF model '{name}'. "
                    f"Err1={e} Err2={e2}"
                ) from e2

    def _encode_text_tokens(
        text_list,
        tokenizer,
        model,
        device,
        batch_size=8,
        max_length=512,
        desc="[LM] token encoding",
        mask_special_tokens=True
    ):
        """
        抽取 token-level LM embedding。

        输入：
            text_list: list[str]

        输出：
            token_embeds: [N, max_length, D]
            pad_mask:     [N, max_length]，True 表示需要被 attention 忽略
        """
        all_tokens = []
        all_masks = []

        model.eval()

        with torch.no_grad():
            for i in tqdm(
                range(0, len(text_list), batch_size),
                desc=desc,
                unit="batch"
            ):
                batch = [str(x) for x in text_list[i:i + batch_size]]

                enc = tokenizer(
                    batch,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                    return_special_tokens_mask=True
                )

                # special_tokens_mask 是 tokenizer 的辅助输出，只能用于构造 mask
                # 不能传入 model(**enc)
                special_tokens_mask = enc.pop("special_tokens_mask", None)

                enc = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in enc.items()
                }

                if special_tokens_mask is not None:
                    special_tokens_mask = special_tokens_mask.to(device)

                try:
                    out = model(**enc)
                except TypeError as e:
                    # 某些模型不接受 token_type_ids，自动删除后重试
                    if "token_type_ids" in str(e) and "token_type_ids" in enc:
                        enc.pop("token_type_ids")
                        out = model(**enc)
                    else:
                        raise e

                if hasattr(out, "last_hidden_state"):
                    hs = out.last_hidden_state
                elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                    hs = out[0]
                else:
                    raise RuntimeError("Unsupported model output structure for token-level LM encoding.")

                attention_mask = enc["attention_mask"].bool()
                pad_mask = ~attention_mask

                if mask_special_tokens and special_tokens_mask is not None:
                    special_mask = special_tokens_mask.bool()
                    pad_mask = pad_mask | special_mask

                all_tokens.append(hs.detach().cpu().numpy().astype(np.float32))
                all_masks.append(pad_mask.detach().cpu().numpy().astype(bool))

        if len(all_tokens) == 0:
            hidden_size = int(getattr(model.config, "hidden_size", 0))
            return (
                np.zeros((0, max_length, hidden_size), dtype=np.float32),
                np.ones((0, max_length), dtype=bool)
            )

        token_embeds = np.concatenate(all_tokens, axis=0).astype(np.float32)
        token_masks = np.concatenate(all_masks, axis=0).astype(bool)

        return token_embeds, token_masks

    def _make_unimol2_repr():
        """
        初始化 Uni-Mol2 表征器。
        先按你旧代码的 model='unimol2' 写法尝试；
        如果当前 unimol_tools 版本使用 model_name='unimolv2'，则自动兜底。
        """
        try:
            return UniMolRepr(
                model="unimol2",
                model_size=unimol2_size,
                use_cuda=use_cuda_for_unimol
            )
        except TypeError:
            return UniMolRepr(
                model_name="unimolv2",
                model_size=unimol2_size,
                use_cuda=use_cuda_for_unimol
            )

    def _extract_atomic_reprs_from_output(repr_output, expected_len=None):
        """
        从 Uni-Mol get_repr(..., return_atomic_reprs=True) 的返回结果中取 atomic representations。

        兼容可能出现的几种结构：
            dict['atomic_reprs']
            dict['atomic_reprs'][0]
            list / tuple 包一层
            np.ndarray / torch.Tensor
        """
        obj = repr_output

        if isinstance(obj, dict):
            if "atomic_reprs" in obj:
                obj = obj["atomic_reprs"]
            elif "atomic_repr" in obj:
                obj = obj["atomic_repr"]
            elif "cls_repr" in obj and allow_pooled_3d_fallback:
                obj = obj["cls_repr"]
            else:
                raise RuntimeError(
                    f"Uni-Mol output does not contain atomic_reprs. Keys={list(obj.keys())}"
                )

        if isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                raise RuntimeError("Uni-Mol atomic_reprs is empty.")
            obj = obj[0]

        arr = _as_numpy(obj)

        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]

        if arr.ndim == 1:
            if allow_pooled_3d_fallback:
                arr = arr.reshape(1, -1)
            else:
                raise RuntimeError(
                    "Uni-Mol returned a pooled 1D vector instead of atom-level tokens. "
                    "Set allow_pooled_3d_fallback=True only for debugging, not for final revision."
                )

        if arr.ndim != 2:
            raise RuntimeError(f"Unexpected atomic representation shape: {arr.shape}")

        if expected_len is not None and expected_len > 0:
            if arr.shape[0] >= expected_len:
                arr = arr[:expected_len]
            else:
                # 如果返回 token 数少于原子数，保留返回值，但后续会 padding。
                arr = arr

        return arr.astype(np.float32)

    def _get_unimol_atomic_tokens(model3d, atoms, xyz, label="mol"):
        """
        用 Uni-Mol2 抽取原子级 token。
        优先使用原子符号；失败后用原子序号兜底。
        """
        data_symbol = {
            "atoms": atoms,
            "coordinates": xyz
        }

        try:
            repr_output = model3d.get_repr(
                data_symbol,
                return_atomic_reprs=True
            )
            return _extract_atomic_reprs_from_output(
                repr_output,
                expected_len=len(atoms)
            )
        except Exception as e_symbol:
            data_number = {
                "atoms": _to_atomic_numbers(atoms),
                "coordinates": xyz
            }

            try:
                repr_output = model3d.get_repr(
                    data_number,
                    return_atomic_reprs=True
                )
                return _extract_atomic_reprs_from_output(
                    repr_output,
                    expected_len=len(atoms)
                )
            except Exception as e_number:
                raise RuntimeError(
                    f"Failed to extract Uni-Mol atom-level tokens for {label}. "
                    f"symbol_error={e_symbol}; number_error={e_number}"
                ) from e_number

    def _pad_token_arrays(arr_list, max_len):
        """
        把不同长度的 token 序列补齐成固定长度。

        输入：
            arr_list: list，每个元素形状为 [Li, D]
            max_len: 固定 token 长度

        输出：
            out:  [N, max_len, D]
            mask: [N, max_len]，True 表示 padding，需要被 attention 忽略
        """
        if len(arr_list) == 0:
            return np.zeros((0, max_len, 0), dtype=np.float32), np.ones((0, max_len), dtype=bool)

        dim = int(arr_list[0].shape[-1])
        n = len(arr_list)

        out = np.zeros((n, max_len, dim), dtype=np.float32)
        mask = np.ones((n, max_len), dtype=bool)

        for i, x in enumerate(arr_list):
            x = np.asarray(x, dtype=np.float32)

            if x.ndim != 2:
                raise ValueError(f"Token array at index {i} must be [L,D], got shape={x.shape}")

            if x.shape[-1] != dim:
                raise ValueError(
                    f"Token dim mismatch at index {i}: expected {dim}, got {x.shape[-1]}"
                )

            L = min(x.shape[0], max_len)
            out[i, :L, :] = x[:L]
            mask[i, :L] = False

        return out, mask

    # =========================
    # 路径初始化
    # =========================
    root_base = Path(root_base)
    out_mm = Path(out_mm)
    out_mm.mkdir(parents=True, exist_ok=True)

    # =========================
    # 处理单个 split
    # =========================
    def _process_one_split(sp: str) -> dict:
        lmdb_root = root_base / sp

        p_1dnpz = out_mm / f"si30_{sp}_1d_lm_token.npz"
        p_3dnpz = out_mm / f"si30_{sp}_3d_token.npz"

        # =========================
        # 缓存直读
        # =========================
        if (not force_refresh) and p_1dnpz.exists() and p_3dnpz.exists():
            d1 = np.load(p_1dnpz, allow_pickle=True)
            d3 = np.load(p_3dnpz, allow_pickle=True)

            ids = d1["ids"]
            smiles = d1["smiles"]
            seq = d1["seq"]
            y = d1["pkd"].astype(np.float32)

            seq_chain_policy = d1["seq_chain_policy"] if "seq_chain_policy" in d1.files else np.array(["unknown"] * len(ids), dtype=object)

            drug_lm_tokens = d1["drug_lm_tokens"].astype(np.float32)
            drug_lm_mask = d1["drug_lm_mask"].astype(bool)

            prot_lm_tokens = d1["prot_lm_tokens"].astype(np.float32)
            prot_lm_mask = d1["prot_lm_mask"].astype(bool)

            lig_3d_tokens = d3["lig_3d_tokens"].astype(np.float32)
            lig_3d_mask = d3["lig_3d_mask"].astype(bool)

            poc_3d_tokens = d3["poc_3d_tokens"].astype(np.float32)
            poc_3d_mask = d3["poc_3d_mask"].astype(bool)

            N = len(ids)

            assert len(smiles) == N and len(seq) == N and len(y) == N, "1D 元信息长度不一致"
            assert drug_lm_tokens.shape[0] == N and prot_lm_tokens.shape[0] == N, "LM token 行数不一致"
            assert drug_lm_mask.shape[0] == N and prot_lm_mask.shape[0] == N, "LM mask 行数不一致"
            assert lig_3d_tokens.shape[0] == N and poc_3d_tokens.shape[0] == N, "3D token 行数不一致"
            assert lig_3d_mask.shape[0] == N and poc_3d_mask.shape[0] == N, "3D mask 行数不一致"

            return {
                "ids": ids,
                "y": y,
                "smiles": smiles,
                "seq": seq,
                "seq_chain_policy": seq_chain_policy,

                "drug_lm_tokens": drug_lm_tokens,
                "drug_lm_mask": drug_lm_mask,

                "prot_lm_tokens": prot_lm_tokens,
                "prot_lm_mask": prot_lm_mask,

                "lig_3d_tokens": lig_3d_tokens,
                "lig_3d_mask": lig_3d_mask,

                "poc_3d_tokens": poc_3d_tokens,
                "poc_3d_mask": poc_3d_mask,

                "g_lig": None,
                "g_prot": None,
            }

        # =========================
        # 正常构建
        # =========================
        ds = da.LMDBDataset(str(lmdb_root))
        print(f"[si-30|{sp}] LMDB samples = {len(ds)}")

        model3d = _make_unimol2_repr()
        model3d.model.eval()

        source_atoms = "atoms_protein"

        ids_list = []
        smiles_list = []
        seq_list = []
        seq_policy_list = []
        y_list = []

        lig_token_list = []
        poc_token_list = []

        n_total = 0
        n_ok = 0
        n_fail = 0

        breakdown = {
            "missing_field": 0,
            "smiles": 0,
            "protein_seq": 0,
            "empty3d": 0,
            "unimol3d": 0,
            "other": 0
        }

        DEBUG_PRINT_N = 8
        debug_shown = 0
        printed_backend = False

        for i, sample in enumerate(
            tqdm(ds, desc=f"[si-30|{sp}] token-level 1D+3D", unit="sample")
        ):
            n_total += 1

            try:
                smi = sample["smiles"]
                y = float(sample["scores"]["neglog_aff"])
                sid = sample.get("id", f"{sp}_{i}")

                if source_atoms not in sample:
                    breakdown["missing_field"] += 1
                    raise KeyError(f"missing_field:{source_atoms}; have={sorted(sample.keys())}")

                if "atoms_ligand" not in sample or "atoms_pocket" not in sample:
                    breakdown["missing_field"] += 1
                    raise KeyError(f"missing_field:atoms_ligand/atoms_pocket; have={sorted(sample.keys())}")

                df_prot = sample[source_atoms]
                lig_df = sample["atoms_ligand"]
                poc_df = sample["atoms_pocket"]

                # ===== 1D：使用全蛋白序列 =====
                # 优先使用 ATOM3D 样本自带的 sample["seq"]。
                # 如果 sample["seq"] 缺失，则从 atoms_protein 的所有链重建完整序列。
                # 注意：这里不使用 longest_only=True，也不只保留 pocket chain。
                seq_full, seq_policy = _get_full_protein_sequence(sample, df_prot)

                if not seq_full:
                    breakdown["protein_seq"] += 1
                    raise ValueError("empty full target sequence from sample['seq'] or atoms_protein")

                # ===== 3D：配体和口袋原子 =====
                lig_atoms = lig_df["element"].tolist()
                lig_xyz = lig_df[["x", "y", "z"]].values.tolist()

                poc_atoms = poc_df["element"].tolist()
                poc_xyz = poc_df[["x", "y", "z"]].values.tolist()

                if len(lig_atoms) == 0 or len(poc_atoms) == 0:
                    breakdown["empty3d"] += 1
                    raise ValueError("empty ligand or pocket atoms")

                lig_tok = _get_unimol_atomic_tokens(
                    model3d,
                    lig_atoms,
                    lig_xyz,
                    label=f"ligand:{sid}"
                )

                poc_tok = _get_unimol_atomic_tokens(
                    model3d,
                    poc_atoms,
                    poc_xyz,
                    label=f"pocket:{sid}"
                )

                if lig_tok.shape[0] == 0 or poc_tok.shape[0] == 0:
                    breakdown["empty3d"] += 1
                    raise ValueError("empty Uni-Mol atomic token output")

                if not printed_backend:
                    backend = type(model3d.model).__name__
                    print(f"[UniMol backend] {backend}")
                    print(f"[UniMol token dims] ligand={lig_tok.shape}, pocket={poc_tok.shape}")
                    printed_backend = True

                ids_list.append(sid)
                smiles_list.append(smi)
                seq_list.append(seq_full)
                seq_policy_list.append(seq_policy)
                y_list.append(y)

                lig_token_list.append(lig_tok)
                poc_token_list.append(poc_tok)

                n_ok += 1

            except KeyError as e:
                n_fail += 1
                breakdown["missing_field"] += 1
                if debug_shown < DEBUG_PRINT_N:
                    print(f"[DEBUG][{sp}] KeyError at #{i} id={sample.get('id', '')}: {e}")
                    debug_shown += 1

            except Exception as e:
                n_fail += 1
                msg = str(e).lower()

                if "invalid smiles" in msg:
                    breakdown["smiles"] += 1
                elif "sequence" in msg:
                    breakdown["protein_seq"] += 1
                elif "unimol" in msg or "atomic" in msg or "repr" in msg:
                    breakdown["unimol3d"] += 1
                elif "empty ligand" in msg or "empty pocket" in msg or "empty3d" in msg:
                    breakdown["empty3d"] += 1
                else:
                    breakdown["other"] += 1

                if debug_shown < DEBUG_PRINT_N:
                    print(f"[DEBUG][{sp}] Exception at #{i} id={sample.get('id', '')}: {e}")
                    debug_shown += 1

        if n_ok == 0:
            print(f"[FAIL][si-30|{sp}] breakdown: {breakdown}")
            raise RuntimeError(f"[si-30|{sp}] No valid samples. Check inputs at {lmdb_root}")

        ids_arr = np.array(ids_list, dtype=object)
        smiles_arr = np.array(smiles_list, dtype=object)
        seq_arr = np.array(seq_list, dtype=object)
        seq_policy_arr = np.array(seq_policy_list, dtype=object)
        y_arr = np.array(y_list, dtype=np.float32)

        N = len(ids_arr)

        assert len(smiles_arr) == N and len(seq_arr) == N and len(y_arr) == N, "元信息长度不一致"
        assert len(lig_token_list) == N and len(poc_token_list) == N, "3D token list 长度不一致"

        print(f"[si-30|{sp}] total={n_total} | ok={n_ok} | fail={n_fail} | breakdown={breakdown}")
        print(f"[si-30|{sp}] sequence source policies:")
        unique_policy, policy_count = np.unique(seq_policy_arr.astype(str), return_counts=True)
        for p, c in zip(unique_policy[:10], policy_count[:10]):
            print(f"  - {p}: {c}")

        # =========================
        # 3D token padding
        # =========================
        lig_3d_tokens, lig_3d_mask = _pad_token_arrays(
            lig_token_list,
            max_len=lig_max_atoms
        )

        poc_3d_tokens, poc_3d_mask = _pad_token_arrays(
            poc_token_list,
            max_len=pocket_max_atoms
        )

        assert lig_3d_tokens.shape[0] == N and poc_3d_tokens.shape[0] == N, "3D token 行数与样本数不一致"

        # =========================
        # 1D LM token embedding
        # =========================
        lm_device = torch.device(
            "cuda" if (use_cuda_for_unimol and torch.cuda.is_available()) else "cpu"
        )

        print(f"[LM] using device: {lm_device}")

        print(f"[LM] loading ChemBERTa model: {chemberta_model_name} (use_safetensors={use_safetensors})")
        chem_tok = AutoTokenizer.from_pretrained(chemberta_model_name)
        chem_model = _hf_load_model(
            chemberta_model_name,
            lm_device,
            use_safetensors
        )

        drug_lm_tokens, drug_lm_mask = _encode_text_tokens(
            smiles_arr.tolist(),
            chem_tok,
            chem_model,
            lm_device,
            batch_size=lm_batch_size,
            max_length=chem_max_len,
            desc=f"[LM|{sp}] ChemBERTa token encoding",
            mask_special_tokens=mask_special_tokens
        )

        try:
            del chem_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        print(f"[LM] loading ESM model: {esm2_model_name} (use_safetensors={use_safetensors})")
        esm_tok = AutoTokenizer.from_pretrained(esm2_model_name)
        esm_model = _hf_load_model(
            esm2_model_name,
            lm_device,
            use_safetensors
        )

        prot_lm_tokens, prot_lm_mask = _encode_text_tokens(
            seq_arr.tolist(),
            esm_tok,
            esm_model,
            lm_device,
            batch_size=max(1, min(2, lm_batch_size)),
            max_length=prot_max_len,
            desc=f"[LM|{sp}] ESM-2 token encoding",
            mask_special_tokens=mask_special_tokens
        )

        try:
            del esm_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        assert drug_lm_tokens.shape[0] == N and prot_lm_tokens.shape[0] == N, "LM token 行数与样本数不一致"
        assert drug_lm_mask.shape[0] == N and prot_lm_mask.shape[0] == N, "LM mask 行数与样本数不一致"

        print(f"[CACHE|{sp}] drug_lm_tokens={drug_lm_tokens.shape}, drug_lm_mask={drug_lm_mask.shape}")
        print(f"[CACHE|{sp}] prot_lm_tokens={prot_lm_tokens.shape}, prot_lm_mask={prot_lm_mask.shape}")
        print(f"[CACHE|{sp}] lig_3d_tokens={lig_3d_tokens.shape}, lig_3d_mask={lig_3d_mask.shape}")
        print(f"[CACHE|{sp}] poc_3d_tokens={poc_3d_tokens.shape}, poc_3d_mask={poc_3d_mask.shape}")

        # =========================
        # 保存缓存
        # =========================
        np.savez_compressed(
            p_1dnpz,
            ids=ids_arr,
            smiles=smiles_arr,
            seq=seq_arr,
            seq_chain_policy=seq_policy_arr,
            pkd=y_arr,

            drug_lm_tokens=drug_lm_tokens,
            drug_lm_mask=drug_lm_mask,

            prot_lm_tokens=prot_lm_tokens,
            prot_lm_mask=prot_lm_mask,
        )

        np.savez_compressed(
            p_3dnpz,
            ids=ids_arr,
            pkd=y_arr,

            lig_3d_tokens=lig_3d_tokens,
            lig_3d_mask=lig_3d_mask,

            poc_3d_tokens=poc_3d_tokens,
            poc_3d_mask=poc_3d_mask,
        )

        return {
            "ids": ids_arr,
            "y": y_arr,
            "smiles": smiles_arr,
            "seq": seq_arr,
            "seq_chain_policy": seq_policy_arr,

            "drug_lm_tokens": drug_lm_tokens,
            "drug_lm_mask": drug_lm_mask,

            "prot_lm_tokens": prot_lm_tokens,
            "prot_lm_mask": prot_lm_mask,

            "lig_3d_tokens": lig_3d_tokens,
            "lig_3d_mask": lig_3d_mask,

            "poc_3d_tokens": poc_3d_tokens,
            "poc_3d_mask": poc_3d_mask,

            "g_lig": None,
            "g_prot": None,
        }

    # =========================
    # 顶层调度
    # =========================
    if split in ("train", "val", "test"):
        return {split: _process_one_split(split)}

    elif split == "all":
        parts = {
            sp: _process_one_split(sp)
            for sp in ("train", "val", "test")
        }

        def _cat(*xs):
            return np.concatenate(xs, axis=0)

        all_pkg = {
            "ids": _cat(
                parts["train"]["ids"],
                parts["val"]["ids"],
                parts["test"]["ids"]
            ),

            "y": _cat(
                parts["train"]["y"],
                parts["val"]["y"],
                parts["test"]["y"]
            ).astype(np.float32),

            "smiles": _cat(
                parts["train"]["smiles"],
                parts["val"]["smiles"],
                parts["test"]["smiles"]
            ),

            "seq": _cat(
                parts["train"]["seq"],
                parts["val"]["seq"],
                parts["test"]["seq"]
            ),

            "seq_chain_policy": _cat(
                parts["train"]["seq_chain_policy"],
                parts["val"]["seq_chain_policy"],
                parts["test"]["seq_chain_policy"]
            ),

            "drug_lm_tokens": _cat(
                parts["train"]["drug_lm_tokens"],
                parts["val"]["drug_lm_tokens"],
                parts["test"]["drug_lm_tokens"]
            ).astype(np.float32),

            "drug_lm_mask": _cat(
                parts["train"]["drug_lm_mask"],
                parts["val"]["drug_lm_mask"],
                parts["test"]["drug_lm_mask"]
            ).astype(bool),

            "prot_lm_tokens": _cat(
                parts["train"]["prot_lm_tokens"],
                parts["val"]["prot_lm_tokens"],
                parts["test"]["prot_lm_tokens"]
            ).astype(np.float32),

            "prot_lm_mask": _cat(
                parts["train"]["prot_lm_mask"],
                parts["val"]["prot_lm_mask"],
                parts["test"]["prot_lm_mask"]
            ).astype(bool),

            "lig_3d_tokens": _cat(
                parts["train"]["lig_3d_tokens"],
                parts["val"]["lig_3d_tokens"],
                parts["test"]["lig_3d_tokens"]
            ).astype(np.float32),

            "lig_3d_mask": _cat(
                parts["train"]["lig_3d_mask"],
                parts["val"]["lig_3d_mask"],
                parts["test"]["lig_3d_mask"]
            ).astype(bool),

            "poc_3d_tokens": _cat(
                parts["train"]["poc_3d_tokens"],
                parts["val"]["poc_3d_tokens"],
                parts["test"]["poc_3d_tokens"]
            ).astype(np.float32),

            "poc_3d_mask": _cat(
                parts["train"]["poc_3d_mask"],
                parts["val"]["poc_3d_mask"],
                parts["test"]["poc_3d_mask"]
            ).astype(bool),

            "g_lig": None,
            "g_prot": None,
        }

        return {"all": all_pkg}

    else:
        raise ValueError(
            f"split 必须是 'train'/'val'/'test'/'all' 之一，收到: {split!r}"
        )



def LoadData_atom3d_si60_multimodal_lm_token(
    root_base: str,                                              # 行：指向 .../split-by-sequence-identity-60/data
    split: str = "train",                                        # 行：'train' | 'val' | 'test' | 'all'
    out_mm: str = "../dataset/ATOM3D/processed_mm_si60_lm_token", # 行：token 版缓存目录，避免覆盖旧 pooled 缓存
    unimol2_size: str = "unimol2_small",                         # 行：Uni-Mol2 模型规格，保持你原来的写法
    # 下面这些 2D 相关参数保留占位，避免外部旧调用报错
    contact_threshold: float = 8.0,
    dis_min: float = 1.0,
    prot_self_loop: bool = False,
    bond_bidirectional: bool = True,
    prefer_model: int = None,
    force_refresh: bool = False,                                  # 行：是否忽略缓存并强制重建
    use_cuda_for_unimol: bool = True,                             # 行：Uni-Mol2 是否使用 GPU
    # --- LM 模型相关 ---
    esm2_model_name: str = "facebook/esm2_t33_650M_UR50D",         # 行：ESM-2 checkpoint
    chemberta_model_name: str = "DeepChem/ChemBERTa-77M-MTR",      # 行：ChemBERTa checkpoint
    lm_batch_size: int = 8,                                        # 行：LM token 前向 batch_size，比 pooled 版更吃显存
    chem_max_len: int = 128,                                       # 行：SMILES token 最大长度
    prot_max_len: int = 1024,                                      # 行：蛋白 token 最大长度；显存不够可先改 512
    lig_max_atoms: int = 128,                                      # 行：配体最多保留多少个原子 token
    pocket_max_atoms: int = 256,                                   # 行：口袋最多保留多少个原子 token
    use_safetensors: bool = True,                                  # 行：是否优先使用 safetensors
    mask_special_tokens: bool = True,                              # 行：是否在 attention 中屏蔽 CLS/SEP/BOS/EOS 等特殊 token
    allow_pooled_3d_fallback: bool = False                         # 行：若 Uni-Mol2 原子级表示失败，是否允许退化为 1 个 pooled token
) -> dict:
    """
    ATOM3D SI-60 token-level 多模态数据加载函数。

    与旧版 LoadData_atom3d_si60_multimodal_lm 的区别：
      1. 1D 不再保存 CLS / pooled 向量，而是保存 token-level last_hidden_state：
            drug_lm_tokens : [N, chem_max_len, D_drug_lm]
            drug_lm_mask   : [N, chem_max_len]，True 表示 padding/special token，需要在 attention 中忽略

            prot_lm_tokens : [N, prot_max_len, D_prot_lm]
            prot_lm_mask   : [N, prot_max_len]，True 表示 padding/special token，需要在 attention 中忽略

      2. 3D 不再保存单个 pooled 向量，而是尽量保存 Uni-Mol2 atom-level representations：
            lig_3d_tokens  : [N, lig_max_atoms, D_3d]
            lig_3d_mask    : [N, lig_max_atoms]

            poc_3d_tokens  : [N, pocket_max_atoms, D_3d]
            poc_3d_mask    : [N, pocket_max_atoms]

      3. 多链蛋白处理：
            1D 分支使用 ATOM3D 样本中提供的 sample["seq"] 作为全蛋白序列；
            不再使用 longest_only=True，也不再任意保留最长链；
            若 sample["seq"] 缺失或为空，则从 atoms_protein 的所有链中重建完整蛋白序列作为兜底。

    返回：
        {split: {
            'ids', 'y', 'smiles', 'seq', 'seq_chain_policy',
            'drug_lm_tokens', 'drug_lm_mask',
            'prot_lm_tokens', 'prot_lm_mask',
            'lig_3d_tokens', 'lig_3d_mask',
            'poc_3d_tokens', 'poc_3d_mask',
            'g_lig', 'g_prot'
        }}
    """

    # =========================
    # 依赖导入
    # =========================
    from pathlib import Path
    import numpy as np
    import torch
    from tqdm import tqdm
    import atom3d.datasets as da
    from unimol_tools import UniMolRepr
    from transformers import AutoTokenizer, AutoModel

    # 注意：
    # 当前 token 版 1D 分支优先使用 sample["seq"] 作为全蛋白序列；
    # 不再依赖 atoms_to_sequence，也不再使用 longest_only=True。

    # =========================
    # 原子符号到原子序号的兜底表
    # =========================
    _PERIODIC = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
        'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
        'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
        'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54
    }

    def _to_atomic_numbers(sym_list):
        """把原子符号列表转成原子序号列表，供部分 Uni-Mol 版本兜底使用。"""
        return [int(_PERIODIC.get(str(s), 0)) for s in sym_list]

    def _as_numpy(x):
        """把 torch.Tensor / list / np.ndarray 统一转成 numpy.float32。"""
        if torch.is_tensor(x):
            return x.detach().cpu().numpy().astype(np.float32)
        return np.asarray(x, dtype=np.float32)

    def _detect_chain_col(df):
        """
        自动判断 DataFrame 中哪个字段是 chain id。
        不同 ATOM3D / PDB 解析版本字段名可能不完全一致，所以这里做兼容。
        """
        candidates = [
            "chain", "chain_id", "chainID", "chain_name",
            "asym_id", "auth_asym_id", "label_asym_id"
        ]
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def _unique_nonempty_values(df, col):
        """提取某个列中的非空唯一值，并统一转成字符串。"""
        if col is None or col not in df.columns:
            return []
        vals = []
        for v in df[col].tolist():
            if v is None:
                continue
            s = str(v).strip()
            if s == "" or s.lower() == "nan":
                continue
            vals.append(s)
        return sorted(set(vals))

    # =========================
    # 三字母氨基酸转一字母氨基酸
    # =========================
    _AA3_TO_1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",

        # 常见修饰残基
        "MSE": "M", "SEC": "C", "PYL": "K",
        "HSD": "H", "HSE": "H", "HSP": "H",
        "HID": "H", "HIE": "H", "HIP": "H",
        "CYX": "C", "CME": "C", "CSO": "C",
        "SEP": "S", "TPO": "T", "PTR": "Y",
    }

    def _clean_protein_sequence(seq):
        """
        清洗蛋白序列。

        输入可能是：
            1. 字符串
            2. list / tuple
            3. dict，例如多链序列

        输出：
            只保留标准氨基酸字符的字符串。
        """
        if seq is None:
            return ""

        if isinstance(seq, dict):
            parts = []

            for k in sorted(seq.keys()):
                v = seq[k]

                if v is None:
                    continue

                parts.append(str(v))

            seq = "".join(parts)

        elif isinstance(seq, (list, tuple)):
            seq = "".join([str(x) for x in seq if x is not None])

        else:
            seq = str(seq)

        seq = seq.upper()

        valid = set("ACDEFGHIKLMNPQRSTVWY")

        seq = "".join([c for c in seq if c in valid])

        return seq

    def _atoms_df_to_full_sequence_direct(df_prot):
        """
        从 atoms_protein 的所有链中重建完整蛋白序列。

        这是 sample["seq"] 缺失时的兜底方案。
        注意：
            这里使用所有 protein chains；
            不使用 pocket chain 过滤；
            不使用 longest_only=True。
        """
        if df_prot is None or len(df_prot) == 0:
            return ""

        required_cols = ["chain", "residue", "resname"]

        for col in required_cols:
            if col not in df_prot.columns:
                print(f"[SEQ-ERROR] missing column: {col}")
                print("[SEQ-ERROR] available columns:", list(df_prot.columns))
                return ""

        tmp = df_prot.copy()

        tmp["chain"] = tmp["chain"].astype(str)

        tmp["resname"] = tmp["resname"].astype(str).str.upper()

        tmp["_order"] = np.arange(len(tmp))

        if "insertion_code" in tmp.columns:
            tmp["insertion_code"] = tmp["insertion_code"].fillna("").astype(str)

            residue_df = tmp.drop_duplicates(
                subset=["chain", "residue", "insertion_code"],
                keep="first"
            ).copy()

            residue_df = residue_df.sort_values(
                by=["chain", "residue", "insertion_code", "_order"]
            )
        else:
            residue_df = tmp.drop_duplicates(
                subset=["chain", "residue"],
                keep="first"
            ).copy()

            residue_df = residue_df.sort_values(
                by=["chain", "residue", "_order"]
            )

        seq_chars = []

        for aa3 in residue_df["resname"].tolist():
            aa1 = _AA3_TO_1.get(aa3, "")

            if aa1:
                seq_chars.append(aa1)

        return "".join(seq_chars)

    def _get_full_protein_sequence(sample, df_prot):
        """
        获取 1D 分支使用的全蛋白序列。

        优先级：
            1. 优先使用 ATOM3D 样本自带的 sample["seq"]。
            2. 如果 sample["seq"] 为空，则从 atoms_protein 的所有链重建序列。

        返回：
            seq_full: 全蛋白序列
            policy:   记录序列来源，方便写日志和回复审稿人
        """
        seq_from_sample = _clean_protein_sequence(sample.get("seq", None))

        if seq_from_sample:
            return seq_from_sample, "full_protein_sequence_from_sample_seq"

        seq_from_atoms = _atoms_df_to_full_sequence_direct(df_prot)

        if seq_from_atoms:
            return seq_from_atoms, "full_protein_sequence_rebuilt_from_atoms_protein_all_chains"

        return "", "failed_to_extract_full_protein_sequence"

    def _hf_load_model(name: str, device: torch.device, use_safetensors: bool = True):
        """加载 HuggingFace AutoModel，并兼容 use_safetensors 参数。"""
        try:
            model = AutoModel.from_pretrained(name, use_safetensors=use_safetensors)
            model.to(device)
            return model
        except TypeError:
            model = AutoModel.from_pretrained(name)
            model.to(device)
            return model
        except Exception as e:
            msg = str(e).lower()

            if ("torch" in msg and "2.6" in msg) or ("vulnerability" in msg) or ("cve" in msg):
                raise RuntimeError(
                    f"Loading HF model '{name}' failed due to transformers / torch safety check:\n{e}\n"
                    "Possible solutions:\n"
                    "  1) 优先使用 safetensors；\n"
                    "  2) 升级 torch 到 >=2.6；\n"
                    "  3) 使用兼容版本 transformers。"
                ) from e

            try:
                model = AutoModel.from_pretrained(name)
                model.to(device)
                return model
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load HF model '{name}'. "
                    f"Err1={e} Err2={e2}"
                ) from e2

    def _encode_text_tokens(
        text_list,
        tokenizer,
        model,
        device,
        batch_size=8,
        max_length=512,
        desc="[LM] token encoding",
        mask_special_tokens=True
    ):
        """
        抽取 token-level LM embedding。

        输入：
            text_list: list[str]

        输出：
            token_embeds: [N, max_length, D]
            pad_mask:     [N, max_length]，True 表示需要被 attention 忽略
        """
        all_tokens = []
        all_masks = []

        model.eval()

        with torch.no_grad():
            for i in tqdm(
                range(0, len(text_list), batch_size),
                desc=desc,
                unit="batch"
            ):
                batch = [str(x) for x in text_list[i:i + batch_size]]

                enc = tokenizer(
                    batch,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                    return_special_tokens_mask=True
                )

                # special_tokens_mask 是 tokenizer 的辅助输出，只能用于构造 mask
                # 不能传入 model(**enc)
                special_tokens_mask = enc.pop("special_tokens_mask", None)

                enc = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in enc.items()
                }

                if special_tokens_mask is not None:
                    special_tokens_mask = special_tokens_mask.to(device)

                try:
                    out = model(**enc)
                except TypeError as e:
                    # 某些模型不接受 token_type_ids，自动删除后重试
                    if "token_type_ids" in str(e) and "token_type_ids" in enc:
                        enc.pop("token_type_ids")
                        out = model(**enc)
                    else:
                        raise e

                if hasattr(out, "last_hidden_state"):
                    hs = out.last_hidden_state
                elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                    hs = out[0]
                else:
                    raise RuntimeError("Unsupported model output structure for token-level LM encoding.")

                attention_mask = enc["attention_mask"].bool()
                pad_mask = ~attention_mask

                if mask_special_tokens and special_tokens_mask is not None:
                    special_mask = special_tokens_mask.bool()
                    pad_mask = pad_mask | special_mask

                all_tokens.append(hs.detach().cpu().numpy().astype(np.float32))
                all_masks.append(pad_mask.detach().cpu().numpy().astype(bool))

        if len(all_tokens) == 0:
            hidden_size = int(getattr(model.config, "hidden_size", 0))
            return (
                np.zeros((0, max_length, hidden_size), dtype=np.float32),
                np.ones((0, max_length), dtype=bool)
            )

        token_embeds = np.concatenate(all_tokens, axis=0).astype(np.float32)
        token_masks = np.concatenate(all_masks, axis=0).astype(bool)

        return token_embeds, token_masks

    def _make_unimol2_repr():
        """
        初始化 Uni-Mol2 表征器。
        先按你旧代码的 model='unimol2' 写法尝试；
        如果当前 unimol_tools 版本使用 model_name='unimolv2'，则自动兜底。
        """
        try:
            return UniMolRepr(
                model="unimol2",
                model_size=unimol2_size,
                use_cuda=use_cuda_for_unimol
            )
        except TypeError:
            return UniMolRepr(
                model_name="unimolv2",
                model_size=unimol2_size,
                use_cuda=use_cuda_for_unimol
            )

    def _extract_atomic_reprs_from_output(repr_output, expected_len=None):
        """
        从 Uni-Mol get_repr(..., return_atomic_reprs=True) 的返回结果中取 atomic representations。

        兼容可能出现的几种结构：
            dict['atomic_reprs']
            dict['atomic_reprs'][0]
            list / tuple 包一层
            np.ndarray / torch.Tensor
        """
        obj = repr_output

        if isinstance(obj, dict):
            if "atomic_reprs" in obj:
                obj = obj["atomic_reprs"]
            elif "atomic_repr" in obj:
                obj = obj["atomic_repr"]
            elif "cls_repr" in obj and allow_pooled_3d_fallback:
                obj = obj["cls_repr"]
            else:
                raise RuntimeError(
                    f"Uni-Mol output does not contain atomic_reprs. Keys={list(obj.keys())}"
                )

        if isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                raise RuntimeError("Uni-Mol atomic_reprs is empty.")
            obj = obj[0]

        arr = _as_numpy(obj)

        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]

        if arr.ndim == 1:
            if allow_pooled_3d_fallback:
                arr = arr.reshape(1, -1)
            else:
                raise RuntimeError(
                    "Uni-Mol returned a pooled 1D vector instead of atom-level tokens. "
                    "Set allow_pooled_3d_fallback=True only for debugging, not for final revision."
                )

        if arr.ndim != 2:
            raise RuntimeError(f"Unexpected atomic representation shape: {arr.shape}")

        if expected_len is not None and expected_len > 0:
            if arr.shape[0] >= expected_len:
                arr = arr[:expected_len]
            else:
                # 如果返回 token 数少于原子数，保留返回值，但后续会 padding。
                arr = arr

        return arr.astype(np.float32)

    def _get_unimol_atomic_tokens(model3d, atoms, xyz, label="mol"):
        """
        用 Uni-Mol2 抽取原子级 token。
        优先使用原子符号；失败后用原子序号兜底。
        """
        data_symbol = {
            "atoms": atoms,
            "coordinates": xyz
        }

        try:
            repr_output = model3d.get_repr(
                data_symbol,
                return_atomic_reprs=True
            )
            return _extract_atomic_reprs_from_output(
                repr_output,
                expected_len=len(atoms)
            )
        except Exception as e_symbol:
            data_number = {
                "atoms": _to_atomic_numbers(atoms),
                "coordinates": xyz
            }

            try:
                repr_output = model3d.get_repr(
                    data_number,
                    return_atomic_reprs=True
                )
                return _extract_atomic_reprs_from_output(
                    repr_output,
                    expected_len=len(atoms)
                )
            except Exception as e_number:
                raise RuntimeError(
                    f"Failed to extract Uni-Mol atom-level tokens for {label}. "
                    f"symbol_error={e_symbol}; number_error={e_number}"
                ) from e_number

    def _pad_token_arrays(arr_list, max_len):
        """
        把不同长度的 token 序列补齐成固定长度。

        输入：
            arr_list: list，每个元素形状为 [Li, D]
            max_len: 固定 token 长度

        输出：
            out:  [N, max_len, D]
            mask: [N, max_len]，True 表示 padding，需要被 attention 忽略
        """
        if len(arr_list) == 0:
            return np.zeros((0, max_len, 0), dtype=np.float32), np.ones((0, max_len), dtype=bool)

        dim = int(arr_list[0].shape[-1])
        n = len(arr_list)

        out = np.zeros((n, max_len, dim), dtype=np.float32)
        mask = np.ones((n, max_len), dtype=bool)

        for i, x in enumerate(arr_list):
            x = np.asarray(x, dtype=np.float32)

            if x.ndim != 2:
                raise ValueError(f"Token array at index {i} must be [L,D], got shape={x.shape}")

            if x.shape[-1] != dim:
                raise ValueError(
                    f"Token dim mismatch at index {i}: expected {dim}, got {x.shape[-1]}"
                )

            L = min(x.shape[0], max_len)
            out[i, :L, :] = x[:L]
            mask[i, :L] = False

        return out, mask

    # =========================
    # 路径初始化
    # =========================
    root_base = Path(root_base)
    out_mm = Path(out_mm)
    out_mm.mkdir(parents=True, exist_ok=True)

    # =========================
    # 处理单个 split
    # =========================
    def _process_one_split(sp: str) -> dict:
        lmdb_root = root_base / sp

        p_1dnpz = out_mm / f"si60_{sp}_1d_lm_token.npz"
        p_3dnpz = out_mm / f"si60_{sp}_3d_token.npz"

        # =========================
        # 缓存直读
        # =========================
        if (not force_refresh) and p_1dnpz.exists() and p_3dnpz.exists():
            d1 = np.load(p_1dnpz, allow_pickle=True)
            d3 = np.load(p_3dnpz, allow_pickle=True)

            ids = d1["ids"]
            smiles = d1["smiles"]
            seq = d1["seq"]
            y = d1["pkd"].astype(np.float32)

            seq_chain_policy = d1["seq_chain_policy"] if "seq_chain_policy" in d1.files else np.array(["unknown"] * len(ids), dtype=object)

            drug_lm_tokens = d1["drug_lm_tokens"].astype(np.float32)
            drug_lm_mask = d1["drug_lm_mask"].astype(bool)

            prot_lm_tokens = d1["prot_lm_tokens"].astype(np.float32)
            prot_lm_mask = d1["prot_lm_mask"].astype(bool)

            lig_3d_tokens = d3["lig_3d_tokens"].astype(np.float32)
            lig_3d_mask = d3["lig_3d_mask"].astype(bool)

            poc_3d_tokens = d3["poc_3d_tokens"].astype(np.float32)
            poc_3d_mask = d3["poc_3d_mask"].astype(bool)

            N = len(ids)

            assert len(smiles) == N and len(seq) == N and len(y) == N, "1D 元信息长度不一致"
            assert drug_lm_tokens.shape[0] == N and prot_lm_tokens.shape[0] == N, "LM token 行数不一致"
            assert drug_lm_mask.shape[0] == N and prot_lm_mask.shape[0] == N, "LM mask 行数不一致"
            assert lig_3d_tokens.shape[0] == N and poc_3d_tokens.shape[0] == N, "3D token 行数不一致"
            assert lig_3d_mask.shape[0] == N and poc_3d_mask.shape[0] == N, "3D mask 行数不一致"

            return {
                "ids": ids,
                "y": y,
                "smiles": smiles,
                "seq": seq,
                "seq_chain_policy": seq_chain_policy,

                "drug_lm_tokens": drug_lm_tokens,
                "drug_lm_mask": drug_lm_mask,

                "prot_lm_tokens": prot_lm_tokens,
                "prot_lm_mask": prot_lm_mask,

                "lig_3d_tokens": lig_3d_tokens,
                "lig_3d_mask": lig_3d_mask,

                "poc_3d_tokens": poc_3d_tokens,
                "poc_3d_mask": poc_3d_mask,

                "g_lig": None,
                "g_prot": None,
            }

        # =========================
        # 正常构建
        # =========================
        ds = da.LMDBDataset(str(lmdb_root))
        print(f"[si-60|{sp}] LMDB samples = {len(ds)}")

        model3d = _make_unimol2_repr()
        model3d.model.eval()

        source_atoms = "atoms_protein"

        ids_list = []
        smiles_list = []
        seq_list = []
        seq_policy_list = []
        y_list = []

        lig_token_list = []
        poc_token_list = []

        n_total = 0
        n_ok = 0
        n_fail = 0

        breakdown = {
            "missing_field": 0,
            "smiles": 0,
            "protein_seq": 0,
            "empty3d": 0,
            "unimol3d": 0,
            "other": 0
        }

        DEBUG_PRINT_N = 8
        debug_shown = 0
        printed_backend = False

        for i, sample in enumerate(
            tqdm(ds, desc=f"[si-60|{sp}] token-level 1D+3D", unit="sample")
        ):
            n_total += 1

            try:
                smi = sample["smiles"]
                y = float(sample["scores"]["neglog_aff"])
                sid = sample.get("id", f"{sp}_{i}")

                if source_atoms not in sample:
                    breakdown["missing_field"] += 1
                    raise KeyError(f"missing_field:{source_atoms}; have={sorted(sample.keys())}")

                if "atoms_ligand" not in sample or "atoms_pocket" not in sample:
                    breakdown["missing_field"] += 1
                    raise KeyError(f"missing_field:atoms_ligand/atoms_pocket; have={sorted(sample.keys())}")

                df_prot = sample[source_atoms]
                lig_df = sample["atoms_ligand"]
                poc_df = sample["atoms_pocket"]

                # ===== 1D：使用全蛋白序列 =====
                # 优先使用 ATOM3D 样本自带的 sample["seq"]。
                # 如果 sample["seq"] 缺失，则从 atoms_protein 的所有链重建完整序列。
                # 注意：这里不使用 longest_only=True，也不只保留 pocket chain。
                seq_full, seq_policy = _get_full_protein_sequence(sample, df_prot)

                if not seq_full:
                    breakdown["protein_seq"] += 1
                    raise ValueError("empty full target sequence from sample['seq'] or atoms_protein")

                # ===== 3D：配体和口袋原子 =====
                lig_atoms = lig_df["element"].tolist()
                lig_xyz = lig_df[["x", "y", "z"]].values.tolist()

                poc_atoms = poc_df["element"].tolist()
                poc_xyz = poc_df[["x", "y", "z"]].values.tolist()

                if len(lig_atoms) == 0 or len(poc_atoms) == 0:
                    breakdown["empty3d"] += 1
                    raise ValueError("empty ligand or pocket atoms")

                lig_tok = _get_unimol_atomic_tokens(
                    model3d,
                    lig_atoms,
                    lig_xyz,
                    label=f"ligand:{sid}"
                )

                poc_tok = _get_unimol_atomic_tokens(
                    model3d,
                    poc_atoms,
                    poc_xyz,
                    label=f"pocket:{sid}"
                )

                if lig_tok.shape[0] == 0 or poc_tok.shape[0] == 0:
                    breakdown["empty3d"] += 1
                    raise ValueError("empty Uni-Mol atomic token output")

                if not printed_backend:
                    backend = type(model3d.model).__name__
                    print(f"[UniMol backend] {backend}")
                    print(f"[UniMol token dims] ligand={lig_tok.shape}, pocket={poc_tok.shape}")
                    printed_backend = True

                ids_list.append(sid)
                smiles_list.append(smi)
                seq_list.append(seq_full)
                seq_policy_list.append(seq_policy)
                y_list.append(y)

                lig_token_list.append(lig_tok)
                poc_token_list.append(poc_tok)

                n_ok += 1

            except KeyError as e:
                n_fail += 1
                breakdown["missing_field"] += 1
                if debug_shown < DEBUG_PRINT_N:
                    print(f"[DEBUG][{sp}] KeyError at #{i} id={sample.get('id', '')}: {e}")
                    debug_shown += 1

            except Exception as e:
                n_fail += 1
                msg = str(e).lower()

                if "invalid smiles" in msg:
                    breakdown["smiles"] += 1
                elif "sequence" in msg:
                    breakdown["protein_seq"] += 1
                elif "unimol" in msg or "atomic" in msg or "repr" in msg:
                    breakdown["unimol3d"] += 1
                elif "empty ligand" in msg or "empty pocket" in msg or "empty3d" in msg:
                    breakdown["empty3d"] += 1
                else:
                    breakdown["other"] += 1

                if debug_shown < DEBUG_PRINT_N:
                    print(f"[DEBUG][{sp}] Exception at #{i} id={sample.get('id', '')}: {e}")
                    debug_shown += 1

        if n_ok == 0:
            print(f"[FAIL][si-60|{sp}] breakdown: {breakdown}")
            raise RuntimeError(f"[si-60|{sp}] No valid samples. Check inputs at {lmdb_root}")

        ids_arr = np.array(ids_list, dtype=object)
        smiles_arr = np.array(smiles_list, dtype=object)
        seq_arr = np.array(seq_list, dtype=object)
        seq_policy_arr = np.array(seq_policy_list, dtype=object)
        y_arr = np.array(y_list, dtype=np.float32)

        N = len(ids_arr)

        assert len(smiles_arr) == N and len(seq_arr) == N and len(y_arr) == N, "元信息长度不一致"
        assert len(lig_token_list) == N and len(poc_token_list) == N, "3D token list 长度不一致"

        print(f"[si-60|{sp}] total={n_total} | ok={n_ok} | fail={n_fail} | breakdown={breakdown}")
        print(f"[si-60|{sp}] sequence source policies:")
        unique_policy, policy_count = np.unique(seq_policy_arr.astype(str), return_counts=True)
        for p, c in zip(unique_policy[:10], policy_count[:10]):
            print(f"  - {p}: {c}")

        # =========================
        # 3D token padding
        # =========================
        lig_3d_tokens, lig_3d_mask = _pad_token_arrays(
            lig_token_list,
            max_len=lig_max_atoms
        )

        poc_3d_tokens, poc_3d_mask = _pad_token_arrays(
            poc_token_list,
            max_len=pocket_max_atoms
        )

        assert lig_3d_tokens.shape[0] == N and poc_3d_tokens.shape[0] == N, "3D token 行数与样本数不一致"

        # =========================
        # 1D LM token embedding
        # =========================
        lm_device = torch.device(
            "cuda" if (use_cuda_for_unimol and torch.cuda.is_available()) else "cpu"
        )

        print(f"[LM] using device: {lm_device}")

        print(f"[LM] loading ChemBERTa model: {chemberta_model_name} (use_safetensors={use_safetensors})")
        chem_tok = AutoTokenizer.from_pretrained(chemberta_model_name)
        chem_model = _hf_load_model(
            chemberta_model_name,
            lm_device,
            use_safetensors
        )

        drug_lm_tokens, drug_lm_mask = _encode_text_tokens(
            smiles_arr.tolist(),
            chem_tok,
            chem_model,
            lm_device,
            batch_size=lm_batch_size,
            max_length=chem_max_len,
            desc=f"[LM|{sp}] ChemBERTa token encoding",
            mask_special_tokens=mask_special_tokens
        )

        try:
            del chem_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        print(f"[LM] loading ESM model: {esm2_model_name} (use_safetensors={use_safetensors})")
        esm_tok = AutoTokenizer.from_pretrained(esm2_model_name)
        esm_model = _hf_load_model(
            esm2_model_name,
            lm_device,
            use_safetensors
        )

        prot_lm_tokens, prot_lm_mask = _encode_text_tokens(
            seq_arr.tolist(),
            esm_tok,
            esm_model,
            lm_device,
            batch_size=max(1, min(2, lm_batch_size)),
            max_length=prot_max_len,
            desc=f"[LM|{sp}] ESM-2 token encoding",
            mask_special_tokens=mask_special_tokens
        )

        try:
            del esm_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        assert drug_lm_tokens.shape[0] == N and prot_lm_tokens.shape[0] == N, "LM token 行数与样本数不一致"
        assert drug_lm_mask.shape[0] == N and prot_lm_mask.shape[0] == N, "LM mask 行数与样本数不一致"

        print(f"[CACHE|{sp}] drug_lm_tokens={drug_lm_tokens.shape}, drug_lm_mask={drug_lm_mask.shape}")
        print(f"[CACHE|{sp}] prot_lm_tokens={prot_lm_tokens.shape}, prot_lm_mask={prot_lm_mask.shape}")
        print(f"[CACHE|{sp}] lig_3d_tokens={lig_3d_tokens.shape}, lig_3d_mask={lig_3d_mask.shape}")
        print(f"[CACHE|{sp}] poc_3d_tokens={poc_3d_tokens.shape}, poc_3d_mask={poc_3d_mask.shape}")

        # =========================
        # 保存缓存
        # =========================
        np.savez_compressed(
            p_1dnpz,
            ids=ids_arr,
            smiles=smiles_arr,
            seq=seq_arr,
            seq_chain_policy=seq_policy_arr,
            pkd=y_arr,

            drug_lm_tokens=drug_lm_tokens,
            drug_lm_mask=drug_lm_mask,

            prot_lm_tokens=prot_lm_tokens,
            prot_lm_mask=prot_lm_mask,
        )

        np.savez_compressed(
            p_3dnpz,
            ids=ids_arr,
            pkd=y_arr,

            lig_3d_tokens=lig_3d_tokens,
            lig_3d_mask=lig_3d_mask,

            poc_3d_tokens=poc_3d_tokens,
            poc_3d_mask=poc_3d_mask,
        )

        return {
            "ids": ids_arr,
            "y": y_arr,
            "smiles": smiles_arr,
            "seq": seq_arr,
            "seq_chain_policy": seq_policy_arr,

            "drug_lm_tokens": drug_lm_tokens,
            "drug_lm_mask": drug_lm_mask,

            "prot_lm_tokens": prot_lm_tokens,
            "prot_lm_mask": prot_lm_mask,

            "lig_3d_tokens": lig_3d_tokens,
            "lig_3d_mask": lig_3d_mask,

            "poc_3d_tokens": poc_3d_tokens,
            "poc_3d_mask": poc_3d_mask,

            "g_lig": None,
            "g_prot": None,
        }

    # =========================
    # 顶层调度
    # =========================
    if split in ("train", "val", "test"):
        return {split: _process_one_split(split)}

    elif split == "all":
        parts = {
            sp: _process_one_split(sp)
            for sp in ("train", "val", "test")
        }

        def _cat(*xs):
            return np.concatenate(xs, axis=0)

        all_pkg = {
            "ids": _cat(
                parts["train"]["ids"],
                parts["val"]["ids"],
                parts["test"]["ids"]
            ),

            "y": _cat(
                parts["train"]["y"],
                parts["val"]["y"],
                parts["test"]["y"]
            ).astype(np.float32),

            "smiles": _cat(
                parts["train"]["smiles"],
                parts["val"]["smiles"],
                parts["test"]["smiles"]
            ),

            "seq": _cat(
                parts["train"]["seq"],
                parts["val"]["seq"],
                parts["test"]["seq"]
            ),

            "seq_chain_policy": _cat(
                parts["train"]["seq_chain_policy"],
                parts["val"]["seq_chain_policy"],
                parts["test"]["seq_chain_policy"]
            ),

            "drug_lm_tokens": _cat(
                parts["train"]["drug_lm_tokens"],
                parts["val"]["drug_lm_tokens"],
                parts["test"]["drug_lm_tokens"]
            ).astype(np.float32),

            "drug_lm_mask": _cat(
                parts["train"]["drug_lm_mask"],
                parts["val"]["drug_lm_mask"],
                parts["test"]["drug_lm_mask"]
            ).astype(bool),

            "prot_lm_tokens": _cat(
                parts["train"]["prot_lm_tokens"],
                parts["val"]["prot_lm_tokens"],
                parts["test"]["prot_lm_tokens"]
            ).astype(np.float32),

            "prot_lm_mask": _cat(
                parts["train"]["prot_lm_mask"],
                parts["val"]["prot_lm_mask"],
                parts["test"]["prot_lm_mask"]
            ).astype(bool),

            "lig_3d_tokens": _cat(
                parts["train"]["lig_3d_tokens"],
                parts["val"]["lig_3d_tokens"],
                parts["test"]["lig_3d_tokens"]
            ).astype(np.float32),

            "lig_3d_mask": _cat(
                parts["train"]["lig_3d_mask"],
                parts["val"]["lig_3d_mask"],
                parts["test"]["lig_3d_mask"]
            ).astype(bool),

            "poc_3d_tokens": _cat(
                parts["train"]["poc_3d_tokens"],
                parts["val"]["poc_3d_tokens"],
                parts["test"]["poc_3d_tokens"]
            ).astype(np.float32),

            "poc_3d_mask": _cat(
                parts["train"]["poc_3d_mask"],
                parts["val"]["poc_3d_mask"],
                parts["test"]["poc_3d_mask"]
            ).astype(bool),

            "g_lig": None,
            "g_prot": None,
        }

        return {"all": all_pkg}

    else:
        raise ValueError(
            f"split 必须是 'train'/'val'/'test'/'all' 之一，收到: {split!r}"
        )


# -*- coding: utf-8 -*-
"""
Davis token-level 1D loader with strict ligand-scaffold and protein-disjoint split.

Recommended usage:
    from util_load import LoadData_davis_lm_1d_token_scaffold_seqsplit

This function is designed to replace the old pair-level random split + pooled CLS embedding
loader for Davis. It returns unique ligand/protein token banks plus pair indices to avoid
expanding huge protein token tensors to every ligand-protein pair.
"""


def LoadData_davis_lm_1d_token_scaffold_seqsplit(
    data_dir: str,
    split: str = "all",
    out_1d: str = "../dataset/davis/processed_lm_1d_token_scaffold_seqsplit",
    logspace_trans: bool = True,
    # --- LM models ---
    esm2_model_name: str = "facebook/esm2_t33_650M_UR50D",
    chemberta_model_name: str = "DeepChem/ChemBERTa-77M-MTR",
    lm_batch_size: int = 4,
    chem_max_len: int = 128,
    prot_max_len: int = 1024,
    use_safetensors: bool = True,
    mask_special_tokens: bool = True,
    # --- strict split ---
    split_seed: int = 2023,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    force_refresh: bool = False,
    # --- compatibility / memory option ---
    return_pair_level_tokens: bool = False,
) -> dict:
    """
    Davis + token-level LM loader with strict ligand scaffold / protein disjoint split.

    Why this new loader is needed:
        1. The old loader used CLS/first-token pooled vectors [N, D], which cannot support
           token-level modeling.
        2. The old loader randomly split ligand-protein pairs, so the same ligand scaffold
           and the same protein could appear in train/val/test, which is vulnerable to
           data-leakage criticism.
        3. Davis is a 1D-only benchmark and has no 3D pocket branch; therefore, the
           multi-chain 1D/3D mismatch issue does not apply to Davis. The protein sequence
           used here is the full sequence from proteins.txt.

    Split protocol:
        - Ligands are grouped by Bemis-Murcko scaffold using RDKit.
        - Proteins are grouped by cleaned full protein sequence.
        - Ligand scaffold groups and protein sequence groups are separately assigned to
          train/val/test.
        - A pair is kept in a split only when BOTH its ligand and protein belong to that
          split. Cross-partition pairs are excluded. This guarantees:
              train/val/test have disjoint ligand scaffolds;
              train/val/test have disjoint protein sequences.

    Cache files:
        davis_unique_lm_tokens.npz
        davis_train_pairs_scaffold_seqsplit.npz
        davis_val_pairs_scaffold_seqsplit.npz
        davis_test_pairs_scaffold_seqsplit.npz
        davis_split_audit_scaffold_seqsplit.txt

    Return for one split:
        {
            'ids', 'y', 'smiles', 'seq',
            'pair_lig_idx', 'pair_pro_idx',
            'drug_lm_tokens_bank', 'drug_lm_mask_bank',
            'prot_lm_tokens_bank', 'prot_lm_mask_bank',
            'ligand_scaffold', 'split_protocol',
            optionally 'drug_lm_tokens', 'drug_lm_mask', 'prot_lm_tokens', 'prot_lm_mask'
        }

    Important:
        By default this loader DOES NOT expand unique protein token embeddings to pair-level
        arrays, because Davis contains many ligand-protein pairs and pair-level protein
        token expansion can require huge memory. In the training Dataset, use
        pair_lig_idx/pair_pro_idx to index the token banks.
    """
    from pathlib import Path
    import json
    import pickle
    from collections import OrderedDict, defaultdict

    import numpy as np
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModel

    data_dir = Path(data_dir)
    out_1d = Path(out_1d)
    out_1d.mkdir(parents=True, exist_ok=True)

    token_bank_path = out_1d / "davis_unique_lm_tokens.npz"
    split_meta_path = out_1d / "davis_split_meta_scaffold_seqsplit.json"
    audit_path = out_1d / "davis_split_audit_scaffold_seqsplit.txt"

    def _pair_cache_path(sp: str):
        return out_1d / f"davis_{sp}_pairs_scaffold_seqsplit.npz"

    def _clean_protein_sequence(seq):
        if seq is None:
            return ""
        seq = str(seq).upper()
        valid = set("ACDEFGHIKLMNPQRSTVWY")
        return "".join([c for c in seq if c in valid])

    def _clean_smiles(smi):
        if smi is None:
            return ""
        return str(smi).strip()

    def _load_raw_davis():
        lig_path = data_dir / "ligands_can.txt"
        pro_path = data_dir / "proteins.txt"
        y_path = data_dir / "Y"

        with lig_path.open("r", encoding="utf-8") as f:
            lig_dict = json.load(f, object_pairs_hook=OrderedDict)
        with pro_path.open("r", encoding="utf-8") as f:
            pro_dict = json.load(f, object_pairs_hook=OrderedDict)
        with y_path.open("rb") as f:
            Y = pickle.load(f, encoding="latin1")

        lig_ids = np.array(list(lig_dict.keys()), dtype=object)
        pro_ids = np.array(list(pro_dict.keys()), dtype=object)
        lig_list = np.array([_clean_smiles(x) for x in lig_dict.values()], dtype=object)
        pro_list = np.array([_clean_protein_sequence(x) for x in pro_dict.values()], dtype=object)

        Y = np.asarray(Y, dtype=np.float64)
        if logspace_trans:
            Y = -np.log10(Y / 1e9)

        return lig_ids, lig_list, pro_ids, pro_list, Y

    def _get_murcko_scaffold(smi: str) -> str:
        """
        Return Bemis-Murcko scaffold. If RDKit cannot parse the molecule or the scaffold
        is empty, fall back to canonical SMILES / raw SMILES so that every ligand belongs
        to a deterministic group.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                return f"INVALID::{str(smi)}"
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            if scaf is None or str(scaf).strip() == "":
                can = Chem.MolToSmiles(mol, canonical=True)
                return f"NO_SCAFFOLD::{can}"
            return str(scaf)
        except Exception:
            # RDKit not available or unexpected chemistry parsing error.
            return f"FALLBACK::{str(smi)}"

    def _split_groups(groups: dict, total_items: int):
        """
        Greedy group split. A whole group is assigned to one split, never broken.
        """
        rng = np.random.RandomState(split_seed)
        items = list(groups.items())
        rng.shuffle(items)
        items = sorted(items, key=lambda kv: len(kv[1]), reverse=True)

        n_train_target = int(total_items * train_ratio)
        n_val_target = int(total_items * val_ratio)

        split_to_indices = {"train": [], "val": [], "test": []}
        split_to_groups = {"train": [], "val": [], "test": []}

        for g, idxs in items:
            idxs = list(idxs)
            if len(split_to_indices["train"]) + len(idxs) <= n_train_target:
                sp = "train"
            elif len(split_to_indices["val"]) + len(idxs) <= n_val_target:
                sp = "val"
            else:
                sp = "test"
            split_to_indices[sp].extend(idxs)
            split_to_groups[sp].append(g)

        return (
            {k: np.array(v, dtype=np.int64) for k, v in split_to_indices.items()},
            {k: set(v) for k, v in split_to_groups.items()},
        )

    def _hf_load_model(name: str, device: torch.device, use_safetensors: bool = True):
        try:
            model = AutoModel.from_pretrained(name, use_safetensors=use_safetensors)
            model.to(device)
            return model
        except TypeError:
            model = AutoModel.from_pretrained(name)
            model.to(device)
            return model
        except Exception as e:
            msg = str(e).lower()
            if ("torch" in msg and "2.6" in msg) or ("vulnerability" in msg) or ("cve" in msg):
                raise RuntimeError(
                    f"Loading HF model '{name}' failed due to transformers / torch safety check:\n{e}\n"
                    "Possible solutions: use safetensors, upgrade torch, or use a compatible transformers version."
                ) from e
            model = AutoModel.from_pretrained(name)
            model.to(device)
            return model

    def _encode_text_tokens(
        text_list,
        tokenizer,
        model,
        device,
        batch_size=4,
        max_length=512,
        desc="[LM] token encoding",
        mask_special_tokens=True,
    ):
        all_tokens = []
        all_masks = []
        model.eval()

        with torch.no_grad():
            for i in tqdm(range(0, len(text_list), batch_size), desc=desc, unit="batch"):
                batch = [str(x) for x in text_list[i:i + batch_size]]
                enc = tokenizer(
                    batch,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                    return_special_tokens_mask=True,
                )

                special_tokens_mask = enc.pop("special_tokens_mask", None)
                enc = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in enc.items()}
                if special_tokens_mask is not None:
                    special_tokens_mask = special_tokens_mask.to(device)

                try:
                    out = model(**enc)
                except TypeError as e:
                    if "token_type_ids" in str(e) and "token_type_ids" in enc:
                        enc.pop("token_type_ids")
                        out = model(**enc)
                    else:
                        raise e

                if hasattr(out, "last_hidden_state"):
                    hs = out.last_hidden_state
                elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                    hs = out[0]
                else:
                    raise RuntimeError("Unsupported model output structure for token-level LM encoding.")

                attention_mask = enc["attention_mask"].bool()
                pad_mask = ~attention_mask
                if mask_special_tokens and special_tokens_mask is not None:
                    pad_mask = pad_mask | special_tokens_mask.bool()

                all_tokens.append(hs.detach().cpu().numpy().astype(np.float32))
                all_masks.append(pad_mask.detach().cpu().numpy().astype(bool))

        if len(all_tokens) == 0:
            hidden_size = int(getattr(model.config, "hidden_size", 0))
            return (
                np.zeros((0, max_length, hidden_size), dtype=np.float32),
                np.ones((0, max_length), dtype=bool),
            )

        return (
            np.concatenate(all_tokens, axis=0).astype(np.float32),
            np.concatenate(all_masks, axis=0).astype(bool),
        )

    def _make_pair_table(lig_list, pro_list, Y):
        pair_lig_idx = []
        pair_pro_idx = []
        y_list = []
        ids_list = []
        smiles_list = []
        seq_list = []

        n_lig, n_pro = Y.shape
        for i in range(n_lig):
            for j in range(n_pro):
                y_ij = Y[i, j]
                if np.isnan(y_ij):
                    continue
                pair_lig_idx.append(i)
                pair_pro_idx.append(j)
                y_list.append(float(y_ij))
                ids_list.append(f"L{i}_P{j}")
                smiles_list.append(lig_list[i])
                seq_list.append(pro_list[j])

        return {
            "pair_lig_idx": np.array(pair_lig_idx, dtype=np.int64),
            "pair_pro_idx": np.array(pair_pro_idx, dtype=np.int64),
            "y": np.array(y_list, dtype=np.float32),
            "ids": np.array(ids_list, dtype=object),
            "smiles": np.array(smiles_list, dtype=object),
            "seq": np.array(seq_list, dtype=object),
        }

    def _build_strict_split_indices(lig_list, pro_list, pair_table, scaffolds):
        # 1) Ligand groups by Murcko scaffold.
        lig_groups = defaultdict(list)
        for i, scaf in enumerate(scaffolds):
            lig_groups[str(scaf)].append(i)

        # 2) Protein groups by exact cleaned full sequence.
        pro_groups = defaultdict(list)
        for j, seq in enumerate(pro_list):
            pro_groups[str(seq)].append(j)

        lig_split, lig_group_split = _split_groups(lig_groups, len(lig_list))
        pro_split, pro_group_split = _split_groups(pro_groups, len(pro_list))

        lig_to_split = {}
        for sp, arr in lig_split.items():
            for x in arr:
                lig_to_split[int(x)] = sp

        pro_to_split = {}
        for sp, arr in pro_split.items():
            for x in arr:
                pro_to_split[int(x)] = sp

        pair_split_indices = {"train": [], "val": [], "test": []}
        excluded = []

        for k, (li, pj) in enumerate(zip(pair_table["pair_lig_idx"], pair_table["pair_pro_idx"])):
            lsp = lig_to_split[int(li)]
            psp = pro_to_split[int(pj)]
            if lsp == psp:
                pair_split_indices[lsp].append(k)
            else:
                excluded.append(k)

        pair_split_indices = {k: np.array(v, dtype=np.int64) for k, v in pair_split_indices.items()}
        excluded = np.array(excluded, dtype=np.int64)

        meta = {
            "split_protocol": "Davis strict ligand-scaffold and protein-sequence-disjoint product split",
            "split_seed": int(split_seed),
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "n_ligands": int(len(lig_list)),
            "n_proteins": int(len(pro_list)),
            "n_scaffold_groups": int(len(lig_groups)),
            "n_protein_sequence_groups": int(len(pro_groups)),
            "ligand_counts": {sp: int(len(arr)) for sp, arr in lig_split.items()},
            "protein_counts": {sp: int(len(arr)) for sp, arr in pro_split.items()},
            "pair_counts": {sp: int(len(arr)) for sp, arr in pair_split_indices.items()},
            "excluded_cross_partition_pairs": int(len(excluded)),
        }

        return pair_split_indices, excluded, lig_split, pro_split, meta

    def _audit_and_save(pair_split_indices, pair_table, scaffolds, save_path):
        lines = []
        lines.append("========== Davis Strict Split Audit ==========")
        lines.append("Protocol: ligand Murcko scaffold split + protein full-sequence disjoint split")
        lines.append("A pair is kept only when ligand split == protein split; cross-partition pairs are excluded.")
        lines.append("")

        split_names = ["train", "val", "test"]
        split_sets = {}
        for sp in split_names:
            idx = pair_split_indices[sp]
            lig_idx = pair_table["pair_lig_idx"][idx]
            pro_idx = pair_table["pair_pro_idx"][idx]
            split_sets[sp] = {
                "ids": set([str(x) for x in pair_table["ids"][idx]]),
                "smiles": set([str(x) for x in pair_table["smiles"][idx]]),
                "seq": set([str(x) for x in pair_table["seq"][idx]]),
                "scaffold": set([str(scaffolds[int(i)]) for i in lig_idx]),
                "lig_idx": set([int(x) for x in lig_idx]),
                "pro_idx": set([int(x) for x in pro_idx]),
            }
            lines.append(
                f"[{sp}] pairs={len(idx)} | ligands={len(split_sets[sp]['lig_idx'])} | "
                f"proteins={len(split_sets[sp]['pro_idx'])} | scaffolds={len(split_sets[sp]['scaffold'])}"
            )

        lines.append("")
        for i, a in enumerate(split_names):
            for b in split_names[i + 1:]:
                lines.append(f"--- {a} vs {b} ---")
                for field in ["ids", "scaffold", "seq", "smiles", "lig_idx", "pro_idx"]:
                    ov = sorted(split_sets[a][field].intersection(split_sets[b][field]))
                    lines.append(f"{field}_overlap={len(ov)} | examples={ov[:5]}")
                lines.append("")

        text = "\n".join(lines)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(text)

    def _save_pair_split(sp, pair_indices, pair_table, scaffolds):
        p = _pair_cache_path(sp)
        idx = np.asarray(pair_indices, dtype=np.int64)
        lig_idx = pair_table["pair_lig_idx"][idx]
        pro_idx = pair_table["pair_pro_idx"][idx]
        np.savez_compressed(
            p,
            ids=pair_table["ids"][idx],
            y=pair_table["y"][idx].astype(np.float32),
            smiles=pair_table["smiles"][idx],
            seq=pair_table["seq"][idx],
            pair_lig_idx=lig_idx.astype(np.int64),
            pair_pro_idx=pro_idx.astype(np.int64),
            ligand_scaffold=np.array([scaffolds[int(i)] for i in lig_idx], dtype=object),
        )

    def _build_and_cache_all():
        lig_ids, lig_list, pro_ids, pro_list, Y = _load_raw_davis()

        print(f"[DAVIS] ligands={len(lig_list)}, proteins={len(pro_list)}, Y={Y.shape}")
        scaffolds = np.array([_get_murcko_scaffold(smi) for smi in lig_list], dtype=object)

        pair_table = _make_pair_table(lig_list, pro_list, Y)
        print(f"[DAVIS] valid measured pairs={len(pair_table['y'])}")

        pair_split_indices, excluded, lig_split, pro_split, meta = _build_strict_split_indices(
            lig_list=lig_list,
            pro_list=pro_list,
            pair_table=pair_table,
            scaffolds=scaffolds,
        )

        for sp in ("train", "val", "test"):
            if len(pair_split_indices[sp]) == 0:
                raise RuntimeError(
                    f"Davis strict split produced empty {sp} set. "
                    "Try changing split_seed/train_ratio/val_ratio or use a less strict protocol."
                )

        # Save split meta.
        with open(split_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # Save pair split files.
        for sp in ("train", "val", "test"):
            _save_pair_split(sp, pair_split_indices[sp], pair_table, scaffolds)

        _audit_and_save(pair_split_indices, pair_table, scaffolds, audit_path)

        # Encode unique ligand/protein token banks.
        lm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LM|DAVIS] using device: {lm_device}")

        print(f"[LM|DAVIS] loading ChemBERTa: {chemberta_model_name}")
        chem_tok = AutoTokenizer.from_pretrained(chemberta_model_name)
        chem_model = _hf_load_model(chemberta_model_name, lm_device, use_safetensors)
        drug_lm_tokens_bank, drug_lm_mask_bank = _encode_text_tokens(
            lig_list.tolist(),
            chem_tok,
            chem_model,
            lm_device,
            batch_size=lm_batch_size,
            max_length=chem_max_len,
            desc="[LM|DAVIS] ChemBERTa token encoding",
            mask_special_tokens=mask_special_tokens,
        )
        try:
            del chem_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        print(f"[LM|DAVIS] loading ESM-2: {esm2_model_name}")
        esm_tok = AutoTokenizer.from_pretrained(esm2_model_name)
        esm_model = _hf_load_model(esm2_model_name, lm_device, use_safetensors)
        prot_lm_tokens_bank, prot_lm_mask_bank = _encode_text_tokens(
            pro_list.tolist(),
            esm_tok,
            esm_model,
            lm_device,
            batch_size=max(1, min(2, lm_batch_size)),
            max_length=prot_max_len,
            desc="[LM|DAVIS] ESM-2 token encoding",
            mask_special_tokens=mask_special_tokens,
        )
        try:
            del esm_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        np.savez_compressed(
            token_bank_path,
            lig_ids=lig_ids,
            lig_smiles=lig_list,
            lig_scaffold=scaffolds,
            pro_ids=pro_ids,
            pro_seq=pro_list,
            drug_lm_tokens_bank=drug_lm_tokens_bank.astype(np.float32),
            drug_lm_mask_bank=drug_lm_mask_bank.astype(bool),
            prot_lm_tokens_bank=prot_lm_tokens_bank.astype(np.float32),
            prot_lm_mask_bank=prot_lm_mask_bank.astype(bool),
        )

        print("[CACHE|DAVIS] strict token cache done.")

    def _cache_ready():
        if force_refresh:
            return False
        if not token_bank_path.exists():
            return False
        for sp in ("train", "val", "test"):
            if not _pair_cache_path(sp).exists():
                return False
        return True

    if split not in ("train", "val", "test", "all"):
        raise ValueError(f"split must be 'train'/'val'/'test'/'all', got {split!r}")

    if not _cache_ready():
        _build_and_cache_all()

    token_bank = np.load(token_bank_path, allow_pickle=True)

    def _load_split(sp: str):
        d = np.load(_pair_cache_path(sp), allow_pickle=True)

        pair_lig_idx = d["pair_lig_idx"].astype(np.int64)
        pair_pro_idx = d["pair_pro_idx"].astype(np.int64)

        pkg = {
            "ids": d["ids"],
            "y": d["y"].astype(np.float32),
            "smiles": d["smiles"],
            "seq": d["seq"],
            "pair_lig_idx": pair_lig_idx,
            "pair_pro_idx": pair_pro_idx,
            "ligand_scaffold": d["ligand_scaffold"],
            "drug_lm_tokens_bank": token_bank["drug_lm_tokens_bank"].astype(np.float32),
            "drug_lm_mask_bank": token_bank["drug_lm_mask_bank"].astype(bool),
            "prot_lm_tokens_bank": token_bank["prot_lm_tokens_bank"].astype(np.float32),
            "prot_lm_mask_bank": token_bank["prot_lm_mask_bank"].astype(bool),
            "lig_ids": token_bank["lig_ids"],
            "pro_ids": token_bank["pro_ids"],
            "split_protocol": "Davis strict ligand-scaffold and protein-sequence-disjoint product split",
            "g_lig": None,
            "g_prot": None,
        }

        if return_pair_level_tokens:
            # Not recommended for large token banks; useful only for a quick compatibility test.
            pkg["drug_lm_tokens"] = pkg["drug_lm_tokens_bank"][pair_lig_idx]
            pkg["drug_lm_mask"] = pkg["drug_lm_mask_bank"][pair_lig_idx]
            pkg["prot_lm_tokens"] = pkg["prot_lm_tokens_bank"][pair_pro_idx]
            pkg["prot_lm_mask"] = pkg["prot_lm_mask_bank"][pair_pro_idx]

        return pkg

    if split in ("train", "val", "test"):
        return {split: _load_split(split)}

    parts = {sp: _load_split(sp) for sp in ("train", "val", "test")}
    return {"all": parts}


# -*- coding: utf-8 -*-
# =========================================================
# KIBA token-level 1D loader with strict ligand-scaffold and
# protein-sequence-disjoint split.
#
# This function is designed to be copied into util_load.py.
# It follows the same design as the Davis cold-start loader:
#   1. token-level ChemBERTa / ESM-2 embeddings;
#   2. ligand Murcko scaffold split;
#   3. protein full-sequence-disjoint split;
#   4. pair is retained only when ligand split == protein split;
#   5. unique token banks are stored once and indexed by pair_lig_idx / pair_pro_idx.
# =========================================================


def LoadData_kiba_lm_1d_token_scaffold_seqsplit(
    data_dir: str,
    split: str = "all",
    out_1d: str = "../dataset/kiba/processed_lm_1d_token_scaffold_seqsplit",
    # KIBA labels are usually already KIBA scores, so no log transform by default.
    # Keep this option only for compatibility with Davis-style code.
    logspace_trans: bool = False,
    # --- LM models ---
    esm2_model_name: str = "facebook/esm2_t33_650M_UR50D",
    chemberta_model_name: str = "DeepChem/ChemBERTa-77M-MTR",
    lm_batch_size: int = 4,
    chem_max_len: int = 128,
    prot_max_len: int = 1024,
    use_safetensors: bool = True,
    mask_special_tokens: bool = True,
    # --- strict split ---
    split_seed: int = 2023,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    force_refresh: bool = False,
    # --- compatibility / memory option ---
    return_pair_level_tokens: bool = False,
) -> dict:
    """
    KIBA + token-level LM loader with strict ligand scaffold / protein sequence-disjoint split.

    Why this loader is needed:
        1. The old LM loaders usually store CLS/first-token pooled vectors [N, D], which
           cannot support token-level modeling.
        2. Pair-level random split lets the same ligand scaffold and the same protein
           sequence appear in train/val/test, which is vulnerable to leakage criticism.
        3. KIBA is a 1D-only benchmark and has no 3D pocket branch; therefore, the
           multi-chain 1D/3D mismatch issue does not apply here. The protein sequence
           used here is the full sequence from proteins.txt.

    Split protocol:
        - Ligands are grouped by Bemis-Murcko scaffold using RDKit.
        - Proteins are grouped by cleaned full protein sequence.
        - Ligand scaffold groups and protein sequence groups are separately assigned to
          train/val/test.
        - A pair is kept in a split only when BOTH its ligand and protein belong to that
          split. Cross-partition pairs are excluded. This guarantees:
              train/val/test have disjoint ligand scaffolds;
              train/val/test have disjoint protein sequences.

    Expected raw files under data_dir:
        ligands_can.txt : JSON dict {ligand_id: SMILES}
        proteins.txt    : JSON dict {protein_id: sequence}
        Y               : pickle affinity matrix, usually [n_ligand, n_protein]

    Return for one split:
        {
            'ids', 'y', 'smiles', 'seq',
            'pair_lig_idx', 'pair_pro_idx',
            'drug_lm_tokens_bank', 'drug_lm_mask_bank',
            'prot_lm_tokens_bank', 'prot_lm_mask_bank',
            'ligand_scaffold', 'split_protocol',
            optionally 'drug_lm_tokens', 'drug_lm_mask', 'prot_lm_tokens', 'prot_lm_mask'
        }

    Important:
        By default this loader DOES NOT expand unique protein token embeddings to pair-level
        arrays, because KIBA can contain many measured ligand-protein pairs. In the training
        Dataset, use pair_lig_idx/pair_pro_idx to index the token banks.
    """
    from pathlib import Path
    import json
    import pickle
    from collections import OrderedDict, defaultdict

    import numpy as np
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModel

    data_dir = Path(data_dir)
    out_1d = Path(out_1d)
    out_1d.mkdir(parents=True, exist_ok=True)

    token_bank_path = out_1d / "kiba_unique_lm_tokens.npz"
    split_meta_path = out_1d / "kiba_split_meta_scaffold_seqsplit.json"
    audit_path = out_1d / "kiba_split_audit_scaffold_seqsplit.txt"

    def _pair_cache_path(sp: str):
        return out_1d / f"kiba_{sp}_pairs_scaffold_seqsplit.npz"

    def _clean_protein_sequence(seq):
        if seq is None:
            return ""
        seq = str(seq).upper()
        valid = set("ACDEFGHIKLMNPQRSTVWY")
        return "".join([c for c in seq if c in valid])

    def _clean_smiles(smi):
        if smi is None:
            return ""
        return str(smi).strip()

    def _load_raw_kiba():
        lig_path = data_dir / "ligands_can.txt"
        pro_path = data_dir / "proteins.txt"
        y_path = data_dir / "Y"

        if not lig_path.exists():
            raise FileNotFoundError(f"Cannot find {lig_path}")
        if not pro_path.exists():
            raise FileNotFoundError(f"Cannot find {pro_path}")
        if not y_path.exists():
            raise FileNotFoundError(f"Cannot find {y_path}")

        with lig_path.open("r", encoding="utf-8") as f:
            lig_dict = json.load(f, object_pairs_hook=OrderedDict)
        with pro_path.open("r", encoding="utf-8") as f:
            pro_dict = json.load(f, object_pairs_hook=OrderedDict)
        with y_path.open("rb") as f:
            Y = pickle.load(f, encoding="latin1")

        lig_ids = np.array(list(lig_dict.keys()), dtype=object)
        pro_ids = np.array(list(pro_dict.keys()), dtype=object)
        lig_list = np.array([_clean_smiles(x) for x in lig_dict.values()], dtype=object)
        pro_list = np.array([_clean_protein_sequence(x) for x in pro_dict.values()], dtype=object)

        Y = np.asarray(Y, dtype=np.float64)

        # Standard DeepDTA-style Y is [n_ligand, n_protein].
        # If a local copy is transposed, fix it automatically.
        if Y.shape != (len(lig_list), len(pro_list)):
            if Y.T.shape == (len(lig_list), len(pro_list)):
                print(f"[KIBA] Y shape {Y.shape} looks transposed; using Y.T")
                Y = Y.T
            else:
                raise ValueError(
                    f"Unexpected KIBA Y shape={Y.shape}; expected "
                    f"({len(lig_list)}, {len(pro_list)})"
                )

        # Usually do NOT apply Davis-style pKd transform to KIBA.
        # This option exists only for compatibility.
        if logspace_trans:
            Y = -np.log10(Y / 1e9)

        return lig_ids, lig_list, pro_ids, pro_list, Y

    def _get_murcko_scaffold(smi: str) -> str:
        """
        Return Bemis-Murcko scaffold. If RDKit cannot parse the molecule or the scaffold
        is empty, fall back to canonical SMILES / raw SMILES so that every ligand belongs
        to a deterministic group.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                return f"INVALID::{str(smi)}"
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            if scaf is None or str(scaf).strip() == "":
                can = Chem.MolToSmiles(mol, canonical=True)
                return f"NO_SCAFFOLD::{can}"
            return str(scaf)
        except Exception:
            return f"FALLBACK::{str(smi)}"

    def _split_groups(groups: dict, total_items: int):
        """
        Greedy group split. A whole group is assigned to one split, never broken.
        """
        rng = np.random.RandomState(split_seed)
        items = list(groups.items())
        rng.shuffle(items)
        items = sorted(items, key=lambda kv: len(kv[1]), reverse=True)

        n_train_target = int(total_items * train_ratio)
        n_val_target = int(total_items * val_ratio)

        split_to_indices = {"train": [], "val": [], "test": []}
        split_to_groups = {"train": [], "val": [], "test": []}

        for g, idxs in items:
            idxs = list(idxs)
            if len(split_to_indices["train"]) + len(idxs) <= n_train_target:
                sp = "train"
            elif len(split_to_indices["val"]) + len(idxs) <= n_val_target:
                sp = "val"
            else:
                sp = "test"
            split_to_indices[sp].extend(idxs)
            split_to_groups[sp].append(g)

        return (
            {k: np.array(v, dtype=np.int64) for k, v in split_to_indices.items()},
            {k: set(v) for k, v in split_to_groups.items()},
        )

    def _hf_load_model(name: str, device: torch.device, use_safetensors: bool = True):
        try:
            model = AutoModel.from_pretrained(name, use_safetensors=use_safetensors)
            model.to(device)
            return model
        except TypeError:
            model = AutoModel.from_pretrained(name)
            model.to(device)
            return model
        except Exception as e:
            msg = str(e).lower()
            if ("torch" in msg and "2.6" in msg) or ("vulnerability" in msg) or ("cve" in msg):
                raise RuntimeError(
                    f"Loading HF model '{name}' failed due to transformers / torch safety check:\n{e}\n"
                    "Possible solutions: use safetensors, upgrade torch, or use a compatible transformers version."
                ) from e
            model = AutoModel.from_pretrained(name)
            model.to(device)
            return model

    def _encode_text_tokens(
        text_list,
        tokenizer,
        model,
        device,
        batch_size=4,
        max_length=512,
        desc="[LM] token encoding",
        mask_special_tokens=True,
    ):
        all_tokens = []
        all_masks = []
        model.eval()

        with torch.no_grad():
            for i in tqdm(range(0, len(text_list), batch_size), desc=desc, unit="batch"):
                batch = [str(x) for x in text_list[i:i + batch_size]]
                enc = tokenizer(
                    batch,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                    return_special_tokens_mask=True,
                )

                special_tokens_mask = enc.pop("special_tokens_mask", None)
                enc = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in enc.items()}
                if special_tokens_mask is not None:
                    special_tokens_mask = special_tokens_mask.to(device)

                try:
                    out = model(**enc)
                except TypeError as e:
                    if "token_type_ids" in str(e) and "token_type_ids" in enc:
                        enc.pop("token_type_ids")
                        out = model(**enc)
                    else:
                        raise e

                if hasattr(out, "last_hidden_state"):
                    hs = out.last_hidden_state
                elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                    hs = out[0]
                else:
                    raise RuntimeError("Unsupported model output structure for token-level LM encoding.")

                attention_mask = enc["attention_mask"].bool()
                pad_mask = ~attention_mask
                if mask_special_tokens and special_tokens_mask is not None:
                    pad_mask = pad_mask | special_tokens_mask.bool()

                all_tokens.append(hs.detach().cpu().numpy().astype(np.float32))
                all_masks.append(pad_mask.detach().cpu().numpy().astype(bool))

        if len(all_tokens) == 0:
            hidden_size = int(getattr(model.config, "hidden_size", 0))
            return (
                np.zeros((0, max_length, hidden_size), dtype=np.float32),
                np.ones((0, max_length), dtype=bool),
            )

        return (
            np.concatenate(all_tokens, axis=0).astype(np.float32),
            np.concatenate(all_masks, axis=0).astype(bool),
        )

    def _make_pair_table(lig_list, pro_list, Y):
        pair_lig_idx = []
        pair_pro_idx = []
        y_list = []
        ids_list = []
        smiles_list = []
        seq_list = []

        n_lig, n_pro = Y.shape
        for i in range(n_lig):
            for j in range(n_pro):
                y_ij = Y[i, j]
                if np.isnan(y_ij):
                    continue
                pair_lig_idx.append(i)
                pair_pro_idx.append(j)
                y_list.append(float(y_ij))
                ids_list.append(f"L{i}_P{j}")
                smiles_list.append(lig_list[i])
                seq_list.append(pro_list[j])

        return {
            "pair_lig_idx": np.array(pair_lig_idx, dtype=np.int64),
            "pair_pro_idx": np.array(pair_pro_idx, dtype=np.int64),
            "y": np.array(y_list, dtype=np.float32),
            "ids": np.array(ids_list, dtype=object),
            "smiles": np.array(smiles_list, dtype=object),
            "seq": np.array(seq_list, dtype=object),
        }

    def _build_strict_split_indices(lig_list, pro_list, pair_table, scaffolds):
        # 1) Ligand groups by Murcko scaffold.
        lig_groups = defaultdict(list)
        for i, scaf in enumerate(scaffolds):
            lig_groups[str(scaf)].append(i)

        # 2) Protein groups by exact cleaned full sequence.
        pro_groups = defaultdict(list)
        for j, seq in enumerate(pro_list):
            pro_groups[str(seq)].append(j)

        lig_split, lig_group_split = _split_groups(lig_groups, len(lig_list))
        pro_split, pro_group_split = _split_groups(pro_groups, len(pro_list))

        lig_to_split = {}
        for sp, arr in lig_split.items():
            for x in arr:
                lig_to_split[int(x)] = sp

        pro_to_split = {}
        for sp, arr in pro_split.items():
            for x in arr:
                pro_to_split[int(x)] = sp

        pair_split_indices = {"train": [], "val": [], "test": []}
        excluded = []

        for k, (li, pj) in enumerate(zip(pair_table["pair_lig_idx"], pair_table["pair_pro_idx"])):
            lsp = lig_to_split[int(li)]
            psp = pro_to_split[int(pj)]
            if lsp == psp:
                pair_split_indices[lsp].append(k)
            else:
                excluded.append(k)

        pair_split_indices = {k: np.array(v, dtype=np.int64) for k, v in pair_split_indices.items()}
        excluded = np.array(excluded, dtype=np.int64)

        meta = {
            "split_protocol": "KIBA strict ligand-scaffold and protein-sequence-disjoint product split",
            "split_seed": int(split_seed),
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "test_ratio": float(1.0 - train_ratio - val_ratio),
            "n_ligands": int(len(lig_list)),
            "n_proteins": int(len(pro_list)),
            "n_scaffold_groups": int(len(lig_groups)),
            "n_protein_sequence_groups": int(len(pro_groups)),
            "ligand_counts": {sp: int(len(arr)) for sp, arr in lig_split.items()},
            "protein_counts": {sp: int(len(arr)) for sp, arr in pro_split.items()},
            "pair_counts": {sp: int(len(arr)) for sp, arr in pair_split_indices.items()},
            "excluded_cross_partition_pairs": int(len(excluded)),
        }

        return pair_split_indices, excluded, lig_split, pro_split, meta

    def _audit_and_save(pair_split_indices, pair_table, scaffolds, save_path):
        lines = []
        lines.append("========== KIBA Strict Split Audit ==========")
        lines.append("Protocol: ligand Murcko scaffold split + protein full-sequence disjoint split")
        lines.append("A pair is kept only when ligand split == protein split; cross-partition pairs are excluded.")
        lines.append("")

        split_names = ["train", "val", "test"]
        split_sets = {}
        for sp in split_names:
            idx = pair_split_indices[sp]
            lig_idx = pair_table["pair_lig_idx"][idx]
            pro_idx = pair_table["pair_pro_idx"][idx]
            split_sets[sp] = {
                "ids": set([str(x) for x in pair_table["ids"][idx]]),
                "smiles": set([str(x) for x in pair_table["smiles"][idx]]),
                "seq": set([str(x) for x in pair_table["seq"][idx]]),
                "scaffold": set([str(scaffolds[int(i)]) for i in lig_idx]),
                "lig_idx": set([int(x) for x in lig_idx]),
                "pro_idx": set([int(x) for x in pro_idx]),
            }
            lines.append(
                f"[{sp}] pairs={len(idx)} | ligands={len(split_sets[sp]['lig_idx'])} | "
                f"proteins={len(split_sets[sp]['pro_idx'])} | scaffolds={len(split_sets[sp]['scaffold'])}"
            )

        lines.append("")
        for i, a in enumerate(split_names):
            for b in split_names[i + 1:]:
                lines.append(f"--- {a} vs {b} ---")
                for field in ["ids", "scaffold", "seq", "smiles", "lig_idx", "pro_idx"]:
                    ov = sorted(split_sets[a][field].intersection(split_sets[b][field]))
                    lines.append(f"{field}_overlap={len(ov)} | examples={ov[:5]}")
                lines.append("")

        text = "\n".join(lines)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(text)

    def _save_pair_split(sp, pair_indices, pair_table, scaffolds):
        p = _pair_cache_path(sp)
        idx = np.asarray(pair_indices, dtype=np.int64)
        lig_idx = pair_table["pair_lig_idx"][idx]
        pro_idx = pair_table["pair_pro_idx"][idx]
        np.savez_compressed(
            p,
            ids=pair_table["ids"][idx],
            y=pair_table["y"][idx].astype(np.float32),
            smiles=pair_table["smiles"][idx],
            seq=pair_table["seq"][idx],
            pair_lig_idx=lig_idx.astype(np.int64),
            pair_pro_idx=pro_idx.astype(np.int64),
            ligand_scaffold=np.array([scaffolds[int(i)] for i in lig_idx], dtype=object),
        )

    def _build_and_cache_all():
        lig_ids, lig_list, pro_ids, pro_list, Y = _load_raw_kiba()

        print(f"[KIBA] ligands={len(lig_list)}, proteins={len(pro_list)}, Y={Y.shape}")
        scaffolds = np.array([_get_murcko_scaffold(smi) for smi in lig_list], dtype=object)

        pair_table = _make_pair_table(lig_list, pro_list, Y)
        print(f"[KIBA] valid measured pairs={len(pair_table['y'])}")

        pair_split_indices, excluded, lig_split, pro_split, meta = _build_strict_split_indices(
            lig_list=lig_list,
            pro_list=pro_list,
            pair_table=pair_table,
            scaffolds=scaffolds,
        )

        for sp in ("train", "val", "test"):
            if len(pair_split_indices[sp]) == 0:
                raise RuntimeError(
                    f"KIBA strict split produced empty {sp} set. "
                    "Try changing split_seed/train_ratio/val_ratio or use a less strict protocol."
                )

        with open(split_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        for sp in ("train", "val", "test"):
            _save_pair_split(sp, pair_split_indices[sp], pair_table, scaffolds)

        _audit_and_save(pair_split_indices, pair_table, scaffolds, audit_path)

        lm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LM|KIBA] using device: {lm_device}")

        print(f"[LM|KIBA] loading ChemBERTa: {chemberta_model_name}")
        chem_tok = AutoTokenizer.from_pretrained(chemberta_model_name)
        chem_model = _hf_load_model(chemberta_model_name, lm_device, use_safetensors)
        drug_lm_tokens_bank, drug_lm_mask_bank = _encode_text_tokens(
            lig_list.tolist(),
            chem_tok,
            chem_model,
            lm_device,
            batch_size=lm_batch_size,
            max_length=chem_max_len,
            desc="[LM|KIBA] ChemBERTa token encoding",
            mask_special_tokens=mask_special_tokens,
        )
        try:
            del chem_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        print(f"[LM|KIBA] loading ESM-2: {esm2_model_name}")
        esm_tok = AutoTokenizer.from_pretrained(esm2_model_name)
        esm_model = _hf_load_model(esm2_model_name, lm_device, use_safetensors)
        prot_lm_tokens_bank, prot_lm_mask_bank = _encode_text_tokens(
            pro_list.tolist(),
            esm_tok,
            esm_model,
            lm_device,
            batch_size=max(1, min(2, lm_batch_size)),
            max_length=prot_max_len,
            desc="[LM|KIBA] ESM-2 token encoding",
            mask_special_tokens=mask_special_tokens,
        )
        try:
            del esm_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        np.savez_compressed(
            token_bank_path,
            lig_ids=lig_ids,
            lig_smiles=lig_list,
            lig_scaffold=scaffolds,
            pro_ids=pro_ids,
            pro_seq=pro_list,
            drug_lm_tokens_bank=drug_lm_tokens_bank.astype(np.float32),
            drug_lm_mask_bank=drug_lm_mask_bank.astype(bool),
            prot_lm_tokens_bank=prot_lm_tokens_bank.astype(np.float32),
            prot_lm_mask_bank=prot_lm_mask_bank.astype(bool),
        )

        print("[CACHE|KIBA] strict token cache done.")

    def _cache_ready():
        if force_refresh:
            return False
        if not token_bank_path.exists():
            return False
        for sp in ("train", "val", "test"):
            if not _pair_cache_path(sp).exists():
                return False
        return True

    if split not in ("train", "val", "test", "all"):
        raise ValueError(f"split must be 'train'/'val'/'test'/'all', got {split!r}")

    if not _cache_ready():
        _build_and_cache_all()

    token_bank = np.load(token_bank_path, allow_pickle=True)

    def _load_split(sp: str):
        d = np.load(_pair_cache_path(sp), allow_pickle=True)

        pair_lig_idx = d["pair_lig_idx"].astype(np.int64)
        pair_pro_idx = d["pair_pro_idx"].astype(np.int64)

        pkg = {
            "ids": d["ids"],
            "y": d["y"].astype(np.float32),
            "smiles": d["smiles"],
            "seq": d["seq"],
            "pair_lig_idx": pair_lig_idx,
            "pair_pro_idx": pair_pro_idx,
            "ligand_scaffold": d["ligand_scaffold"],
            "drug_lm_tokens_bank": token_bank["drug_lm_tokens_bank"].astype(np.float32),
            "drug_lm_mask_bank": token_bank["drug_lm_mask_bank"].astype(bool),
            "prot_lm_tokens_bank": token_bank["prot_lm_tokens_bank"].astype(np.float32),
            "prot_lm_mask_bank": token_bank["prot_lm_mask_bank"].astype(bool),
            "lig_ids": token_bank["lig_ids"],
            "pro_ids": token_bank["pro_ids"],
            "split_protocol": "KIBA strict ligand-scaffold and protein-sequence-disjoint product split",
            "g_lig": None,
            "g_prot": None,
        }

        if return_pair_level_tokens:
            pkg["drug_lm_tokens"] = pkg["drug_lm_tokens_bank"][pair_lig_idx]
            pkg["drug_lm_mask"] = pkg["drug_lm_mask_bank"][pair_lig_idx]
            pkg["prot_lm_tokens"] = pkg["prot_lm_tokens_bank"][pair_pro_idx]
            pkg["prot_lm_mask"] = pkg["prot_lm_mask_bank"][pair_pro_idx]

        return pkg

    if split in ("train", "val", "test"):
        return {split: _load_split(split)}

    parts = {sp: _load_split(sp) for sp in ("train", "val", "test")}
    return {"all": parts}


def LoadData_bindingdb_lm_1d_token_scaffold_seqsplit(
    data_dir: str,
    split: str = "all",
    out_1d: str = "../dataset/bindingdb/processed_lm_1d_token_scaffold_seqsplit",
    logspace_trans: bool = False,
    # --- LM models ---
    esm2_model_name: str = "facebook/esm2_t33_650M_UR50D",
    chemberta_model_name: str = "DeepChem/ChemBERTa-77M-MTR",
    lm_batch_size: int = 4,
    chem_max_len: int = 128,
    prot_max_len: int = 1024,
    use_safetensors: bool = True,
    mask_special_tokens: bool = True,
    # --- strict split ---
    split_seed: int = 2023,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    force_refresh: bool = False,
    # --- compatibility / memory option ---
    return_pair_level_tokens: bool = False,
) -> dict:
    """
    BindingDB + token-level LM loader with strict ligand scaffold / protein sequence-disjoint split.

    这个版本主要解决两类问题：
        1. 旧版 BindingDB loader 只保存 CLS/first-token pooled 向量 [N, D]，
           不能支持后续真正的 token-level 模型。
        2. 旧版 BindingDB loader 保留原 bindingdb_test.csv 作为固定 test，
           并只在原 train CSV 内随机划分 train/val，无法保证 ligand scaffold
           与 protein sequence 在 train/val/test 之间严格不重叠。

    新版划分策略：
        - 将 bindingdb_train.csv 与 bindingdb_test.csv 合并为原始样本池；
        - 以 Bemis-Murcko scaffold 对 ligand 分组；
        - 以清洗后的完整蛋白序列对 protein 分组；
        - ligand scaffold groups 与 protein sequence groups 分别划分到 train/val/test；
        - 只有当某个样本的 ligand split 与 protein split 完全一致时，才保留到该 split；
          否则该 pair 会被排除。
        - 这样可以保证：
              train/val/test 之间 ligand scaffold 不重叠；
              train/val/test 之间 protein sequence 不重叠。

    说明：
        这个 loader 是 BindingDB 的 strict cold-start / leakage-controlled 版本。
        它不是旧版固定 bindingdb_test.csv 协议的直接替代。
        如果论文中还需要 common benchmark protocol，可继续保留旧 loader；
        本 loader 用于回应审稿人的严格 scaffold + sequence-disjoint 评估要求。

    输入 CSV 需要包含：
        compound_iso_smiles
        target_sequence
        affinity

    缓存文件：
        bindingdb_unique_lm_tokens.npz
        bindingdb_train_pairs_scaffold_seqsplit.npz
        bindingdb_val_pairs_scaffold_seqsplit.npz
        bindingdb_test_pairs_scaffold_seqsplit.npz
        bindingdb_split_meta_scaffold_seqsplit.json
        bindingdb_split_audit_scaffold_seqsplit.txt

    单个 split 返回：
        {
            'ids', 'y', 'smiles', 'seq',
            'pair_lig_idx', 'pair_pro_idx',
            'drug_lm_tokens_bank', 'drug_lm_mask_bank',
            'prot_lm_tokens_bank', 'prot_lm_mask_bank',
            'ligand_scaffold', 'source_csv', 'split_protocol',
            'lig_ids', 'pro_ids',
            'g_lig', 'g_prot',
            optionally:
                'drug_lm_tokens', 'drug_lm_mask',
                'prot_lm_tokens', 'prot_lm_mask'
        }

    重要：
        默认不会把 unique ligand/protein token bank 扩展到 pair-level，
        因为 BindingDB 样本量大，pair-level 展开会消耗大量内存。
        后续 Dataset 请根据 pair_lig_idx / pair_pro_idx 从 token bank 中取样。
    """
    from pathlib import Path
    import json
    from collections import OrderedDict, defaultdict

    import numpy as np
    import pandas as pd
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModel

    data_dir = Path(data_dir)
    out_1d = Path(out_1d)
    out_1d.mkdir(parents=True, exist_ok=True)

    token_bank_path = out_1d / "bindingdb_unique_lm_tokens.npz"
    split_meta_path = out_1d / "bindingdb_split_meta_scaffold_seqsplit.json"
    audit_path = out_1d / "bindingdb_split_audit_scaffold_seqsplit.txt"

    def _pair_cache_path(sp: str):
        return out_1d / f"bindingdb_{sp}_pairs_scaffold_seqsplit.npz"

    def _clean_protein_sequence(seq):
        if seq is None:
            return ""
        seq = str(seq).upper()
        valid = set("ACDEFGHIKLMNPQRSTVWY")
        return "".join([c for c in seq if c in valid])

    def _clean_smiles(smi):
        if smi is None:
            return ""
        return str(smi).strip()

    def _load_raw_bindingdb():
        """
        读取 train/test CSV，并合并为一个原始样本池。
        为了严格 scaffold/sequence-disjoint，后续会重新划分 train/val/test。
        """
        train_path = data_dir / "bindingdb_train.csv"
        test_path = data_dir / "bindingdb_test.csv"

        if not train_path.exists():
            raise FileNotFoundError(f"Cannot find {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Cannot find {test_path}")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        required_cols = ["compound_iso_smiles", "target_sequence", "affinity"]
        for col in required_cols:
            if col not in train_df.columns:
                raise ValueError(f"bindingdb_train.csv 缺少列: {col}")
            if col not in test_df.columns:
                raise ValueError(f"bindingdb_test.csv 缺少列: {col}")

        rows = []
        stats = {
            "raw_train_rows": int(len(train_df)),
            "raw_test_rows": int(len(test_df)),
            "kept_rows": 0,
            "drop_empty_smiles": 0,
            "drop_empty_sequence": 0,
            "drop_invalid_affinity": 0,
        }

        for source_name, df in (("train_csv", train_df), ("test_csv", test_df)):
            for row_idx, row in df.iterrows():
                smi = _clean_smiles(row["compound_iso_smiles"])
                seq = _clean_protein_sequence(row["target_sequence"])

                if smi == "":
                    stats["drop_empty_smiles"] += 1
                    continue

                if seq == "":
                    stats["drop_empty_sequence"] += 1
                    continue

                try:
                    y = float(row["affinity"])
                except Exception:
                    stats["drop_invalid_affinity"] += 1
                    continue

                if not np.isfinite(y):
                    stats["drop_invalid_affinity"] += 1
                    continue

                if logspace_trans:
                    y = float(np.log10(y + 1e-8))

                rows.append({
                    "id": f"{source_name}_{row_idx}",
                    "smiles": smi,
                    "seq": seq,
                    "y": y,
                    "source_csv": source_name,
                })

        stats["kept_rows"] = int(len(rows))

        if len(rows) == 0:
            raise RuntimeError("BindingDB 清洗后没有有效样本，请检查 CSV 字段与 affinity 数值。")

        ids = np.array([r["id"] for r in rows], dtype=object)
        smiles = np.array([r["smiles"] for r in rows], dtype=object)
        seq = np.array([r["seq"] for r in rows], dtype=object)
        y = np.array([r["y"] for r in rows], dtype=np.float32)
        source_csv = np.array([r["source_csv"] for r in rows], dtype=object)

        return ids, smiles, seq, y, source_csv, stats

    def _get_murcko_scaffold(smi: str) -> str:
        """
        计算 Bemis-Murcko scaffold。
        若 RDKit 无法解析或 scaffold 为空，则使用确定性的 fallback key。
        """
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold

            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                return f"INVALID::{str(smi)}"

            scaf = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol,
                includeChirality=False
            )

            if scaf is None or str(scaf).strip() == "":
                can = Chem.MolToSmiles(mol, canonical=True)
                return f"NO_SCAFFOLD::{can}"

            return str(scaf)

        except Exception:
            return f"FALLBACK::{str(smi)}"

    def _split_groups(groups: dict, total_items: int):
        """
        按完整 group 划分，group 不会被拆开。
        目标比例按 unique ligand / unique protein 数量近似控制。
        """
        rng = np.random.RandomState(split_seed)
        items = list(groups.items())
        rng.shuffle(items)
        items = sorted(items, key=lambda kv: len(kv[1]), reverse=True)

        n_train_target = int(total_items * train_ratio)
        n_val_target = int(total_items * val_ratio)

        split_to_indices = {"train": [], "val": [], "test": []}
        split_to_groups = {"train": [], "val": [], "test": []}

        for group_key, idxs in items:
            idxs = list(idxs)

            if len(split_to_indices["train"]) + len(idxs) <= n_train_target:
                sp = "train"
            elif len(split_to_indices["val"]) + len(idxs) <= n_val_target:
                sp = "val"
            else:
                sp = "test"

            split_to_indices[sp].extend(idxs)
            split_to_groups[sp].append(group_key)

        return (
            {k: np.array(v, dtype=np.int64) for k, v in split_to_indices.items()},
            {k: set(v) for k, v in split_to_groups.items()},
        )

    def _hf_load_model(name: str, device: torch.device, use_safetensors: bool = True):
        try:
            model = AutoModel.from_pretrained(name, use_safetensors=use_safetensors)
            model.to(device)
            return model
        except TypeError:
            model = AutoModel.from_pretrained(name)
            model.to(device)
            return model
        except Exception as e:
            msg = str(e).lower()
            if ("torch" in msg and "2.6" in msg) or ("vulnerability" in msg) or ("cve" in msg):
                raise RuntimeError(
                    f"Loading HF model '{name}' failed due to transformers / torch safety check:\n{e}\n"
                    "Possible solutions: use safetensors, upgrade torch, or use a compatible transformers version."
                ) from e
            model = AutoModel.from_pretrained(name)
            model.to(device)
            return model

    def _encode_text_tokens(
        text_list,
        tokenizer,
        model,
        device,
        batch_size=4,
        max_length=512,
        desc="[LM] token encoding",
        mask_special_tokens=True,
    ):
        """
        对 unique SMILES / sequence 生成 token-level last_hidden_state 与 padding mask。
        """
        all_tokens = []
        all_masks = []
        model.eval()

        with torch.no_grad():
            for i in tqdm(range(0, len(text_list), batch_size), desc=desc, unit="batch"):
                batch = [str(x) for x in text_list[i:i + batch_size]]
                enc = tokenizer(
                    batch,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                    return_special_tokens_mask=True,
                )

                special_tokens_mask = enc.pop("special_tokens_mask", None)
                enc = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in enc.items()
                }
                if special_tokens_mask is not None:
                    special_tokens_mask = special_tokens_mask.to(device)

                try:
                    out = model(**enc)
                except TypeError as e:
                    if "token_type_ids" in str(e) and "token_type_ids" in enc:
                        enc.pop("token_type_ids")
                        out = model(**enc)
                    else:
                        raise e

                if hasattr(out, "last_hidden_state"):
                    hs = out.last_hidden_state
                elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                    hs = out[0]
                else:
                    raise RuntimeError("Unsupported model output structure for token-level LM encoding.")

                attention_mask = enc["attention_mask"].bool()
                pad_mask = ~attention_mask

                if mask_special_tokens and special_tokens_mask is not None:
                    pad_mask = pad_mask | special_tokens_mask.bool()

                all_tokens.append(hs.detach().cpu().numpy().astype(np.float32))
                all_masks.append(pad_mask.detach().cpu().numpy().astype(bool))

        if len(all_tokens) == 0:
            hidden_size = int(getattr(model.config, "hidden_size", 0))
            return (
                np.zeros((0, max_length, hidden_size), dtype=np.float32),
                np.ones((0, max_length), dtype=bool),
            )

        return (
            np.concatenate(all_tokens, axis=0).astype(np.float32),
            np.concatenate(all_masks, axis=0).astype(bool),
        )

    def _make_pair_table(ids, smiles, seq, y, source_csv):
        """
        BindingDB 原始数据本身就是 pair-level 表格。
        这里构造 unique ligand/protein bank 索引。
        """
        lig_list = np.array(
            list(OrderedDict.fromkeys(smiles.tolist()).keys()),
            dtype=object
        )
        pro_list = np.array(
            list(OrderedDict.fromkeys(seq.tolist()).keys()),
            dtype=object
        )

        lig2idx = {str(smi): i for i, smi in enumerate(lig_list)}
        pro2idx = {str(s): i for i, s in enumerate(pro_list)}

        pair_lig_idx = np.array([lig2idx[str(smi)] for smi in smiles], dtype=np.int64)
        pair_pro_idx = np.array([pro2idx[str(s)] for s in seq], dtype=np.int64)

        pair_table = {
            "ids": ids,
            "y": y.astype(np.float32),
            "smiles": smiles,
            "seq": seq,
            "source_csv": source_csv,
            "pair_lig_idx": pair_lig_idx,
            "pair_pro_idx": pair_pro_idx,
        }

        return lig_list, pro_list, pair_table

    def _build_strict_split_indices(lig_list, pro_list, pair_table, scaffolds):
        # 1) Ligand groups by Murcko scaffold.
        lig_groups = defaultdict(list)
        for i, scaf in enumerate(scaffolds):
            lig_groups[str(scaf)].append(i)

        # 2) Protein groups by exact cleaned full sequence.
        pro_groups = defaultdict(list)
        for j, seq_text in enumerate(pro_list):
            pro_groups[str(seq_text)].append(j)

        lig_split, lig_group_split = _split_groups(lig_groups, len(lig_list))
        pro_split, pro_group_split = _split_groups(pro_groups, len(pro_list))

        lig_to_split = {}
        for sp, arr in lig_split.items():
            for x in arr:
                lig_to_split[int(x)] = sp

        pro_to_split = {}
        for sp, arr in pro_split.items():
            for x in arr:
                pro_to_split[int(x)] = sp

        pair_split_indices = {"train": [], "val": [], "test": []}
        excluded = []

        for k, (lig_idx, pro_idx) in enumerate(
            zip(pair_table["pair_lig_idx"], pair_table["pair_pro_idx"])
        ):
            lsp = lig_to_split[int(lig_idx)]
            psp = pro_to_split[int(pro_idx)]

            if lsp == psp:
                pair_split_indices[lsp].append(k)
            else:
                excluded.append(k)

        pair_split_indices = {
            k: np.array(v, dtype=np.int64)
            for k, v in pair_split_indices.items()
        }
        excluded = np.array(excluded, dtype=np.int64)

        meta = {
            "split_protocol": (
                "BindingDB strict ligand-scaffold and protein-sequence-disjoint "
                "product split over merged train/test CSV pool"
            ),
            "split_seed": int(split_seed),
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "test_ratio": float(1.0 - train_ratio - val_ratio),
            "n_pairs_before_strict_filter": int(len(pair_table["y"])),
            "n_pairs_excluded_cross_partition": int(len(excluded)),
            "n_pairs_after_strict_filter": int(sum(len(v) for v in pair_split_indices.values())),
            "n_ligands": int(len(lig_list)),
            "n_proteins": int(len(pro_list)),
            "n_scaffold_groups": int(len(lig_groups)),
            "n_protein_sequence_groups": int(len(pro_groups)),
            "ligand_counts": {sp: int(len(arr)) for sp, arr in lig_split.items()},
            "protein_counts": {sp: int(len(arr)) for sp, arr in pro_split.items()},
            "pair_counts": {sp: int(len(arr)) for sp, arr in pair_split_indices.items()},
            "source_csv_counts_after_filter": {
                sp: {
                    "train_csv": int(np.sum(pair_table["source_csv"][idx] == "train_csv")),
                    "test_csv": int(np.sum(pair_table["source_csv"][idx] == "test_csv")),
                }
                for sp, idx in pair_split_indices.items()
            },
        }

        return pair_split_indices, excluded, lig_split, pro_split, meta

    def _audit_and_save(pair_split_indices, pair_table, scaffolds, save_path):
        lines = []
        lines.append("========== BindingDB Strict Split Audit ==========")
        lines.append(
            "Protocol: ligand Murcko scaffold split + protein full-sequence disjoint split"
        )
        lines.append(
            "Raw bindingdb_train.csv and bindingdb_test.csv are merged into one pool, "
            "then re-split under the strict protocol."
        )
        lines.append(
            "A pair is kept only when ligand split == protein split; "
            "cross-partition pairs are excluded."
        )
        lines.append("")

        split_names = ["train", "val", "test"]
        split_sets = {}

        for sp in split_names:
            idx = pair_split_indices[sp]
            lig_idx = pair_table["pair_lig_idx"][idx]
            pro_idx = pair_table["pair_pro_idx"][idx]

            split_sets[sp] = {
                "ids": set([str(x) for x in pair_table["ids"][idx]]),
                "smiles": set([str(x) for x in pair_table["smiles"][idx]]),
                "seq": set([str(x) for x in pair_table["seq"][idx]]),
                "scaffold": set([str(scaffolds[int(i)]) for i in lig_idx]),
                "lig_idx": set([int(x) for x in lig_idx]),
                "pro_idx": set([int(x) for x in pro_idx]),
            }

            n_from_train_csv = int(np.sum(pair_table["source_csv"][idx] == "train_csv"))
            n_from_test_csv = int(np.sum(pair_table["source_csv"][idx] == "test_csv"))

            lines.append(
                f"[{sp}] pairs={len(idx)} | ligands={len(split_sets[sp]['lig_idx'])} | "
                f"proteins={len(split_sets[sp]['pro_idx'])} | "
                f"scaffolds={len(split_sets[sp]['scaffold'])} | "
                f"from_train_csv={n_from_train_csv} | from_test_csv={n_from_test_csv}"
            )

        lines.append("")

        for i, a in enumerate(split_names):
            for b in split_names[i + 1:]:
                lines.append(f"--- {a} vs {b} ---")
                for field in ["ids", "scaffold", "seq", "smiles", "lig_idx", "pro_idx"]:
                    overlap = sorted(split_sets[a][field].intersection(split_sets[b][field]))
                    lines.append(
                        f"{field}_overlap={len(overlap)} | examples={overlap[:5]}"
                    )
                lines.append("")

        text = "\n".join(lines)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(text)

    def _save_pair_split(sp, pair_indices, pair_table, scaffolds):
        p = _pair_cache_path(sp)
        idx = np.asarray(pair_indices, dtype=np.int64)
        lig_idx = pair_table["pair_lig_idx"][idx]
        pro_idx = pair_table["pair_pro_idx"][idx]

        np.savez_compressed(
            p,
            ids=pair_table["ids"][idx],
            y=pair_table["y"][idx].astype(np.float32),
            smiles=pair_table["smiles"][idx],
            seq=pair_table["seq"][idx],
            source_csv=pair_table["source_csv"][idx],
            pair_lig_idx=lig_idx.astype(np.int64),
            pair_pro_idx=pro_idx.astype(np.int64),
            ligand_scaffold=np.array(
                [scaffolds[int(i)] for i in lig_idx],
                dtype=object
            ),
        )

    def _build_and_cache_all():
        ids, smiles, seq, y, source_csv, raw_stats = _load_raw_bindingdb()

        print(
            f"[BindingDB] valid rows={len(y)} | "
            f"raw_train={raw_stats['raw_train_rows']} | raw_test={raw_stats['raw_test_rows']} | "
            f"drop_empty_smiles={raw_stats['drop_empty_smiles']} | "
            f"drop_empty_sequence={raw_stats['drop_empty_sequence']} | "
            f"drop_invalid_affinity={raw_stats['drop_invalid_affinity']}"
        )

        lig_list, pro_list, pair_table = _make_pair_table(
            ids=ids,
            smiles=smiles,
            seq=seq,
            y=y,
            source_csv=source_csv,
        )

        print(
            f"[BindingDB] unique ligands={len(lig_list)}, "
            f"unique proteins={len(pro_list)}, pairs={len(pair_table['y'])}"
        )

        scaffolds = np.array(
            [_get_murcko_scaffold(smi) for smi in lig_list],
            dtype=object
        )

        pair_split_indices, excluded, lig_split, pro_split, meta = _build_strict_split_indices(
            lig_list=lig_list,
            pro_list=pro_list,
            pair_table=pair_table,
            scaffolds=scaffolds,
        )

        meta["raw_stats"] = raw_stats

        for sp in ("train", "val", "test"):
            if len(pair_split_indices[sp]) == 0:
                raise RuntimeError(
                    f"BindingDB strict split produced empty {sp} set. "
                    "Try changing split_seed/train_ratio/val_ratio."
                )

        with open(split_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        for sp in ("train", "val", "test"):
            _save_pair_split(sp, pair_split_indices[sp], pair_table, scaffolds)

        _audit_and_save(pair_split_indices, pair_table, scaffolds, audit_path)

        # Encode unique ligand/protein token banks.
        lm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LM|BindingDB] using device: {lm_device}")

        print(f"[LM|BindingDB] loading ChemBERTa: {chemberta_model_name}")
        chem_tok = AutoTokenizer.from_pretrained(chemberta_model_name)
        chem_model = _hf_load_model(
            chemberta_model_name,
            lm_device,
            use_safetensors
        )

        drug_lm_tokens_bank, drug_lm_mask_bank = _encode_text_tokens(
            lig_list.tolist(),
            chem_tok,
            chem_model,
            lm_device,
            batch_size=lm_batch_size,
            max_length=chem_max_len,
            desc="[LM|BindingDB] ChemBERTa token encoding",
            mask_special_tokens=mask_special_tokens,
        )

        try:
            del chem_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        print(f"[LM|BindingDB] loading ESM-2: {esm2_model_name}")
        esm_tok = AutoTokenizer.from_pretrained(esm2_model_name)
        esm_model = _hf_load_model(
            esm2_model_name,
            lm_device,
            use_safetensors
        )

        prot_lm_tokens_bank, prot_lm_mask_bank = _encode_text_tokens(
            pro_list.tolist(),
            esm_tok,
            esm_model,
            lm_device,
            batch_size=max(1, min(2, lm_batch_size)),
            max_length=prot_max_len,
            desc="[LM|BindingDB] ESM-2 token encoding",
            mask_special_tokens=mask_special_tokens,
        )

        try:
            del esm_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        lig_ids = np.array([f"L{i}" for i in range(len(lig_list))], dtype=object)
        pro_ids = np.array([f"P{i}" for i in range(len(pro_list))], dtype=object)

        np.savez_compressed(
            token_bank_path,
            lig_ids=lig_ids,
            lig_smiles=lig_list,
            lig_scaffold=scaffolds,
            pro_ids=pro_ids,
            pro_seq=pro_list,
            drug_lm_tokens_bank=drug_lm_tokens_bank.astype(np.float32),
            drug_lm_mask_bank=drug_lm_mask_bank.astype(bool),
            prot_lm_tokens_bank=prot_lm_tokens_bank.astype(np.float32),
            prot_lm_mask_bank=prot_lm_mask_bank.astype(bool),
        )

        print("[CACHE|BindingDB] strict token cache done.")

    def _cache_ready():
        if force_refresh:
            return False

        if not token_bank_path.exists():
            return False

        for sp in ("train", "val", "test"):
            if not _pair_cache_path(sp).exists():
                return False

        return True

    if split not in ("train", "val", "test", "all"):
        raise ValueError(f"split must be 'train'/'val'/'test'/'all', got {split!r}")

    if not _cache_ready():
        _build_and_cache_all()

    token_bank = np.load(token_bank_path, allow_pickle=True)

    def _load_split(sp: str):
        d = np.load(_pair_cache_path(sp), allow_pickle=True)

        pair_lig_idx = d["pair_lig_idx"].astype(np.int64)
        pair_pro_idx = d["pair_pro_idx"].astype(np.int64)

        source_csv = (
            d["source_csv"]
            if "source_csv" in d.files
            else np.array(["unknown"] * len(d["ids"]), dtype=object)
        )

        pkg = {
            "ids": d["ids"],
            "y": d["y"].astype(np.float32),
            "smiles": d["smiles"],
            "seq": d["seq"],
            "source_csv": source_csv,
            "pair_lig_idx": pair_lig_idx,
            "pair_pro_idx": pair_pro_idx,
            "ligand_scaffold": d["ligand_scaffold"],

            "drug_lm_tokens_bank": token_bank["drug_lm_tokens_bank"].astype(np.float32),
            "drug_lm_mask_bank": token_bank["drug_lm_mask_bank"].astype(bool),
            "prot_lm_tokens_bank": token_bank["prot_lm_tokens_bank"].astype(np.float32),
            "prot_lm_mask_bank": token_bank["prot_lm_mask_bank"].astype(bool),

            "lig_ids": token_bank["lig_ids"],
            "pro_ids": token_bank["pro_ids"],
            "split_protocol": (
                "BindingDB strict ligand-scaffold and protein-sequence-disjoint "
                "product split over merged train/test CSV pool"
            ),
            "g_lig": None,
            "g_prot": None,
        }

        if return_pair_level_tokens:
            pkg["drug_lm_tokens"] = pkg["drug_lm_tokens_bank"][pair_lig_idx]
            pkg["drug_lm_mask"] = pkg["drug_lm_mask_bank"][pair_lig_idx]
            pkg["prot_lm_tokens"] = pkg["prot_lm_tokens_bank"][pair_pro_idx]
            pkg["prot_lm_mask"] = pkg["prot_lm_mask_bank"][pair_pro_idx]

        return pkg

    if split in ("train", "val", "test"):
        return {split: _load_split(split)}

    parts = {sp: _load_split(sp) for sp in ("train", "val", "test")}
    return {"all": parts}


def LoadData_fdavis_lm_1d_token_scaffold_seqsplit(
    data_dir: str,
    split: str = "all",
    out_1d: str = "../dataset/fdavis/processed_lm_1d_token_scaffold_seqsplit",
    logspace_trans: bool = False,
    # --- LM models ---
    esm2_model_name: str = "facebook/esm2_t33_650M_UR50D",
    chemberta_model_name: str = "DeepChem/ChemBERTa-77M-MTR",
    lm_batch_size: int = 4,
    chem_max_len: int = 128,
    prot_max_len: int = 1024,
    use_safetensors: bool = True,
    mask_special_tokens: bool = True,
    # --- strict split ---
    split_seed: int = 2023,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    force_refresh: bool = False,
    # --- compatibility / memory option ---
    return_pair_level_tokens: bool = False,
) -> dict:
    """
    FDavis + token-level LM loader with strict ligand-scaffold / protein-sequence-disjoint split.

    主要修改：
        1. 将旧版 CLS/first-token pooled 向量 [N, D] 改为 token-level 表征 [N_unique, L, D]；
        2. 采用 token bank + pair index，避免把蛋白 token 展开到所有 pair，节省内存；
        3. 将旧版随机 pair split 改为严格的：
              - ligand Bemis-Murcko scaffold disjoint；
              - protein full-sequence disjoint；
              - 仅保留 ligand split 与 protein split 一致的 pair；
           从而满足更严格的 cold-start / leakage-controlled 评估要求。

    原始文件：
        affi_info.txt，制表符分隔，无表头。沿用旧版列定义：
            col 0 : 样本 ID 或其它字段
            col 1 : 配体 SMILES
            col 2 : 目标 ID，可忽略
            col 3 : 蛋白序列
            col 4 : affinity 标签

    缓存文件：
        fdavis_unique_lm_tokens.npz
        fdavis_train_pairs_scaffold_seqsplit.npz
        fdavis_val_pairs_scaffold_seqsplit.npz
        fdavis_test_pairs_scaffold_seqsplit.npz
        fdavis_split_meta_scaffold_seqsplit.json
        fdavis_split_audit_scaffold_seqsplit.txt

    返回单个 split：
        {
            'ids', 'y', 'smiles', 'seq',
            'pair_lig_idx', 'pair_pro_idx',
            'drug_lm_tokens_bank', 'drug_lm_mask_bank',
            'prot_lm_tokens_bank', 'prot_lm_mask_bank',
            'ligand_scaffold', 'split_protocol',
            'lig_ids', 'pro_ids',
            'g_lig', 'g_prot',
            optionally:
                'drug_lm_tokens', 'drug_lm_mask',
                'prot_lm_tokens', 'prot_lm_mask'
        }

    注意：
        return_pair_level_tokens=False 是推荐设置。
        若改为 True，会额外展开 pair-level token，便于快速兼容旧训练代码，
        但会显著增加内存占用。
    """
    from pathlib import Path
    import json
    from collections import OrderedDict, defaultdict

    import numpy as np
    import pandas as pd
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModel

    data_dir = Path(data_dir)
    out_1d = Path(out_1d)
    out_1d.mkdir(parents=True, exist_ok=True)

    token_bank_path = out_1d / "fdavis_unique_lm_tokens.npz"
    split_meta_path = out_1d / "fdavis_split_meta_scaffold_seqsplit.json"
    audit_path = out_1d / "fdavis_split_audit_scaffold_seqsplit.txt"

    def _pair_cache_path(sp: str):
        return out_1d / f"fdavis_{sp}_pairs_scaffold_seqsplit.npz"

    def _clean_protein_sequence(seq):
        if seq is None:
            return ""
        seq = str(seq).upper()
        valid = set("ACDEFGHIKLMNPQRSTVWY")
        return "".join([c for c in seq if c in valid])

    def _clean_smiles(smi):
        if smi is None:
            return ""
        return str(smi).strip()

    def _load_raw_fdavis():
        affi_path = data_dir / "affi_info.txt"
        if not affi_path.exists():
            raise FileNotFoundError(f"Cannot find {affi_path}")

        df = pd.read_csv(affi_path, sep="\t", header=None)
        if df.shape[1] < 5:
            raise ValueError(
                f"affi_info.txt 至少需要 5 列，但当前只有 {df.shape[1]} 列。"
            )

        rows = []
        stats = {
            "raw_rows": int(len(df)),
            "kept_rows": 0,
            "drop_empty_smiles": 0,
            "drop_empty_sequence": 0,
            "drop_invalid_affinity": 0,
        }

        for row_idx, row in df.iterrows():
            smi = _clean_smiles(row.iloc[1])
            seq = _clean_protein_sequence(row.iloc[3])

            if smi == "":
                stats["drop_empty_smiles"] += 1
                continue

            if seq == "":
                stats["drop_empty_sequence"] += 1
                continue

            try:
                y = float(row.iloc[4])
            except Exception:
                stats["drop_invalid_affinity"] += 1
                continue

            if not np.isfinite(y):
                stats["drop_invalid_affinity"] += 1
                continue

            if logspace_trans:
                y = float(np.log10(y + 1e-8))

            rows.append({
                "id": f"fd_{row_idx}",
                "smiles": smi,
                "seq": seq,
                "y": y,
            })

        stats["kept_rows"] = int(len(rows))

        if len(rows) == 0:
            raise RuntimeError("FDavis 清洗后没有有效样本，请检查 affi_info.txt。")

        ids = np.array([r["id"] for r in rows], dtype=object)
        smiles = np.array([r["smiles"] for r in rows], dtype=object)
        seq = np.array([r["seq"] for r in rows], dtype=object)
        y = np.array([r["y"] for r in rows], dtype=np.float32)

        return ids, smiles, seq, y, stats

    def _get_murcko_scaffold(smi: str) -> str:
        """
        返回 Bemis-Murcko scaffold。
        若 RDKit 无法解析或 scaffold 为空，则使用稳定 fallback key。
        """
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold

            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                return f"INVALID::{str(smi)}"

            scaf = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol,
                includeChirality=False,
            )

            if scaf is None or str(scaf).strip() == "":
                can = Chem.MolToSmiles(mol, canonical=True)
                return f"NO_SCAFFOLD::{can}"

            return str(scaf)

        except Exception:
            return f"FALLBACK::{str(smi)}"

    def _split_groups(groups: dict, total_items: int):
        """
        按完整 group 划分，group 不会被拆开。
        目标比例依据 unique ligand / unique protein 数量近似控制。
        """
        if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
            raise ValueError(
                f"train_ratio={train_ratio}, val_ratio={val_ratio} 不合法；"
                "要求 train_ratio>0, val_ratio>=0, 且 train_ratio+val_ratio<1。"
            )

        rng = np.random.RandomState(split_seed)
        items = list(groups.items())
        rng.shuffle(items)
        items = sorted(items, key=lambda kv: len(kv[1]), reverse=True)

        n_train_target = int(total_items * train_ratio)
        n_val_target = int(total_items * val_ratio)

        split_to_indices = {"train": [], "val": [], "test": []}
        split_to_groups = {"train": [], "val": [], "test": []}

        for group_key, idxs in items:
            idxs = list(idxs)

            if len(split_to_indices["train"]) + len(idxs) <= n_train_target:
                sp = "train"
            elif len(split_to_indices["val"]) + len(idxs) <= n_val_target:
                sp = "val"
            else:
                sp = "test"

            split_to_indices[sp].extend(idxs)
            split_to_groups[sp].append(group_key)

        return (
            {k: np.array(v, dtype=np.int64) for k, v in split_to_indices.items()},
            {k: set(v) for k, v in split_to_groups.items()},
        )

    def _hf_load_model(name: str, device: torch.device, use_safetensors: bool = True):
        try:
            model = AutoModel.from_pretrained(
                name,
                use_safetensors=use_safetensors,
            )
            model.to(device)
            return model

        except TypeError:
            model = AutoModel.from_pretrained(name)
            model.to(device)
            return model

        except Exception as e:
            msg = str(e).lower()
            if ("torch" in msg and "2.6" in msg) or ("vulnerability" in msg) or ("cve" in msg):
                raise RuntimeError(
                    f"Loading HF model '{name}' failed due to transformers / torch safety check:\n{e}\n"
                    "Possible solutions: use safetensors, upgrade torch, or use a compatible transformers version."
                ) from e

            model = AutoModel.from_pretrained(name)
            model.to(device)
            return model

    def _encode_text_tokens(
        text_list,
        tokenizer,
        model,
        device,
        batch_size=4,
        max_length=512,
        desc="[LM] token encoding",
        mask_special_tokens=True,
    ):
        """
        对 unique SMILES / sequence 生成 token-level last_hidden_state 与 padding mask。
        mask 中 True 表示该位置需要在后续 attention 中被忽略。
        """
        all_tokens = []
        all_masks = []
        model.eval()

        with torch.no_grad():
            for i in tqdm(range(0, len(text_list), batch_size), desc=desc, unit="batch"):
                batch = [str(x) for x in text_list[i:i + batch_size]]
                enc = tokenizer(
                    batch,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                    return_special_tokens_mask=True,
                )

                special_tokens_mask = enc.pop("special_tokens_mask", None)
                enc = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in enc.items()
                }
                if special_tokens_mask is not None:
                    special_tokens_mask = special_tokens_mask.to(device)

                try:
                    out = model(**enc)
                except TypeError as e:
                    if "token_type_ids" in str(e) and "token_type_ids" in enc:
                        enc.pop("token_type_ids")
                        out = model(**enc)
                    else:
                        raise e

                if hasattr(out, "last_hidden_state"):
                    hs = out.last_hidden_state
                elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                    hs = out[0]
                else:
                    raise RuntimeError("Unsupported model output structure for token-level LM encoding.")

                attention_mask = enc["attention_mask"].bool()
                pad_mask = ~attention_mask

                if mask_special_tokens and special_tokens_mask is not None:
                    pad_mask = pad_mask | special_tokens_mask.bool()

                all_tokens.append(hs.detach().cpu().numpy().astype(np.float32))
                all_masks.append(pad_mask.detach().cpu().numpy().astype(bool))

        if len(all_tokens) == 0:
            hidden_size = int(getattr(model.config, "hidden_size", 0))
            return (
                np.zeros((0, max_length, hidden_size), dtype=np.float32),
                np.ones((0, max_length), dtype=bool),
            )

        return (
            np.concatenate(all_tokens, axis=0).astype(np.float32),
            np.concatenate(all_masks, axis=0).astype(bool),
        )

    def _make_pair_table(ids, smiles, seq, y):
        """
        FDavis 原始数据本身就是 pair-level 表格。
        这里构造 unique ligand/protein bank 索引。
        """
        lig_list = np.array(
            list(OrderedDict.fromkeys(smiles.tolist()).keys()),
            dtype=object,
        )
        pro_list = np.array(
            list(OrderedDict.fromkeys(seq.tolist()).keys()),
            dtype=object,
        )

        lig2idx = {str(smi): i for i, smi in enumerate(lig_list)}
        pro2idx = {str(s): i for i, s in enumerate(pro_list)}

        pair_lig_idx = np.array([lig2idx[str(smi)] for smi in smiles], dtype=np.int64)
        pair_pro_idx = np.array([pro2idx[str(s)] for s in seq], dtype=np.int64)

        pair_table = {
            "ids": ids,
            "y": y.astype(np.float32),
            "smiles": smiles,
            "seq": seq,
            "pair_lig_idx": pair_lig_idx,
            "pair_pro_idx": pair_pro_idx,
        }

        return lig_list, pro_list, pair_table

    def _build_strict_split_indices(lig_list, pro_list, pair_table, scaffolds):
        # 1) Ligand groups by Murcko scaffold.
        lig_groups = defaultdict(list)
        for i, scaf in enumerate(scaffolds):
            lig_groups[str(scaf)].append(i)

        # 2) Protein groups by exact cleaned full sequence.
        pro_groups = defaultdict(list)
        for j, seq_text in enumerate(pro_list):
            pro_groups[str(seq_text)].append(j)

        lig_split, _ = _split_groups(lig_groups, len(lig_list))
        pro_split, _ = _split_groups(pro_groups, len(pro_list))

        lig_to_split = {}
        for sp, arr in lig_split.items():
            for x in arr:
                lig_to_split[int(x)] = sp

        pro_to_split = {}
        for sp, arr in pro_split.items():
            for x in arr:
                pro_to_split[int(x)] = sp

        pair_split_indices = {"train": [], "val": [], "test": []}
        excluded = []

        for k, (lig_idx, pro_idx) in enumerate(
            zip(pair_table["pair_lig_idx"], pair_table["pair_pro_idx"])
        ):
            lsp = lig_to_split[int(lig_idx)]
            psp = pro_to_split[int(pro_idx)]

            if lsp == psp:
                pair_split_indices[lsp].append(k)
            else:
                excluded.append(k)

        pair_split_indices = {
            k: np.array(v, dtype=np.int64)
            for k, v in pair_split_indices.items()
        }
        excluded = np.array(excluded, dtype=np.int64)

        meta = {
            "split_protocol": (
                "FDavis strict ligand-scaffold and protein-sequence-disjoint product split"
            ),
            "split_seed": int(split_seed),
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "test_ratio": float(1.0 - train_ratio - val_ratio),
            "n_pairs_before_strict_filter": int(len(pair_table["y"])),
            "n_pairs_excluded_cross_partition": int(len(excluded)),
            "n_pairs_after_strict_filter": int(sum(len(v) for v in pair_split_indices.values())),
            "n_ligands": int(len(lig_list)),
            "n_proteins": int(len(pro_list)),
            "n_scaffold_groups": int(len(lig_groups)),
            "n_protein_sequence_groups": int(len(pro_groups)),
            "ligand_counts": {sp: int(len(arr)) for sp, arr in lig_split.items()},
            "protein_counts": {sp: int(len(arr)) for sp, arr in pro_split.items()},
            "pair_counts": {sp: int(len(arr)) for sp, arr in pair_split_indices.items()},
        }

        return pair_split_indices, excluded, lig_split, pro_split, meta

    def _audit_and_save(pair_split_indices, pair_table, scaffolds, save_path):
        lines = []
        lines.append("========== FDavis Strict Split Audit ==========")
        lines.append(
            "Protocol: ligand Murcko scaffold split + protein full-sequence disjoint split"
        )
        lines.append(
            "A pair is kept only when ligand split == protein split; "
            "cross-partition pairs are excluded."
        )
        lines.append("")

        split_names = ["train", "val", "test"]
        split_sets = {}

        for sp in split_names:
            idx = pair_split_indices[sp]
            lig_idx = pair_table["pair_lig_idx"][idx]
            pro_idx = pair_table["pair_pro_idx"][idx]

            split_sets[sp] = {
                "ids": set([str(x) for x in pair_table["ids"][idx]]),
                "smiles": set([str(x) for x in pair_table["smiles"][idx]]),
                "seq": set([str(x) for x in pair_table["seq"][idx]]),
                "scaffold": set([str(scaffolds[int(i)]) for i in lig_idx]),
                "lig_idx": set([int(x) for x in lig_idx]),
                "pro_idx": set([int(x) for x in pro_idx]),
            }

            lines.append(
                f"[{sp}] pairs={len(idx)} | ligands={len(split_sets[sp]['lig_idx'])} | "
                f"proteins={len(split_sets[sp]['pro_idx'])} | "
                f"scaffolds={len(split_sets[sp]['scaffold'])}"
            )

        lines.append("")

        for i, a in enumerate(split_names):
            for b in split_names[i + 1:]:
                lines.append(f"--- {a} vs {b} ---")
                for field in ["ids", "scaffold", "seq", "smiles", "lig_idx", "pro_idx"]:
                    overlap = sorted(split_sets[a][field].intersection(split_sets[b][field]))
                    lines.append(
                        f"{field}_overlap={len(overlap)} | examples={overlap[:5]}"
                    )
                lines.append("")

        text = "\n".join(lines)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(text)

    def _save_pair_split(sp, pair_indices, pair_table, scaffolds):
        p = _pair_cache_path(sp)
        idx = np.asarray(pair_indices, dtype=np.int64)
        lig_idx = pair_table["pair_lig_idx"][idx]
        pro_idx = pair_table["pair_pro_idx"][idx]

        np.savez_compressed(
            p,
            ids=pair_table["ids"][idx],
            y=pair_table["y"][idx].astype(np.float32),
            smiles=pair_table["smiles"][idx],
            seq=pair_table["seq"][idx],
            pair_lig_idx=lig_idx.astype(np.int64),
            pair_pro_idx=pro_idx.astype(np.int64),
            ligand_scaffold=np.array(
                [scaffolds[int(i)] for i in lig_idx],
                dtype=object,
            ),
        )

    def _build_and_cache_all():
        ids, smiles, seq, y, raw_stats = _load_raw_fdavis()

        print(
            f"[FDavis] valid rows={len(y)} | raw_rows={raw_stats['raw_rows']} | "
            f"drop_empty_smiles={raw_stats['drop_empty_smiles']} | "
            f"drop_empty_sequence={raw_stats['drop_empty_sequence']} | "
            f"drop_invalid_affinity={raw_stats['drop_invalid_affinity']}"
        )

        lig_list, pro_list, pair_table = _make_pair_table(
            ids=ids,
            smiles=smiles,
            seq=seq,
            y=y,
        )

        print(
            f"[FDavis] unique ligands={len(lig_list)}, "
            f"unique proteins={len(pro_list)}, pairs={len(pair_table['y'])}"
        )

        scaffolds = np.array(
            [_get_murcko_scaffold(smi) for smi in lig_list],
            dtype=object,
        )

        pair_split_indices, excluded, lig_split, pro_split, meta = _build_strict_split_indices(
            lig_list=lig_list,
            pro_list=pro_list,
            pair_table=pair_table,
            scaffolds=scaffolds,
        )

        meta["raw_stats"] = raw_stats

        for sp in ("train", "val", "test"):
            if len(pair_split_indices[sp]) == 0:
                raise RuntimeError(
                    f"FDavis strict split produced empty {sp} set. "
                    "Try changing split_seed/train_ratio/val_ratio."
                )

        with open(split_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        for sp in ("train", "val", "test"):
            _save_pair_split(sp, pair_split_indices[sp], pair_table, scaffolds)

        _audit_and_save(pair_split_indices, pair_table, scaffolds, audit_path)

        # Encode unique ligand/protein token banks.
        lm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LM|FDavis] using device: {lm_device}")

        print(f"[LM|FDavis] loading ChemBERTa: {chemberta_model_name}")
        chem_tok = AutoTokenizer.from_pretrained(chemberta_model_name)
        chem_model = _hf_load_model(
            chemberta_model_name,
            lm_device,
            use_safetensors,
        )

        drug_lm_tokens_bank, drug_lm_mask_bank = _encode_text_tokens(
            lig_list.tolist(),
            chem_tok,
            chem_model,
            lm_device,
            batch_size=lm_batch_size,
            max_length=chem_max_len,
            desc="[LM|FDavis] ChemBERTa token encoding",
            mask_special_tokens=mask_special_tokens,
        )

        try:
            del chem_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        print(f"[LM|FDavis] loading ESM-2: {esm2_model_name}")
        esm_tok = AutoTokenizer.from_pretrained(esm2_model_name)
        esm_model = _hf_load_model(
            esm2_model_name,
            lm_device,
            use_safetensors,
        )

        prot_lm_tokens_bank, prot_lm_mask_bank = _encode_text_tokens(
            pro_list.tolist(),
            esm_tok,
            esm_model,
            lm_device,
            batch_size=max(1, min(2, lm_batch_size)),
            max_length=prot_max_len,
            desc="[LM|FDavis] ESM-2 token encoding",
            mask_special_tokens=mask_special_tokens,
        )

        try:
            del esm_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        lig_ids = np.array([f"L{i}" for i in range(len(lig_list))], dtype=object)
        pro_ids = np.array([f"P{i}" for i in range(len(pro_list))], dtype=object)

        np.savez_compressed(
            token_bank_path,
            lig_ids=lig_ids,
            lig_smiles=lig_list,
            lig_scaffold=scaffolds,
            pro_ids=pro_ids,
            pro_seq=pro_list,
            drug_lm_tokens_bank=drug_lm_tokens_bank.astype(np.float32),
            drug_lm_mask_bank=drug_lm_mask_bank.astype(bool),
            prot_lm_tokens_bank=prot_lm_tokens_bank.astype(np.float32),
            prot_lm_mask_bank=prot_lm_mask_bank.astype(bool),
        )

        print("[CACHE|FDavis] strict token cache done.")

    def _cache_ready():
        if force_refresh:
            return False

        if not token_bank_path.exists():
            return False

        for sp in ("train", "val", "test"):
            if not _pair_cache_path(sp).exists():
                return False

        return True

    if split not in ("train", "val", "test", "all"):
        raise ValueError(f"split must be 'train'/'val'/'test'/'all', got {split!r}")

    if not _cache_ready():
        _build_and_cache_all()

    token_bank = np.load(token_bank_path, allow_pickle=True)

    def _load_split(sp: str):
        d = np.load(_pair_cache_path(sp), allow_pickle=True)

        pair_lig_idx = d["pair_lig_idx"].astype(np.int64)
        pair_pro_idx = d["pair_pro_idx"].astype(np.int64)

        pkg = {
            "ids": d["ids"],
            "y": d["y"].astype(np.float32),
            "smiles": d["smiles"],
            "seq": d["seq"],
            "pair_lig_idx": pair_lig_idx,
            "pair_pro_idx": pair_pro_idx,
            "ligand_scaffold": d["ligand_scaffold"],

            "drug_lm_tokens_bank": token_bank["drug_lm_tokens_bank"].astype(np.float32),
            "drug_lm_mask_bank": token_bank["drug_lm_mask_bank"].astype(bool),
            "prot_lm_tokens_bank": token_bank["prot_lm_tokens_bank"].astype(np.float32),
            "prot_lm_mask_bank": token_bank["prot_lm_mask_bank"].astype(bool),

            "lig_ids": token_bank["lig_ids"],
            "pro_ids": token_bank["pro_ids"],
            "split_protocol": (
                "FDavis strict ligand-scaffold and protein-sequence-disjoint product split"
            ),
            "g_lig": None,
            "g_prot": None,
        }

        if return_pair_level_tokens:
            pkg["drug_lm_tokens"] = pkg["drug_lm_tokens_bank"][pair_lig_idx]
            pkg["drug_lm_mask"] = pkg["drug_lm_mask_bank"][pair_lig_idx]
            pkg["prot_lm_tokens"] = pkg["prot_lm_tokens_bank"][pair_pro_idx]
            pkg["prot_lm_mask"] = pkg["prot_lm_mask_bank"][pair_pro_idx]

        return pkg

    if split in ("train", "val", "test"):
        return {split: _load_split(split)}

    parts = {sp: _load_split(sp) for sp in ("train", "val", "test")}
    return {"all": parts}


def LoadData_metz_lm_1d_token_scaffold_seqsplit(
    data_dir: str,
    split: str = "all",
    out_1d: str = "../dataset/metz/processed_lm_1d_token_scaffold_seqsplit",
    logspace_trans: bool = False,
    # --- LM models ---
    esm2_model_name: str = "facebook/esm2_t33_650M_UR50D",
    chemberta_model_name: str = "DeepChem/ChemBERTa-77M-MTR",
    lm_batch_size: int = 4,
    chem_max_len: int = 128,
    prot_max_len: int = 1024,
    use_safetensors: bool = True,
    mask_special_tokens: bool = True,
    # --- strict split ---
    split_seed: int = 2023,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    force_refresh: bool = False,
    # --- compatibility / memory option ---
    return_pair_level_tokens: bool = False,
) -> dict:
    """
    Metz + token-level LM loader with strict ligand-scaffold / protein-sequence-disjoint split.

    主要修改：
        1. 旧版 loader 使用 CLS / first-token pooled 向量 [N, D]；
           本版改为 token-level 表征 [N_unique, L, D]。
        2. 旧版 loader 对 pair 随机划分；
           本版改为 scaffold-disjoint + protein-sequence-disjoint 严格划分。
        3. 使用 token bank + pair index 形式，避免把蛋白 token 展开到所有 pair，
           后续训练时可直接根据 pair_lig_idx / pair_pro_idx 索引 token bank。

    原始文件：
        drug_info.txt:
            col 0 : drug_id
            col 2 : ligand SMILES

        targ_info.txt:
            col 0 : target_id
            col 2 : protein sequence

        affi_info.txt:
            col 0 : drug_id
            col 1 : target_id
            col 2 : affinity

    严格划分协议：
        - unique ligand 按 Bemis-Murcko scaffold 分组；
        - unique protein 按清洗后的完整蛋白序列分组；
        - ligand groups 与 protein groups 分别划分到 train/val/test；
        - 只有 ligand split == protein split 的 pair 才保留；
        - 跨分区 pair 会被排除。

    这保证：
        - train / val / test 之间 ligand scaffold 不重叠；
        - train / val / test 之间 protein sequence 不重叠。

    缓存文件：
        metz_unique_lm_tokens.npz
        metz_train_pairs_scaffold_seqsplit.npz
        metz_val_pairs_scaffold_seqsplit.npz
        metz_test_pairs_scaffold_seqsplit.npz
        metz_split_meta_scaffold_seqsplit.json
        metz_split_audit_scaffold_seqsplit.txt

    单个 split 返回：
        {
            'ids', 'y', 'smiles', 'seq',
            'pair_lig_idx', 'pair_pro_idx',
            'drug_lm_tokens_bank', 'drug_lm_mask_bank',
            'prot_lm_tokens_bank', 'prot_lm_mask_bank',
            'ligand_scaffold', 'split_protocol',
            'lig_ids', 'pro_ids',
            'g_lig', 'g_prot',
            optionally:
                'drug_lm_tokens', 'drug_lm_mask',
                'prot_lm_tokens', 'prot_lm_mask'
        }

    注意：
        return_pair_level_tokens=False 是推荐设置。
        若设置为 True，会把 token bank 展开为 pair-level token，
        便于临时兼容旧写法，但会增加内存占用。
    """
    from pathlib import Path
    import json
    from collections import OrderedDict, defaultdict

    import numpy as np
    import pandas as pd
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModel

    data_dir = Path(data_dir)
    out_1d = Path(out_1d)
    out_1d.mkdir(parents=True, exist_ok=True)

    token_bank_path = out_1d / "metz_unique_lm_tokens.npz"
    split_meta_path = out_1d / "metz_split_meta_scaffold_seqsplit.json"
    audit_path = out_1d / "metz_split_audit_scaffold_seqsplit.txt"

    def _pair_cache_path(sp: str):
        return out_1d / f"metz_{sp}_pairs_scaffold_seqsplit.npz"

    def _clean_protein_sequence(seq):
        if seq is None:
            return ""
        seq = str(seq).upper()
        valid = set("ACDEFGHIKLMNPQRSTVWY")
        return "".join([c for c in seq if c in valid])

    def _clean_smiles(smi):
        if smi is None:
            return ""
        return str(smi).strip()

    def _load_raw_metz():
        """
        读取 Metz 原始文件并构造有效 pair。

        关键：
            drug_id / target_id 必须保留 pandas 原始类型，
            不要提前转成 str，否则 1 与 1.0 会匹配失败。
        """
        drug_path = data_dir / "drug_info.txt"
        targ_path = data_dir / "targ_info.txt"
        affi_path = data_dir / "affi_info.txt"

        for p in (drug_path, targ_path, affi_path):
            if not p.exists():
                raise FileNotFoundError(f"Cannot find {p}")

        drug_df = pd.read_csv(drug_path, sep="\t", header=None)
        targ_df = pd.read_csv(targ_path, sep="\t", header=None)
        affi_df = pd.read_csv(affi_path, sep="\t", header=None)

        if drug_df.shape[1] < 3:
            raise ValueError(
                f"drug_info.txt 至少需要 3 列，但当前只有 {drug_df.shape[1]} 列。"
            )
        if targ_df.shape[1] < 3:
            raise ValueError(
                f"targ_info.txt 至少需要 3 列，但当前只有 {targ_df.shape[1]} 列。"
            )
        if affi_df.shape[1] < 3:
            raise ValueError(
                f"affi_info.txt 至少需要 3 列，但当前只有 {affi_df.shape[1]} 列。"
            )

        # 这里严格沿用你旧版 loader 的 ID 匹配方式：
        # 不把 ID 转成 str，直接用原始 pandas 值作为键。
        drug_dict = dict(zip(drug_df[0], drug_df[2]))  # drug_id -> SMILES
        prot_dict = dict(zip(targ_df[0], targ_df[2]))  # target_id -> protein sequence

        rows = []
        stats = {
            "raw_drug_rows": int(len(drug_df)),
            "raw_target_rows": int(len(targ_df)),
            "raw_affinity_rows": int(len(affi_df)),
            "kept_pairs": 0,
            "drop_missing_drug": 0,
            "drop_missing_target": 0,
            "drop_empty_smiles": 0,
            "drop_empty_sequence": 0,
            "drop_invalid_affinity": 0,
        }

        missing_drug_examples = []
        missing_target_examples = []

        for row_idx, row in affi_df.iterrows():
            # 保留原始类型，不转 str
            drug_id = row.iloc[0]
            target_id = row.iloc[1]

            if drug_id not in drug_dict:
                stats["drop_missing_drug"] += 1
                if len(missing_drug_examples) < 5:
                    missing_drug_examples.append(repr(drug_id))
                continue

            if target_id not in prot_dict:
                stats["drop_missing_target"] += 1
                if len(missing_target_examples) < 5:
                    missing_target_examples.append(repr(target_id))
                continue

            smi = _clean_smiles(drug_dict[drug_id])
            seq = _clean_protein_sequence(prot_dict[target_id])

            if smi == "":
                stats["drop_empty_smiles"] += 1
                continue

            if seq == "":
                stats["drop_empty_sequence"] += 1
                continue

            try:
                y = float(row.iloc[2])
            except Exception:
                stats["drop_invalid_affinity"] += 1
                continue

            if not np.isfinite(y):
                stats["drop_invalid_affinity"] += 1
                continue

            if logspace_trans:
                y = float(np.log10(y + 1e-8))

            rows.append({
                "id": f"metz_{row_idx}",
                "drug_id": drug_id,
                "target_id": target_id,
                "smiles": smi,
                "seq": seq,
                "y": y,
            })

        stats["kept_pairs"] = int(len(rows))

        print(
            f"[Metz|raw] affi_rows={stats['raw_affinity_rows']} | "
            f"kept_pairs={stats['kept_pairs']} | "
            f"missing_drug={stats['drop_missing_drug']} | "
            f"missing_target={stats['drop_missing_target']} | "
            f"empty_smiles={stats['drop_empty_smiles']} | "
            f"empty_seq={stats['drop_empty_sequence']} | "
            f"invalid_affinity={stats['drop_invalid_affinity']}"
        )

        if stats["drop_missing_drug"] > 0:
            print(f"[Metz|debug] missing drug id examples: {missing_drug_examples}")

        if stats["drop_missing_target"] > 0:
            print(f"[Metz|debug] missing target id examples: {missing_target_examples}")

        if len(rows) == 0:
            print("[Metz|debug] drug_dict key examples:", list(drug_dict.keys())[:5])
            print("[Metz|debug] prot_dict key examples:", list(prot_dict.keys())[:5])
            raise RuntimeError("Metz 清洗后没有有效 pair，请检查三个原始文件。")

        ids = np.array([r["id"] for r in rows], dtype=object)
        drug_ids_pair = np.array([r["drug_id"] for r in rows], dtype=object)
        target_ids_pair = np.array([r["target_id"] for r in rows], dtype=object)
        smiles = np.array([r["smiles"] for r in rows], dtype=object)
        seq = np.array([r["seq"] for r in rows], dtype=object)
        y = np.array([r["y"] for r in rows], dtype=np.float32)

        return ids, drug_ids_pair, target_ids_pair, smiles, seq, y, stats

    def _get_murcko_scaffold(smi: str) -> str:
        """返回 Bemis-Murcko scaffold；失败时使用稳定 fallback key。"""
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold

            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                return f"INVALID::{str(smi)}"

            scaf = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol,
                includeChirality=False,
            )

            if scaf is None or str(scaf).strip() == "":
                can = Chem.MolToSmiles(mol, canonical=True)
                return f"NO_SCAFFOLD::{can}"

            return str(scaf)
        except Exception:
            return f"FALLBACK::{str(smi)}"

    def _split_groups(groups: dict, total_items: int):
        """按完整 group 划分；group 不会被拆开。"""
        if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
            raise ValueError(
                f"train_ratio={train_ratio}, val_ratio={val_ratio} 不合法；"
                "要求 train_ratio>0, val_ratio>=0, 且 train_ratio+val_ratio<1。"
            )

        rng = np.random.RandomState(split_seed)
        items = list(groups.items())
        rng.shuffle(items)
        items = sorted(items, key=lambda kv: len(kv[1]), reverse=True)

        n_train_target = int(total_items * train_ratio)
        n_val_target = int(total_items * val_ratio)

        split_to_indices = {"train": [], "val": [], "test": []}
        split_to_groups = {"train": [], "val": [], "test": []}

        for group_key, idxs in items:
            idxs = list(idxs)

            if len(split_to_indices["train"]) + len(idxs) <= n_train_target:
                sp = "train"
            elif len(split_to_indices["val"]) + len(idxs) <= n_val_target:
                sp = "val"
            else:
                sp = "test"

            split_to_indices[sp].extend(idxs)
            split_to_groups[sp].append(group_key)

        return (
            {k: np.array(v, dtype=np.int64) for k, v in split_to_indices.items()},
            {k: set(v) for k, v in split_to_groups.items()},
        )

    def _hf_load_model(name: str, device: torch.device, use_safetensors: bool = True):
        try:
            model = AutoModel.from_pretrained(
                name,
                use_safetensors=use_safetensors,
            )
            model.to(device)
            return model
        except TypeError:
            model = AutoModel.from_pretrained(name)
            model.to(device)
            return model
        except Exception as e:
            msg = str(e).lower()
            if ("torch" in msg and "2.6" in msg) or ("vulnerability" in msg) or ("cve" in msg):
                raise RuntimeError(
                    f"Loading HF model '{name}' failed due to transformers / torch safety check:\n{e}\n"
                    "Possible solutions: use safetensors, upgrade torch, or use a compatible transformers version."
                ) from e
            model = AutoModel.from_pretrained(name)
            model.to(device)
            return model

    def _encode_text_tokens(
        text_list,
        tokenizer,
        model,
        device,
        batch_size=4,
        max_length=512,
        desc="[LM] token encoding",
        mask_special_tokens=True,
    ):
        """
        生成 token-level last_hidden_state 与 padding mask。
        mask 中 True 表示后续 attention 中需要忽略该 token。
        """
        all_tokens = []
        all_masks = []
        model.eval()

        with torch.no_grad():
            for i in tqdm(range(0, len(text_list), batch_size), desc=desc, unit="batch"):
                batch = [str(x) for x in text_list[i:i + batch_size]]
                enc = tokenizer(
                    batch,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                    return_special_tokens_mask=True,
                )

                special_tokens_mask = enc.pop("special_tokens_mask", None)
                enc = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in enc.items()
                }
                if special_tokens_mask is not None:
                    special_tokens_mask = special_tokens_mask.to(device)

                try:
                    out = model(**enc)
                except TypeError as e:
                    if "token_type_ids" in str(e) and "token_type_ids" in enc:
                        enc.pop("token_type_ids")
                        out = model(**enc)
                    else:
                        raise e

                if hasattr(out, "last_hidden_state"):
                    hs = out.last_hidden_state
                elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                    hs = out[0]
                else:
                    raise RuntimeError("Unsupported model output structure for token-level LM encoding.")

                attention_mask = enc["attention_mask"].bool()
                pad_mask = ~attention_mask

                if mask_special_tokens and special_tokens_mask is not None:
                    pad_mask = pad_mask | special_tokens_mask.bool()

                all_tokens.append(hs.detach().cpu().numpy().astype(np.float32))
                all_masks.append(pad_mask.detach().cpu().numpy().astype(bool))

        if len(all_tokens) == 0:
            hidden_size = int(getattr(model.config, "hidden_size", 0))
            return (
                np.zeros((0, max_length, hidden_size), dtype=np.float32),
                np.ones((0, max_length), dtype=bool),
            )

        return (
            np.concatenate(all_tokens, axis=0).astype(np.float32),
            np.concatenate(all_masks, axis=0).astype(bool),
        )

    def _make_pair_table(ids, drug_ids_pair, target_ids_pair, smiles, seq, y):
        """构造 unique ligand/protein token bank 索引。"""
        lig_list = np.array(
            list(OrderedDict.fromkeys(smiles.tolist()).keys()),
            dtype=object,
        )
        pro_list = np.array(
            list(OrderedDict.fromkeys(seq.tolist()).keys()),
            dtype=object,
        )

        lig2idx = {str(smi): i for i, smi in enumerate(lig_list)}
        pro2idx = {str(s): i for i, s in enumerate(pro_list)}

        pair_lig_idx = np.array([lig2idx[str(smi)] for smi in smiles], dtype=np.int64)
        pair_pro_idx = np.array([pro2idx[str(s)] for s in seq], dtype=np.int64)

        pair_table = {
            "ids": ids,
            "drug_ids_pair": drug_ids_pair,
            "target_ids_pair": target_ids_pair,
            "y": y.astype(np.float32),
            "smiles": smiles,
            "seq": seq,
            "pair_lig_idx": pair_lig_idx,
            "pair_pro_idx": pair_pro_idx,
        }

        return lig_list, pro_list, pair_table

    def _build_strict_split_indices(lig_list, pro_list, pair_table, scaffolds):
        # 1) Ligand groups by Murcko scaffold.
        lig_groups = defaultdict(list)
        for i, scaf in enumerate(scaffolds):
            lig_groups[str(scaf)].append(i)

        # 2) Protein groups by exact cleaned full sequence.
        pro_groups = defaultdict(list)
        for j, seq_text in enumerate(pro_list):
            pro_groups[str(seq_text)].append(j)

        lig_split, _ = _split_groups(lig_groups, len(lig_list))
        pro_split, _ = _split_groups(pro_groups, len(pro_list))

        lig_to_split = {}
        for sp, arr in lig_split.items():
            for x in arr:
                lig_to_split[int(x)] = sp

        pro_to_split = {}
        for sp, arr in pro_split.items():
            for x in arr:
                pro_to_split[int(x)] = sp

        pair_split_indices = {"train": [], "val": [], "test": []}
        excluded = []

        for k, (lig_idx, pro_idx) in enumerate(
            zip(pair_table["pair_lig_idx"], pair_table["pair_pro_idx"])
        ):
            lsp = lig_to_split[int(lig_idx)]
            psp = pro_to_split[int(pro_idx)]

            if lsp == psp:
                pair_split_indices[lsp].append(k)
            else:
                excluded.append(k)

        pair_split_indices = {
            k: np.array(v, dtype=np.int64)
            for k, v in pair_split_indices.items()
        }
        excluded = np.array(excluded, dtype=np.int64)

        meta = {
            "split_protocol": (
                "Metz strict ligand-scaffold and protein-sequence-disjoint product split"
            ),
            "split_seed": int(split_seed),
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "test_ratio": float(1.0 - train_ratio - val_ratio),
            "n_pairs_before_strict_filter": int(len(pair_table["y"])),
            "n_pairs_excluded_cross_partition": int(len(excluded)),
            "n_pairs_after_strict_filter": int(sum(len(v) for v in pair_split_indices.values())),
            "n_ligands": int(len(lig_list)),
            "n_proteins": int(len(pro_list)),
            "n_scaffold_groups": int(len(lig_groups)),
            "n_protein_sequence_groups": int(len(pro_groups)),
            "ligand_counts": {sp: int(len(arr)) for sp, arr in lig_split.items()},
            "protein_counts": {sp: int(len(arr)) for sp, arr in pro_split.items()},
            "pair_counts": {sp: int(len(arr)) for sp, arr in pair_split_indices.items()},
        }

        return pair_split_indices, excluded, lig_split, pro_split, meta

    def _audit_and_save(pair_split_indices, pair_table, scaffolds, save_path):
        lines = []
        lines.append("========== Metz Strict Split Audit ==========")
        lines.append(
            "Protocol: ligand Murcko scaffold split + protein full-sequence disjoint split"
        )
        lines.append(
            "A pair is kept only when ligand split == protein split; "
            "cross-partition pairs are excluded."
        )
        lines.append("")

        split_names = ["train", "val", "test"]
        split_sets = {}

        for sp in split_names:
            idx = pair_split_indices[sp]
            lig_idx = pair_table["pair_lig_idx"][idx]
            pro_idx = pair_table["pair_pro_idx"][idx]

            split_sets[sp] = {
                "ids": set([str(x) for x in pair_table["ids"][idx]]),
                "smiles": set([str(x) for x in pair_table["smiles"][idx]]),
                "seq": set([str(x) for x in pair_table["seq"][idx]]),
                "scaffold": set([str(scaffolds[int(i)]) for i in lig_idx]),
                "lig_idx": set([int(x) for x in lig_idx]),
                "pro_idx": set([int(x) for x in pro_idx]),
            }

            lines.append(
                f"[{sp}] pairs={len(idx)} | ligands={len(split_sets[sp]['lig_idx'])} | "
                f"proteins={len(split_sets[sp]['pro_idx'])} | "
                f"scaffolds={len(split_sets[sp]['scaffold'])}"
            )

        lines.append("")

        for i, a in enumerate(split_names):
            for b in split_names[i + 1:]:
                lines.append(f"--- {a} vs {b} ---")
                for field in ["ids", "scaffold", "seq", "smiles", "lig_idx", "pro_idx"]:
                    overlap = sorted(split_sets[a][field].intersection(split_sets[b][field]))
                    lines.append(
                        f"{field}_overlap={len(overlap)} | examples={overlap[:5]}"
                    )
                lines.append("")

        text = "\n".join(lines)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(text)

    def _save_pair_split(sp, pair_indices, pair_table, scaffolds):
        p = _pair_cache_path(sp)
        idx = np.asarray(pair_indices, dtype=np.int64)
        lig_idx = pair_table["pair_lig_idx"][idx]
        pro_idx = pair_table["pair_pro_idx"][idx]

        np.savez_compressed(
            p,
            ids=pair_table["ids"][idx],
            drug_ids_pair=pair_table["drug_ids_pair"][idx],
            target_ids_pair=pair_table["target_ids_pair"][idx],
            y=pair_table["y"][idx].astype(np.float32),
            smiles=pair_table["smiles"][idx],
            seq=pair_table["seq"][idx],
            pair_lig_idx=lig_idx.astype(np.int64),
            pair_pro_idx=pro_idx.astype(np.int64),
            ligand_scaffold=np.array(
                [scaffolds[int(i)] for i in lig_idx],
                dtype=object,
            ),
        )

    def _build_and_cache_all():
        (
            ids,
            drug_ids_pair,
            target_ids_pair,
            smiles,
            seq,
            y,
            raw_stats,
        ) = _load_raw_metz()

        print(
            f"[Metz] valid pairs={len(y)} | raw_affinity_rows={raw_stats['raw_affinity_rows']} | "
            f"drop_missing_drug={raw_stats['drop_missing_drug']} | "
            f"drop_missing_target={raw_stats['drop_missing_target']} | "
            f"drop_empty_smiles={raw_stats['drop_empty_smiles']} | "
            f"drop_empty_sequence={raw_stats['drop_empty_sequence']} | "
            f"drop_invalid_affinity={raw_stats['drop_invalid_affinity']}"
        )

        lig_list, pro_list, pair_table = _make_pair_table(
            ids=ids,
            drug_ids_pair=drug_ids_pair,
            target_ids_pair=target_ids_pair,
            smiles=smiles,
            seq=seq,
            y=y,
        )

        print(
            f"[Metz] unique ligands={len(lig_list)}, "
            f"unique proteins={len(pro_list)}, pairs={len(pair_table['y'])}"
        )

        scaffolds = np.array(
            [_get_murcko_scaffold(smi) for smi in lig_list],
            dtype=object,
        )

        pair_split_indices, excluded, lig_split, pro_split, meta = _build_strict_split_indices(
            lig_list=lig_list,
            pro_list=pro_list,
            pair_table=pair_table,
            scaffolds=scaffolds,
        )

        meta["raw_stats"] = raw_stats

        for sp in ("train", "val", "test"):
            if len(pair_split_indices[sp]) == 0:
                raise RuntimeError(
                    f"Metz strict split produced empty {sp} set. "
                    "Try changing split_seed/train_ratio/val_ratio."
                )

        with open(split_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        for sp in ("train", "val", "test"):
            _save_pair_split(sp, pair_split_indices[sp], pair_table, scaffolds)

        _audit_and_save(pair_split_indices, pair_table, scaffolds, audit_path)

        # Encode unique ligand/protein token banks.
        lm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LM|Metz] using device: {lm_device}")

        print(f"[LM|Metz] loading ChemBERTa: {chemberta_model_name}")
        chem_tok = AutoTokenizer.from_pretrained(chemberta_model_name)
        chem_model = _hf_load_model(
            chemberta_model_name,
            lm_device,
            use_safetensors,
        )

        drug_lm_tokens_bank, drug_lm_mask_bank = _encode_text_tokens(
            lig_list.tolist(),
            chem_tok,
            chem_model,
            lm_device,
            batch_size=lm_batch_size,
            max_length=chem_max_len,
            desc="[LM|Metz] ChemBERTa token encoding",
            mask_special_tokens=mask_special_tokens,
        )

        try:
            del chem_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        print(f"[LM|Metz] loading ESM-2: {esm2_model_name}")
        esm_tok = AutoTokenizer.from_pretrained(esm2_model_name)
        esm_model = _hf_load_model(
            esm2_model_name,
            lm_device,
            use_safetensors,
        )

        prot_lm_tokens_bank, prot_lm_mask_bank = _encode_text_tokens(
            pro_list.tolist(),
            esm_tok,
            esm_model,
            lm_device,
            batch_size=max(1, min(2, lm_batch_size)),
            max_length=prot_max_len,
            desc="[LM|Metz] ESM-2 token encoding",
            mask_special_tokens=mask_special_tokens,
        )

        try:
            del esm_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        lig_ids = np.array([f"L{i}" for i in range(len(lig_list))], dtype=object)
        pro_ids = np.array([f"P{i}" for i in range(len(pro_list))], dtype=object)

        np.savez_compressed(
            token_bank_path,
            lig_ids=lig_ids,
            lig_smiles=lig_list,
            lig_scaffold=scaffolds,
            pro_ids=pro_ids,
            pro_seq=pro_list,
            drug_lm_tokens_bank=drug_lm_tokens_bank.astype(np.float32),
            drug_lm_mask_bank=drug_lm_mask_bank.astype(bool),
            prot_lm_tokens_bank=prot_lm_tokens_bank.astype(np.float32),
            prot_lm_mask_bank=prot_lm_mask_bank.astype(bool),
        )

        print("[CACHE|Metz] strict token cache done.")

    def _cache_ready():
        if force_refresh:
            return False

        if not token_bank_path.exists():
            return False

        for sp in ("train", "val", "test"):
            if not _pair_cache_path(sp).exists():
                return False

        return True

    if split not in ("train", "val", "test", "all"):
        raise ValueError(f"split must be 'train'/'val'/'test'/'all', got {split!r}")

    if not _cache_ready():
        _build_and_cache_all()

    token_bank = np.load(token_bank_path, allow_pickle=True)

    def _load_split(sp: str):
        d = np.load(_pair_cache_path(sp), allow_pickle=True)

        pair_lig_idx = d["pair_lig_idx"].astype(np.int64)
        pair_pro_idx = d["pair_pro_idx"].astype(np.int64)

        pkg = {
            "ids": d["ids"],
            "drug_ids_pair": d["drug_ids_pair"],
            "target_ids_pair": d["target_ids_pair"],
            "y": d["y"].astype(np.float32),
            "smiles": d["smiles"],
            "seq": d["seq"],
            "pair_lig_idx": pair_lig_idx,
            "pair_pro_idx": pair_pro_idx,
            "ligand_scaffold": d["ligand_scaffold"],

            "drug_lm_tokens_bank": token_bank["drug_lm_tokens_bank"].astype(np.float32),
            "drug_lm_mask_bank": token_bank["drug_lm_mask_bank"].astype(bool),
            "prot_lm_tokens_bank": token_bank["prot_lm_tokens_bank"].astype(np.float32),
            "prot_lm_mask_bank": token_bank["prot_lm_mask_bank"].astype(bool),

            "lig_ids": token_bank["lig_ids"],
            "pro_ids": token_bank["pro_ids"],
            "split_protocol": (
                "Metz strict ligand-scaffold and protein-sequence-disjoint product split"
            ),
            "g_lig": None,
            "g_prot": None,
        }

        if return_pair_level_tokens:
            pkg["drug_lm_tokens"] = pkg["drug_lm_tokens_bank"][pair_lig_idx]
            pkg["drug_lm_mask"] = pkg["drug_lm_mask_bank"][pair_lig_idx]
            pkg["prot_lm_tokens"] = pkg["prot_lm_tokens_bank"][pair_pro_idx]
            pkg["prot_lm_mask"] = pkg["prot_lm_mask_bank"][pair_pro_idx]

        return pkg

    if split in ("train", "val", "test"):
        return {split: _load_split(split)}

    parts = {sp: _load_split(sp) for sp in ("train", "val", "test")}
    return {"all": parts}


def LoadData_toxcast_lm_1d_token_scaffold_seqsplit(
    data_dir: str,
    split: str = "all",
    out_1d: str = "../dataset/toxcast/processed_lm_1d_token_scaffold_seqsplit",
    logspace_trans: bool = True,
    # --- LM models ---
    esm2_model_name: str = "facebook/esm2_t33_650M_UR50D",
    chemberta_model_name: str = "DeepChem/ChemBERTa-77M-MTR",
    lm_batch_size: int = 4,
    chem_max_len: int = 128,
    prot_max_len: int = 1024,
    use_safetensors: bool = True,
    mask_special_tokens: bool = True,
    # --- strict split ---
    split_seed: int = 2023,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    force_refresh: bool = False,
    # --- compatibility / memory option ---
    return_pair_level_tokens: bool = False,
) -> dict:
    """
    ToxCast + token-level LM loader with strict ligand-scaffold / protein-sequence-disjoint split.

    主要修改：
        1. 旧版 loader 使用 CLS / first-token pooled 向量 [N, D]；
           本版改为 token-level 表征 [N_unique, L, D]。
        2. 旧版 loader 保留 data_test.csv 为固定 test，
           并仅在 data_train.csv 内随机划分 train/val；
           本版为回应严格 cold-start 评估，将 train/test 两个 CSV 合并后重新划分。
        3. 使用 token bank + pair index 形式，避免把蛋白 token 展开到所有 pair，
           后续训练时可直接根据 pair_lig_idx / pair_pro_idx 索引 token bank。

    原始文件：
        data_train.csv / data_test.csv，均需包含：
            smiles
            sequence
            label

    严格划分协议：
        - 将 data_train.csv 与 data_test.csv 合并为原始样本池；
        - unique ligand 按 Bemis-Murcko scaffold 分组；
        - unique protein 按清洗后的完整蛋白序列分组；
        - ligand groups 与 protein groups 分别划分到 train/val/test；
        - 只有 ligand split == protein split 的 pair 才保留；
        - 跨分区 pair 会被排除。

    这保证：
        - train / val / test 之间 ligand scaffold 不重叠；
        - train / val / test 之间 protein sequence 不重叠。

    说明：
        本函数是 ToxCast 的 strict cold-start / leakage-controlled 版本，
        不是旧版固定 data_test.csv 协议的直接替代。
        若论文中需要保留原 benchmark protocol，可继续保留旧 loader；
        本函数用于补充严格 scaffold + sequence-disjoint 评估。

    缓存文件：
        toxcast_unique_lm_tokens.npz
        toxcast_train_pairs_scaffold_seqsplit.npz
        toxcast_val_pairs_scaffold_seqsplit.npz
        toxcast_test_pairs_scaffold_seqsplit.npz
        toxcast_split_meta_scaffold_seqsplit.json
        toxcast_split_audit_scaffold_seqsplit.txt

    单个 split 返回：
        {
            'ids', 'y', 'smiles', 'seq', 'source_csv',
            'pair_lig_idx', 'pair_pro_idx',
            'drug_lm_tokens_bank', 'drug_lm_mask_bank',
            'prot_lm_tokens_bank', 'prot_lm_mask_bank',
            'ligand_scaffold', 'split_protocol',
            'lig_ids', 'pro_ids',
            'g_lig', 'g_prot',
            optionally:
                'drug_lm_tokens', 'drug_lm_mask',
                'prot_lm_tokens', 'prot_lm_mask'
        }

    注意：
        return_pair_level_tokens=False 是推荐设置。
        若设置为 True，会把 token bank 展开为 pair-level token，
        便于临时兼容旧写法，但会增加内存占用。
    """
    from pathlib import Path
    import json
    from collections import OrderedDict, defaultdict

    import numpy as np
    import pandas as pd
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModel

    data_dir = Path(data_dir)
    out_1d = Path(out_1d)
    out_1d.mkdir(parents=True, exist_ok=True)

    token_bank_path = out_1d / "toxcast_unique_lm_tokens.npz"
    split_meta_path = out_1d / "toxcast_split_meta_scaffold_seqsplit.json"
    audit_path = out_1d / "toxcast_split_audit_scaffold_seqsplit.txt"

    def _pair_cache_path(sp: str):
        return out_1d / f"toxcast_{sp}_pairs_scaffold_seqsplit.npz"

    def _clean_protein_sequence(seq):
        if seq is None:
            return ""
        seq = str(seq).upper()
        valid = set("ACDEFGHIKLMNPQRSTVWY")
        return "".join([c for c in seq if c in valid])

    def _clean_smiles(smi):
        if smi is None:
            return ""
        return str(smi).strip()

    def _load_raw_toxcast():
        """
        读取 ToxCast train/test CSV，并合并为统一样本池。
        后续重新执行严格 scaffold + sequence-disjoint 划分。
        """
        train_path = data_dir / "data_train.csv"
        test_path = data_dir / "data_test.csv"

        if not train_path.exists():
            raise FileNotFoundError(f"Cannot find {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Cannot find {test_path}")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        required_cols = ["smiles", "sequence", "label"]
        for col in required_cols:
            if col not in train_df.columns:
                raise ValueError(f"data_train.csv 缺少列: {col}")
            if col not in test_df.columns:
                raise ValueError(f"data_test.csv 缺少列: {col}")

        rows = []
        stats = {
            "raw_train_rows": int(len(train_df)),
            "raw_test_rows": int(len(test_df)),
            "kept_rows": 0,
            "drop_empty_smiles": 0,
            "drop_empty_sequence": 0,
            "drop_invalid_label": 0,
        }

        for source_name, df in (("train_csv", train_df), ("test_csv", test_df)):
            for row_idx, row in df.iterrows():
                smi = _clean_smiles(row["smiles"])
                seq = _clean_protein_sequence(row["sequence"])

                if smi == "":
                    stats["drop_empty_smiles"] += 1
                    continue

                if seq == "":
                    stats["drop_empty_sequence"] += 1
                    continue

                try:
                    y = float(row["label"])
                except Exception:
                    stats["drop_invalid_label"] += 1
                    continue

                if not np.isfinite(y):
                    stats["drop_invalid_label"] += 1
                    continue

                if logspace_trans:
                    y = float(np.log10(y + 1e-8))

                rows.append({
                    "id": f"{source_name}_{row_idx}",
                    "smiles": smi,
                    "seq": seq,
                    "y": y,
                    "source_csv": source_name,
                })

        stats["kept_rows"] = int(len(rows))

        if len(rows) == 0:
            raise RuntimeError("ToxCast 清洗后没有有效样本，请检查 data_train.csv / data_test.csv。")

        ids = np.array([r["id"] for r in rows], dtype=object)
        smiles = np.array([r["smiles"] for r in rows], dtype=object)
        seq = np.array([r["seq"] for r in rows], dtype=object)
        y = np.array([r["y"] for r in rows], dtype=np.float32)
        source_csv = np.array([r["source_csv"] for r in rows], dtype=object)

        return ids, smiles, seq, y, source_csv, stats

    def _get_murcko_scaffold(smi: str) -> str:
        """返回 Bemis-Murcko scaffold；失败时使用稳定 fallback key。"""
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold

            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                return f"INVALID::{str(smi)}"

            scaf = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol,
                includeChirality=False,
            )

            if scaf is None or str(scaf).strip() == "":
                can = Chem.MolToSmiles(mol, canonical=True)
                return f"NO_SCAFFOLD::{can}"

            return str(scaf)
        except Exception:
            return f"FALLBACK::{str(smi)}"

    def _split_groups(groups: dict, total_items: int):
        """按完整 group 划分；group 不会被拆开。"""
        if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
            raise ValueError(
                f"train_ratio={train_ratio}, val_ratio={val_ratio} 不合法；"
                "要求 train_ratio>0, val_ratio>=0, 且 train_ratio+val_ratio<1。"
            )

        rng = np.random.RandomState(split_seed)
        items = list(groups.items())
        rng.shuffle(items)
        items = sorted(items, key=lambda kv: len(kv[1]), reverse=True)

        n_train_target = int(total_items * train_ratio)
        n_val_target = int(total_items * val_ratio)

        split_to_indices = {"train": [], "val": [], "test": []}
        split_to_groups = {"train": [], "val": [], "test": []}

        for group_key, idxs in items:
            idxs = list(idxs)

            if len(split_to_indices["train"]) + len(idxs) <= n_train_target:
                sp = "train"
            elif len(split_to_indices["val"]) + len(idxs) <= n_val_target:
                sp = "val"
            else:
                sp = "test"

            split_to_indices[sp].extend(idxs)
            split_to_groups[sp].append(group_key)

        return (
            {k: np.array(v, dtype=np.int64) for k, v in split_to_indices.items()},
            {k: set(v) for k, v in split_to_groups.items()},
        )

    def _hf_load_model(name: str, device: torch.device, use_safetensors: bool = True):
        try:
            model = AutoModel.from_pretrained(
                name,
                use_safetensors=use_safetensors,
            )
            model.to(device)
            return model
        except TypeError:
            model = AutoModel.from_pretrained(name)
            model.to(device)
            return model
        except Exception as e:
            msg = str(e).lower()
            if ("torch" in msg and "2.6" in msg) or ("vulnerability" in msg) or ("cve" in msg):
                raise RuntimeError(
                    f"Loading HF model '{name}' failed due to transformers / torch safety check:\n{e}\n"
                    "Possible solutions: use safetensors, upgrade torch, or use a compatible transformers version."
                ) from e
            model = AutoModel.from_pretrained(name)
            model.to(device)
            return model

    def _encode_text_tokens(
        text_list,
        tokenizer,
        model,
        device,
        batch_size=4,
        max_length=512,
        desc="[LM] token encoding",
        mask_special_tokens=True,
    ):
        """
        生成 token-level last_hidden_state 与 padding mask。
        mask 中 True 表示后续 attention 中需要忽略该 token。
        """
        all_tokens = []
        all_masks = []
        model.eval()

        with torch.no_grad():
            for i in tqdm(range(0, len(text_list), batch_size), desc=desc, unit="batch"):
                batch = [str(x) for x in text_list[i:i + batch_size]]
                enc = tokenizer(
                    batch,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                    return_special_tokens_mask=True,
                )

                special_tokens_mask = enc.pop("special_tokens_mask", None)
                enc = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in enc.items()
                }
                if special_tokens_mask is not None:
                    special_tokens_mask = special_tokens_mask.to(device)

                try:
                    out = model(**enc)
                except TypeError as e:
                    if "token_type_ids" in str(e) and "token_type_ids" in enc:
                        enc.pop("token_type_ids")
                        out = model(**enc)
                    else:
                        raise e

                if hasattr(out, "last_hidden_state"):
                    hs = out.last_hidden_state
                elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                    hs = out[0]
                else:
                    raise RuntimeError("Unsupported model output structure for token-level LM encoding.")

                attention_mask = enc["attention_mask"].bool()
                pad_mask = ~attention_mask

                if mask_special_tokens and special_tokens_mask is not None:
                    pad_mask = pad_mask | special_tokens_mask.bool()

                all_tokens.append(hs.detach().cpu().numpy().astype(np.float32))
                all_masks.append(pad_mask.detach().cpu().numpy().astype(bool))

        if len(all_tokens) == 0:
            hidden_size = int(getattr(model.config, "hidden_size", 0))
            return (
                np.zeros((0, max_length, hidden_size), dtype=np.float32),
                np.ones((0, max_length), dtype=bool),
            )

        return (
            np.concatenate(all_tokens, axis=0).astype(np.float32),
            np.concatenate(all_masks, axis=0).astype(bool),
        )

    def _make_pair_table(ids, smiles, seq, y, source_csv):
        """构造 unique ligand/protein token bank 索引。"""
        lig_list = np.array(
            list(OrderedDict.fromkeys(smiles.tolist()).keys()),
            dtype=object,
        )
        pro_list = np.array(
            list(OrderedDict.fromkeys(seq.tolist()).keys()),
            dtype=object,
        )

        lig2idx = {str(smi): i for i, smi in enumerate(lig_list)}
        pro2idx = {str(s): i for i, s in enumerate(pro_list)}

        pair_lig_idx = np.array([lig2idx[str(smi)] for smi in smiles], dtype=np.int64)
        pair_pro_idx = np.array([pro2idx[str(s)] for s in seq], dtype=np.int64)

        pair_table = {
            "ids": ids,
            "y": y.astype(np.float32),
            "smiles": smiles,
            "seq": seq,
            "source_csv": source_csv,
            "pair_lig_idx": pair_lig_idx,
            "pair_pro_idx": pair_pro_idx,
        }

        return lig_list, pro_list, pair_table

    def _build_strict_split_indices(lig_list, pro_list, pair_table, scaffolds):
        lig_groups = defaultdict(list)
        for i, scaf in enumerate(scaffolds):
            lig_groups[str(scaf)].append(i)

        pro_groups = defaultdict(list)
        for j, seq_text in enumerate(pro_list):
            pro_groups[str(seq_text)].append(j)

        lig_split, _ = _split_groups(lig_groups, len(lig_list))
        pro_split, _ = _split_groups(pro_groups, len(pro_list))

        lig_to_split = {}
        for sp, arr in lig_split.items():
            for x in arr:
                lig_to_split[int(x)] = sp

        pro_to_split = {}
        for sp, arr in pro_split.items():
            for x in arr:
                pro_to_split[int(x)] = sp

        pair_split_indices = {"train": [], "val": [], "test": []}
        excluded = []

        for k, (lig_idx, pro_idx) in enumerate(
            zip(pair_table["pair_lig_idx"], pair_table["pair_pro_idx"])
        ):
            lsp = lig_to_split[int(lig_idx)]
            psp = pro_to_split[int(pro_idx)]

            if lsp == psp:
                pair_split_indices[lsp].append(k)
            else:
                excluded.append(k)

        pair_split_indices = {
            k: np.array(v, dtype=np.int64)
            for k, v in pair_split_indices.items()
        }
        excluded = np.array(excluded, dtype=np.int64)

        meta = {
            "split_protocol": (
                "ToxCast strict ligand-scaffold and protein-sequence-disjoint product split "
                "over merged data_train/data_test CSV pool"
            ),
            "split_seed": int(split_seed),
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "test_ratio": float(1.0 - train_ratio - val_ratio),
            "n_pairs_before_strict_filter": int(len(pair_table["y"])),
            "n_pairs_excluded_cross_partition": int(len(excluded)),
            "n_pairs_after_strict_filter": int(sum(len(v) for v in pair_split_indices.values())),
            "n_ligands": int(len(lig_list)),
            "n_proteins": int(len(pro_list)),
            "n_scaffold_groups": int(len(lig_groups)),
            "n_protein_sequence_groups": int(len(pro_groups)),
            "ligand_counts": {sp: int(len(arr)) for sp, arr in lig_split.items()},
            "protein_counts": {sp: int(len(arr)) for sp, arr in pro_split.items()},
            "pair_counts": {sp: int(len(arr)) for sp, arr in pair_split_indices.items()},
            "source_csv_counts_after_filter": {
                sp: {
                    "train_csv": int(np.sum(pair_table["source_csv"][idx] == "train_csv")),
                    "test_csv": int(np.sum(pair_table["source_csv"][idx] == "test_csv")),
                }
                for sp, idx in pair_split_indices.items()
            },
        }

        return pair_split_indices, excluded, lig_split, pro_split, meta

    def _audit_and_save(pair_split_indices, pair_table, scaffolds, save_path):
        lines = []
        lines.append("========== ToxCast Strict Split Audit ==========")
        lines.append(
            "Protocol: ligand Murcko scaffold split + protein full-sequence disjoint split"
        )
        lines.append(
            "Raw data_train.csv and data_test.csv are merged into one pool, "
            "then re-split under the strict protocol."
        )
        lines.append(
            "A pair is kept only when ligand split == protein split; "
            "cross-partition pairs are excluded."
        )
        lines.append("")

        split_names = ["train", "val", "test"]
        split_sets = {}

        for sp in split_names:
            idx = pair_split_indices[sp]
            lig_idx = pair_table["pair_lig_idx"][idx]
            pro_idx = pair_table["pair_pro_idx"][idx]

            split_sets[sp] = {
                "ids": set([str(x) for x in pair_table["ids"][idx]]),
                "smiles": set([str(x) for x in pair_table["smiles"][idx]]),
                "seq": set([str(x) for x in pair_table["seq"][idx]]),
                "scaffold": set([str(scaffolds[int(i)]) for i in lig_idx]),
                "lig_idx": set([int(x) for x in lig_idx]),
                "pro_idx": set([int(x) for x in pro_idx]),
            }

            n_from_train_csv = int(np.sum(pair_table["source_csv"][idx] == "train_csv"))
            n_from_test_csv = int(np.sum(pair_table["source_csv"][idx] == "test_csv"))

            lines.append(
                f"[{sp}] pairs={len(idx)} | ligands={len(split_sets[sp]['lig_idx'])} | "
                f"proteins={len(split_sets[sp]['pro_idx'])} | "
                f"scaffolds={len(split_sets[sp]['scaffold'])} | "
                f"from_train_csv={n_from_train_csv} | from_test_csv={n_from_test_csv}"
            )

        lines.append("")

        for i, a in enumerate(split_names):
            for b in split_names[i + 1:]:
                lines.append(f"--- {a} vs {b} ---")
                for field in ["ids", "scaffold", "seq", "smiles", "lig_idx", "pro_idx"]:
                    overlap = sorted(split_sets[a][field].intersection(split_sets[b][field]))
                    lines.append(
                        f"{field}_overlap={len(overlap)} | examples={overlap[:5]}"
                    )
                lines.append("")

        text = "\n".join(lines)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(text)

    def _save_pair_split(sp, pair_indices, pair_table, scaffolds):
        p = _pair_cache_path(sp)
        idx = np.asarray(pair_indices, dtype=np.int64)
        lig_idx = pair_table["pair_lig_idx"][idx]
        pro_idx = pair_table["pair_pro_idx"][idx]

        np.savez_compressed(
            p,
            ids=pair_table["ids"][idx],
            y=pair_table["y"][idx].astype(np.float32),
            smiles=pair_table["smiles"][idx],
            seq=pair_table["seq"][idx],
            source_csv=pair_table["source_csv"][idx],
            pair_lig_idx=lig_idx.astype(np.int64),
            pair_pro_idx=pro_idx.astype(np.int64),
            ligand_scaffold=np.array(
                [scaffolds[int(i)] for i in lig_idx],
                dtype=object,
            ),
        )

    def _build_and_cache_all():
        ids, smiles, seq, y, source_csv, raw_stats = _load_raw_toxcast()

        print(
            f"[ToxCast] valid rows={len(y)} | "
            f"raw_train={raw_stats['raw_train_rows']} | raw_test={raw_stats['raw_test_rows']} | "
            f"drop_empty_smiles={raw_stats['drop_empty_smiles']} | "
            f"drop_empty_sequence={raw_stats['drop_empty_sequence']} | "
            f"drop_invalid_label={raw_stats['drop_invalid_label']}"
        )

        lig_list, pro_list, pair_table = _make_pair_table(
            ids=ids,
            smiles=smiles,
            seq=seq,
            y=y,
            source_csv=source_csv,
        )

        print(
            f"[ToxCast] unique ligands={len(lig_list)}, "
            f"unique proteins={len(pro_list)}, pairs={len(pair_table['y'])}"
        )

        scaffolds = np.array(
            [_get_murcko_scaffold(smi) for smi in lig_list],
            dtype=object,
        )

        pair_split_indices, excluded, lig_split, pro_split, meta = _build_strict_split_indices(
            lig_list=lig_list,
            pro_list=pro_list,
            pair_table=pair_table,
            scaffolds=scaffolds,
        )

        meta["raw_stats"] = raw_stats

        for sp in ("train", "val", "test"):
            if len(pair_split_indices[sp]) == 0:
                raise RuntimeError(
                    f"ToxCast strict split produced empty {sp} set. "
                    "Try changing split_seed/train_ratio/val_ratio."
                )

        with open(split_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        for sp in ("train", "val", "test"):
            _save_pair_split(sp, pair_split_indices[sp], pair_table, scaffolds)

        _audit_and_save(pair_split_indices, pair_table, scaffolds, audit_path)

        lm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LM|ToxCast] using device: {lm_device}")

        print(f"[LM|ToxCast] loading ChemBERTa: {chemberta_model_name}")
        chem_tok = AutoTokenizer.from_pretrained(chemberta_model_name)
        chem_model = _hf_load_model(
            chemberta_model_name,
            lm_device,
            use_safetensors,
        )

        drug_lm_tokens_bank, drug_lm_mask_bank = _encode_text_tokens(
            lig_list.tolist(),
            chem_tok,
            chem_model,
            lm_device,
            batch_size=lm_batch_size,
            max_length=chem_max_len,
            desc="[LM|ToxCast] ChemBERTa token encoding",
            mask_special_tokens=mask_special_tokens,
        )

        try:
            del chem_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        print(f"[LM|ToxCast] loading ESM-2: {esm2_model_name}")
        esm_tok = AutoTokenizer.from_pretrained(esm2_model_name)
        esm_model = _hf_load_model(
            esm2_model_name,
            lm_device,
            use_safetensors,
        )

        prot_lm_tokens_bank, prot_lm_mask_bank = _encode_text_tokens(
            pro_list.tolist(),
            esm_tok,
            esm_model,
            lm_device,
            batch_size=max(1, min(2, lm_batch_size)),
            max_length=prot_max_len,
            desc="[LM|ToxCast] ESM-2 token encoding",
            mask_special_tokens=mask_special_tokens,
        )

        try:
            del esm_model
            torch.cuda.empty_cache()
        except Exception:
            pass

        lig_ids = np.array([f"L{i}" for i in range(len(lig_list))], dtype=object)
        pro_ids = np.array([f"P{i}" for i in range(len(pro_list))], dtype=object)

        np.savez_compressed(
            token_bank_path,
            lig_ids=lig_ids,
            lig_smiles=lig_list,
            lig_scaffold=scaffolds,
            pro_ids=pro_ids,
            pro_seq=pro_list,
            drug_lm_tokens_bank=drug_lm_tokens_bank.astype(np.float32),
            drug_lm_mask_bank=drug_lm_mask_bank.astype(bool),
            prot_lm_tokens_bank=prot_lm_tokens_bank.astype(np.float32),
            prot_lm_mask_bank=prot_lm_mask_bank.astype(bool),
        )

        print("[CACHE|ToxCast] strict token cache done.")

    def _cache_ready():
        if force_refresh:
            return False

        if not token_bank_path.exists():
            return False

        for sp in ("train", "val", "test"):
            if not _pair_cache_path(sp).exists():
                return False

        return True

    if split not in ("train", "val", "test", "all"):
        raise ValueError(f"split must be 'train'/'val'/'test'/'all', got {split!r}")

    if not _cache_ready():
        _build_and_cache_all()

    token_bank = np.load(token_bank_path, allow_pickle=True)

    def _load_split(sp: str):
        d = np.load(_pair_cache_path(sp), allow_pickle=True)

        pair_lig_idx = d["pair_lig_idx"].astype(np.int64)
        pair_pro_idx = d["pair_pro_idx"].astype(np.int64)

        pkg = {
            "ids": d["ids"],
            "y": d["y"].astype(np.float32),
            "smiles": d["smiles"],
            "seq": d["seq"],
            "source_csv": d["source_csv"],
            "pair_lig_idx": pair_lig_idx,
            "pair_pro_idx": pair_pro_idx,
            "ligand_scaffold": d["ligand_scaffold"],

            "drug_lm_tokens_bank": token_bank["drug_lm_tokens_bank"].astype(np.float32),
            "drug_lm_mask_bank": token_bank["drug_lm_mask_bank"].astype(bool),
            "prot_lm_tokens_bank": token_bank["prot_lm_tokens_bank"].astype(np.float32),
            "prot_lm_mask_bank": token_bank["prot_lm_mask_bank"].astype(bool),

            "lig_ids": token_bank["lig_ids"],
            "pro_ids": token_bank["pro_ids"],
            "split_protocol": (
                "ToxCast strict ligand-scaffold and protein-sequence-disjoint product split "
                "over merged data_train/data_test CSV pool"
            ),
            "g_lig": None,
            "g_prot": None,
        }

        if return_pair_level_tokens:
            pkg["drug_lm_tokens"] = pkg["drug_lm_tokens_bank"][pair_lig_idx]
            pkg["drug_lm_mask"] = pkg["drug_lm_mask_bank"][pair_lig_idx]
            pkg["prot_lm_tokens"] = pkg["prot_lm_tokens_bank"][pair_pro_idx]
            pkg["prot_lm_mask"] = pkg["prot_lm_mask_bank"][pair_pro_idx]

        return pkg

    if split in ("train", "val", "test"):
        return {split: _load_split(split)}

    parts = {sp: _load_split(sp) for sp in ("train", "val", "test")}
    return {"all": parts}
