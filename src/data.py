"""Data loading, splitting, and Dataset.

Key change vs. baseline:
    - Baseline: train_test_split 8:2, no held-out test; val set doubles as submission.
    - Here:     (1) carve out a real held-out test set first (never touched),
                (2) run StratifiedKFold on the remainder so every fold sees
                    a balanced mix of the highest-weight label.
"""
import json
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from .config import CFG, EVAL_FIELDS
from .features import extract_features, NUM_FEATURES


# 主辦方 train / val 兩個資料集對同一類別命名不一致（其他三個欄位都對齊）：
#   train_1000.json: `longer_than_5_years`
#   val_1000.json:   `more_than_5_years`
# 統一到 train（與 config.EVAL_FIELDS 對齊），避免 ESGDataset 在 val 上找不到 label。
# 若未來主辦方又出現新的不一致，直接在這個 dict 加 mapping 即可。
_LABEL_NORM = {
    "verification_timeline": {"more_than_5_years": "longer_than_5_years"},
}


def load_raw(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for d in data:
        for field, mapping in _LABEL_NORM.items():
            v = d.get(field)
            if v in mapping:
                d[field] = mapping[v]
    return data


def _build_multilabel_matrix(data: List[dict]) -> np.ndarray:
    """One-hot concat across all 4 EVAL_FIELDS → [N, sum(num_classes)] binary matrix.
    Each row has exactly `len(EVAL_FIELDS)` ones.
    """
    columns = [(f, lab) for f, labs in EVAL_FIELDS.items() for lab in labs]
    col_to_idx = {c: i for i, c in enumerate(columns)}
    y = np.zeros((len(data), len(columns)), dtype=np.int32)
    for i, d in enumerate(data):
        for f in EVAL_FIELDS:
            y[i, col_to_idx[(f, d[f])]] = 1
    return y


def _safe_stratify_labels(data: List[dict], field: str, min_count: int) -> Optional[List[str]]:
    """Return stratify labels, or None (with warning) if any class has < min_count samples."""
    counts = Counter(d[field] for d in data)
    rare = {lab: c for lab, c in counts.items() if c < min_count}
    if rare:
        print(f"⚠️  single-field stratify '{field}' 放棄：{rare} < {min_count}，退回隨機切分")
        return None
    return [d[field] for d in data]


def holdout_test_split(data: List[dict], test_size: float, stratify_field: str,
                       seed: int) -> Tuple[List[dict], List[dict]]:
    """Carve out a held-out test set balanced across ALL 4 fields if possible."""
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        y = _build_multilabel_matrix(data)
        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed,
        )
        idx = np.arange(len(data))
        tr_idx, te_idx = next(msss.split(idx, y))
        print(f"✅ held-out 用 MultilabelStratifiedShuffleSplit，四個欄位同時分層")
        return [data[i] for i in tr_idx], [data[i] for i in te_idx]
    except ImportError:
        print("⚠️  iterative-stratification 未安裝，退回單欄位 stratify。")
        print("    建議: !pip install iterative-stratification")
        strat = _safe_stratify_labels(data, stratify_field, min_count=2)
        return train_test_split(
            data, test_size=test_size, random_state=seed, stratify=strat,
        )


def kfold_indices(data: List[dict], n_splits: int, stratify_field: str, seed: int):
    """Yield (train_idx, val_idx). Prefers MultilabelStratifiedKFold across all 4 fields."""
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        y = _build_multilabel_matrix(data)
        mskf = MultilabelStratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed,
        )
        idx = np.arange(len(data))
        print(f"✅ K-fold 用 MultilabelStratifiedKFold(n={n_splits})，四個欄位同時分層")
        for tr, va in mskf.split(idx, y):
            yield tr, va
        return
    except ImportError:
        print("⚠️  iterative-stratification 未安裝，退回單欄位 StratifiedKFold。")

    y1 = _safe_stratify_labels(data, stratify_field, min_count=n_splits)
    if y1 is None:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, va in kf.split(data):
            yield tr, va
        return
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(data, y1):
        yield tr, va


SPAN_LABEL_IDS = {
    "O": 0, "B-PROMISE": 1, "I-PROMISE": 2, "B-EVIDENCE": 3, "I-EVIDENCE": 4,
}
SPAN_IGNORE = -100   # F.cross_entropy(ignore_index=-100) → 不算 loss

# esg_type 前綴映射 — 支援單一或複合類別（"E", "E;S;G", "S;G" ...）
_ESG_NAMES = {"E": "環境", "S": "社會", "G": "治理"}


def _build_esg_prefix(et: str) -> str:
    """Build prefix supporting compound types like 'E;S;G' or 'S;G'.

    Examples:
        "E"       → "環境類："
        "E;S;G"   → "環境、社會、治理類："
        "S;G"     → "社會、治理類："
        ""        → ""
    """
    if not et:
        return ""
    parts = [t.strip() for t in et.split(";")]
    names = [_ESG_NAMES[t] for t in parts if t in _ESG_NAMES]
    if not names:
        return ""
    return "、".join(names) + "類："


def _maybe_add_esg_prefix(sample: dict, text: str) -> str:
    if not CFG.use_esg_prefix:
        return text
    return _build_esg_prefix(sample.get("esg_type", "")) + text


def _assign_bio(span_labels: List[int], offsets, text_span, b_id: int, i_id: int) -> None:
    """In-place overwrite: 對 offset 與 [char_start, char_end) 相交的 token 標 BIO。"""
    if text_span is None:
        return
    s, e = text_span
    first = True
    for i, (toks, toke) in enumerate(offsets):
        if toks == toke == 0:                       # 特殊/padding
            continue
        if toke > s and toks < e:                   # 有交集
            span_labels[i] = b_id if first else i_id
            first = False


def _find_substr_span(text: str, substr: str):
    """Return (start_char, end_char) or None if substr empty/not found."""
    if not substr:
        return None
    idx = text.find(substr)
    if idx < 0:
        return None
    return (idx, idx + len(substr))


class ESGDataset(Dataset):
    """PyTorch Dataset.

    如果 tokenizer 是 fast 版本且 `CFG.use_span_aux=True`，會額外產出 `span_labels`
    供輔助 token classification 任務使用。否則 span_labels 全為 ignore (-100)。
    """

    def __init__(self, data: List[dict], tokenizer, label2id: Dict,
                 mask_prob: float = 0.0, aug_prob: float = 0.0, augmenter=None):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.mask_prob = mask_prob
        self.aug_prob = aug_prob
        self.augmenter = augmenter
        self._can_span = CFG.use_span_aux and getattr(tokenizer, "is_fast", False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample["data"]

        augmented = False
        if self.augmenter is not None and self.aug_prob > 0 and random.random() < self.aug_prob:
            # Span-preserving augmentation：把 promise / evidence 原文當 protected_spans 傳入，
            # augmenter 會用佔位符保護它們，augment 後還原 → augmented text 仍包含原 span 子字串，
            # 後續 text.find(span_string) 仍能對齊，span_labels 不必被丟棄。
            protected = []
            for key in ("promise_string", "evidence_string"):
                s = sample.get(key, "")
                if s:
                    protected.append(s)
            text = self.augmenter(text, protected_spans=protected)
            augmented = True

        # ESG 類別前綴（E/S/G → "環境類：/社會類：/治理類："）
        # 在 augment 之後加，確保前綴不會被 augmenter 換掉；span 對齊用 text.find
        # 會自動把 offset 向後平移，不會錯位。
        text = _maybe_add_esg_prefix(sample, text)

        enc_kwargs = dict(
            truncation=True, max_length=CFG.max_len,
            padding="max_length", return_tensors="pt",
        )
        if self._can_span:
            enc_kwargs["return_offsets_mapping"] = True

        enc = self.tokenizer(text, **enc_kwargs)
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Span labels — 由於 augmenter 採 span-preserving（保留 promise / evidence 原子字串），
        # augment 後仍能用 text.find 對齊，不必再丟棄 supervision。
        if self._can_span:
            offsets = enc["offset_mapping"].squeeze(0).tolist()
            span_labels = [SPAN_IGNORE] * CFG.max_len
            # 先把 attention_mask=1 但非特殊符的位置設為 O（0）
            for i, (s, e) in enumerate(offsets):
                if attention_mask[i].item() == 1 and not (s == 0 and e == 0):
                    span_labels[i] = SPAN_LABEL_IDS["O"]
            _assign_bio(span_labels, offsets,
                        _find_substr_span(text, sample.get("promise_string", "")),
                        SPAN_LABEL_IDS["B-PROMISE"], SPAN_LABEL_IDS["I-PROMISE"])
            _assign_bio(span_labels, offsets,
                        _find_substr_span(text, sample.get("evidence_string", "")),
                        SPAN_LABEL_IDS["B-EVIDENCE"], SPAN_LABEL_IDS["I-EVIDENCE"])
            span_labels = torch.tensor(span_labels, dtype=torch.long)
        else:
            span_labels = torch.full((CFG.max_len,), SPAN_IGNORE, dtype=torch.long)

        # Random masking（在 span label 之後做，不影響 offset 計算）
        if self.mask_prob > 0:
            special = (input_ids == 101) | (input_ids == 102) | (input_ids == 0)
            prob = torch.full(input_ids.shape, self.mask_prob)
            prob.masked_fill_(special, value=0.0)
            masked = torch.bernoulli(prob).bool()
            input_ids[masked] = 103  # [MASK]

        labels = {f: torch.tensor(self.label2id[f][sample[f]], dtype=torch.long)
                  for f in self.label2id}

        # 手工特徵：傳整個 sample dict（含 data + page_number），語意更穩定
        if CFG.use_hand_features:
            feats = torch.tensor(extract_features(sample), dtype=torch.float)
        else:
            feats = torch.zeros(CFG.num_hand_features, dtype=torch.float)

        return {
            "input_ids": input_ids, "attention_mask": attention_mask,
            "labels": labels, "span_labels": span_labels, "features": feats,
        }


def compute_sample_weights(data: List[dict],
                           cap: Optional[float] = None,
                           alpha: Optional[float] = None,
                           mode: Optional[str] = None) -> List[float]:
    """Per-sample weight for WeightedRandomSampler.

    `mode` 控制計算邏輯（從 CFG.sampler_mode 讀取若未指定）：

    1. "promise_balanced"（**推薦**，搭配 cascade heads 用）：
       promise=No 樣本權重 = CFG.promise_no_oversample_ratio（預設 3.0）
       promise=Yes 樣本權重 = 1.0
       目的：穩定 promise head 看到的正負樣本比，幫 cascade 下游有可靠的 promise_probs。

    2. "inverse_freq"（舊版，實測在 1000 筆上傷分）：
       weight = min((N / (num_classes × class_count)) ** alpha, cap)
       Per-field max 保留：每個樣本看「在 4 個欄位中最稀有那個」的權重。
       alpha=0.5 是 sqrt 平滑（Misleading n=1 → 14×）；alpha=1.0 是純 inverse freq。
    """
    if mode is None:
        mode = getattr(CFG, "sampler_mode", "inverse_freq")

    if mode == "promise_balanced":
        ratio = getattr(CFG, "promise_no_oversample_ratio", 3.0)
        return [ratio if d.get("promise_status") == "No" else 1.0 for d in data]

    # mode == "inverse_freq"
    if cap is None:
        cap = CFG.sampler_cap
    if alpha is None:
        alpha = CFG.sampler_alpha
    n = len(data)
    counts = {f: Counter(d[f] for d in data) for f in EVAL_FIELDS}
    weights: List[float] = []
    for d in data:
        per_field = []
        for f, labs in EVAL_FIELDS.items():
            c = counts[f].get(d[f], 0)
            if c == 0:
                per_field.append(1.0)
            else:
                per_field.append((n / (len(labs) * c)) ** alpha)
        weights.append(min(max(per_field), cap))
    return weights


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = {f: torch.stack([b["labels"][f] for b in batch]) for f in EVAL_FIELDS}
    span_labels = torch.stack([b["span_labels"] for b in batch])
    features = torch.stack([b["features"] for b in batch])
    return {
        "input_ids": input_ids, "attention_mask": attention_mask,
        "labels": labels, "span_labels": span_labels, "features": features,
    }
