"""Contextual synonym augmentation with ESG-term + span placeholder protection.

Two-layer protection:
    1. ESG_KEYWORDS — fixed domain terms that shouldn't be substituted
    2. protected_spans — dynamic per-sample spans (e.g. promise_string,
       evidence_string) — critical because the span aux task needs to still
       locate these substrings in the augmented text to produce BIO labels.
"""
from typing import List, Optional

import torch
from transformers import BertTokenizer

# Audit 後（train n=1000）從 6 個擴充到 41 個。挑選原則：
#   - hit_rate ≥ 10 樣本（避免 AI 幻覺字）
#   - 英文縮寫一律納入（tokenizer 會切到 character piece 級，保護後 augmenter 不動）
#   - 包括 N/A 反向訊號字（GRI / 利害關係人 / 重大議題）— 保護目的是保留原始字串，
#     不論訊號正反向，被 augmenter 隨機替換才是真正破壞
ESG_KEYWORDS: List[str] = [
    # 既有 6 字
    "碳中和", "永續", "董事會", "綠色能源", "減碳", "企業社會責任",
    # 排放範疇
    "範疇一", "範疇二", "範疇三",
    # 碳 / 溫室氣體
    "溫室氣體", "碳排放", "碳足跡", "碳定價", "碳費", "淨零",
    # ESG 認證 / 框架（英文縮寫 — 切分問題嚴重）
    "ISO", "TCFD", "SASB", "GRI", "CDP", "RE100", "SBTi", "PCAF", "TNFD",
    "ESG", "SDGs", "SDG", "永續報告書",
    # 國際框架
    "巴黎協定", "聯合國", "永續發展目標",
    # 領域名詞
    "再生能源", "綠電", "循環經濟", "生物多樣性",
    "利害關係人", "重大性", "重大議題",
    "資訊安全", "個人資料保護", "供應鏈管理",
]


def _apply_monkey_patch() -> None:
    """nlpaug expects an older BertTokenizer private method — alias it."""
    if not hasattr(BertTokenizer, "_convert_token_to_id"):
        BertTokenizer._convert_token_to_id = BertTokenizer.convert_tokens_to_ids


class ContextualAugmenter:
    def __init__(self, model_name: str, aug_p: float = 0.1, device: Optional[str] = None):
        _apply_monkey_patch()
        import nlpaug.augmenter.word as naw  # imported lazily; optional dep

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.aug = naw.ContextualWordEmbsAug(
            model_path=model_name, model_type="bert",
            device=device, action="substitute", aug_p=aug_p,
        )
        # 按長度 desc 排序：先替換長字串才不會被短子字串吃掉
        # （e.g. 永續報告書 vs 永續，SDGs vs SDG）
        # Placeholder 用 `__P{i}__` 不用 `__ESG_{i}__`，因為 ESG 本身就是 protected term，
        # 用 ESG 開頭的 placeholder 會被後面的 ESG 替換步驟二次破壞。
        sorted_kw = sorted(ESG_KEYWORDS, key=len, reverse=True)
        self._placeholders = {f"__P{i}__": kw for i, kw in enumerate(sorted_kw)}

    def __call__(self, text: str, protected_spans: Optional[List[str]] = None) -> str:
        """Augment text while keeping ESG keywords + any provided spans verbatim.

        Args:
            text: input text to augment.
            protected_spans: optional list of exact substrings that must survive
                augmentation untouched. Typically `[promise_string, evidence_string]`.
                Substrings are replaced with unique placeholders before augmentation
                and restored afterwards. Empty / missing spans are skipped.
        Returns:
            Augmented text with all protected substrings preserved.
        """
        # Build dynamic span placeholders（格式與 ESG 相同方便 nlpaug 一致看待為 rare token）
        span_map = {}
        if protected_spans:
            for i, s in enumerate(protected_spans):
                if s and s in text:
                    span_map[f"__SPAN_{i}__"] = s

        # Apply both placeholder passes (span first → ESG second；reverse 時相反)
        tmp = text
        for ph, s in span_map.items():
            tmp = tmp.replace(s, ph)
        for ph, kw in self._placeholders.items():
            tmp = tmp.replace(kw, ph)

        try:
            out = self.aug.augment(tmp)
            if isinstance(out, list):
                out = out[0]
        except Exception:
            out = tmp

        # Restore（ESG 先還，因為可能嵌在 span 裡面；若真嵌則 span 內保留原 ESG 字）
        for ph, kw in self._placeholders.items():
            out = out.replace(ph, kw)
        for ph, s in span_map.items():
            out = out.replace(ph, s)
        return out
