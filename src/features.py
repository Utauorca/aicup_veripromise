"""Hand-crafted domain features.

從 ESG 報告每筆樣本抓出人工判讀會用到的訊號，作為模型 pooled representation 之外
的輔助輸入。

設計原則：
    - Raw counts：14 個 regex 類別 + 1 個文字長度
    - Derived features：5 個比例/密度指標（future/past ratio + 4 個 density）
    - 所有 count 過 log1p 壓縮極端值；ratio / density 不再 log
"""
import math
import re
from typing import List, Union


# ── Base regex patterns（對應 raw counts）────────────────────────────
# Audit 迭代結果（train n=1000）：
#   1) 移除 hit < 4% 的 `contradiction`、`completion`
#   2) 加 `numeric_specific`（hit 57.7%，evidence_quality C/NC 3.1×）
#   3) 擴充 `evidence_verbs`（hit 34→55%，C/NC 1.33→1.62×；其他 4 個原 pattern
#      擴充後都把鑑別力稀釋掉，所以不動）
#   4) 加 `certification`（hit 10.9%，evidence_quality C/NC 14.15× — 最強訊號）
#   5) 加 `timeline_marker`（hit 13.7%，verification_timeline 5.39× — 補時程任務空洞）
#   6) Per-keyword 清理：刪除 9 個 dead/反向訊號的字
#      - evidence_verbs: 部署 / 採行 / 完備（各 < 0.5% hit）
#      - certification:  碳中和宣告（0% hit, AI 幻覺）/ GRI / CDP / PCAF（N/A > Clear 反向）
#      - timeline_marker: 訂於 / 未來 X 年內（各 < 0.2% hit）
PATTERNS = {
    "future_verbs":     re.compile(r"將|會|擬|計畫|預計|規劃|朝|致力於"),
    "percentages":      re.compile(r"\d+\.?\d*\s*%"),
    "years":            re.compile(r"20[2-4]\d\s*年"),
    "amounts":          re.compile(r"\d+\s*(?:億|萬|千萬)\s*元"),
    "vague":            re.compile(r"持續|積極|努力|適當|相關"),
    "goals":            re.compile(r"目標|KPI|指標"),
    "past_markers":     re.compile(r"已|目前|現行|迄今|截至|本期|累計|過去|自\s*20\d\d"),
    "evidence_verbs":   re.compile(r"推行|落實|導入|執行|建置|完成|達成|設置|設立|簽署|訂定|制定|通過|取得"),
    "numeric_specific": re.compile(r"\d+\.\d+|[1-9]\d{2,}"),
    "certification":    re.compile(r"ISO\s*\d+|TCFD|SASB|RE100|SBTi|永續會計準則"),
    "timeline_marker":  re.compile(r"於\s*20[2-4]\d\s*年(?:前|底|起|內)?|至\s*20[2-4]\d|預計於"),
    # ── 第 7 輪：從 evidence_string 做 chi-square mining 挖出的 3 個 pattern ──
    # `strategic_vague`: Not Clear 強訊號（卓越 lift 24.9× / 主軸 5.0× / 中長期 4.4× /
    #                    碳策略 5.0×），補強現有 `vague` 抓不到的「戰略空話」
    # `long_term_goal`:  longer_than_5_years 強訊號（科學基礎 16.8× / 路徑 17.9× /
    #                    再生能源 4.8× / 2030/2050 等），補強 timeline_marker
    # `near_term_action`: within_2_years 唯一可救援字（n=27 樣本中 取得+認證 lift 4.4×）
    "strategic_vague":  re.compile(r"卓越|主軸|共創|中長期|碳策略|核心|藍圖|優化|期望"),
    "long_term_goal":   re.compile(r"2030|2050|2040|淨零|科學基礎|減量目標|減量路徑|脫碳|再生能源"),
    "near_term_action": re.compile(r"取得.{0,8}認證|簽署.{0,8}合約|供應商.{0,5}(?:配合|管理)|短期內"),
}

NUM_RAW = len(PATTERNS)         # 14
NUM_DERIVED = 5                 # future_ratio + concrete_density + 3 density
NUM_LENGTH = 1                  # length
NUM_FEATURES = NUM_RAW + NUM_DERIVED + NUM_LENGTH   # 20


def extract_features(sample: Union[dict, str]) -> List[float]:
    """Return a fixed-length float list (length = NUM_FEATURES = 20).

    Layout:
        [0:14]   raw counts (log1p'd)
        [14]     future_ratio [0,1]
        [15]     concrete_density (capped at 10)
        [16]     vague_density (capped at 10)        — 去長度污染後的空話訊號
        [17]     goals_density (capped at 10)
        [18]     evidence_verbs_density (capped at 10)
        [19]     log1p(length)

    Args:
        sample: dict (full sample with `data`) or str (legacy: just text)
    """
    if isinstance(sample, dict):
        text = sample.get("data", "")
    else:
        text = sample

    # 1. Raw counts
    counts = {name: len(p.findall(text)) for name, p in PATTERNS.items()}
    raw = [float(counts[name]) for name in PATTERNS.keys()]

    # 2. Derived features
    fw = counts["future_verbs"]
    pm = counts["past_markers"]
    future_ratio = fw / (fw + pm + 1.0)

    len_per_100 = max(len(text) / 100.0, 1.0)
    concrete = counts["percentages"] + counts["years"] + counts["amounts"]
    concrete_density = concrete / len_per_100
    vague_density = counts["vague"] / len_per_100
    goals_density = counts["goals"] / len_per_100
    evidence_verbs_density = counts["evidence_verbs"] / len_per_100

    # 3. Assemble
    vec = [math.log1p(x) for x in raw]
    vec.append(future_ratio)
    vec.append(min(concrete_density, 10.0))
    vec.append(min(vague_density, 10.0))
    vec.append(min(goals_density, 10.0))
    vec.append(min(evidence_verbs_density, 10.0))
    vec.append(math.log1p(len(text)))
    return vec
