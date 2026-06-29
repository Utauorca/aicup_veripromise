# AI CUP 2026 春季賽 — ESG 永續承諾驗證競賽

**Public Leaderboard**: **0.6243** ／ **Private Leaderboard**: **0.6359 (Rank 24)**

> 隊伍：TEAM_9910 ｜ 隊員：謝柏陞（隊長）

本 repo 為 AICup 2026 春季賽「ESG 永續承諾驗證競賽」之完整實作。任務為對中文 ESG 永續報告書句子做 4 個 task 的多任務分類：`promise_status`、`verification_timeline`、`evidence_status`、`evidence_quality`。

## 模型架構簡述

採用 **multi-task RoBERTa（A+MASK）+ 3-best multi-seed ensemble**：

- Backbone: `hfl/chinese-roberta-wwm-ext`（+ 5 epoch DAPT 領域 pretraining）
- **Cascade Task Heads（A）**：上游 task 的 softmax 機率（detach）顯式 concat 進下游 head 的 input
- **Hierarchical Loss Masking（MASK）**：promise=No 樣本不參與下游 task loss
- **Span Auxiliary Task** + Span-weighted pooling + Cross-Attention
- **20 維 ESG 領域 hand-crafted features**（14 regex pattern + 5 derived + length）
- **3-best multi-seed ensemble**：seed=1106910 / 910 / 9910（drop 經實證無增益的 seed=1106）

詳細技術說明見報告 PDF。

## 環境

| 項目 | 版本 |
|---|---|
| OS | Windows 11（推論）+ Google Colab T4/L4（訓練）|
| Python | 3.10 |
| PyTorch | 2.6.0 + CUDA 12.4 |
| transformers | latest |
| iterative-stratification | latest |

完整套件見 `requirements.txt`（請見下方）。

## 安裝

```bash
# 1. 克隆 repo
git clone https://github.com/<your-user>/aicup-2026-veripromise-esg.git
cd aicup-2026-veripromise-esg

# 2. 建虛擬環境（建議 Python 3.10）
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS/Linux

# 3. 安裝套件
pip install -r requirements.txt
```

## 資料準備

主辦競賽資料**未隨 repo 提供**（依比賽規範不公開）。請將以下三個檔放到 `aicup_veripromise/`：

```
aicup_veripromise/
├── vpesg4k_train_1000.json         (主辦提供)
├── vpesg4k_val_1000.json           (主辦提供)
└── vpesg4k_test_2000.json          (主辦提供)
```

## 模型權重下載

由於單一 ckpt 約 400 MB、4 seed × 5 fold = 共 20 個 ckpt 約 8 GB，未隨 repo 上傳。

請從以下 Google Drive 連結下載（資料夾內含 3 個 ckpt zip）：

- **Google Drive**：https://drive.google.com/drive/folders/1vQY_Gl3vN0GSj9fNgMv0CkHZrxc0IVts?usp=sharing
- 把 3 個 zip 直接放到 `aicup_veripromise/leadboard/` 資料夾，檔名應為：
  - `checkpoints_ema_AMASK_1106910.zip`
  - `checkpoints_ema_AMASK_910.zip`
  - `checkpoints_ema_AMASK_9910.zip`

## 流程：從零訓練到提交

### Step 1 — DAPT（Domain-Adaptive Pre-training）

訓練主 notebook 內含 DAPT 步驟，5 epoch MLM continued pretraining。

### Step 2 — 主訓練（5-fold × 4 seed）

```
aicup_veripromise/VeriPromiseESG_2026_Refactored.ipynb
```

在 Colab 上跑，每個 seed 約 3.5 小時：
1. 改 `src/config.py` 的 `seed: int = ?`
2. 跑完整 notebook → 產生 5 個 fold 的 EMA-only checkpoint
3. 重複以上換 seed = 1106910 / 910 / 9910

### Step 3 — 推論與提交

```
aicup_veripromise/leadboard/predict_submission.ipynb
```

本地端跑（RTX 3060 約 5 分鐘）：
1. 確認 `leadboard/` 內有 3 個 ckpt zip
2. 跑完整 notebook
3. 提交產出的 `submission_3best_no1106.csv`

## 程式碼結構

```
aicup_veripromise/
├── src/
│   ├── config.py          # 全部超參數與 flag
│   ├── data.py            # 資料載入、K-fold split、augmentation
│   ├── model.py           # MultiTaskRoberta + cascade head
│   ├── train.py           # 訓練 loop（FGM + EMA + AMP + hierarchical loss mask）
│   ├── features.py        # 20 維手工特徵抽取
│   ├── augment.py         # span-preserving augmenter
│   ├── ema.py             # ExponentialMovingAverage
│   └── fgm.py             # FGM adversarial training
├── VeriPromiseESG_2026_Refactored.ipynb   # 主訓練 notebook
├── leadboard/
│   └── predict_submission.ipynb           # 推論 + CSV 提交產出
├── requirements.txt
├── README.md              # 本文件
└── .gitignore
```

## 重要超參數

關鍵設定列於 `src/config.py`：

| 旗標 | 預設值 | 說明 |
|---|---|---|
| `use_cascade_heads` | True | Cascade Task Heads（A） |
| `use_hierarchical_loss_mask` | True | Hierarchical Loss Masking（MASK） |
| `use_span_aux` | True | Span Auxiliary Task |
| `use_span_pooling` | True | Span-weighted pooling |
| `use_cross_attention` | True | Promise/Evidence Cross-Attention |
| `use_hand_features` | True | 20 維 ESG 手工特徵 |
| `use_esg_prefix` | True | ESG 文字前綴注入 |
| `loss_type` | "ce" | weighted cross-entropy（focal 實測退步已關） |
| `use_rdrop` | False | R-Drop（實測無增益已關） |
| `use_weighted_sampler` | False | Class-balanced sampler（實測退步已關） |
| `seed` | 1106910 / 1106 / 910 / 9910 | 4 個訓練 seed |

## 已實證無效的方法（保留 flag 給 ablation）

| 方法 | 結果 |
|---|---|
| R-Drop | OOF +0.05pp, LB -0.02pp 無感 |
| Focal Loss (γ=1.5) | 前 2 fold 比 CE 差 |
| Ordinal regression for VT | 退步 |
| Per-task Best Subset Selection | LB -0.28pp |
| TTA (mask=0.05) | mean \|Δprob\| < 0.005 |
| Distribution Calibration (CORN) | OOF -0.68pp |
| cRT Stage 2 (EQ head only) | OOF -0.33pp |
| LGBM stacking on OOF | LB -0.98pp |

## 隊伍資訊

- **隊伍**：TEAM_9910
- **隊員**：謝柏陞（隊長）
- **指導教授**：（如有請補）
- **Public Leaderboard**：0.6243（3-best ensemble）
- **Private Leaderboard**：**0.6359 / Rank 24**（比賽最終評分）

## AI 輔助工具聲明

報告撰寫與部分技術文獻整理過程使用了 Anthropic Claude（Claude 4.x）進行討論、debug 與文獻搜尋；模型訓練腳本、推論程式碼、實驗設計、創新方法的構思與實作皆由隊伍完成。

## 參考文獻

詳見報告 PDF「捌、使用的外部資源與參考文獻」段。
