"""Generate the refactored notebook from source cell definitions.

Run once with `python _build_notebook.py` to produce the .ipynb.
Keeping cells as Python strings here is far more maintainable than
hand-editing huge JSON.
"""
import json
from pathlib import Path

OUT = Path("VeriPromiseESG_2026_Refactored.ipynb")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CELLS = [
    md("""# VeriPromiseESG 2026 — Refactored Baseline
## RoBERTa + Multi-Task + **StratifiedKFold / EMA / Soft-Voting Ensemble**

> 本版本相對於原始 baseline 的三個主要改進：
>
> 1. **資料切分** — 先切出 15% 的真正 held-out 測試集（訓練全程不碰），剩下 85% 再用
>    `StratifiedKFold(n_splits=5)` 依 `evidence_quality`（權重最高 0.35）做分層切分。
>    原 baseline 的 val set 同時被當作 early-stop 依據與 submission → 有資料洩漏疑慮。
> 2. **EMA (Exponential Moving Average)** — 對小資料集（~1000 筆）尤其有效，
>    shadow weights 會平滑訓練中 per-step 的噪音，通常比最後一個 epoch 的權重泛化更好。
> 3. **Soft-Voting Ensemble** — 取代原本的 majority voting。每個 fold 的 checkpoint
>    輸出 softmax 機率後做平均，再 argmax，保留每個 fold 的 confidence 資訊。
>
> 其餘架構（MultiTaskRoberta、LLRD、FGM、Weighted CE、Contextual Augmentation）
> 與 baseline 相同，差別只是把它們拆進 `src/` 各自獨立的模組，便於維護與 ablation。
"""),

    md("""## Step 1 — 安裝套件 & 下載資料

`transformers`, `torch`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `nlpaug`, `tqdm`
皆為必需。Colab 環境通常已預裝大部分。
"""),

    code("""# !pip install -q transformers scikit-learn pandas matplotlib seaborn nlpaug tqdm

import urllib.request

DATA_URL = "https://raw.githubusercontent.com/veripromiseesg/veripromiseesgdataset/ac91c1c8b5d116edf6fc44cccc1ee3b618f5a207/vpesg4ktrain1000v1.json"
urllib.request.urlretrieve(DATA_URL, "vpesg4k_train_1000.json")
print("✅ 資料下載完成")
"""),

    md("""## Step 2 — 載入模組化的 `src/` 套件

所有超參數、模型、訓練迴圈、ensemble 邏輯都已拆到 `src/` 下：

| 模組 | 職責 |
|------|------|
| `src/config.py`    | 全域超參數（`CFG`）與任務欄位定義 |
| `src/seed.py`      | 統一 random/numpy/torch seeding |
| `src/data.py`      | 資料切分（held-out + KFold）與 `ESGDataset` |
| `src/augment.py`   | ContextualWordEmbsAug + ESG 術語保護 |
| `src/model.py`     | `MultiTaskRoberta` + LLRD optimizer |
| `src/fgm.py`       | FGM 對抗訓練 |
| `src/ema.py`       | **新增** — 模型權重 EMA |
| `src/train.py`     | `train_one_epoch` / `predict_logits` / `evaluate_hybrid` |
| `src/ensemble.py`  | **新增** — Soft-voting + 邏輯約束 |

改超參只需改 `src/config.py::CFG`，不用翻整個 notebook。
"""),

    code("""import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.config import CFG, EVAL_FIELDS, FIELD_WEIGHTS, LABEL2ID, ID2LABEL, NUM_LABELS
from src.seed import set_seed
from src.data import (
    load_raw, holdout_test_split, kfold_indices, ESGDataset, collate_fn,
    compute_sample_weights,
)
from src.augment import ContextualAugmenter
from src.model import MultiTaskRoberta, get_llrd_optimizer
from src.ema import ModelEMA
from src.train import (
    compute_class_weights, train_one_epoch,
    predict_logits, logits_to_preds, evaluate_hybrid,
)
from src.ensemble import soft_vote, probs_to_preds, apply_logic_constraints, compute_log_priors

set_seed(CFG.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(CFG.ckpt_dir, exist_ok=True)

print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Model: {CFG.model_name} | MAX_LEN={CFG.max_len} | BATCH={CFG.batch_size} | EPOCHS={CFG.epochs}")
print(f"Speed: AMP={CFG.amp} ({CFG.amp_dtype}) | num_workers={CFG.num_workers} | pin_memory={CFG.pin_memory}")
print(f"Splits: {int((1-CFG.test_size)*100)}% for {CFG.n_splits}-fold CV, {int(CFG.test_size*100)}% held-out test")
"""),

    md("""## Step 3 — 資料切分（改進點 1）

### 為什麼要有真正的 held-out test？

原 baseline：
```
all_data ─┬─ train (80%)  ──▶ fit
          └─ val   (20%)  ──▶ early-stop + submission
```
`val` 同時被模型「看過」（用來選最佳 epoch）又被拿去提交 → 分數樂觀偏差。

本版：
```
all_data ─┬─ trainval (85%) ──▶ StratifiedKFold × 5 ──▶ 5 個 best.pt
          └─ test     (15%) ──▶ 訓練全程完全不碰，只在最後做一次 ensemble 評估
```

### 為什麼 stratify 在 `evidence_quality`？

加權總分中 `evidence_quality=0.35` 權重最高；若隨機切分，某個小類別（例如 `Misleading`）
可能整批掉到 test set，fold 也可能看不到，Macro F1 會劇烈變動。分層切分確保
每個 fold 與 held-out test 的類別分布都接近全集。
"""),

    code("""all_data = load_raw(CFG.data_path)
trainval, test_data = holdout_test_split(
    all_data, test_size=CFG.test_size,
    stratify_field=CFG.stratify_field, seed=CFG.seed,
)
print(f"全集 {len(all_data)} 筆 → trainval {len(trainval)} / held-out test {len(test_data)}")

# 簡易 EDA：標籤分佈
train_df = pd.DataFrame(trainval)
for f in EVAL_FIELDS:
    print(f"\\n{f}:\\n{train_df[f].value_counts().to_string()}")
"""),

    md("""## Step 4 — Tokenizer 與資料增強器

Augmenter 只在「訓練時」以機率 `CFG.aug_prob=0.10` 被呼叫；驗證與測試集不做任何
augmentation，避免評估噪音。ESG 專業術語（碳中和、永續、董事會…）會以佔位符保護，
防止 contextual 模型把關鍵字替換成不相干的詞。
"""),

    code("""# use_fast=True 才能拿到 offset_mapping，span aux 任務需要它對齊字元位置
tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, use_fast=True)
print(f"tokenizer is_fast = {tokenizer.is_fast}")

# 初始化 augmenter（耗時約 10-30s，只做一次）
try:
    augmenter = ContextualAugmenter(model_name=CFG.model_name, aug_p=CFG.aug_prob)
    print("✅ Augmenter ready")
except Exception as e:
    print(f"⚠️  Augmenter init failed ({e}); 將以 None 執行，訓練照常進行")
    augmenter = None
"""),

    md("""## Step 5 — K-Fold 訓練主迴圈（改進點 2：EMA + Early Stopping + AMP）

### 加速三件套（T4 GPU）

| 旋鈕 | 目前值 | 效果 |
|------|--------|------|
| `CFG.amp = True` | fp16 | autocast + GradScaler，T4 的 Tensor Core 吃 fp16 → **~1.8× 加速** |
| `CFG.batch_size = 8` | 原 4 | AMP 釋放記憶體後可以推大 batch，GPU 利用率更高 |
| `CFG.max_len = 384` | 原 512 | attention 是 O(L²)，384 相對 512 → **~1.8× 加速** |
| `CFG.num_workers = 2` | 原 0 | tokenize 丟到副 thread，GPU 不等 CPU |
| `CFG.n_splits = 3` | 原 5 | 快速迭代用；定稿前記得改回 5 |

**預期**：原本 5 fold × 20 epoch ≈ 90 分鐘 → 現在 3 fold × 15 epoch with AMP ≈ 15–20 分鐘。

### 長尾類別處理：Focal Loss + Weighted Sampler

`evidence_quality` 的 `Misleading` / `Not Clear` 類別在 1000 筆裡只有個位數比例，
標準 CE 會被多數類別淹沒。兩個改動疊加對付：

1. **Focal Loss**（`CFG.loss_type="focal"`, `γ=2.0`）
   - `(1 − p_t)^γ` 讓已學會的樣本 loss 幾乎歸零，梯度專注在難樣本
   - 與 class_weights 疊加 (`α_t`)，稀有類別再多一層加權
2. **WeightedRandomSampler**（`CFG.use_weighted_sampler=True`）
   - 每個樣本的取樣機率 ∝ 其「最稀有欄位類別」的 inverse frequency（capped 5×）
   - 每個 epoch 看到更多稀有樣本（帶 replacement 抽樣）
   - 與 Focal Loss 正交 — sampler 在 data 層面，Focal 在 loss 層面

### 輔助 span 任務：用 `promise_string` / `evidence_string`

原資料集每筆都有 `promise_string` 和 `evidence_string`，**是 `data` 的子字串**，
標示哪一段是承諾、哪一段是佐證。baseline 完全沒用到這個監督訊號。本版：

- 用 fast tokenizer 的 `offset_mapping`，對齊 char span 到 token span
- 產生 BIO 標籤：`O / B-PROMISE / I-PROMISE / B-EVIDENCE / I-EVIDENCE`
- 模型加一個 token classification head，輸出 `[B, L, 5]`
- Total loss = task_loss + `λ · span_loss`（λ=`CFG.span_loss_weight=0.3`）

效果：強迫 backbone 先「看懂」哪段是承諾/佐證 → 4 個 sentence-level 任務判斷更準，
特別對 `evidence_status` 和 `evidence_quality` 有直接助益（這兩欄位本質就是在問
「有沒有佐證」、「佐證品質」）。

推論時 span_head 的輸出會被丟棄，只取 4 個主任務分類。


### EMA 怎麼運作？

```
w_shadow ← decay · w_shadow + (1 − decay) · w_current
```

`decay=0.999` 時，shadow 權重相當於近 1/(1-0.999) = 1000 個 step 的指數加權平均。
訓練完之後，用 shadow 權重做預測通常比用最後一個 step 的權重更穩定（對小資料集尤其明顯）。

### Early Stopping

相對於硬拉一個 epoch 數字，我們用 **高上限 + 自動停止**：

- `CFG.epochs = 15` 作為上限
- `CFG.patience = 3`：val 分數連續 3 個 epoch 沒進步就停下這個 fold

好處：
- 好訓的 fold 可能 epoch 10 就收斂停下，壞訓的 fold 不會硬跑到 20
- 每個 fold 自適應，不同隨機性下都能跑到自己的甜蜜點
- 整體訓練時間比硬跑固定 epoch 更可控

設 `CFG.patience = 0` 可以關閉 early stopping 回到原本行為。

### 每個 fold 會儲存什麼？

`checkpoints/fold_{k}/best.pt` 裡面存兩份權重：
```python
{"model": model.state_dict(), "ema": ema.shadow}
```
Ensemble 階段會用 `"ema"` 那份做 soft voting（你也可以切換成 `"model"` 比較差異）。
"""),

    code("""def build_model_and_optim(num_train_steps):
    model = MultiTaskRoberta(NUM_LABELS).to(device)
    optimizer = get_llrd_optimizer(
        model, lr=CFG.lr, weight_decay=CFG.weight_decay, lr_decay=CFG.llrd_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(CFG.warmup_ratio * num_train_steps),
        num_training_steps=num_train_steps,
    )
    return model, optimizer, scheduler


fold_results = []

for fold, (tr_idx, va_idx) in enumerate(kfold_indices(
    trainval, n_splits=CFG.n_splits,
    stratify_field=CFG.stratify_field, seed=CFG.seed,
)):
    print(f"\\n{'='*60}\\nFold {fold+1}/{CFG.n_splits}\\n{'='*60}")
    set_seed(CFG.seed + fold)  # 讓每個 fold 內部的隨機性可重現但不同

    tr = [trainval[i] for i in tr_idx]
    va = [trainval[i] for i in va_idx]

    train_ds = ESGDataset(tr, tokenizer, LABEL2ID,
                          mask_prob=CFG.mask_prob, aug_prob=CFG.aug_prob,
                          augmenter=augmenter)
    val_ds   = ESGDataset(va, tokenizer, LABEL2ID, mask_prob=0.0, aug_prob=0.0)
    if CFG.use_weighted_sampler:
        sample_w = compute_sample_weights(tr, cap=CFG.sampler_cap)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_w, num_samples=len(tr), replacement=True,
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=CFG.batch_size, sampler=sampler,
            collate_fn=collate_fn,
            num_workers=CFG.num_workers, pin_memory=CFG.pin_memory,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=CFG.batch_size, shuffle=True, collate_fn=collate_fn,
            num_workers=CFG.num_workers, pin_memory=CFG.pin_memory,
        )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=CFG.batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=CFG.num_workers, pin_memory=CFG.pin_memory,
    )

    total_steps = len(train_loader) * CFG.epochs
    model, optimizer, scheduler = build_model_and_optim(total_steps)
    ema = ModelEMA(model, decay=CFG.ema_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=CFG.amp and device.type == "cuda")
    class_weights = compute_class_weights(pd.DataFrame(tr))

    best_score = 0.0
    no_improve = 0
    best_epoch = 0
    fold_dir = os.path.join(CFG.ckpt_dir, f"fold_{fold+1}")
    os.makedirs(fold_dir, exist_ok=True)
    ckpt_path = os.path.join(fold_dir, "best.pt")

    for epoch in range(CFG.epochs):
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            class_weights, ema=ema, fgm_epsilon=CFG.fgm_epsilon, scaler=scaler,
        )

        # 用 EMA 權重評估
        ema.apply_shadow(model)
        logits = predict_logits(model, val_loader, device)
        ema.restore(model)

        val_preds = logits_to_preds(logits)
        res = evaluate_hybrid(va, val_preds)
        score = res["final_weighted_score"]

        improved = score > best_score
        print(f"  Epoch {epoch+1}/{CFG.epochs} | loss={avg_loss:.4f} | val_weighted_F1={score:.4f}"
              + ("  ⭐" if improved else f"  (no_improve={no_improve+1}/{CFG.patience})"))

        if improved:
            best_score = score
            best_epoch = epoch + 1
            no_improve = 0
            torch.save({"model": model.state_dict(), "ema": ema.shadow}, ckpt_path)
        else:
            no_improve += 1
            if CFG.patience > 0 and no_improve >= CFG.patience:
                print(f"  ⏹️  Early stop at epoch {epoch+1} (best was epoch {best_epoch})")
                break

    fold_results.append({"fold": fold + 1, "best_score": best_score,
                         "best_epoch": best_epoch, "ckpt": ckpt_path})
    print(f"  Fold {fold+1} best = {best_score:.4f} @ epoch {best_epoch}")

print("\\n=== K-Fold 完成 ===")
for r in fold_results:
    print(f"Fold {r['fold']}: {r['best_score']:.4f}  @ epoch {r['best_epoch']}  ({r['ckpt']})")
print(f"平均 CV 分數: {sum(r['best_score'] for r in fold_results)/len(fold_results):.4f}")
"""),

    md("""## Step 6 — Soft-Voting Ensemble（改進點 3）

### Hard vs Soft Voting

**Hard (原 baseline)**：
```python
# fold_1 預測 "Yes", fold_2 預測 "Yes", fold_3 預測 "No"
# → 多數決 = "Yes"
```
當每個 fold 都「剛好猜對」時沒差，但丟掉信心度資訊。若 fold_1 以 51% 猜 "Yes"、
fold_2 以 95% 猜 "No"，hard voting 依然是 "Yes"，明顯不合理。

**Soft (本版)**：
```python
# fold_1 softmax = [Yes: 0.51, No: 0.49]
# fold_2 softmax = [Yes: 0.05, No: 0.95]
# fold_3 softmax = [Yes: 0.55, No: 0.45]
# 平均       = [Yes: 0.37, No: 0.63]  → 最終預測 "No"
```
通常能帶來 0.5–1.5 個百分點的穩定提升。

### 邏輯約束後處理

比賽規則上，若 `promise_status = "No"`，其他三個欄位必為 `"N/A"`。模型不一定學得乾淨，
在推論端加一行後處理可把不合法組合強制修正。
"""),

    code("""from src.ensemble import discover_fold_ckpts

# 測試集 loader
test_ds = ESGDataset(test_data, tokenizer, LABEL2ID, mask_prob=0.0, aug_prob=0.0)
test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=CFG.batch_size, shuffle=False, collate_fn=collate_fn,
    num_workers=CFG.num_workers, pin_memory=CFG.pin_memory,
)

# 以空白模型作為載入容器
inference_model = MultiTaskRoberta(NUM_LABELS).to(device)
ckpts = [r["ckpt"] for r in fold_results]

# 按 fold 的 val 分數計算 ensemble weights（softmax with temperature beta）
import math
scores = [r["best_score"] for r in fold_results]
if CFG.ensemble_weight_beta > 0:
    mx = max(scores)
    exp_w = [math.exp((s - mx) * CFG.ensemble_weight_beta) for s in scores]
    s_total = sum(exp_w)
    fold_weights = [w / s_total for w in exp_w]
    print(f"使用 weighted ensemble（beta={CFG.ensemble_weight_beta}）")
else:
    fold_weights = None
    print(f"使用 uniform ensemble")
print(f"將 ensemble {len(ckpts)} 個 fold 的 EMA 權重")

probs = soft_vote(inference_model, ckpts, test_loader, device,
                  use_ema_key=True, fold_weights=fold_weights)

# Prior correction：以訓練集的 class prior 修正模型對主類別的 bias
log_priors = compute_log_priors(trainval)
print(f"Prior correction alpha = {CFG.prior_correction_alpha}")
for f, lp in log_priors.items():
    rounded = [round(float(x), 2) for x in lp]
    print(f"  {f:<22} log_prior = {rounded}")

test_preds = probs_to_preds(probs, log_priors=log_priors,
                            alpha=CFG.prior_correction_alpha)
# 硬規則 + 信心度規則
test_preds = apply_logic_constraints(test_preds, probs=probs,
                                     conf_threshold=CFG.post_process_conf_threshold)

test_results = evaluate_hybrid(test_data, test_preds)
print(f"\\n🎯 Held-out Test Weighted F1 = {test_results['final_weighted_score']:.5f}")
for f in EVAL_FIELDS:
    r = test_results[f]
    print(f"  {f:<22} macro={r['macro_f1']:.4f}  micro={r['micro_f1']:.4f}  (w={r['weight']})")
"""),

    md("""## Step 6.5 — Confusion Matrix Diagnostic

看看每個欄位**具體錯在哪裡**。Row = 真實 label，Column = 模型預測。對角線是答對的，
off-diagonal 是錯誤，可以據此設計有針對性的後處理規則或下一輪改進方向。
"""),

    code("""from sklearn.metrics import confusion_matrix
import numpy as np

for field in EVAL_FIELDS:
    labels = EVAL_FIELDS[field]
    y_true = [d[field] for d in test_data]
    y_pred = [p[field] for p in test_preds]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_pct = (cm / row_sums * 100).round(1)

    print(f"\\n{'='*60}")
    print(f"  {field}  (weight={FIELD_WEIGHTS[field]})")
    print(f"{'='*60}")
    # 表頭
    header = " " * 22 + "".join([f"→ {l[:15]:<18}" for l in labels]) + "| total"
    print(header)
    for i, true_lab in enumerate(labels):
        row = f"  true={true_lab[:16]:<18} "
        for j in range(len(labels)):
            tag = "●" if i == j else " "
            row += f"{tag} {cm[i,j]:>3} ({cm_pct[i,j]:>4.1f}%)   "
        row += f"| {cm.sum(axis=1)[i]:>4}"
        print(row)
"""),

    md("""## Step 7 — 輸出 submission

把原始欄位與預測欄位合併後，寫入 `prediction.json`。
"""),

    code("""output = []
for orig, pred in zip(test_data, test_preds):
    item = dict(orig)
    item.update(pred)
    output.append(item)

with open(CFG.output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"✅ 已寫入 {CFG.output_path}  ({len(output)} 筆)")
"""),

    md("""## 改進點 ablation 建議

想量化每個改進的貢獻，可在 `src/config.py` 調整下列旋鈕再重跑，對照 held-out 分數：

| 實驗 | 設定 | 預期效果 |
|------|------|---------|
| baseline | `n_splits=1`（改回 train_test_split）、`ema_decay=0` | 接近原 notebook |
| +KFold only | `n_splits=5`、`ema_decay=0`、hard voting | CV 穩定性提升 |
| +EMA only | `n_splits=1`、`ema_decay=0.999` | 單 fold 內泛化提升 |
| +Soft voting | 全開 | 本版完整設定 |

另外可嘗試的方向（皆與本重構正交）：
- 升級 backbone → `hfl/chinese-roberta-wwm-ext-large` 或 `hfl/chinese-lert-large`
- FGM → PGD / AWP
- R-Drop / multi-sample dropout 取代單一 dropout
- Sliding window 對 > 512 token 的長文作 chunk 平均
"""),
]


def main():
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {OUT} with {len(CELLS)} cells")


if __name__ == "__main__":
    main()
