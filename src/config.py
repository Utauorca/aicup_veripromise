"""Central config. All hyperparameters live here; edit once, effect everywhere."""
from dataclasses import dataclass, field
from typing import Dict, List


EVAL_FIELDS: Dict[str, List[str]] = {
    "promise_status": ["Yes", "No"],
    "verification_timeline": [
        "already", "within_2_years", "between_2_and_5_years",
        "longer_than_5_years", "N/A",
    ],
    "evidence_status": ["Yes", "No", "N/A"],
    "evidence_quality": ["Clear", "Not Clear", "Misleading", "N/A"],
}

FIELD_WEIGHTS: Dict[str, float] = {
    "promise_status": 0.2,
    "verification_timeline": 0.15,
    "evidence_status": 0.3,
    "evidence_quality": 0.35,
}

LABEL2ID = {f: {lab: i for i, lab in enumerate(labs)} for f, labs in EVAL_FIELDS.items()}
ID2LABEL = {f: {i: lab for i, lab in enumerate(labs)} for f, labs in EVAL_FIELDS.items()}
NUM_LABELS = {f: len(labs) for f, labs in EVAL_FIELDS.items()}


@dataclass
class Config:
    # Model
    model_name: str = "hfl/chinese-roberta-wwm-ext"
    max_len: int = 512

    # Training
    batch_size: int = 8           # base 在 T4 (15GB) 可承受
    epochs: int = 40              # 原 30；3-fold 都跑滿 30 還在爬，再給 10 個 epoch
    patience: int = 5             # 連續 N 個 epoch val 沒進步就停；設 0 則關閉 early stopping
    lr: float = 3e-5              # 原 2e-5；batch 4→8 時線性 scaling 拉高 ~1.5×
    weight_decay: float = 0.1
    llrd_decay: float = 0.9
    warmup_ratio: float = 0.1
    fgm_epsilon: float = 1.0
    ema_decay: float = 0.999
    grad_clip: float = 1.0

    # Loss & sampling — 針對 evidence_quality 長尾分布
    # 實測 focal + weighted sampler 組合反而使分數下降，因此預設關閉。
    # 程式碼保留在 src/train.py 與 src/data.py，改 flag 即可重新啟用做 ablation。
    # 實測 2026-06-14：focal_gamma=1.5 前 2 fold 都比 CE 差 → 關閉
    # 推測原因：你的 EQ 長尾不是「簡單樣本主導 gradient」造成的，是「資料根本不足」
    #   Misleading n=1 不管 loss 怎麼 weight 都學不起來
    loss_type: str = "ce"              # "ce" = weighted cross-entropy, "focal" = Focal Loss
    focal_gamma: float = 1.5           # 只在 loss_type=="focal" 時生效（1.5 = 溫和版，2.0 = 論文標準）
    # Hierarchical loss masking — 利用 outcome 嵌套關係：
    #   promise=No 樣本 → timeline / evidence_status / evidence_quality 都是 forced N/A，
    #     這些樣本對 3 個下游 task 沒訓練意義，loss 應該跳過。
    #   evidence=No/N/A 樣本 → evidence_quality forced N/A，同理。
    # 開啟後 task head 只在「真有資訊」的樣本子集上學梯度，預期 evidence_quality 大幅改善。
    use_hierarchical_loss_mask: bool = True
    # WeightedRandomSampler — 開啟後依 sampler_mode 走不同邏輯
    # （舊版 inverse_freq 兩輪都實測「傷分數」，因為把 Misleading n=1 過度放大）
    # promise_balanced 已實作但目前先關閉（過擬合風險高、訓練時間長），
    # 只留 Hierarchical Loss Masking + Cascading Task Heads 兩條改善。
    # 想啟用 promise_balanced：use_weighted_sampler=True 即可。
    use_weighted_sampler: bool = False
    sampler_mode: str = "promise_balanced"   # "inverse_freq" 或 "promise_balanced"
    sampler_cap: float = 5.0                 # 只在 sampler_mode="inverse_freq" 生效
    sampler_alpha: float = 0.5               # 只在 sampler_mode="inverse_freq" 生效
    # promise_balanced 模式專用：promise=No 樣本的 oversample 倍率
    # 3.0 大致讓每個 batch 有 ~40% promise=No（vs 原本的 ~19%）
    promise_no_oversample_ratio: float = 3.0
    label_smoothing: float = 0.0       # 實測 0.1 會讓整體分數下降，關閉（0.0 = 標準 CE）
    # Cap on per-class inverse-frequency weight — 防止極稀有類別（如 Misleading n=1
    # 在 trainval 中 → naive weight ~200×）造成梯度爆衝。10.0 是經驗安全值。
    class_weight_cap: float = 10.0

    # R-Drop — 同一筆 input forward 兩次，dropout mask 不同 → KL 一致性約束
    # 實測 2026-06-14：OOF +0.05pp、leaderboard −0.02pp，效果 = noise → 關閉
    # 程式碼保留在 src/train.py，改 use_rdrop=True 即可重新啟用做 ablation
    use_rdrop: bool = False
    rdrop_alpha: float = 1.0

    # Ensemble weighting — soft voting 時按 fold 的 val 分數加權
    # beta=0  → 均勻平均（等同原本行為）
    # beta=5  → 適度放大好 fold 的貢獻
    # beta→∞ → 等於只用最好 fold（winner-take-all）
    ensemble_weight_beta: float = 5.0

    # Inference-time prior correction — 推論時減去 alpha · log_prior 修正類別不均衡 bias
    # 實測在本任務（multilabel stratified split）下 alpha>0 反而傷分數，因稀有類別
    # log_prior 太極端（within_2_years=-4.35, Misleading=-6.75）會被過度推升 → 關閉
    prior_correction_alpha: float = 0.0

    # Confidence-based post-processing — 實測沒帶來增益，關閉（threshold=0 永不觸發）
    post_process_conf_threshold: float = 0.0

    # Auxiliary span task — 用 promise_string / evidence_string 當 BIO token labels
    use_span_aux: bool = True
    span_loss_weight: float = 0.3     # 試過 0.5，evidence_quality macro 下降 → 退回 0.3
    num_span_labels: int = 5          # O, B-PROMISE, I-PROMISE, B-EVIDENCE, I-EVIDENCE

    # Predicted-span pooling — 用 span_head 的 softmax 當 attention，產生專屬的
    # promise_vec 和 evidence_vec 給 task heads。test time 不需要 evidence_string。
    use_span_pooling: bool = True
    span_pool_dim: int = 128          # 各投影到 128 維（兩個 → 加 256 維給 task head）

    # Cross-attention between promise_vec & evidence_vec — 讓兩個向量互相豐富，
    # 模擬「evidence 是否支持 promise」的判斷流程
    use_cross_attention: bool = True
    cross_attn_dim: int = 256         # 768 → 256 → 768 bottleneck，控制參數量
    cross_attn_heads: int = 4         # multi-head attention heads

    # Task cross-attention — 讓 4 個 task 預測「同時進行」、互相 attention
    # 實測沒帶來增益（與 cross_attention 部分功能重疊），關閉
    use_task_xattn: bool = False
    task_xattn_dim: int = 128
    task_xattn_heads: int = 4

    # Cascading task heads — 用「上游 task 的預測機率」當下游 task head 的額外輸入。
    # 直接利用 outcome 嵌套：
    #   promise_status     → 看 [combined]
    #   verification_timeline → 看 [combined, promise_probs]
    #   evidence_status    → 看 [combined, promise_probs]
    #   evidence_quality   → 看 [combined, promise_probs, evidence_probs]
    # 上游 softmax 在 detach 後送入下游，每個 head 仍獨立訓練（梯度不交叉）。
    # 與 use_task_xattn 互斥（兩者都修改 head input）。
    use_cascade_heads: bool = True

    # CORN-style Ordinal Regression for verification_timeline
    # 文獻指引預期 +0.5~1.5pp，但 2026-06-16 實測效果很差 → 撤回
    # 推測原因：timeline 的「次序失準成本」可能跟模型認知不一致（e.g. already 跟 N/A 在實際語意上更接近）
    # 程式碼保留在 src/model.py (OrdinalTimelineHead) + src/train.py (_ordinal_timeline_per_sample_loss)
    # 改 use_ordinal_timeline=True 即可重新啟用做 ablation
    use_ordinal_timeline: bool = False

    # Feature injection — esg_type (E/S/G) 當文字前綴
    use_esg_prefix: bool = True

    # Hand-crafted domain features — 14 raw counts + 5 derived + length
    use_hand_features: bool = True
    num_hand_features: int = 20         # 與 src/features.py::NUM_FEATURES 對應
    feature_proj_dim: int = 64          # pooled(768) + feature_emb(64) = 832 進 task heads

    # Speed knobs
    amp: bool = True              # 混合精度 (fp16 on T4)，約 1.5–2× 加速
    amp_dtype: str = "fp16"       # T4: fp16 / A100,L4: bf16
    num_workers: int = 2          # DataLoader 多 thread tokenize
    pin_memory: bool = True

    # Data split: held-out test + K-fold on the rest
    test_size: float = 0.15          # true held-out, never touched during training
    n_splits: int = 5                # 5-fold CV，每 fold train 680 筆，分數比 3-fold 穩
    stratify_field: str = "evidence_quality"  # highest weight → stratify on this
    seed: int = 182109
    # Augmentation
    # Span-preserving augmenter 啟用後，aug_prob 可安全拉高 — span_aux 的監督不會丟失
    aug_prob: float = 0.30        # 原 0.10；太低等於沒增強
    mask_prob: float = 0.15

    # Paths
    data_path: str = "vpesg4k_train_1000.json"
    val_path: str = "vpesg4k_val_1000.json"   # 外部獨立 held-out，取代從 train 切 15%
    ckpt_dir: str = "checkpoints"
    output_path: str = "prediction.json"


CFG = Config()
