"""Soft-voting ensemble across K-fold checkpoints.

Baseline used majority voting over hard labels — throws away confidence.
Soft voting averages logits (after softmax) then argmax, which keeps each
fold's certainty and empirically gives a small but reliable lift.
"""
import glob
import math
import os
from collections import Counter
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .config import CFG, EVAL_FIELDS, ID2LABEL, LABEL2ID, NUM_LABELS
from .train import predict_logits


def compute_log_priors(train_data: List[dict], eps: float = 1e-6) -> Dict[str, torch.Tensor]:
    """Compute log P(class) per field from training data, for inference-time bias correction.

    Returns dict[field] → tensor of shape [num_classes]，順序對齊 EVAL_FIELDS[field]。
    """
    log_priors = {}
    for field, labels in EVAL_FIELDS.items():
        counts = Counter(d[field] for d in train_data)
        total = sum(counts.values())
        priors = [(counts.get(lab, 0) + eps) / total for lab in labels]
        log_priors[field] = torch.tensor([math.log(p) for p in priors], dtype=torch.float)
    return log_priors


def soft_vote(model, ckpt_paths: List[str], loader, device,
              use_ema_key: bool = False,
              fold_weights: List[float] = None) -> Dict[str, torch.Tensor]:
    """Load each checkpoint into `model`, predict logits, weighted-average softmax probs.

    Args:
        fold_weights: optional list of per-fold weights, same length as ckpt_paths.
            If None → uniform averaging (equal weight). If provided, will be
            normalised so they sum to 1 before accumulation.

    Returns per-field probability tensors [N, num_labels].
    """
    assert ckpt_paths, "No checkpoints given"
    n_ckpt = len(ckpt_paths)
    if fold_weights is None:
        fold_weights = [1.0 / n_ckpt] * n_ckpt
    else:
        assert len(fold_weights) == n_ckpt, \
            f"fold_weights size {len(fold_weights)} != ckpts {n_ckpt}"
        total = sum(fold_weights)
        fold_weights = [w / total for w in fold_weights]
        print(f"Ensemble weights: {[round(w,3) for w in fold_weights]}")

    accum = {f: torch.zeros(0) for f in EVAL_FIELDS}
    first = True
    for path, w in zip(ckpt_paths, fold_weights):
        state = torch.load(path, map_location=device, weights_only=False)
        if use_ema_key and isinstance(state, dict) and "ema" in state:
            model_state = state["ema"]
        elif isinstance(state, dict) and "model" in state:
            model_state = state["model"]
        else:
            model_state = state

        incompat = model.load_state_dict(model_state, strict=False)
        # Drop known-harmless missing keys (non-float buffers that the model
        # re-initialises in its constructor).
        missing = [k for k in incompat.missing_keys
                   if not any(tag in k for tag in ("position_ids", "token_type_ids"))]
        unexpected = list(incompat.unexpected_keys)
        if missing or unexpected:
            raise RuntimeError(
                f"Architecture / checkpoint mismatch when loading {path}:\n"
                f"  missing keys (model expects but ckpt has not): {missing[:8]}\n"
                f"  unexpected keys (ckpt has but model does not): {unexpected[:8]}\n"
                f"Hint: probably stale `MultiTaskRoberta` class binding after reload. "
                f"Restart runtime or re-run the imports cell, then retry."
            )

        logits = predict_logits(model, loader, device)
        probs = {f: F.softmax(logits[f], dim=-1) * w for f in EVAL_FIELDS}

        if first:
            accum = probs
            first = False
        else:
            for f in EVAL_FIELDS:
                accum[f] = accum[f] + probs[f]

    # 權重已在迴圈內乘入並 normalize 過，不再除以 k
    return accum


def probs_to_preds(probs: Dict[str, torch.Tensor],
                   log_priors: Optional[Dict[str, torch.Tensor]] = None,
                   alpha: float = 0.0) -> List[Dict[str, str]]:
    """Argmax over per-field probabilities, with optional prior correction.

    Args:
        probs: dict[field] → tensor [N, num_classes] of softmax probs
        log_priors: dict[field] → tensor [num_classes] of log P(class) from train data
        alpha: prior correction strength. 0 = no correction. 0.5 = soft. 1.0 = full Bayes.

    The correction is `log_probs - alpha * log_prior`：減弱模型對主類別的偏見，
    提升稀有類別的 recall。
    """
    n = next(iter(probs.values())).shape[0]
    use_corr = log_priors is not None and alpha > 0
    preds = []
    for i in range(n):
        p = {}
        for f in EVAL_FIELDS:
            scores = torch.log(probs[f][i].clamp(min=1e-9))
            if use_corr:
                scores = scores - alpha * log_priors[f]
            p[f] = ID2LABEL[f][int(scores.argmax().item())]
        preds.append(p)
    return preds


def apply_logic_constraints(preds: List[Dict[str, str]],
                            probs: Optional[Dict[str, torch.Tensor]] = None,
                            conf_threshold: float = 0.5) -> List[Dict[str, str]]:
    """硬性邏輯約束 + 可選的信心度後處理。

    硬規則（資料集 100% 成立）:
      1. promise_status == "No"  → 下游全為 "N/A"
      2. evidence_status ∈ {"No", "N/A"}  → evidence_quality 必為 "N/A"

    信心度規則（probs 提供時）:
      3. evidence_quality 預測 "Clear" 但其機率 < conf_threshold
         且 evidence_status 預測 "Yes" 的機率也 < conf_threshold
         → 降級為 "Not Clear"（針對 confusion matrix 顯示的 "Clear over-prediction" pattern）
    """
    use_conf = probs is not None
    eq_clear_id = LABEL2ID["evidence_quality"]["Clear"] if use_conf else None
    es_yes_id = LABEL2ID["evidence_status"]["Yes"] if use_conf else None

    for i, item in enumerate(preds):
        # 規則 1
        if item["promise_status"] == "No":
            item["verification_timeline"] = "N/A"
            item["evidence_status"] = "N/A"
            item["evidence_quality"] = "N/A"
        # 規則 2
        if item["evidence_status"] in ("No", "N/A"):
            item["evidence_quality"] = "N/A"
        # 規則 3（僅在 probs 提供時生效）
        if use_conf and item["evidence_quality"] == "Clear":
            eq_conf = float(probs["evidence_quality"][i, eq_clear_id])
            es_conf = float(probs["evidence_status"][i, es_yes_id])
            if eq_conf < conf_threshold and es_conf < conf_threshold:
                item["evidence_quality"] = "Not Clear"
    return preds


def discover_fold_ckpts(ckpt_dir: str, pattern: str = "fold_*/best.pt") -> List[str]:
    paths = sorted(glob.glob(os.path.join(ckpt_dir, pattern)))
    return paths
