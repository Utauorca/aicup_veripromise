"""Training loop with FGM + EMA + AMP, evaluation, and logits-returning prediction."""
from collections import Counter
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

from .config import CFG, EVAL_FIELDS, FIELD_WEIGHTS, LABEL2ID, ID2LABEL
from .ema import ModelEMA
from .fgm import FGM


def _amp_dtype():
    return torch.float16 if CFG.amp_dtype == "fp16" else torch.bfloat16


class FocalLoss(nn.Module):
    """Multi-class Focal Loss with optional class weights.

    FL = -α_t · (1 − p_t)^γ · log(p_t)
      • (1 − p_t)^γ  讓已經學會的樣本（p_t 接近 1）loss 衰減，專注在難樣本
      • α_t (class weight) 解決長尾分布，對稀有類別加權
      • γ=0 退化為標準 (weighted) cross-entropy

    `reduction="none"` 回傳 per-sample loss，方便外層套 hierarchical mask。
    """

    def __init__(self, weight: torch.Tensor = None, gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)                                # [N, C]
        log_p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)           # [N]
        p_t = log_p_t.exp()
        focal = -((1 - p_t) ** self.gamma) * log_p_t                         # [N]
        if self.weight is not None:
            w = self.weight.to(logits.device)[targets]                       # [N]
            focal = focal * w
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal  # "none"


def _build_criteria(class_weights: Dict[str, torch.Tensor], device) -> Dict[str, nn.Module]:
    """所有 criteria 都用 reduction='none' 回 per-sample loss，外層手動處理 mask + reduction。"""
    criteria: Dict[str, nn.Module] = {}
    for f, w in class_weights.items():
        w = w.to(device)
        if CFG.loss_type == "focal":
            criteria[f] = FocalLoss(weight=w, gamma=CFG.focal_gamma, reduction="none")
        else:
            # Label smoothing 軟化 one-hot 目標，減輕模糊邊界任務的過擬合
            criteria[f] = nn.CrossEntropyLoss(
                weight=w, label_smoothing=CFG.label_smoothing, reduction="none",
            )
    return criteria


def _compute_task_masks(labels: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """根據真實 label 算每個 task 的「有資訊」mask。

    嵌套規則（train+val 100% 遵守）:
      promise=No 樣本 → 其他 3 個 task 是 forced N/A，無資訊
      evidence ∈ {No, N/A} → evidence_quality forced N/A，無資訊
    """
    promise_yes_id = LABEL2ID["promise_status"]["Yes"]
    evidence_yes_id = LABEL2ID["evidence_status"]["Yes"]

    is_promise_yes = (labels["promise_status"] == promise_yes_id).float()
    is_evidence_yes = (labels["evidence_status"] == evidence_yes_id).float()

    return {
        "promise_status": torch.ones_like(is_promise_yes),        # 全部訓
        "verification_timeline": is_promise_yes,
        "evidence_status": is_promise_yes,
        "evidence_quality": is_evidence_yes,
    }


def _ordinal_timeline_per_sample_loss(
    na_logit: torch.Tensor,
    ordinal_logits: torch.Tensor,
    target: torch.Tensor,
    na_id: int = 4,
    num_ord_classes: int = 4,
) -> torch.Tensor:
    """CORN-style per-sample ordinal loss + 一個 N/A binary classifier loss。

    對每筆樣本：
      L_i = BCE(na_logit_i, [target_i == N/A])  ← N/A binary
          + Σ_k BCE(ord_logit_i_k, [target_i > k])  for k in {0..K-2}  ← CORN
      ord_loss 對 N/A 樣本歸零（用 1 - is_na 遮罩）

    回傳 [B] per-sample loss tensor，配合外層 hierarchical mask 取平均。
    """
    is_na = (target == na_id).float()                                        # [B]
    na_loss = F.binary_cross_entropy_with_logits(
        na_logit.squeeze(-1).float(), is_na, reduction="none",
    )                                                                        # [B]

    # 把 N/A 的 ord_target 截到合法範圍（dummy；之後被 mask 抹掉）
    ord_target = target.clamp(max=num_ord_classes - 1).long()
    ord_loss = torch.zeros_like(is_na)
    for k in range(num_ord_classes - 1):
        binary_target = (ord_target > k).float()
        bce_k = F.binary_cross_entropy_with_logits(
            ordinal_logits[:, k].float(), binary_target, reduction="none",
        )
        ord_loss = ord_loss + bce_k
    ord_loss = ord_loss * (1 - is_na)                                        # N/A 樣本不參與 ordinal

    return na_loss + ord_loss


def _compute_total_task_loss(criteria, task_logits, labels, n_tasks, masks=None):
    """加總 4 個任務的 weighted loss。若提供 masks，套 per-task mask 並對有效樣本取平均。

    若 timeline 用 ordinal mode，task_logits 會帶 `_verification_timeline_ordinal` 額外 key
    存 (na_logit, ordinal_logits)，套 CORN loss 取代標準 CE。
    """
    total = None
    ordinal_raw = task_logits.get("_verification_timeline_ordinal")
    for f in labels:
        if f == "verification_timeline" and ordinal_raw is not None:
            na_logit, ord_logits = ordinal_raw
            per_sample = _ordinal_timeline_per_sample_loss(
                na_logit, ord_logits, labels[f],
                na_id=LABEL2ID[f]["N/A"], num_ord_classes=4,
            )
        else:
            per_sample = criteria[f](task_logits[f], labels[f])              # [B]
        if masks is not None and masks.get(f) is not None:
            m = masks[f]
            denom = m.sum().clamp(min=1.0)
            task_loss = (per_sample * m).sum() / denom
        else:
            task_loss = per_sample.mean()
        weighted = n_tasks * FIELD_WEIGHTS[f] * task_loss
        total = weighted if total is None else total + weighted
    return total


def _symmetric_kl(logits_1: torch.Tensor, logits_2: torch.Tensor,
                  mask: torch.Tensor = None) -> torch.Tensor:
    """0.5 * (KL(p1||p2) + KL(p2||p1))，per-sample 後依 mask 取有效平均。

    全程 fp32 計算 KL，避免 fp16 下 log_softmax 數值不穩。
    """
    log_p1 = F.log_softmax(logits_1.float(), dim=-1)
    log_p2 = F.log_softmax(logits_2.float(), dim=-1)
    p1 = log_p1.exp()
    p2 = log_p2.exp()
    kl_12 = (p1 * (log_p1 - log_p2)).sum(dim=-1)                            # [B]
    kl_21 = (p2 * (log_p2 - log_p1)).sum(dim=-1)                            # [B]
    per_sample = 0.5 * (kl_12 + kl_21)
    if mask is not None:
        denom = mask.sum().clamp(min=1.0)
        return (per_sample * mask).sum() / denom
    return per_sample.mean()


def _compute_total_rdrop_kl(task_logits_1, task_logits_2, n_tasks, masks=None):
    """加總 4 個 task 的對稱 KL，與 task loss 同樣乘 FIELD_WEIGHTS。

    忽略以 `_` 開頭的 internal key（如 `_verification_timeline_ordinal`），這些是 tuple 不是 logits。
    """
    total = None
    for f in task_logits_1:
        if f.startswith("_"):
            continue
        m = masks.get(f) if masks is not None else None
        kl = _symmetric_kl(task_logits_1[f], task_logits_2[f], m)
        weighted = n_tasks * FIELD_WEIGHTS[f] * kl
        total = weighted if total is None else total + weighted
    return total


def compute_class_weights(train_df: pd.DataFrame) -> Dict[str, torch.Tensor]:
    class_weights = {}
    cap = CFG.class_weight_cap
    for field, labels in EVAL_FIELDS.items():
        counts = Counter(train_df[field])
        total = len(train_df)
        w = [min(total / (len(labels) * counts.get(lab, 0)), cap) if counts.get(lab, 0) > 0 else 1.0
             for lab in labels]
        class_weights[field] = torch.tensor(w, dtype=torch.float)
    return class_weights


def train_one_epoch(model, loader, optimizer, scheduler, device,
                    class_weights, ema: ModelEMA, fgm_epsilon: float,
                    scaler: "torch.cuda.amp.GradScaler" = None):
    """One training epoch with optional AMP + FGM.

    AMP pattern with FGM:
      1. Clean forward under autocast → scaled backward.
      2. `scaler.unscale_()` so `.grad` contains true (unscaled) gradients — FGM needs this
         to compute its perturbation on the word embedding gradient.
      3. FGM attack (adds epsilon·grad/‖grad‖ to word_embeddings).
      4. Adversarial forward under autocast → unscaled `.backward()` accumulates
         adversarial grads on top of the already-unscaled main grads.
      5. Restore original embeddings, clip, `scaler.step()` (detects grads already unscaled).
    """
    model.train()
    fgm = FGM(model)
    criteria = _build_criteria(class_weights, device)
    total = 0.0
    use_amp = CFG.amp and scaler is not None and device.type == "cuda"
    amp_dtype = _amp_dtype()
    pbar = tqdm(loader, desc="Training")
    lam = CFG.span_loss_weight if CFG.use_span_aux else 0.0
    use_rdrop = getattr(CFG, "use_rdrop", False) and getattr(CFG, "rdrop_alpha", 0.0) > 0
    rdrop_alpha = CFG.rdrop_alpha if use_rdrop else 0.0
    for batch in pbar:
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = {f: v.to(device, non_blocking=True) for f, v in batch["labels"].items()}
        span_labels = batch["span_labels"].to(device, non_blocking=True)
        feats = batch["features"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Task loss 對齊 FIELD_WEIGHTS（evidence_quality 0.35 / evidence_status 0.30 /
        # promise_status 0.20 / verification_timeline 0.15）。乘 n_tasks 讓平均權重 = 1，
        # 保持與原本 unweighted sum 同量級，不必再調 lr / grad_clip。
        n_tasks = len(EVAL_FIELDS)

        # Hierarchical mask：promise=No 樣本不參與 timeline/evidence_status/evidence_quality
        # 的 loss；evidence ∈ {No, N/A} 不參與 evidence_quality 的 loss。
        masks = _compute_task_masks(labels) if getattr(
            CFG, "use_hierarchical_loss_mask", False
        ) else None

        # --- Clean pass ---
        # use_rdrop=True 時 forward 兩次（dropout mask 自然不同），加上對稱 KL 一致性 loss。
        # 為了不改變梯度量級（避免重調 lr / grad_clip），CE 用 0.5*(L1+L2)。
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            task_logits, span_logits = model(ids, mask, feats)
            loss = _compute_total_task_loss(criteria, task_logits, labels, n_tasks, masks)
            if lam > 0:
                loss = loss + lam * F.cross_entropy(
                    span_logits.view(-1, span_logits.size(-1)),
                    span_labels.view(-1),
                    ignore_index=-100,
                )

            if use_rdrop:
                task_logits_2, span_logits_2 = model(ids, mask, feats)
                loss_2 = _compute_total_task_loss(
                    criteria, task_logits_2, labels, n_tasks, masks,
                )
                if lam > 0:
                    loss_2 = loss_2 + lam * F.cross_entropy(
                        span_logits_2.view(-1, span_logits_2.size(-1)),
                        span_labels.view(-1),
                        ignore_index=-100,
                    )
                kl = _compute_total_rdrop_kl(
                    task_logits, task_logits_2, n_tasks, masks,
                )
                loss = 0.5 * (loss + loss_2) + rdrop_alpha * kl
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)   # 之後 FGM 要拿真的 grad
        else:
            loss.backward()

        # --- Adversarial pass ---
        fgm.attack(epsilon=fgm_epsilon, emb_name="word_embeddings")
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            task_logits_adv, span_logits_adv = model(ids, mask, feats)
            loss_adv = _compute_total_task_loss(criteria, task_logits_adv, labels, n_tasks, masks)
            if lam > 0:
                loss_adv = loss_adv + lam * F.cross_entropy(
                    span_logits_adv.view(-1, span_logits_adv.size(-1)),
                    span_labels.view(-1),
                    ignore_index=-100,
                )
        loss_adv.backward()   # grads 直接以 unscaled 累加
        fgm.restore(emb_name="word_embeddings")

        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if ema is not None:
            ema.update(model)

        total += loss.item()
        pbar.set_postfix(loss=loss.item())
    return total / len(loader)


@torch.no_grad()
def predict_logits(model, loader, device) -> Dict[str, torch.Tensor]:
    """Return raw task logits per field, shape [N, num_labels]. Span logits are discarded."""
    model.eval()
    collected = {f: [] for f in EVAL_FIELDS}
    use_amp = CFG.amp and device.type == "cuda"
    amp_dtype = _amp_dtype()
    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        feats = batch["features"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            task_logits, _ = model(ids, mask, feats)    # 丟棄 span_logits
        for f in EVAL_FIELDS:
            collected[f].append(task_logits[f].detach().float().cpu())
    return {f: torch.cat(v, dim=0) for f, v in collected.items()}


def logits_to_preds(logits_by_field: Dict[str, torch.Tensor]) -> List[Dict[str, str]]:
    n = next(iter(logits_by_field.values())).shape[0]
    preds = []
    for i in range(n):
        p = {}
        for f in EVAL_FIELDS:
            p[f] = ID2LABEL[f][int(logits_by_field[f][i].argmax().item())]
        preds.append(p)
    return preds


def evaluate_hybrid(gt: List[dict], pred: List[dict]) -> dict:
    assert len(gt) == len(pred)
    results = {}
    weighted = 0.0
    for field, labels in EVAL_FIELDS.items():
        y_true = [g[field] for g in gt]
        y_pred = [p[field] for p in pred]
        macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        micro = f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)
        report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
        w = FIELD_WEIGHTS.get(field, 0)
        weighted += macro * w
        results[field] = {"macro_f1": macro, "micro_f1": micro, "report": report, "weight": w}
    results["final_weighted_score"] = weighted
    return results
