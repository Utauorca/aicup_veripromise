"""MultiTaskRoberta + layerwise LR decay optimizer."""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .config import CFG


class CrossAttentionBlock(nn.Module):
    """Tiny transformer block over a 2-element sequence (promise, evidence).

    `nn.MultiheadAttention(...)(x, x, x)` over a length-2 sequence makes each element
    attend to itself AND the other → effectively cross-attention.

    Bottleneck design (hidden_dim → attn_dim → hidden_dim) keeps params manageable
    on a 1000-sample dataset. Residual connection from input preserves original info.
    """

    def __init__(self, hidden_dim: int, attn_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.proj_down = nn.Linear(hidden_dim, attn_dim)
        self.attn = nn.MultiheadAttention(
            attn_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(attn_dim)
        self.ff = nn.Sequential(
            nn.Linear(attn_dim, attn_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim * 2, attn_dim),
        )
        self.norm2 = nn.LayerNorm(attn_dim)
        self.proj_up = nn.Linear(attn_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, hidden_dim]，第 0 個是 promise_vec，第 1 個是 evidence_vec
        proj = self.proj_down(x)                                    # [B, 2, attn_dim]
        attn_out, _ = self.attn(proj, proj, proj, need_weights=False)
        proj = self.norm1(proj + attn_out)
        ff_out = self.ff(proj)
        proj = self.norm2(proj + ff_out)
        out = self.proj_up(proj)                                    # [B, 2, hidden_dim]
        return x + out                                               # 外層 residual（保留原資訊）


class TaskCrossAttention(nn.Module):
    """讓 4 個 task 在預測前互相 attention。

    Pipeline:
        shared_vec [B, in_dim]
            ↓ 4 個 task-specific Linear 投影
        [B, 4, attn_dim]   ← 4 個 task vector 組成 mini-sequence
            ↓ Multi-head self-attention（每個 task 同時看到自己 + 其他 3 個）
        [B, 4, attn_dim]   ← 各 task 帶有其他 task 的上下文資訊
            ↓ unstack
        分別給對應 task head 當額外輸入

    精神：取代過去 4 個 head 各自為政，讓「promise_status 思考時知道 evidence
    那邊在想什麼」、「evidence_quality 思考時知道 timeline 怎麼判」。
    """

    def __init__(self, in_dim: int, attn_dim: int, num_heads: int,
                 num_tasks: int, dropout: float = 0.1):
        super().__init__()
        self.num_tasks = num_tasks
        # 每個 task 一個獨立投影，產生 task-specific representation
        self.task_projs = nn.ModuleList([
            nn.Linear(in_dim, attn_dim) for _ in range(num_tasks)
        ])
        self.attn = nn.MultiheadAttention(
            attn_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(attn_dim)
        self.ff = nn.Sequential(
            nn.Linear(attn_dim, attn_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim * 2, attn_dim),
        )
        self.norm2 = nn.LayerNorm(attn_dim)

    def forward(self, shared: torch.Tensor) -> torch.Tensor:
        # shared: [B, in_dim]
        # 4 個 task projection → stack 為 [B, num_tasks, attn_dim]
        task_vecs = torch.stack([proj(shared) for proj in self.task_projs], dim=1)
        # Self-attention over 4 task elements
        attn_out, _ = self.attn(task_vecs, task_vecs, task_vecs, need_weights=False)
        x = self.norm1(task_vecs + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x   # [B, num_tasks, attn_dim]


class OrdinalTimelineHead(nn.Module):
    """CORN-style ordinal regression head for verification_timeline.

    時間軸有天然順序 (already < within_2y < 2_5y < longer_than_5y)，N/A 不在序上。
    用 1 個 N/A binary classifier + (K-1=3) 個 CORN binary classifiers 取代 5-class CE。
    每個 ordinal binary classifier 預測 P(rank > k) for k in {0, 1, 2}。

    Forward 回傳 (na_logit, ordinal_logits)。
    `to_5class_probs` 把它轉成下游需要的 [B, 5] probability vector。
    """

    def __init__(self, in_dim: int, hidden: int, num_ord_classes: int = 4):
        super().__init__()
        self.num_ord_classes = num_ord_classes
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.na_classifier = nn.Linear(hidden, 1)
        self.ordinal_classifier = nn.Linear(hidden, num_ord_classes - 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        na_logit = self.na_classifier(h)              # [B, 1]
        ordinal_logits = self.ordinal_classifier(h)   # [B, K-1]
        return na_logit, ordinal_logits

    @staticmethod
    def to_5class_probs(na_logit: torch.Tensor, ordinal_logits: torch.Tensor,
                        num_ord_classes: int = 4) -> torch.Tensor:
        """Convert CORN outputs → [B, num_ord_classes + 1] probability vector.

        非 N/A 機率分配:
          P(0) = 1 - σ(o_0)
          P(k) = ∏_{i<k} σ(o_i) · (1 - σ(o_k))  for k in {1..K-2}
          P(K-1) = ∏ σ(o_i)
        Final:
          p_5class[:, 0..K-1] = (1 - σ(na)) · p_ord
          p_5class[:, K]      = σ(na)
        """
        na_prob = torch.sigmoid(na_logit)                                    # [B, 1]
        ord_sig = torch.sigmoid(ordinal_logits)                              # [B, K-1]
        K = num_ord_classes
        B = na_logit.size(0)

        # cumprod[k] = σ(o_0) * ... * σ(o_k)
        cumprod = torch.cumprod(ord_sig, dim=-1)                             # [B, K-1]
        p_ord = na_logit.new_zeros(B, K)
        p_ord[:, 0] = 1 - ord_sig[:, 0]
        for k in range(1, K - 1):
            p_ord[:, k] = cumprod[:, k - 1] * (1 - ord_sig[:, k])
        p_ord[:, K - 1] = cumprod[:, K - 2]

        non_na_prob = 1 - na_prob                                            # [B, 1]
        p_5class = na_logit.new_zeros(B, K + 1)
        p_5class[:, :K] = non_na_prob * p_ord
        p_5class[:, K] = na_prob.squeeze(-1)
        return p_5class


class AttentionPooler(nn.Module):
    """Additive attention pooling over a variable-length sequence.

    每個 token 算一個純量分數，softmax 後作為權重對整個序列加權平均。比單用 [CLS]
    更能抓到散落在序列中後段的關鍵 token（ESG 承諾常在段落後半出現）。
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, L, H], attention_mask: [B, L]
        scores = self.attn(hidden_states).squeeze(-1)                # [B, L]
        scores = scores.masked_fill(~attention_mask.bool(), -1e4)    # padding 不參與
        weights = scores.softmax(dim=-1).unsqueeze(-1)               # [B, L, 1]
        return (hidden_states * weights).sum(dim=1)                  # [B, H]


class MultiTaskRoberta(nn.Module):
    def __init__(self, num_labels_dict: Dict[str, int], model_name: str = None):
        super().__init__()
        model_name = model_name or CFG.model_name
        self.backbone = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        hidden = self.backbone.config.hidden_size
        self.pooler = AttentionPooler(hidden)

        # Hand-crafted features → 低維 embedding
        self.feature_proj = nn.Sequential(
            nn.Linear(CFG.num_hand_features, CFG.feature_proj_dim),
            nn.GELU(),
            nn.LayerNorm(CFG.feature_proj_dim),
        )

        # Span-head predicted-span pooling projections
        if CFG.use_span_pooling:
            self.promise_proj = nn.Sequential(
                nn.Linear(hidden, CFG.span_pool_dim),
                nn.GELU(),
                nn.LayerNorm(CFG.span_pool_dim),
            )
            self.evidence_proj = nn.Sequential(
                nn.Linear(hidden, CFG.span_pool_dim),
                nn.GELU(),
                nn.LayerNorm(CFG.span_pool_dim),
            )
            if CFG.use_cross_attention:
                self.cross_attn = CrossAttentionBlock(
                    hidden_dim=hidden,
                    attn_dim=CFG.cross_attn_dim,
                    num_heads=CFG.cross_attn_heads,
                )

        # Task head 輸入維度
        head_input = hidden + CFG.feature_proj_dim
        if CFG.use_span_pooling:
            head_input += 2 * CFG.span_pool_dim
        if CFG.use_task_xattn:
            head_input += CFG.task_xattn_dim

        # Task cross-attention — 4 個 task 預測 representation 互相 attention
        if CFG.use_task_xattn:
            self._task_field_order = list(num_labels_dict.keys())
            xattn_in_dim = hidden + CFG.feature_proj_dim
            if CFG.use_span_pooling:
                xattn_in_dim += 2 * CFG.span_pool_dim
            self.task_xattn = TaskCrossAttention(
                in_dim=xattn_in_dim,
                attn_dim=CFG.task_xattn_dim,
                num_heads=CFG.task_xattn_heads,
                num_tasks=len(num_labels_dict),
            )

        # 4 個獨立 task heads（per-task Linear chain）
        # 若開啟 cascade，下游 head 的 input 維度會比上游大（多吃 upstream softmax 機率）
        use_cascade = getattr(CFG, "use_cascade_heads", False)
        if use_cascade and CFG.use_task_xattn:
            raise ValueError(
                "use_cascade_heads 與 use_task_xattn 互斥（兩者都改 task head input 維度）"
            )

        n_promise = num_labels_dict["promise_status"]                    # 2
        n_evidence = num_labels_dict["evidence_status"]                  # 3
        if use_cascade:
            head_inputs = {
                "promise_status":         head_input,
                "verification_timeline":  head_input + n_promise,        # +2
                "evidence_status":        head_input + n_promise,        # +2
                "evidence_quality":       head_input + n_promise + n_evidence,  # +5
            }
        else:
            head_inputs = {field: head_input for field in num_labels_dict}

        def _make_head(in_dim: int, out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(hidden, out_dim),
            )

        self.task_heads = nn.ModuleDict({
            field: _make_head(head_inputs[field], n)
            for field, n in num_labels_dict.items()
        })

        # CORN ordinal head 取代 verification_timeline 標準 head（如開啟）
        self.use_ordinal_timeline = getattr(CFG, "use_ordinal_timeline", False)
        if self.use_ordinal_timeline:
            n_timeline = num_labels_dict["verification_timeline"]    # 5 (4 ordinal + 1 N/A)
            self.task_heads["verification_timeline"] = OrdinalTimelineHead(
                in_dim=head_inputs["verification_timeline"],
                hidden=hidden,
                num_ord_classes=n_timeline - 1,                       # 4 個有序類
            )

        # 輔助 token-classification head（BIO 5 類）
        self.span_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, CFG.num_span_labels),
        )

    def _compute_timeline_outputs(self, head_input):
        """處理 timeline head 的 ordinal vs standard 兩種 mode。

        Returns:
            timeline_logits: [B, 5]
              - Ordinal mode: 5-class log-probabilities (softmax(log_probs) ≈ original probs)
              - Standard mode: raw 5-class logits
            ordinal_raw: tuple (na_logit, ordinal_logits) 或 None
              - 給 train.py 算 CORN loss 用
              - Inference 不需要
        """
        if self.use_ordinal_timeline:
            na_logit, ord_logits = self.task_heads["verification_timeline"](head_input)
            probs_5class = OrdinalTimelineHead.to_5class_probs(
                na_logit, ord_logits, num_ord_classes=4,
            ).clamp(min=1e-8)
            # 回傳 log_probs：F.softmax(log_probs) ≈ probs，下游 inference 不用改
            return torch.log(probs_5class), (na_logit, ord_logits)
        else:
            return self.task_heads["verification_timeline"](head_input), None

    def _span_weighted_pool(self, hidden_states, span_logits, attention_mask):
        """Soft pool of hidden_states weighted by span_head softmax probabilities.

        Returns:
            promise_vec [B, H]: hidden_states weighted by P(B-PROMISE or I-PROMISE)
            evidence_vec [B, H]: hidden_states weighted by P(B-EVIDENCE or I-EVIDENCE)
        """
        # span_logits: [B, L, 5]  →  label order: O/B-P/I-P/B-E/I-E
        span_probs = F.softmax(span_logits, dim=-1)                   # [B, L, 5]
        promise_w = span_probs[..., 1] + span_probs[..., 2]           # [B, L]
        evidence_w = span_probs[..., 3] + span_probs[..., 4]          # [B, L]

        # padding / special token 不參與 pooling
        promise_w = promise_w * attention_mask
        evidence_w = evidence_w * attention_mask

        # Normalize（clamp 避免除零；某些樣本可能完全無 promise 預測）
        p_total = promise_w.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [B, 1]
        e_total = evidence_w.sum(dim=1, keepdim=True).clamp(min=1e-6)

        promise_vec = (hidden_states * promise_w.unsqueeze(-1)).sum(dim=1) / p_total
        evidence_vec = (hidden_states * evidence_w.unsqueeze(-1)).sum(dim=1) / e_total
        return promise_vec, evidence_vec

    def forward(self, input_ids, attention_mask, features=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_4 = torch.stack(out.hidden_states[-4:], dim=0).mean(dim=0)  # [B, L, H]
        span_logits = self.span_head(last_4)                              # [B, L, 5]
        pooled = self.pooler(last_4, attention_mask)                     # [B, H]

        # Hand features
        if features is None:
            features = pooled.new_zeros(pooled.size(0), CFG.num_hand_features)
        feat_emb = self.feature_proj(features)

        parts = [pooled, feat_emb]

        # Predicted-span soft pooling
        if CFG.use_span_pooling:
            promise_vec, evidence_vec = self._span_weighted_pool(
                last_4, span_logits, attention_mask.float(),
            )
            if CFG.use_cross_attention:
                mini_seq = torch.stack([promise_vec, evidence_vec], dim=1)
                attended = self.cross_attn(mini_seq)
                promise_vec = attended[:, 0, :]
                evidence_vec = attended[:, 1, :]
            parts.append(self.promise_proj(promise_vec))
            parts.append(self.evidence_proj(evidence_vec))

        combined_base = torch.cat(parts, dim=-1)                          # [B, base_dim]

        # Task cross-attention：4 task representation 互相 attention 後當每個 head 額外輸入
        if CFG.use_task_xattn:
            task_attended = self.task_xattn(combined_base)                # [B, num_tasks, attn_dim]
            task_logits = {}
            for i, field in enumerate(self._task_field_order):
                head_input = torch.cat([combined_base, task_attended[:, i, :]], dim=-1)
                task_logits[field] = self.task_heads[field](head_input)
        elif getattr(CFG, "use_cascade_heads", False):
            # Cascade：直接利用 outcome 嵌套
            #   promise_status   → [combined_base]
            #   timeline / es    → [combined_base, promise_probs]      (上游 detach)
            #   evidence_quality → [combined_base, promise_probs, es_probs]
            promise_logits = self.task_heads["promise_status"](combined_base)
            promise_probs = F.softmax(promise_logits, dim=-1).detach()
            combined_p = torch.cat([combined_base, promise_probs], dim=-1)

            timeline_logits, timeline_ord_raw = self._compute_timeline_outputs(combined_p)
            es_logits = self.task_heads["evidence_status"](combined_p)

            es_probs = F.softmax(es_logits, dim=-1).detach()
            combined_pe = torch.cat([combined_base, promise_probs, es_probs], dim=-1)
            quality_logits = self.task_heads["evidence_quality"](combined_pe)

            task_logits = {
                "promise_status":        promise_logits,
                "verification_timeline": timeline_logits,
                "evidence_status":       es_logits,
                "evidence_quality":      quality_logits,
            }
            if timeline_ord_raw is not None:
                task_logits["_verification_timeline_ordinal"] = timeline_ord_raw
        else:
            task_logits = {}
            for f, head in self.task_heads.items():
                if f == "verification_timeline" and self.use_ordinal_timeline:
                    timeline_logits, timeline_ord_raw = self._compute_timeline_outputs(combined_base)
                    task_logits[f] = timeline_logits
                    task_logits["_verification_timeline_ordinal"] = timeline_ord_raw
                else:
                    task_logits[f] = head(combined_base)

        return task_logits, span_logits


def get_llrd_optimizer(model, lr: float, weight_decay: float, lr_decay: float):
    """Layerwise LR decay — works for both base (12 layers) and large (24 layers)."""
    num_layers = model.backbone.config.num_hidden_layers
    groups = []

    head_params = [p for n, p in model.named_parameters() if "backbone" not in n]
    groups.append({"params": head_params, "lr": lr, "weight_decay": weight_decay})

    for i in range(num_layers - 1, -1, -1):
        layer_lr = lr * (lr_decay ** (num_layers - i))
        layer_params = [p for n, p in model.named_parameters() if f"encoder.layer.{i}." in n]
        if layer_params:
            groups.append({"params": layer_params, "lr": layer_lr, "weight_decay": weight_decay})

    embed_lr = lr * (lr_decay ** (num_layers + 1))
    embed_params = [p for n, p in model.named_parameters() if "embeddings." in n]
    if embed_params:
        groups.append({"params": embed_params, "lr": embed_lr, "weight_decay": weight_decay})

    return torch.optim.AdamW(groups)
