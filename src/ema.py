"""Exponential Moving Average of model weights.

Small-data regimes (~1000 samples) benefit a lot from EMA: the shadow weights
smooth out per-step noise and usually generalise better than the final weights.

Usage:
    ema = ModelEMA(model, decay=0.999)
    # after optimizer.step():
    ema.update(model)
    # at evaluation time:
    ema.apply_shadow()        # swap model weights -> shadow
    preds = predict(model, ...)
    ema.restore()             # swap back to training weights
"""
from copy import deepcopy
import torch


class ModelEMA:
    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.backup = {}

    @torch.no_grad()
    def update(self, model) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply_shadow(self, model) -> None:
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if k in self.shadow}
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)

    def restore(self, model) -> None:
        if not self.backup:
            return
        model.load_state_dict({**model.state_dict(), **self.backup}, strict=False)
        self.backup = {}

    def state_dict(self) -> dict:
        return self.shadow
