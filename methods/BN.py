from copy import deepcopy
from loguru import logger as log

import torch
import torch.nn as nn

__all__ = ["setup"]

class BN_model(nn.Module):
    def __init__(self, model, steps=1):
        super().__init__()
        self.steps = steps
        assert steps > 0, "BN requires >= 1 step(s) to forward and update"
        self.model = model
        self.model_state = deepcopy(self.model.state_dict())

    def forward(self, x, **kwargs):
        outputs = self.model(x)
        return outputs

def configure_model(model, eps, momentum, reset_stats, no_stats):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.eps = eps
            m.momentum = momentum
            if reset_stats:
                m.reset_running_stats()
            if no_stats:
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    return model

def setup(model, cfg):
    log.info("Setup TTA method: BN")
    model = configure_model(model,cfg.ADAPTER.BN.EPS, cfg.OPTIM.MOMENTUM,cfg.ADAPTER.BN.RESET_STATS,cfg.ADAPTER.BN.NO_STATS)
    model = BN_model(
        model, 
        steps=int(cfg.OPTIM.STEPS)
    )
    return model
