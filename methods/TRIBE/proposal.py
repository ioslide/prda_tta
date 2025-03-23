from copy import deepcopy
from loguru import logger as log

import math
import torch
import torch.nn as nn
import torch.jit
import torch.optim as optim
import torch.nn.functional as F

from methods.TRIBE.bn_layers import BalancedRobustBN2dV5, BalancedRobustBN2dEMA, BalancedRobustBN1dV5
from methods.TRIBE.utils import set_named_submodule, get_named_submodule
from methods.TRIBE.custom_transforms import get_tta_transforms

__all__ = ["setup"]

class TRIBE(nn.Module):
    def __init__(self, model, optimizer, cfg):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.steps = self.cfg.OPTIM.STEPS
        assert self.steps > 0, "TRIBE requires >= 1 step(s) to forward and update"

        self.aux_model = self.build_ema(self.model)
        for (name1, param1), (name2, param2) in zip(self.model.named_parameters(), self.aux_model.named_parameters()):
            set_named_submodule(self.aux_model, name2, param1)
            
        self.source_model = self.build_ema(self.model)
        self.transform = get_tta_transforms(self.cfg)

        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    @staticmethod
    def build_ema(model):
        ema_model = deepcopy(model)
        for param in ema_model.parameters():
            param.detach_()
        return ema_model

    def forward(self, x, **kwargs):
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)
        return outputs      

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        with torch.no_grad():
            self.aux_model.eval()
            ema_out = self.aux_model(batch_data)
        
        self.update_model(self.model, self.optimizer, batch_data, ema_out)
        return ema_out
    
    def update_model(self, model, optimizer, batch_data, logit):
        p_l = logit.argmax(dim=1)
        self.source_model.train()
        self.aux_model.train()
        model.train()
        strong_sup_aug = self.transform(batch_data)
        
        self.set_bn_label(self.aux_model, p_l)
        ema_sup_out = self.aux_model(batch_data)

        self.set_bn_label(model, p_l)
        stu_sup_out = model(strong_sup_aug)

        entropy = self.self_softmax_entropy(ema_sup_out)
        entropy_mask = (entropy < self.cfg.ADAPTER.TRIBE.H0 * math.log(self.cfg.CORRUPTION.NUM_CLASS))

        l_sup = torch.nn.functional.cross_entropy(stu_sup_out, ema_sup_out.argmax(dim=-1), reduction='none')[entropy_mask].mean()

        with torch.no_grad():
            self.set_bn_label(self.source_model, p_l)
            source_anchor = self.source_model(batch_data).detach()
        
        l_reg = self.cfg.ADAPTER.TRIBE.LAMBDA * torch.nn.functional.mse_loss(ema_sup_out, source_anchor, reduction='none')[entropy_mask].mean()

        l = (l_sup + l_reg)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        return

    @staticmethod
    def set_bn_label(model, label=None):
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, BalancedRobustBN1dV5) or isinstance(sub_module, BalancedRobustBN2dV5) or isinstance(sub_module, BalancedRobustBN2dEMA):
                sub_module.label = label
        return
    
    @staticmethod
    def self_softmax_entropy(x):
        return -(x.softmax(dim=-1) * x.log_softmax(dim=-1)).sum(dim=-1)

    @staticmethod
    def update_ema_variables(ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
        return ema_model
        

def collect_params(model):
    names = []
    params = []

    for n, p in model.named_parameters():
        if p.requires_grad:
            names.append(n)
            params.append(p)

    return params, names


def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)



def configure_model(cfg,model):
    model.requires_grad_(False)
    normlayer_names = []

    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
            normlayer_names.append(name)
            
    for name in normlayer_names:
        bn_layer = get_named_submodule(model, name)
        if isinstance(bn_layer, nn.BatchNorm2d):
            NewBN = BalancedRobustBN2dV5
            momentum_bn = NewBN(bn_layer,
                                cfg.CORRUPTION.NUM_CLASS,
                                cfg.ADAPTER.TRIBE.ETA,
                                cfg.ADAPTER.TRIBE.GAMMA
                                )
            # NewBN = BalancedRobustBN2dEMA
        elif isinstance(bn_layer, nn.BatchNorm1d):
            NewBN = BalancedRobustBN1dV5
            momentum_bn = NewBN(bn_layer,
                                cfg.CORRUPTION.NUM_CLASS,
                                cfg.ADAPTER.TRIBE.ETA,
                                cfg.ADAPTER.TRIBE.GAMMA
                                )
        else:
            raise RuntimeError()
        
        momentum_bn.requires_grad_(True)
        set_named_submodule(model, name, momentum_bn)
    return model

def setup(model, cfg):
    log.info("Setup TTA method: TRIBE")
    model = configure_model(cfg,model)
    params, param_names = collect_params(model)
    if cfg.OPTIM.METHOD == "SGD":
        optimizer = optim.SGD(
            params, 
            lr=float(cfg.OPTIM.LR),
            dampening=cfg.OPTIM.DAMPENING,
            momentum=float(cfg.OPTIM.MOMENTUM),
            weight_decay=float(cfg.OPTIM.WD),
            nesterov=cfg.OPTIM.NESTEROV
        )
    elif cfg.OPTIM.METHOD == "Adam":
        optimizer = optim.Adam(
            params, 
            lr=float(cfg.OPTIM.LR),
            betas=(cfg.OPTIM.BETA, 0.999),
            weight_decay=float(cfg.OPTIM.WD)
        )
    TTA_model = TRIBE(
        model, 
        optimizer,
        cfg=cfg
    )
    return TTA_model