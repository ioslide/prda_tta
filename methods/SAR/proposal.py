import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.jit
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from loguru import logger as log
from methods.SAR.sam import SAM

__all__ = ["setup"]

def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data

class SAR(nn.Module):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.steps = cfg.OPTIM.STEPS
        self.model, self.optimizer = prepare_model_and_optimizer(model,cfg)
        self.reset_constant_em = 0.2  # threshold e_m for model recovery scheme
        self.ema = None  # to record the moving average of model output entropy, as model recovery criteri
        self.margin_e = math.log(cfg.CORRUPTION.NUM_CLASS) *0.40# margin E_0 for reliable entropy minimization, Eqn. (2)
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        for _ in range(self.steps):
            outputs, ema, reset_flag = self.forward_and_adapt(x, self.model, self.optimizer, self.margin_e, self.reset_constant_em, self.ema)
            if reset_flag:
                self.reset()
            self.ema = ema  # update moving average value of loss

        return outputs

    @torch.enable_grad()
    def forward_and_adapt(self,x, model, optimizer, margin, reset_constant, ema):
        # batch data
        """Forward and adapt model input data.
        Measure entropy of the model prediction, take gradients, and update params.
        """

        optimizer.zero_grad()
        # forward
        outputs = model(x)
        entropys = softmax_entropy(outputs)
        filter_ids_1 = torch.where(entropys < self.margin_e)
        entropys = entropys[filter_ids_1]
        loss = entropys.mean(0)
        loss.backward()

        optimizer.first_step(zero_grad=True)
        outputs2 = model(x)
        entropys2 = softmax_entropy(outputs2)

        entropys2 = entropys2[filter_ids_1]
        filter_ids_2 = torch.where(entropys2 < self.margin_e) 
        loss_second = entropys2[filter_ids_2].mean(0)
        if not np.isnan(loss_second.item()):
            ema = update_ema(ema, loss_second.item()) 
        
        loss_second.backward()
        optimizer.second_step(zero_grad=True)
        # perform model recovery
        reset_flag = False
        if ema is not None:
            if ema < 0.2:
                log.info("ema < 0.2, now reset the model")
                reset_flag = True


        return outputs, ema, reset_flag

    def save(self, ): pass
    def print(self, ): pass 


    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def timeliness_reweighting(ages):
    if isinstance(ages, list):
        ages = torch.tensor(ages).float().cuda()
    return torch.exp(-ages) / (1 + torch.exp(-ages))


def configure_model(model):
    """Configure model for use with SAR."""
    # train mode, because SAR optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what SAR updates
    model.requires_grad_(False)
    # configure norm for SAR updates: enable grad + force batch statisics (this only for BN models)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model


def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def collect_params(model):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names

def prepare_model_and_optimizer(model, cfg):
    model = configure_model(model)
    params, param_names = collect_params(model)

    lr = (0.00025 / 64) * cfg.TEST.BATCH_SIZE * 2 if cfg.TEST.BATCH_SIZE < 32 else 0.00025
    if cfg.TEST.BATCH_SIZE == 1:
        lr = 2*lr
    base_optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
    optimizer = SAM(base_optimizer=base_optimizer, rho=0.05)
    return model, optimizer

def get_named_submodule(model, sub_name: str):
    names = sub_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)

    return module

def set_named_submodule(model, sub_name, value):
    names = sub_name.split(".")
    module = model
    for i in range(len(names)):
        if i != len(names) - 1:
            module = getattr(module, names[i])

        else:
            setattr(module, names[i], value)

def setup(model, cfg):
    model = configure_model(model)
    TTA_model = SAR(
        cfg,
        model
    )
    return TTA_model
