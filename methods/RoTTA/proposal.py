import torch
import torch.nn as nn
import torch.jit
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from loguru import logger as log
from torchvision import transforms
from methods.RoTTA.transformers_cotta import get_tta_transforms
from methods.RoTTA.bn_layers import RobustBN1d, RobustBN2d
import methods.RoTTA.memory as memory

__all__ = ["setup"]

class RoTTA(nn.Module):

    def __init__(self, cfg, base, model, optimizer):
        super().__init__()
        self.cfg = cfg
        self.base = base
        self.model = model
        self.optimizer = optimizer
        self.mem = memory.CSTU(capacity=cfg.ADAPTER.RoTTA.MEMORY_SIZE, num_class=cfg.CORRUPTION.NUM_CLASS, lambda_t=cfg.ADAPTER.RoTTA.LAMBDA_T, lambda_u=cfg.ADAPTER.RoTTA.LAMBDA_U)

        self.transform = get_tta_transforms(cfg.CORRUPTION.DATASET)
        self.nu = cfg.ADAPTER.RoTTA.NU
        self.update_frequency = cfg.ADAPTER.RoTTA.UPDATE_FREQUENCY  # actually the same as the size of memory bank
        self.current_instance = 0

        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        outputs = self.forward_and_adapt(x, self.model, self.optimizer)
        return outputs

    @torch.enable_grad()
    def forward_and_adapt(self, x, model, optimizer):
        # batch data
        with torch.no_grad():
            model.eval()
            self.model_ema.eval()
            ema_out = self.model_ema(x)
            predict = torch.softmax(ema_out, dim=1)
            pseudo_label = torch.argmax(predict, dim=1)
            entropy = torch.sum(- predict * torch.log(predict + 1e-6), dim=1)

        # add into memory
        for i, data in enumerate(x):
            p_l = pseudo_label[i].item()
            uncertainty = entropy[i].item()
            current_instance = (data, p_l, uncertainty)
            self.mem.add_instance(current_instance)
            self.current_instance += 1

            if self.current_instance % self.update_frequency == 0:
                self.update_model(model, optimizer)

        return ema_out

    def update_model(self, model, optimizer):
        model.train()
        self.model_ema.train()
        # get memory data
        sup_data, ages = self.mem.get_memory()
        l_sup = None
        if len(sup_data) > 0:
            sup_data = torch.stack(sup_data)
            strong_sup_aug = self.transform(sup_data)
            ema_sup_out = self.model_ema(sup_data)
            stu_sup_out = model(strong_sup_aug)
            instance_weight = timeliness_reweighting(ages)
            l_sup = (softmax_entropy(stu_sup_out, ema_sup_out) * instance_weight).mean()

        l = l_sup
        if l is not None:
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        self.update_ema_variables(self.model_ema, self.model, self.nu)

    @staticmethod
    def update_ema_variables(ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
        return ema_model
    

@torch.jit.script
def softmax_entropy(x, x_ema):
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

def timeliness_reweighting(ages):
    if isinstance(ages, list):
        ages = torch.tensor(ages).float().cuda()
    return torch.exp(-ages) / (1 + torch.exp(-ages))

def configure_model(model: nn.Module,cfg):

    model.requires_grad_(False)
    normlayer_names = []

    for name, sub_module in model.named_modules():
        if isinstance(sub_module, nn.BatchNorm1d) or isinstance(sub_module, nn.BatchNorm2d):
            normlayer_names.append(name)

    for name in normlayer_names:
        bn_layer = get_named_submodule(model, name)
        if isinstance(bn_layer, nn.BatchNorm1d):
            NewBN = RobustBN1d
        elif isinstance(bn_layer, nn.BatchNorm2d):
            NewBN = RobustBN2d
        else:
            raise RuntimeError()

        momentum_bn = NewBN(bn_layer,cfg.ADAPTER.RoTTA.ALPHA)
        momentum_bn.requires_grad_(True)
        set_named_submodule(model, name, momentum_bn)
    return model

def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


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
    log.info("Setup TTA method: RoTTA")
    base = deepcopy(model)
    model = configure_model(model,cfg)
    if cfg.OPTIM.METHOD == "SGD":
        optimizer = optim.SGD(
            model.parameters(), 
            lr=float(cfg.OPTIM.LR),
            dampening=cfg.OPTIM.DAMPENING,
            momentum=float(cfg.OPTIM.MOMENTUM),
            weight_decay=float(cfg.OPTIM.WD),
            nesterov=cfg.OPTIM.NESTEROV
        )
    elif cfg.OPTIM.METHOD == "Adam":
        optimizer = optim.Adam(
            model.parameters(), 
            lr=float(cfg.OPTIM.LR),
            betas=(cfg.OPTIM.BETA, 0.999),
            weight_decay=float(cfg.OPTIM.WD)
        )
    TTA_model = RoTTA(
        cfg,
        base,
        model, 
        optimizer
    )
    return TTA_model
