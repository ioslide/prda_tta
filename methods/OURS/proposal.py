import torch
import torch.nn as nn
import torch.jit
from copy import deepcopy
from collections import defaultdict
from loguru import logger as log
from methods.OURS.transformers_cotta import get_tta_transforms

__all__ = ["setup"]

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def consistency(x: torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1).mean()

class OURS(nn.Module):
    def __init__(self, cfg,model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = cfg.OPTIM.STEPS
        self.cfg = cfg
        if self.steps <= 0:
            raise ValueError("OURS requires >= 1 step(s) to forward and update")

        self.transforms = get_tta_transforms(cfg.CORRUPTION.DATASET)
        self.model_state, self.optimizer_state, self.model_anchor = self.copy_model_and_optimizer()
        self.model_anchor.eval()
        self.theta_0 = deepcopy(self.model.state_dict())
        self.FIM_0 = None
        self.trainable_params = {k: v for k, v in self.model.named_parameters() if v.requires_grad}
        self.index = 0
        self.mu = self.cfg.ADAPTER.OURS.MU  # Caching for efficiency
        self.lambda1 = self.cfg.ADAPTER.OURS.LAMBDA1
        self.lambda2 = self.cfg.ADAPTER.OURS.LAMBDA2
        self.fim_clip_value = torch.tensor(0.0001) #For FIM clipping
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)
        return outputs

    def copy_model_and_optimizer(self):
        model_state = deepcopy(self.model.state_dict())
        model_anchor = deepcopy(self.model)
        optimizer_state = deepcopy(self.optimizer.state_dict())

        for param in model_anchor.parameters():
            param.detach_()
        return model_state, optimizer_state, model_anchor

    def param_robustness(self):
        FIM_t = defaultdict(lambda: torch.tensor(0.0, device=self.device))

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                FIM_t[name] = torch.pow(param.grad, 2)

        for name, fim_value in FIM_t.items():
            FIM_t[name] = torch.min(fim_value, self.fim_clip_value)

        if self.index == 0:
            self.FIM_0 = FIM_t

        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for npp, p in m.named_parameters():
                    if p.requires_grad and npp in ['weight', 'bias']:
                        key = f"{nm}.{npp}"
                        if key in self.FIM_0 and key in FIM_t:
                            theta_0_val = self.theta_0[key].to(self.device)
                            FIM_0_val = self.FIM_0[key]
                            FIM_t_val = FIM_t[key]

                            FIM_theta_0 = theta_0_val * (FIM_0_val + 1e-8)
                            FIM_theta_1 = p.data * (FIM_t_val + 1e-8)

                            a = (1 - self.mu) * FIM_theta_0 + self.mu * FIM_theta_1
                            b = (1 - self.mu) * (FIM_0_val + 1e-8) + self.mu * (FIM_t_val + 1e-8)
                            p.data = (a / b).to(p.dtype)

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        self.optimizer.zero_grad()
        logits = self.model(x)
        aug_logits = self.model(self.transforms(x))
        with torch.no_grad():
            anchor_logits = self.model_anchor(x)

        loss_0 = softmax_entropy(logits).mean() 
        loss_1 = consistency(logits, aug_logits)
        loss_2 = consistency(logits, anchor_logits)
        loss = loss_0 + self.lambda1 * loss_1 + self.lambda2 * loss_2

        loss.backward()
        self.param_robustness()
        self.optimizer.step()
        self.index += 1
        return logits


def collect_params(model):
    """Collects the parameters of normalization layers."""
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def configure_model(model):
    """Configures the model for test-time adaptation."""
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    return model

def setup(model, cfg):
    log.info("Setup TTA method: OURS")
    model = configure_model(model)
    params, param_names = collect_params(model)
    if cfg.OPTIM.METHOD == "SGD":
        optimizer = torch.optim.SGD(
            params, 
            lr=float(cfg.OPTIM.LR),
            momentum=float(cfg.OPTIM.MOMENTUM),
            weight_decay=float(cfg.OPTIM.WD),
        )
    elif cfg.OPTIM.METHOD == "Adam":
        optimizer = torch.optim.Adam(
            params, 
            lr=float(cfg.OPTIM.LR),
            weight_decay=float(cfg.OPTIM.WD)
        )
    TTA_model = OURS(
        cfg,
        model, 
        optimizer
    )
    return TTA_model
