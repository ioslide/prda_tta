from loguru import logger as log
import torch.nn as nn
import torch
__all__ = ["setup"]

class Source(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x, **kwargs):
        with torch.no_grad():
            return self.model(x)

def setup(model, args):
    log.info("Setup TTA method: Source")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    model = Source(model)
    return model