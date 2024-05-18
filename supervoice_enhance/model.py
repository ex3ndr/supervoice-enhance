import torch

class EnchanceModel(torch.nn.Module):
    def __init__(self, flow, config):
        self.flow = flow
        self.config = config

    def forward(self, *, source, noise, times, target = None):
        return self.flow(
            audio = source,
            noise = noise,
            times = times,
            mask = None,
            target = target
        )