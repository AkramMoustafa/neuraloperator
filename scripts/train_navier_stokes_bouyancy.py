# Author  : Akram Moustafa
# Date    : 08/21/2025
# Purpose : Starting point for testing a two-channel Navier–Stokes dataset 
#           with buoyancy effects, to be used in future model development

import torch
from torch.utils.data import DataLoader, random_split

class ChannelwiseGaussianNormalizer:
    def __init__(self, sample_loader, key, eps=1e-6, max_batches=None):
        sums = sqs = None; n = 0
        with torch.no_grad():
            for bi, batch in enumerate(sample_loader):
                x = batch[key].float()  # [B,C,H,W]
                B, C = x.shape[:2]
                x = x.view(B, C, -1)
                sums = x.sum((0,2)) if sums is None else sums + x.sum((0,2))
                sqs  = (x**2).sum((0,2)) if sqs is None else sqs + (x**2).sum((0,2))
                n += x.shape[0] * x.shape[2]
                if max_batches and (bi+1) >= max_batches: break
        self.mean = (sums / n).view(1,-1,1,1)
        var = (sqs / n).view(1,-1,1,1) - self.mean**2
        self.std  = var.clamp_min(0.).sqrt() + eps

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std  = self.std.to(device)
        return self

    def encode(self, x): return (x - self.mean) / self.std
    def decode(self, x): return x * self.std + self.mean

class SimpleProcessor:
    def __init__(self, inn, outn): self.inn, self.outn = inn, outn
    def to(self, device): self.inn.to(device); self.outn.to(device); return self
    def encode(self, batch):
        return {"input": self.inn.encode(batch["input"]),
                "output": self.outn.encode(batch["output"])}
    def decode(self, batch):
        return {"input": self.inn.decode(batch["input"]),
                "output": self.outn.decode(batch["output"])}

def make_train_test_loaders(dataset, batch_size=4, test_fraction=0.2, num_workers=0):
    """Split dataset, build train/test loaders, and return normalizer data_processor."""
    n_total = len(dataset)
    n_test = max(1, int(test_fraction * n_total))
    n_train = n_total - n_test
    train_ds, test_ds = random_split(dataset, [n_train, n_test])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    stat_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    in_norm  = ChannelwiseGaussianNormalizer(stat_loader, key="input", max_batches=8)
    out_norm = ChannelwiseGaussianNormalizer(stat_loader, key="output", max_batches=8)
    data_processor = SimpleProcessor(in_norm, out_norm)

    test_loaders = {64: test_loader}

    return train_loader, test_loaders, data_processor
