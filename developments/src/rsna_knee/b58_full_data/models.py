"""Shared trainable DINOv2 encoder and distinct, masked source-native heads."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def native_loss(logits, tasks, values, weights):
    """Unknown values have no loss/gradient; soft RSNA targets retain confidence."""
    if not len(tasks) == len(logits) == len(values) == len(weights):
        raise ValueError("native label/task dimensions disagree")
    numerator = logits[0].sum() * 0
    denominator = 0.
    for logit, task, value, weight in zip(logits, tasks, values, weights):
        if weight <= 0:
            continue
        if task["kind"] == "categorical":
            loss = F.cross_entropy(logit.reshape(1, -1), torch.tensor([int(value)], device=logit.device))
        elif task["kind"] == "regression":
            loss = F.smooth_l1_loss(logit.sigmoid().reshape(()), logit.new_tensor(float(value)))
        else:
            loss = F.binary_cross_entropy_with_logits(logit.reshape(()), logit.new_tensor(float(value)))
        numerator = numerator + float(weight) * loss
        denominator += float(weight)
    if denominator <= 0:
        raise ValueError("unlabelled cases belong in SSL, not supervised optimizer steps")
    return numerator / denominator


class NativeHeads(nn.Module):
    def __init__(self, encoder, tasks, chunk_size=2):
        super().__init__()
        self.encoder = encoder.requires_grad_(True)
        self.tasks, self.chunk_size = tasks, chunk_size
        dim = int(encoder.num_features)
        self.heads = nn.ModuleDict({source: nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, len(t["classes"]) if t["kind"] == "categorical" else 1))
            for t in definitions]) for source, definitions in tasks.items()})
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def encode(self, cpu_images):
        x = cpu_images.to(self.mean.device)
        return self.encoder((x - self.mean) / self.std)

    def forward(self, source, volumes):
        # Each volume is yielded/decompressed only when needed. Checkpoint CPU
        # chunks so GPU activations for all series need not remain resident.
        series = []
        for volume in volumes:
            features = []
            for chunk in volume.split(self.chunk_size):
                z = checkpoint(self.encode, chunk, use_reentrant=False, preserve_rng_state=True) if self.training else self.encode(chunk)
                features.append(z)
            series.append(torch.cat(features).mean(0))
        if not series:
            raise ValueError("native diagnosis case has no MRI series")
        features = torch.stack(series).mean(0)
        return [head(features).float() for head in self.heads[source]]
