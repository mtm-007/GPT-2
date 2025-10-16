import sys
import torch
import torch.nn as nn
from torch.nn import functional as F


@torch.no_grad()
def estimate_loss():
    """
    OPTIMIZED: Keep losses on GPU, single sync at end
    """
    out = {}
    model.eval()
    for split in ['train','val']:
        # FIXED: Create tensor on GPU
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            # FIXED: No .item() - keep on GPU
            losses[k] = loss
        # FIXED: Single GPU->CPU sync at the end
        out[split] = losses.mean().item()
    model.train()
    return out