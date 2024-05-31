import torch
import math

def cosine_schedule(base, end, T):
    return end - (end - base) * ((torch.arange(0, T, 1) * math.pi / T).cos() + 1) / 2