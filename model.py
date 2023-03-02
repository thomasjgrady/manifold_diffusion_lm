from dataclasses import dataclass, field
from torch import Tensor
from sphere import UnitSphere
from typing import *

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelConfig:
    vocab_size: int = 50_297
    ctx_width: int = 1024
    n_dim: int = 3
    n_embed: int = 1024
    n_heads: int = 64
    hidden_scale: float = 4.0
    n_hidden: int = field(init=False)
    act: Callable = F.gelu
    n_blocks: int = 4
    device: torch.device = torch.device('cuda')
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.n_hidden = int(self.n_embed*self.hidden_scale)

class CausalSelfAttention(nn.Module):

    def __init__(self, config: ModelConfig) -> None:

        super().__init__()

        self.config = config

        self.lift = nn.Linear(config.n_embed, 3*config.n_embed, bias=False, device=config.device, dtype=config.dtype)
        self.proj = nn.Linear(config.n_embed,   config.n_embed, bias=False, device=config.device, dtype=config.dtype)
        self.mask = torch.tril(torch.ones(config.ctx_width, config.ctx_width, device=config.device, dtype=torch.bool)) \
            .view(1, 1, config.ctx_width, config.ctx_width)
        
    def forward(self, x: Tensor) -> Tensor:

        n_batch, n_tokens, n_embed = x.shape
        n_heads = self.config.n_heads

        q. k, v = self.lift(x).split(n_embed, dim=-1)

        q = q.view(n_batch, n_tokens, n_heads, n_embed // n_heads).transpose(1, 2)
        k = k.view(n_batch, n_tokens, n_heads, n_embed // n_heads).transpose(1, 2)
        v = v.view(n_batch, n_tokens, n_heads, n_embed // n_heads).transpose(1, 2)

        a = (q @ k.transpose(-1, -2))/np.sqrt(n_embed // n_heads)
        a.masked_fill_(self.mask[:,:,:n_tokens,:n_tokens] == 0, float('-inf'))
        a = torch.softmax(a, dim=-1)
        
        y = (a @ v).transpose(1, 2).contiguous().view(n_batch, n_tokens, n_embed)
        y = self.proj(y)

        return y
    
class MLP(nn.Module):

    def __init__(self, config: ModelConfig) -> None:

        super().__init__()

        self.config = config
        self.w0 = nn.Linear(config.n_embed, config.n_hidden, device=config.device, dtype=config.dtype)
        self.w1 = nn.Linear(config.n_embed, config.n_hidden, device=config.device, dtype=config.dtype)
        self.act = config.act

    def forward(self, x: Tensor) -> Tensor:
        x = self.w0(x)
        x = self.act(x)
        x = self.w1(x)
        return x
    
class TransformerBlock(nn.Module):

    def __init__(self, config: ModelConfig) -> None:

        super().__init__()
        
        self.config = config
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.ln0 = nn.LayerNorm(config.n_embed)
        self.ln1 = nn.LayerNorm(config.n_embed)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(x)
        x = self.ln0(x)
        x = x + self.mlp(x)
        x = self.ln1(x)
        return x
    
class Model(nn.Module):

    def __init__(self, config: ModelConfig) -> None:

        super().__init__()

        self.config = config
        self.sphere = UnitSphere(dim=config.n_dim)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_blocks)])
        self.lift = nn.Linear(config.n_dim, config.n_hidden, bias=False, device=config.device, dtype=config.dtype)
        self.proj = nn.Linear(config.n_hidden, config.n_dim, bias=False, device=config.device, dtype=config.dtype)

        self.embeddings_path = f'embeddings_{config.vocab_size}_{config.n_dim}.pt'
        if os.path.exists(self.embeddings_path):
            self.embeddings = torch.load(self.embeddings_path, map_location=config.device)
        else:
            self.embeddings = self.sphere.create_embeddings(config.n_dim).to(device=config.device, dtype=config.dtype)
            torch.save(self.embeddings, self.embeddings_path)

    def forward(self, p: Tensor, x: Tensor, t: float) -> Tensor:
        v = self.lift(p)
        x = self.lift(x)
        for bx, bv in zip(self.blocks_x, self.blocks_v):
            x = bx(x)
            v = v + torch.mean(x, dim=1, keepdim=True)
        v = self.proj(v)
        v = (1-t)*self.sphere.proj_tangent(p, v)
        return v