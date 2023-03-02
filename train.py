from dataclasses import dataclass, field
from model import *

from torch import Tensor

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class TrainConfig:
    data_path: str = os.path.expanduser('~/data/wikitext-103/wiki.train.tokens_bpe.npy')
    batch_size: int = 1
    n_examples: int = 1_000_000
    n_batches: int = field(init=False)
    min_lr: float = 1e-5
    max_lr: float = 1e-4
    weight_decay: float = 1e-4
    max_noise: float = np.pi
    n_warmup: int = 1_000
    n_warmup_batches: int = field(init=False)
    checkpoint_interval: int = 100_000
    checkpoint_interval_batches: int = field(init=False)
    checkpoint_dir: str = 'checkpoints'

    def __post_init__(self) -> None:
        self.n_batches = self.n_examples // self.batch_size
        self.n_warmup_batches = self.n_warmup // self.batch_size
        self.checkpoint_interval_batches = self.checkpoint_interval // self.batch_size

def setup_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """
    Sets up ADAM optimizer w/ weight decay. Taken from Andrej Karpathy
    https://github.com/karpathy/nanoGPT/blob/ae3a8d5fdd3ddb8b13fab182723476523961e3ab/model.py#L269
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (nn.LayerNorm)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
    
    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=config.max_lr)

    return optimizer

def get_train_batch(x: np.ndarray, model_config: ModelConfig, train_config: TrainConfig) -> Tuple[Tensor, Tensor]:
    """
    Gets a training batch by randomly sampling from x.
    """

    ctx_width = np.random.randint(1, model_config.ctx_width)
    js = np.random.randint(0, x.shape[0] - ctx_width - 1, size=train_config.batch_size)
    xs = [torch.tensor(x[j:j+ctx_width].astype(np.int64), dtype=torch.long) for j in js]
    ys = [torch.tensor(x[j+ctx_width:j+ctx_width+1].astype(np.int64), dtype=torch.long) for j in js]
    xs = torch.cat([v.unsqueeze(0) for v in xs], dim=0).to(model_config.device)
    ys = torch.cat([v.unsqueeze(0) for v in ys], dim=0).to(model_config.device)
    
    return xs, ys

def get_lr(i: int, train_config: TrainConfig) -> float:
    if i < train_config.warmup:
        return np.sin(i/train_config.n_warmup_batches)*train_config.max_lr
    else:
        frac = np.cos((i-train_config.n_warmup_batches)/train_config.n_batches)
        return frac*train_config.max_lr + (1-frac)*train_config.min_lr

if __name__ == '__main__':

    from simple_parsing import ArgumentParser
    import time

    parser = ArgumentParser()
    parser.add_arguments(ModelConfig, 'model')
    parser.add_arguments(TrainConfig, 'train')
    parser.add_argument('--id', type=int, default=int(time.time()))

    args = parser.parse_args()

    print(f'training run id = {args.id}', flush=True)

    if not os.path.exists(args.train.checkpoint_dir):
        p = os.path.abspath(args.train.checkpoint_dir)
        print(f'creating checkpoint directory: {p}... ', end='', flush=True)
        os.mkdir(p)
        print('done.', flush=True)

    # Setup from configs

    # Load data as memmapped file
    print('Loading training data... ', end='', flush=True)
    data = np.memmap(args.train.data_path, dtype=np.uint16, mode='r')
    print('done.', flush=True)

    n_tokens = data.shape[0]
    print(f'# training tokens = {n_tokens/1e9:.4f}B', flush=True)

    # Setup model
    print('Setting up model... ', end='', flush=True)
    model = Model(args.model)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.shape) for p in trainable_params])
    print('done.', flush=True)
    print(f'# trainable params = {n_params/1e9:3.4f}B', flush=True)

    # Setup optimizers
    print('Setting up optimizer... ', end='', flush=True)
    optim = setup_optimizer(model, args.train)
    print('done.', flush=True)

    torch.set_anomaly_enabled(True)

    # Main training loop
    for i in range(args.train.n_batches):
        
        # Get training pair
        x, y = get_train_batch(data, args.model, args.train)

        n_batch, n_tokens = x.shape

        # Convert training pair to embedding vectors
        x = torch.index_select(model.embeddings, dim=0, index=x.flatten()).reshape(n_batch, n_tokens, -1)
        y = torch.index_select(model.embeddings, dim=0, index=y.flatten()).reshape(n_batch, 1, -1)

        # Get random timesteps
        t = torch.rand(x.shape[0], 1, 1)

        # Create random noise-scaled vectors in the tangent space of y
        v = torch.rand(y.shape[0], 1, y.shape[-1], device=y.device, dtype=y.dtype)
        v = model.sphere.proj_tangent(y, v)
        v = v/torch.norm(v, dim=-1)

        # Scale by noise level dependent on t
        sigma = t*args.train.max_noise
        v = v*sigma

        # Project to new points on sphere
        p = model.sphere.exp(y, v)

        # Compute log map from p to y
        v_true = model.sphere.log(p, y)
        
        # Predict log map from t
        v_pred = model(p, x, t)

        # Compute loss and backprop
        optim.zero_grad()
        loss = F.mse_loss(v_pred, v_true)
        loss.backward()
        optim.step()

        print(f'batch = {i:08d}/{args.train.n_batches}, loss = {loss.item():2.6f}')