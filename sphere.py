from torch import Tensor

import numpy as np
import torch

class UnitSphere:

    def __init__(self, dim: int, eps: float = 1e-7) -> None:
        self.dim = dim
        self.eps = eps

    def distance(self, p: Tensor, q: Tensor, keepdim: bool = False) -> Tensor:
        ip = torch.einsum('...d,...d->...', p, q)
        d = torch.arccos(ip.clamp(-1.0 + self.eps, 1.0 - self.eps))
        if keepdim:
            d = d.unsqueeze(-1)
        return d

    def exp(self, p: Tensor, v: Tensor) -> Tensor:
        p, v = torch.broadcast_tensors(p, v)
        s = p.shape
        p, v = p.view(s[0], -1), v.view(s[0], -1)
        nv = torch.norm(v, dim=-1)
        zero = torch.tensor(0.0, device=p.device, dtype=p.dtype)
        q = torch.where(torch.isclose(nv, zero, atol=self.eps), p, torch.cos(nv)*p + torch.sin(nv)*v/nv)
        return q.reshape(*s)
    
    def log(self, p: Tensor, q: Tensor) -> Tensor:
        p, q = torch.broadcast_tensors(p, q)
        s = p.shape
        p, q = p.view(s[0], -1), q.view(s[0], -1)
        d = self.distance(p, q, keepdim=True)
        proj = self.proj_tangent(p, q)
        nproj = torch.norm(proj, dim=-1)

        zero = torch.tensor(0.0, device=p.device, dtype=p.dtype)
        v = torch.where(torch.isclose(nproj, zero, atol=self.eps), p, d*proj/nproj)
        return v.reshape(*s)
    
    def proj(self, x: Tensor) -> Tensor:
        return x/torch.norm(x, dim=-1)

    def proj_tangent(self, p: Tensor, x: Tensor) -> Tensor:
        ip = torch.einsum('...d,...d->...', p, x).unsqueeze(-1)
        return x - ip*p
    
    def create_embeddings(self, n: int) -> Tensor:

        # For now, only implement fibonacci lattice in 3-space.
        if self.dim == 3:

            pts = torch.arange(0, n).to(torch.float32)
            pts += 0.5

            phi = torch.acos(1.0 - 2.0*pts/n)
            theta = np.pi * (1 + np.sqrt(5)) * pts

            xs = torch.cos(theta) * torch.sin(phi)
            ys = torch.sin(theta) * torch.sin(phi)
            zs = torch.cos(phi)

            out = torch.cat((
                xs.view(n, 1),
                ys.view(n, 1),
                zs.view(n, 1)
            ), dim=-1)

            return out

        else:
            raise Exception('unimplemented')
