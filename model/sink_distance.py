import torch
import torch.nn as nn
import torch.nn.functional as F
class SinkhornDistance(nn.Module):
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, c):
        C = -c
        x_points = C.shape[-2]
        y_points = C.shape[-1]
        batch_size = C.shape[0]
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False, device=C.device).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False, device=C.device).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        thresh = 1e-12

        # Sinkhorn iterations
        for i in range(self.max_iter):
            if i % 2 == 0:
                u1 = u  # useful to check the update
                u = self.eps * (torch.log(mu) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
                err = (u - u1).abs().sum(-1).mean()
            else:
                v = self.eps * (torch.log(nu) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
                v = v.detach().requires_grad_(False)
                v[v > 9 * 1e8] = 0.0
                v = v.detach().requires_grad_(True)

            if err.item() < thresh:
                break

        U, V = u, v
        pi = torch.exp(self.M(C, U, V))

        return pi, C, U, V

    def M(self, C, u, v):
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def ave(u, u1, tau):
        return tau * u + (1 - tau) * u1