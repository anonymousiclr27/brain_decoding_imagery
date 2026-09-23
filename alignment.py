"""The alignment module, the loss and the model-selection criterion.

The module sits in front of the frozen brain module of the pretrained decoder and is
trained only through a latent objective: nothing downstream is updated.

It has two stages.

`TemporalMixing` recombines the T time samples. The MLP that follows treats the time
samples as independent observations sharing one set of weights, but information about
the imagined content is not spread uniformly over the window, so a learned mixing of
the samples is allowed. It is a 1x1 convolution over the voxel axis with T input and
T output channels and no bias: a wider kernel would be meaningless, since voxels that
are adjacent in the flattened vector are not adjacent in the brain. It is initialised
to the identity, so at step 0 the model is exactly the one without mixing.

`VoxelAdapter` is a one-hidden-layer MLP applied identically to each time sample, from
the input voxels to the number of voxels the frozen brain module expects.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import settings as C


class TemporalMixing(nn.Module):
    """(B, V, T) -> (B, V, T): each output sample is a learned mix of the T inputs."""

    def __init__(self, time_dim=C.T):
        super().__init__()
        self.conv = nn.Conv1d(time_dim, time_dim, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.eye(time_dim).unsqueeze(-1))

    def forward(self, x):
        return self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)


class VoxelAdapter(nn.Module):
    """Temporal mixing, then an MLP shared across time samples."""

    def __init__(self, in_vox, out_vox, hidden=C.HIDDEN, dropout=C.DROPOUT,
                 time_dim=C.T, temporal_mixing=True):
        super().__init__()
        self.mixing = TemporalMixing(time_dim) if temporal_mixing else None
        self.mlp = nn.Sequential(
            nn.Linear(in_vox, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_vox),
        )

    def forward(self, x):                      # x: (B, V, T)
        if self.mixing is not None:
            x = self.mixing(x)
        x = self.mlp(x.permute(0, 2, 1))       # the MLP acts on the voxel axis
        return x.permute(0, 2, 1)              # (B, V_vis, T)


def supcon(pred, target, labels, tau=C.TAU):
    """Supervised contrastive term between predicted and target embeddings.

    Each predicted (imagery) embedding is an anchor; the target (perception)
    embeddings of the same stimulus in the batch are its positives and those of the
    other stimuli its negatives. Without it, the regression term alone pushes weakly
    informative inputs towards a single embedding shared by all stimuli.
    """
    p = F.normalize(pred.flatten(1), dim=1)
    t = F.normalize(target.flatten(1), dim=1)
    sim = p @ t.T / tau
    pos = (labels[:, None] == labels[None, :]).float()
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -((pos * log_prob).sum(1) / pos.sum(1).clamp(min=1)).mean()


def loss_fn(pred, target, labels, lam=C.LAMBDA, tau=C.TAU):
    """MSE towards the perceptual target, plus the contrastive term."""
    return F.mse_loss(pred, target) + lam * supcon(pred, target, labels, tau)


@torch.no_grad()
def validation_specificity(adapter, loader, embed, device):
    """Model-selection criterion: how stimulus-specific the predictions are.

    Predicted and target embeddings are averaged per stimulus, giving two sets of six
    centroids; the criterion is the mean of the diagonal minus the mean of the
    off-diagonal entries of their 6x6 cosine-similarity matrix. Validation loss alone
    would not do: a collapsed solution, one embedding for every stimulus, can have a
    perfectly good MSE.
    """
    adapter.eval()
    P, T_, L = [], [], []
    for xb, yb, lb in loader:
        P.append(embed(adapter(xb.to(device).float())).flatten(1).cpu())
        T_.append(embed(yb.to(device).float()).flatten(1).cpu())
        L.append(lb)
    P, T_, L = torch.cat(P), torch.cat(T_), torch.cat(L).numpy()
    classes = np.unique(L)
    Pc = torch.stack([P[L == c].mean(0) for c in classes])
    Tc = torch.stack([T_[L == c].mean(0) for c in classes])
    S = (F.normalize(Pc, dim=1) @ F.normalize(Tc, dim=1).T).numpy()
    off = ~np.eye(len(classes), dtype=bool)
    return float(np.diag(S).mean() - S[off].mean())
