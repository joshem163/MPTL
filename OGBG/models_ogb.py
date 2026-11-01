"""Models for OGB topological feature experiments.

MLP:      flat 1D feature vector → classification (PH, PTL, MPH-flat, MPTL-flat)
CNN2D:    (C, R, Q) grid image   → classification (MPH, MPTL)
HybridCNN: CNN branch (grid) + MLP branch (flat extra features) → classification

All models accept in_channels from feat_shape directly, so direction='both'
(which doubles channels) is handled automatically — no code change needed.

Mode → feature shape → recommended model:

  direction='sub' or 'super' (single direction):
    PH   → (2R,)       → MLP
    PTL  → (8R,)       → MLP
    MPH  → (2,  R, Q)  → CNN2D or MLP
    MPTL → (8,  R, Q)  → CNN2D  ← recommended

  direction='both' (sublevel + superlevel concatenated):
    PH   → (4R,)       → MLP
    PTL  → (16R,)      → MLP
    MPH  → (4,  R, Q)  → CNN2D or MLP
    MPTL → (16, R, Q)  → CNN2D  ← recommended

Channel layout per direction (MPTL):
  sub or super (8 channels):
    0   : β0,  1   : β1
    2-4 : L0 spectral (λ_min, λ_max, λ_mean)
    5-7 : L1 spectral (λ_min, λ_max, λ_mean)

  both (16 channels):
    0-7 : sublevel   (same layout as above)
    8-15: superlevel (same layout as above)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# MLP  (PH / PTL / flat MPH / flat MPTL)
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Fully-connected MLP for flat topological feature vectors.

    Accepts any flat input; grid inputs are flattened automatically.

    Typical input sizes (R=10 thresholds, Q=4 bond types):
      PH   sub/super : 2R  = 20      both : 4R  = 40
      PTL  sub/super : 8R  = 80      both : 16R = 160
      MPH  sub/super : 2RQ = 80      both : 4RQ = 160
      MPTL sub/super : 8RQ = 320     both : 16RQ= 640
    """
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 3,
                 dropout: float = 0.5, problem_type: str = 'multi-label'):
        super().__init__()
        self.problem_type = problem_type

        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.lins = nn.ModuleList()
        self.bns  = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.lins.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                self.bns.append(nn.BatchNorm1d(dims[i + 1]))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins: lin.reset_parameters()
        for bn  in self.bns:  bn.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)   # flatten grid input if needed
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        if self.problem_type == 'single-label':
            return F.log_softmax(x, dim=-1)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# CNN2D  (MPH / MPTL grid images)
# ─────────────────────────────────────────────────────────────────────────────

class CNN2D(nn.Module):
    """2D convolutional model for Betti / spectral grid images.

    Input shape: (batch, C, R, Q) — C is read from feat_shape automatically.

    C by mode and direction:
      MPH  sub/super : C=2   both : C=4
      MPTL sub/super : C=8   both : C=16

    Channel layout MPTL sub/super (C=8):
      0   : β0,  1   : β1
      2-4 : L0 spectral (λ_min, λ_max, λ_mean)
      5-7 : L1 spectral (λ_min, λ_max, λ_mean)

    Channel layout MPTL both (C=16):
      0-7 : sublevel channels (same as above)
      8-15: superlevel channels (same layout)

    zero_count excluded — equals β0 / β1 exactly (redundant).
    """
    def __init__(self, in_channels: int, grid_h: int, grid_w: int,
                 out_channels: int,
                 conv_channels: tuple = (32, 64),
                 kernel_size: int = 3,
                 hidden_dim: int = 128,
                 dropout: float = 0.5,
                 problem_type: str = 'multi-label'):
        super().__init__()
        self.problem_type = problem_type

        # Conv layers with adaptive padding so small grids (e.g. 8×8) still work
        layers = []
        c_in = in_channels
        for c_out in conv_channels:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ]
            c_in = c_out
        layers.append(nn.AdaptiveAvgPool2d((4, 4)))  # fixed spatial size
        self.conv = nn.Sequential(*layers)

        flat_dim = conv_channels[-1] * 4 * 4
        self.head = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_channels),
        )

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, R, Q)"""
        if x.dim() == 2:
            raise ValueError('CNN2D expects (B, C, R, Q) input, not flat.')
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        if self.problem_type == 'single-label':
            return F.log_softmax(x, dim=-1)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# HybridCNN  (CNN grid branch + MLP extra-feature branch)
# ─────────────────────────────────────────────────────────────────────────────

class HybridCNN(nn.Module):
    """Fuse CNN branch (MPTL grid) with MLP branch (extra flat features).

    Useful if you want to combine the (8 or 16, R, Q) MPTL grid with
    additional hand-crafted molecular descriptors.
    in_channels is set from feat_shape so direction='both' (16 ch) works
    automatically.
    """
    def __init__(self, in_channels: int, grid_h: int, grid_w: int,
                 extra_dim: int,
                 out_channels: int,
                 conv_channels: tuple = (32, 64),
                 kernel_size: int = 3,
                 hidden_dim: int = 128,
                 dropout: float = 0.5,
                 problem_type: str = 'multi-label'):
        super().__init__()
        self.problem_type = problem_type

        # CNN branch
        layers = []
        c_in = in_channels
        for c_out in conv_channels:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ]
            c_in = c_out
        layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        self.conv = nn.Sequential(*layers)
        cnn_flat = conv_channels[-1] * 4 * 4

        # MLP branch for extra features
        self.mlp_branch = nn.Sequential(
            nn.Linear(extra_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Fusion head
        self.head = nn.Sequential(
            nn.Linear(cnn_flat + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_channels),
        )

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, grid: torch.Tensor,
                extra: torch.Tensor) -> torch.Tensor:
        """
        grid  : (B, C, R, Q)
        extra : (B, extra_dim)
        """
        cnn_out = self.conv(grid).view(grid.size(0), -1)
        mlp_out = self.mlp_branch(extra)
        fused   = torch.cat([cnn_out, mlp_out], dim=-1)
        out     = self.head(fused)
        if self.problem_type == 'single-label':
            return F.log_softmax(out, dim=-1)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_type: str, mode: str,
                feat_shape,        # shape of one feature tensor
                out_channels: int,
                hidden_dim: int = 128,
                num_layers: int = 3,
                dropout: float = 0.5,
                conv_channels: tuple = (32, 64),
                problem_type: str = 'multi-label') -> nn.Module:
    """Instantiate the right model for a given mode.

    Parameters
    ----------
    model_type    : 'mlp' | 'cnn' | 'hybrid'
    mode          : 'PH' | 'PTL' | 'MPH' | 'MPTL'
    feat_shape    : shape of one sample tensor — read directly from extracted
                    features so direction='both' is handled automatically:
                      PH/PTL → (flat_dim,)       flat_dim doubles for 'both'
                      MPH/MPTL → (C, R, Q)       C doubles for 'both'
    out_channels  : number of output classes / tasks
    """
    mode = mode.upper()
    model_type = model_type.lower()

    if model_type == 'mlp':
        if len(feat_shape) == 1:
            in_dim = feat_shape[0]
        else:
            in_dim = int(torch.tensor(feat_shape).prod().item())
        return MLP(in_dim, hidden_dim, out_channels, num_layers,
                   dropout, problem_type)

    if model_type == 'cnn':
        if len(feat_shape) != 3:
            raise ValueError(
                f'CNN requires (C, R, Q) feature shape, got {feat_shape}. '
                'Use MPH or MPTL mode.')
        C, R, Q = feat_shape
        return CNN2D(C, R, Q, out_channels,
                     conv_channels=conv_channels,
                     hidden_dim=hidden_dim,
                     dropout=dropout,
                     problem_type=problem_type)

    if model_type == 'hybrid':
        raise ValueError(
            'For hybrid, instantiate HybridCNN directly with '
            'the grid shape and extra_dim.')

    raise ValueError(f'Unknown model_type: {model_type!r}. '
                     'Choose mlp | cnn | hybrid')
