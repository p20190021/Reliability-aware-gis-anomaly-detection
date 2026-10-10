"""
Paper-faithful independent PyTorch reimplementation core for SSF-AE.

Paper
-----
Zhimin Zhang, Chuan Ma, X. Teng, et al.,
"Hyperspectral anomaly detection based on spatial-spectral feature fusion
autoencoder", Neurocomputing 678 (2026) 133162.

STATUS
------
Independent reimplementation/adaptation. This is NOT author-released code.

Paper-supported structure retained
----------------------------------
- Parallel 3D-CNN AE and 3D-ViT AE branches.
- CNN Conv3D kernel 3x3x3.
- Two 2x2x2 max-pooling stages and symmetric upsampling decoder.
- ViT patch size 4x4x4.
- Learnable positional encoding.
- Transformer encoder: self-attention + FFN + residual + LayerNorm.
- Transformer decoder: cross-attention + FFN + residual + LayerNorm.
- Four repeated encoder/decoder blocks, following the "4x" architecture figure.
- Projection / embedding dimension = 64.
- Dropout = 0.1.
- Sigmoid reconstruction output.
- MSE reconstruction objective.
- Dynamic fusion weights based on branch training losses.

Explicit independent-reimplementation choices
---------------------------------------------
The paper leaves several details underspecified:
- CNN widths C1 and C2 are symbolic. We freeze C1=16, C2=32.
- FFN hidden width is not numerically stated. We freeze FFN=128 (=2*d).
- With d=64 and six heads, equal integer splitting is impossible. The paper
  states that increasing head count reduces per-head dimensionality, so this
  implementation uses floor(64/6)=10 dimensions/head (60 internal attention
  dimensions) followed by projection back to 64.
- The initial decoder query is the encoder memory itself. The paper describes
  decoder queries coming from the previous decoder layer and K/V from the
  encoder output, but does not specify a separate learned query token set.
- 3D upsampling interpolation is nearest-neighbour/repetition, a conservative
  implementation of the paper's generic "3D-upsampling" operation.

Tensor convention
-----------------
Input/output: [batch, 1, depth, height, width].

Project adaptation:
- 21 real normalized PCA channels.
- 3 zero-padding channels for model alignment.
- Model depth = 24.
- Spatial window = 40 x 40.
- Only the first 21 real channels contribute to MSE/anomaly scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SSFAEConfig:
    input_depth: int = 24
    real_depth: int = 21
    height: int = 40
    width: int = 40
    cnn_c1: int = 16
    cnn_c2: int = 32
    cnn_kernel: int = 3
    patch_size: int = 4
    embed_dim: int = 64
    num_heads: int = 6
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    ffn_dim: int = 128
    dropout: float = 0.1

    def validate(self) -> None:
        if self.real_depth > self.input_depth:
            raise ValueError("real_depth cannot exceed input_depth.")
        if self.input_depth % 4 != 0:
            raise ValueError("input_depth must be divisible by 4.")
        if self.height % 4 != 0 or self.width % 4 != 0:
            raise ValueError("height/width must be divisible by 4.")
        if self.input_depth % self.patch_size != 0:
            raise ValueError("input_depth must be divisible by patch_size.")
        if self.height % self.patch_size != 0:
            raise ValueError("height must be divisible by patch_size.")
        if self.width % self.patch_size != 0:
            raise ValueError("width must be divisible by patch_size.")
        if self.embed_dim <= 0 or self.num_heads <= 0:
            raise ValueError("embed_dim and num_heads must be positive.")


class CNN3DAutoencoder(nn.Module):
    """Local SSF-AE branch."""

    def __init__(self, cfg: SSFAEConfig):
        super().__init__()
        cfg.validate()
        k = cfg.cnn_kernel
        p = k // 2

        self.encoder = nn.Sequential(
            nn.Conv3d(1, cfg.cnn_c1, kernel_size=k, padding=p),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(cfg.cnn_c1, cfg.cnn_c2, kernel_size=k, padding=p),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        self.decoder_conv1 = nn.ConvTranspose3d(
            cfg.cnn_c2, cfg.cnn_c2, kernel_size=k, padding=p
        )
        self.decoder_conv2 = nn.ConvTranspose3d(
            cfg.cnn_c2, cfg.cnn_c1, kernel_size=k, padding=p
        )
        self.decoder_out = nn.ConvTranspose3d(
            cfg.cnn_c1, 1, kernel_size=k, padding=p
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = F.relu(self.decoder_conv1(z), inplace=False)
        z = F.interpolate(z, scale_factor=2, mode="nearest")
        z = F.relu(self.decoder_conv2(z), inplace=False)
        z = F.interpolate(z, scale_factor=2, mode="nearest")
        z = self.decoder_out(z)
        return torch.sigmoid(z)


class FlexibleMultiheadAttention(nn.Module):
    """Multi-head attention allowing d=64 with h=6."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim < 1:
            raise ValueError("num_heads exceeds embed_dim.")
        self.inner_dim = self.head_dim * num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, self.inner_dim)
        self.k_proj = nn.Linear(embed_dim, self.inner_dim)
        self.v_proj = nn.Linear(embed_dim, self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return (
            x.view(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        q = self._heads(self.q_proj(query))
        k = self._heads(self.k_proj(key))
        v = self._heads(self.v_proj(value))

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(query.shape[0], query.shape[1], self.inner_dim)
        )
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Two fully connected layers with ReLU between them."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x, inplace=False)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, cfg: SSFAEConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.attn = FlexibleMultiheadAttention(
            cfg.embed_dim, cfg.num_heads, cfg.dropout
        )
        self.drop1 = nn.Dropout(cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
        self.ffn = FeedForward(cfg.embed_dim, cfg.ffn_dim, cfg.dropout)
        self.drop2 = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.drop1(self.attn(h, h, h))
        h = self.norm2(x)
        x = x + self.drop2(self.ffn(h))
        return x


class TransformerDecoderBlock(nn.Module):
    """Cross-attention + FFN decoder block."""

    def __init__(self, cfg: SSFAEConfig):
        super().__init__()
        self.norm_q = nn.LayerNorm(cfg.embed_dim)
        self.norm_mem = nn.LayerNorm(cfg.embed_dim)
        self.cross_attn = FlexibleMultiheadAttention(
            cfg.embed_dim, cfg.num_heads, cfg.dropout
        )
        self.drop_cross = nn.Dropout(cfg.dropout)
        self.norm_ffn = nn.LayerNorm(cfg.embed_dim)
        self.ffn = FeedForward(cfg.embed_dim, cfg.ffn_dim, cfg.dropout)
        self.drop_ffn = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(x)
        mem = self.norm_mem(memory)
        x = x + self.drop_cross(self.cross_attn(q, mem, mem))
        h = self.norm_ffn(x)
        x = x + self.drop_ffn(self.ffn(h))
        return x


class ViT3DAutoencoder(nn.Module):
    """Global/contextual SSF-AE branch."""

    def __init__(self, cfg: SSFAEConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        p = cfg.patch_size

        self.patch_embed = nn.Conv3d(
            1, cfg.embed_dim, kernel_size=p, stride=p, padding=0
        )

        self.grid_d = cfg.input_depth // p
        self.grid_h = cfg.height // p
        self.grid_w = cfg.width // p
        self.num_tokens = self.grid_d * self.grid_h * self.grid_w

        self.pos_embed = nn.Parameter(
            torch.empty(1, self.num_tokens, cfg.embed_dim)
        )
        self.input_dropout = nn.Dropout(cfg.dropout)

        self.encoder = nn.ModuleList(
            [TransformerEncoderBlock(cfg) for _ in range(cfg.num_encoder_layers)]
        )
        self.decoder = nn.ModuleList(
            [TransformerDecoderBlock(cfg) for _ in range(cfg.num_decoder_layers)]
        )
        self.final_norm = nn.LayerNorm(cfg.embed_dim)

        # Paper: Dense -> reshape -> 3D-deconvolution -> Sigmoid.
        self.reconstruction_dense = nn.Linear(
            cfg.embed_dim, cfg.embed_dim * (p ** 3)
        )
        self.patch_reconstruct = nn.ConvTranspose3d(
            cfg.embed_dim * (p ** 3),
            1,
            kernel_size=p,
            stride=p,
            padding=0,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.pos_embed)

    def _expanded_tokens_to_grid(self, expanded: torch.Tensor) -> torch.Tensor:
        B, N, C = expanded.shape
        if N != self.num_tokens:
            raise ValueError(f"Expected {self.num_tokens} tokens, got {N}.")
        return (
            expanded.transpose(1, 2)
            .contiguous()
            .view(B, C, self.grid_d, self.grid_h, self.grid_w)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.patch_embed(x)
        tokens = z.flatten(2).transpose(1, 2).contiguous()
        tokens = self.input_dropout(tokens + self.pos_embed)

        memory = tokens
        for block in self.encoder:
            memory = block(memory)

        decoded = memory
        for block in self.decoder:
            decoded = block(decoded, memory)

        decoded = self.final_norm(decoded)
        expanded = self.reconstruction_dense(decoded)
        grid = self._expanded_tokens_to_grid(expanded)
        recon = self.patch_reconstruct(grid)
        return torch.sigmoid(recon)


def crop_real_depth(x: torch.Tensor, real_depth: int = 21) -> torch.Tensor:
    if x.ndim != 5:
        raise ValueError("Expected [B,C,D,H,W] tensor.")
    if real_depth > x.shape[2]:
        raise ValueError("real_depth exceeds tensor depth.")
    return x[:, :, :real_depth, :, :]


def reconstruction_mse(
    target: torch.Tensor,
    recon: torch.Tensor,
    real_depth: int = 21,
) -> torch.Tensor:
    t = crop_real_depth(target, real_depth)
    r = crop_real_depth(recon, real_depth)
    return F.mse_loss(r, t, reduction="mean")


def per_pixel_reconstruction_error(
    target: torch.Tensor,
    recon: torch.Tensor,
    real_depth: int = 21,
) -> torch.Tensor:
    t = crop_real_depth(target, real_depth)
    r = crop_real_depth(recon, real_depth)
    return ((r - t) ** 2).mean(dim=(1, 2))


def dynamic_fusion_weights(
    cnn_training_loss: float | torch.Tensor,
    vit_training_loss: float | torch.Tensor,
    eps: float = 1e-12,
):
    if not torch.is_tensor(cnn_training_loss):
        cnn_training_loss = torch.tensor(float(cnn_training_loss))
    if not torch.is_tensor(vit_training_loss):
        vit_training_loss = torch.tensor(float(vit_training_loss))

    denom = cnn_training_loss + vit_training_loss + eps
    w_cnn = vit_training_loss / denom
    w_vit = cnn_training_loss / denom
    return w_cnn, w_vit


def fused_per_pixel_error(
    target: torch.Tensor,
    cnn_recon: torch.Tensor,
    vit_recon: torch.Tensor,
    cnn_training_loss: float | torch.Tensor,
    vit_training_loss: float | torch.Tensor,
    real_depth: int = 21,
) -> torch.Tensor:
    cnn_err = per_pixel_reconstruction_error(target, cnn_recon, real_depth)
    vit_err = per_pixel_reconstruction_error(target, vit_recon, real_depth)

    w_cnn, w_vit = dynamic_fusion_weights(
        cnn_training_loss, vit_training_loss
    )
    w_cnn = w_cnn.to(cnn_err.device, cnn_err.dtype)
    w_vit = w_vit.to(vit_err.device, vit_err.dtype)

    return w_cnn * cnn_err + w_vit * vit_err


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
