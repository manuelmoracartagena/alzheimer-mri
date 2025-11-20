"""
This Python script defines a Vision Transformer model for classification.
It is configured via the 'configs/vit_config.py' file.
"""

import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import Union, Tuple

from configs.vit_config import MODEL_CONFIG

def pair(t: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.) -> None:
        super().__init__()
        
        self.net: nn.Sequential
        
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self, 
        dim: int, 
        heads: int = 8, 
        dim_head: int = 64, 
        dropout: float = 0.
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads: int = heads
        self.scale: float = dim_head ** -0.5
        self.norm: nn.LayerNorm
        self.attend: nn.Softmax
        self.dropout: nn.Dropout
        self.to_qkv: nn.Linear
        self.to_out: Union[nn.Sequential, nn.Identity]

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(
        self, 
        dim: int, 
        depth: int, 
        heads: int, 
        dim_head: int, 
        mlp_dim: int, 
        dropout: float = 0.
    ) -> None:
        super().__init__()
        
        self.norm: nn.LayerNorm
        self.layers: nn.ModuleList
        
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(
        self, 
        image_size: Union[int, Tuple[int, int]] = MODEL_CONFIG["image_size"], 
        patch_size: Union[int, Tuple[int, int]] = MODEL_CONFIG["patch_size"], 
        channels: int = MODEL_CONFIG["in_channels"], 
        dim: int = MODEL_CONFIG["dim"], 
        depth: int = MODEL_CONFIG["depth"], 
        heads: int = MODEL_CONFIG["heads"], 
        mlp_dim: int = MODEL_CONFIG["mlp_dim"], 
        dim_head: int = MODEL_CONFIG["dim_head"], 
        dropout: float = MODEL_CONFIG["dropout"], 
        emb_dropout: float = MODEL_CONFIG["emb_dropout"], 
        pool: str = MODEL_CONFIG["pool"], 
        linear_out_features: int = MODEL_CONFIG["linear_out_features"]
    ) -> None:
        super().__init__()
        
        self.to_patch_embedding: nn.Sequential
        self.pos_embedding: nn.Parameter
        self.cls_token: nn.Parameter
        self.dropout: nn.Dropout
        self.transformer: Transformer
        self.pool: str
        self.to_latent: nn.Identity
        self.mlp_head: nn.Linear
        
        num_classes = linear_out_features
        
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)