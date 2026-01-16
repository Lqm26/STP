import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
import math
from einops import rearrange
from functools import partial


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.LTA = LTA(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.LTA(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class LTA(nn.Module):
    def __init__(self, dim=768):
        super(LTA, self).__init__()
        self.LTA_Block = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.LTA_Block(x)
        x = x.flatten(2).transpose(1, 2)

        return x



class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(ConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if activation is not None:
            self.activation = nn.ReLU()
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out
    
    
    

class SpatioTemporalPrompt(nn.Module):
    def __init__(self, backbone: nn.Module, num_frames: int = 5):
        super().__init__()
        self.backbone = backbone
        self.num_frames = num_frames

        # --- Configs ---
        dims = [2, 32, 128, 256] # [Input, Stage1, Stage2, Stage3]
        
        # --- 1. Stage 1 (Stem) ---
        # Input: (B*T, 2, H, W) -> Output: (B*T, 32, H/4, W/4)
        self.stem = nn.Sequential(
            ConvLayer(dims[0], dims[0], kernel_size=5, stride=1, padding=2),
            ConvLayer(dims[0], dims[1], kernel_size=8, stride=4, padding=2, norm='BN')
        )
        
        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, dims[1]))
        self.temporal_block1 = Block(
            dim=dims[1], num_heads=4, mlp_ratio=4, qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), linear=False, drop_path=0.03
        )

        # --- 2. Stage 2 ---
        # Input: (B*T, 32, H/4, W/4) -> Output: (B*T, 128, H/8, W/8)
        self.stage2_conv = nn.Sequential(
            ConvLayer(dims[1], dims[1], kernel_size=3, stride=1, padding=1),
            ConvLayer(dims[1], dims[2], kernel_size=6, stride=2, padding=2, norm='BN')
        )
        
        self.temporal_block2 = Block(
            dim=dims[2], num_heads=2, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), linear=False, drop_path=0.06
        )

        # --- 3. Stage 3 ---
        # Input: (B*T, 128, H/8, W/8) -> Output: (B*T, 256, H/16, W/16)
        self.stage3_conv = nn.Sequential(
            ConvLayer(dims[2], dims[2], kernel_size=3, stride=1, padding=1),
            ConvLayer(dims[2], dims[3], kernel_size=6, stride=2, padding=2, norm='BN')
        )

        self._init_weights()
        self._freeze_backbone()

    def _init_weights(self):
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)

    def _freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if 'patch_embed' not in name:
                param.requires_grad = False

    def _forward_temporal(self, x: torch.Tensor, block: nn.Module, 
                          add_embed: bool = False) -> torch.Tensor:
        B_T, C, H, W = x.shape
        B = B_T // self.num_frames
        
        # (B*T, C, H, W) -> (B*H*W, T, C)
        x = rearrange(x, '(b t) c h w -> (b h w) t c', b=B, t=self.num_frames)
        
        if add_embed:
            x = x + self.temporal_embed
            
        x = block(x, 1, self.num_frames)
        
        # (B*H*W, T, C) -> (B*T, C, H, W)
        x = rearrange(x, '(b h w) t c -> (b t) c h w', b=B, h=H, w=W)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        
        # Merge Batch and Time
        x = rearrange(x, 'b t c h w -> (b t) c h w')

        # --- Stage 1 ---
        x = self.stem(x)
        x = self._forward_temporal(x, self.temporal_block1, add_embed=True)
        
        # --- Stage 2 ---
        x = self.stage2_conv(x)
        x = self._forward_temporal(x, self.temporal_block2, add_embed=False)
        
        # --- Stage 3 ---
        x = self.stage3_conv(x) # Output shape: (B*T, 256, H_out, W_out)
        
        p1, p2 = 16, 16 
        
        x = rearrange(x, '(b t) (p1 p2) h w -> b t (h p1) (w p2)', 
                      b=B, t=T, p1=p1, p2=p2)
        
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.backbone(x)
        return x
