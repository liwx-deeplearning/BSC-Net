"""Segmentation models for the static XCA baselines."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = conv3x3(channels, channels, stride)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


ENCODER_CONFIGS: Dict[str, Tuple[Type[nn.Module], List[int]]] = {
    "resnet18": (BasicBlock, [2, 2, 2, 2]),
    "resnet34": (BasicBlock, [3, 4, 6, 3]),
    "resnet50": (Bottleneck, [3, 4, 6, 3]),
}

IMAGENET_WEIGHT_URLS = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
}

class ResNetEncoder(nn.Module):
    """ImageNet-compatible ResNet feature extractor without a classifier."""

    def __init__(self, name: str = "resnet34") -> None:
        super().__init__()
        if name not in ENCODER_CONFIGS:
            raise ValueError(f"Unsupported encoder: {name}")
        block, layers = ENCODER_CONFIGS[name]
        self.name = name
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        expansion = block.expansion
        self.out_channels = [64, 64 * expansion, 128 * expansion, 256 * expansion, 512 * expansion]
        self._initialize()

    def _make_layer(
        self,
        block: Type[nn.Module],
        channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        out_channels = channels * block.expansion
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        modules = [block(self.in_channels, channels, stride, downsample)]
        self.in_channels = out_channels
        modules.extend(block(self.in_channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*modules)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def load_external_weights(self, weight_path: str) -> int:
        """Load matching encoder weights from torchvision ImageNet weights or a local file."""
        if weight_path.lower() == "imagenet":
            checkpoint = torch.hub.load_state_dict_from_url(
                IMAGENET_WEIGHT_URLS[self.name],
                map_location="cpu",
                progress=True,
                check_hash=True,
            )
        else:
            checkpoint = torch.load(Path(weight_path), map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        own_state = self.state_dict()
        selected = {}
        for key, value in state.items():
            key = key.removeprefix("module.").removeprefix("encoder.")
            key = key.removeprefix("backbone.")
            if key.startswith("conv1."):
                key = "stem.0." + key[len("conv1."):]
            elif key.startswith("bn1."):
                key = "stem.1." + key[len("bn1."):]
            if key in own_state and own_state[key].shape == value.shape:
                selected[key] = value
        if not selected:
            raise RuntimeError(f"No compatible encoder tensors found in {weight_path}")
        self.load_state_dict(selected, strict=False)
        return len(selected)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        c1 = self.stem(x)
        c2 = self.layer1(self.maxpool(c1))
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c1, c2, c3, c4, c5]


class InputNormalize(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class AttentionGate(nn.Module):
    def __init__(self, gating_channels: int, skip_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.gating_proj = nn.Sequential(
            nn.Conv2d(gating_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
        )
        self.attn = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gating: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # gating shape: (B, gating_channels, H, W)
        # skip shape: (B, skip_channels, H, W)
        alpha = self.attn(self.relu(self.gating_proj(gating) + self.skip_proj(skip)))
        # alpha shape: (B, 1, H, W)
        gated_skip = skip * alpha
        # gated_skip shape: (B, skip_channels, H, W)
        return gated_skip


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    # x shape: (B, C, H, W)
    batch_size, channels, height, width = x.shape
    if height % window_size != 0 or width % window_size != 0:
        raise ValueError(
            f"Feature size {(height, width)} must be divisible by window_size={window_size}."
        )
    x = x.view(
        batch_size,
        channels,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
    )
    # x shape: (B, C, H/ws, ws, W/ws, ws)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    # windows shape: (B, H/ws, W/ws, ws, ws, C)
    return windows.view(-1, window_size * window_size, channels)


def window_reverse(windows: torch.Tensor, window_size: int, height: int, width: int) -> torch.Tensor:
    # windows shape: (B*num_windows, window_size*window_size, C)
    batch_size = int(windows.shape[0] / ((height // window_size) * (width // window_size)))
    x = windows.view(
        batch_size,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    )
    # x shape: (B, H/ws, W/ws, ws, ws, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    # x shape: (B, C, H/ws, ws, W/ws, ws)
    return x.view(batch_size, -1, height, width)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.relative_position_bias = nn.Parameter(
            torch.zeros(num_heads, (2 * window_size - 1) * (2 * window_size - 1))
        )
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            )
        )
        coords_flat = coords.reshape(2, -1)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = relative_coords + (window_size - 1)
        relative_index = relative_coords[0] * (2 * window_size - 1) + relative_coords[1]
        self.register_buffer("relative_index", relative_index)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x shape: (B*num_windows, N, C)
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_windows,
            tokens,
            3,
            self.num_heads,
            channels // self.num_heads,
        )
        # qkv shape: (B*nw, N, 3, heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # qkv shape: (3, B*nw, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn shape: (B*nw, heads, N, N)
        relative_bias = self.relative_position_bias[:, self.relative_index.flatten()]
        relative_bias = relative_bias.view(self.num_heads, tokens, tokens).unsqueeze(0)
        attn = attn + relative_bias
        if attention_mask is not None:
            # attention_mask shape: (num_windows, N, N)
            num_windows = attention_mask.shape[0]
            if batch_windows % num_windows != 0:
                raise ValueError(
                    f"batch_windows={batch_windows} must be divisible by num_windows={num_windows}."
                )
            batch_size = batch_windows // num_windows
            attn = attn.view(
                batch_size,
                num_windows,
                self.num_heads,
                tokens,
                tokens,
            )
            attn = attn + attention_mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(batch_windows, self.num_heads, tokens, tokens)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(batch_windows, tokens, channels)
        # x shape: (B*nw, N, C)
        return self.proj(x)


class SwinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if shift_size < 0 or shift_size >= window_size:
            raise ValueError(
                f"shift_size must be in [0, window_size), got {shift_size}."
            )
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def build_attention_mask(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self.shift_size == 0:
            return None
        if height % self.window_size != 0 or width % self.window_size != 0:
            raise ValueError(
                f"Feature size {(height, width)} must be divisible by "
                f"window_size={self.window_size}."
            )
        image_mask = torch.zeros((1, height, width, 1), device=device, dtype=dtype)
        height_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        width_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        region_id = 0
        for height_slice in height_slices:
            for width_slice in width_slices:
                image_mask[:, height_slice, width_slice, :] = region_id
                region_id += 1
        # image_mask shape: (1, H, W, 1)
        mask_windows = window_partition(
            image_mask.permute(0, 3, 1, 2),
            self.window_size,
        )
        # mask_windows shape: (num_windows, window_size*window_size, 1)
        mask_windows = mask_windows.squeeze(-1)
        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attention_mask = attention_mask.masked_fill(
            attention_mask != 0,
            float(-100.0),
        ).masked_fill(attention_mask == 0, 0.0)
        # attention_mask shape: (num_windows, N, N)
        return attention_mask

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # x shape: (B, H*W, C)
        batch_size = x.shape[0]
        shortcut = x
        x = self.norm1(x)
        x_2d = x.view(batch_size, height, width, self.dim).permute(0, 3, 1, 2)
        # x_2d shape: (B, C, H, W)
        if self.shift_size > 0:
            x_2d = torch.roll(
                x_2d,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(2, 3),
            )
        windows = window_partition(x_2d, self.window_size)
        # windows shape: (B*num_windows, window_size*window_size, C)
        attention_mask = self.build_attention_mask(
            height,
            width,
            x.device,
            x.dtype,
        )
        windows = self.attn(windows, attention_mask)
        x_2d = window_reverse(windows, self.window_size, height, width)
        if self.shift_size > 0:
            x_2d = torch.roll(
                x_2d,
                shifts=(self.shift_size, self.shift_size),
                dims=(2, 3),
            )
        x = x_2d.permute(0, 2, 3, 1).reshape(batch_size, height * width, self.dim)
        # x shape: (B, H*W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class SwinBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        window_size: int = 4,
        mlp_ratio: float = 4.0,
        gamma_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                SwinBlock(
                    embed_dim,
                    window_size,
                    num_heads,
                    shift_size=0 if index % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                )
                for index in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Conv2d(embed_dim, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, in_channels, H, W)
        identity = x
        batch_size, _, height, width = x.shape
        x = self.in_proj(x)
        # x shape: (B, embed_dim, H, W)
        x = x.flatten(2).transpose(1, 2)
        # x shape: (B, H*W, embed_dim)
        for block in self.blocks:
            x = block(x, height, width)
            # x shape: (B, H*W, embed_dim)
        x = self.norm(x)
        x = x.transpose(1, 2).view(batch_size, -1, height, width)
        # x shape: (B, embed_dim, H, W)
        x = self.out_proj(x)
        # x shape: (B, in_channels, H, W), interpreted as a residual delta
        return identity + self.gamma * x


class UNetDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        attention: str = "none",
        use_gamma: bool = False,
        gamma_init: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if attention not in {"none", "ag"}:
            raise ValueError(f"Unsupported decoder attention: {attention}")
        self.attn = attention
        self.use_gamma = bool(use_gamma and attention != "none")
        fused_channels = in_channels + skip_channels
        self.block = ConvBlock(fused_channels, out_channels)
        self.drop = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        if attention == "ag":
            self.gate = AttentionGate(
                gating_channels=in_channels,
                skip_channels=skip_channels,
                hidden_channels=max(skip_channels // 2, 16),
            )
        if self.use_gamma:
            self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def _blend(self, base: torch.Tensor, refined: torch.Tensor) -> torch.Tensor:
        if self.use_gamma:
            return base + self.gamma * (refined - base)
        return refined

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # x shape: (B, in_channels, H_low, W_low)
        # skip shape: (B, skip_channels, H_skip, W_skip)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        # x shape: (B, in_channels, H_skip, W_skip)
        if self.attn == "ag":
            skip = self._blend(skip, self.gate(gating=x, skip=skip))
            # skip shape: (B, skip_channels, H_skip, W_skip)
        x = torch.cat([x, skip], dim=1)
        # x shape: (B, in_channels + skip_channels, H_skip, W_skip)
        x = self.block(x)
        # x shape: (B, out_channels, H_skip, W_skip)
        return self.drop(x)


class ResNetUNet(nn.Module):
    """U-Net decoder over a configurable ResNet encoder."""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = None,
        use_swin_bottleneck: bool = False,
        swin_embed_dim: int = 384,
        swin_depth: int = 4,
        swin_heads: int = 6,
        swin_window_size: int = 4,
        swin_gamma_init: float = 0.5,
        decoder_attention: str = "none",
        use_attention_gamma: bool = False,
        attention_gamma_init: float = 0.1,
        decoder_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.normalize = InputNormalize()
        self.encoder = ResNetEncoder(encoder_name)
        c1, c2, c3, c4, c5 = self.encoder.out_channels
        self.swin = (
            SwinBottleneck(
                in_channels=c5,
                embed_dim=swin_embed_dim,
                depth=swin_depth,
                num_heads=swin_heads,
                window_size=swin_window_size,
                gamma_init=swin_gamma_init,
            )
            if use_swin_bottleneck
            else None
        )
        decoder_kwargs = {
            "attention": decoder_attention,
            "use_gamma": use_attention_gamma,
            "gamma_init": attention_gamma_init,
            "dropout": decoder_dropout,
        }
        self.decode4 = UNetDecoderBlock(
            c5,
            c4,
            256,
            **decoder_kwargs,
        )
        self.decode3 = UNetDecoderBlock(
            256,
            c3,
            128,
            **decoder_kwargs,
        )
        self.decode2 = UNetDecoderBlock(
            128,
            c2,
            64,
            **decoder_kwargs,
        )
        self.decode1 = UNetDecoderBlock(
            64,
            c1,
            64,
            **decoder_kwargs,
        )
        self.head = nn.Sequential(
            ConvBlock(64, 32),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        if encoder_weights:
            loaded = self.encoder.load_external_weights(encoder_weights)
            print(f"Loaded {loaded} encoder tensors from {encoder_weights}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 3, H, W)
        output_size = x.shape[-2:]
        c1, c2, c3, c4, c5 = self.encoder(self.normalize(x))
        # c1/c2/c3/c4/c5 shapes: (B, C_i, H_i, W_i)
        if self.swin is not None:
            c5 = self.swin(c5)
            # c5 shape: (B, C5, H // 32, W // 32)
        d4 = self.decode4(c5, c4)
        # d4 shape: (B, 256, H // 16, W // 16)
        d3 = self.decode3(d4, c3)
        # d3 shape: (B, 128, H // 8, W // 8)
        d2 = self.decode2(d3, c2)
        # d2 shape: (B, 64, H // 4, W // 4)
        d1 = self.decode1(d2, c1)
        # d1 shape: (B, 64, H // 2, W // 2)
        x = F.interpolate(d1, size=output_size, mode="bilinear", align_corners=False)
        # x shape: (B, 64, H, W)
        logits = self.head(x)
        # logits shape: (B, 1, H, W)
        return logits


class BSCNet(ResNetUNet):
    """Final BSC-Net architecture used for the paper experiments.

    BSC-Net combines a ResNet-34 encoder, a Swin Transformer bottleneck and
    the U-Net decoder. Branch-aware sampling and the optimized vessel loss are
    training strategies and are configured by ``train.py``.
    """

    def __init__(self, encoder_weights: Optional[str] = None) -> None:
        super().__init__(
            encoder_name="resnet34",
            encoder_weights=encoder_weights,
            use_swin_bottleneck=True,
            swin_embed_dim=384,
            swin_depth=4,
            swin_heads=6,
            swin_window_size=4,
            swin_gamma_init=0.1,
            decoder_attention="none",
            decoder_dropout=0.0,
        )

