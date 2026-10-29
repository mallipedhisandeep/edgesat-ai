"""Compact two-head network for EdgeSat AI."""
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class QACNet(nn.Module):
    """Returns land-cover logits (10) and synthetic quality logits (clean/corrupt)."""
    def __init__(self, num_classes: int = 10, width: int = 48):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, 2, 1, bias=False),
                                  nn.BatchNorm2d(width), nn.ReLU6(inplace=True))
        self.features = nn.Sequential(
            ConvBlock(width, width, 1), ConvBlock(width, width * 2, 2),
            ConvBlock(width * 2, width * 2, 1), ConvBlock(width * 2, width * 3, 2),
            ConvBlock(width * 3, width * 3, 1), ConvBlock(width * 3, width * 4, 2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.landcover_head = nn.Linear(width * 4, num_classes)
        self.quality_head = nn.Linear(width * 4, 2)

    def forward(self, x):
        z = self.features(self.stem(x))
        return self.landcover_head(z), self.quality_head(z)


class QACMobileNet(nn.Module):
    """Transfer-learning variant using MobileNetV3-Small as a compact backbone."""
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)
        feature_size = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()
        self.landcover_head = nn.Linear(feature_size, num_classes)
        self.quality_head = nn.Linear(feature_size, 2)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        z = self.backbone((x - self.image_mean) / self.image_std)
        return self.landcover_head(z), self.quality_head(z)
