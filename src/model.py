"""ResNet9 (and a ResNet6 baseline) for plant disease classification.

Architecture follows the project's research paper: convolution blocks with
batch normalisation and ReLU, two residual blocks with skip connections to
defeat vanishing/exploding gradients, max pooling for downsampling, and a
fully connected output layer (softmax is applied at inference, not in the
module, because the loss function expects raw logits).

    conv1: 3   -> 64
    conv2: 64  -> 128, MaxPool(4)
    res1:  [128->128, 128->128]  + skip
    conv3: 128 -> 256, MaxPool(4)
    conv4: 256 -> 512, MaxPool(4)
    res2:  [512->512, 512->512]  + skip
    classifier: AdaptiveMaxPool2d(1) -> Flatten -> Linear(512, num_classes)

One deliberate deviation from the canonical implementation: the classifier ends
in AdaptiveMaxPool2d(1) rather than a fixed MaxPool2d(4). At 256x256 the two
are numerically identical (256 -> 64 -> 16 -> 4 -> 1), but the fixed version
crashes at 64x64 because the feature map reaches 1x1 before the final pool.
The adaptive version lets the same weights-compatible definition train at 64,
128 or 256 px, which is what makes CPU training viable on modest hardware.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels: int, out_channels: int, pool: bool = False) -> nn.Sequential:
    """Conv3x3 -> BatchNorm -> ReLU, optionally followed by MaxPool(4)."""
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class ClassificationBase(nn.Module):
    """Shared training/validation steps for the models below."""

    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        return F.cross_entropy(out, labels)

    @torch.no_grad()
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return {"val_loss": loss.detach(), "val_acc": accuracy(out, labels)}

    def validation_epoch_end(self, outputs):
        return {
            "val_loss": torch.stack([o["val_loss"] for o in outputs]).mean().item(),
            "val_acc": torch.stack([o["val_acc"] for o in outputs]).mean().item(),
        }


def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ResNet9(ClassificationBase):
    """The paper's model: 9 weight layers, two residual blocks."""

    def __init__(self, in_channels: int = 3, num_classes: int = 38):
        super().__init__()
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out          # skip connection
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out          # skip connection
        return self.classifier(out)


class ResNet6(ClassificationBase):
    """Shallower baseline used for the paper's ResNet6-vs-ResNet9 comparison.

    6 weight layers: conv1, conv2, res1 (x2), conv3, linear.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 38):
        super().__init__()
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv3 = conv_block(128, 256, pool=True)

        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out          # skip connection
        out = self.conv3(out)
        return self.classifier(out)


MODELS = {"resnet9": ResNet9, "resnet6": ResNet6}


def build_model(name: str = "resnet9", num_classes: int = 38) -> ClassificationBase:
    key = name.lower()
    if key not in MODELS:
        raise ValueError("unknown model %r (choose from %s)" % (name, sorted(MODELS)))
    return MODELS[key](num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
