from typing import Mapping

import torch
import torch.nn as nn
import torchvision.models as models


class BlackBoxClassifier(nn.Module):
    # Mô hình phân loại trực tiếp từ ảnh sang nhãn chẩn đoán (Black-box EfficientNet-B0 Baseline)
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Nạp mạng EfficientNet-B0 chuẩn với trọng số ImageNet
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features  # 1280 chiều

        # Tách phần trích xuất đặc trưng (features + avgpool)
        self.features = nn.Sequential(
            backbone.features,
            backbone.avgpool,
        )

        # Lớp phân loại chẩn đoán nhị phân (0: Non-Melanoma, 1: Melanoma)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # Trích xuất vector đặc trưng bậc cao (1280 chiều) từ ảnh soi da
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dự đoán trực tiếp logits chẩn đoán từ ảnh đầu vào
        feat = self.extract_features(x)
        logits = self.classifier(feat)
        return logits


def load_blackbox_state_dict(model: BlackBoxClassifier, state_dict: Mapping[str, torch.Tensor]) -> None:
    """Load current or legacy M1 weights; legacy checkpoints registered backbone twice."""
    if any(key.startswith("backbone.") for key in state_dict):
        state_dict = {key: value for key, value in state_dict.items() if not key.startswith("backbone.")}
    model.load_state_dict(state_dict, strict=True)


def get_model(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    # Khởi tạo mô hình EfficientNet-B0 Black-box Baseline
    return BlackBoxClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )
