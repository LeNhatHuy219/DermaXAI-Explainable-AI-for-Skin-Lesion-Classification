from typing import Dict, List, Mapping, Tuple

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


class SoftJointCBM(nn.Module):
    """M3 - Soft Joint CBM: x -> 7 concept heads (soft probability) -> g (Linear) -> y.

    Backbone EfficientNet-B0 (giống M1) trích đặc trưng 1280 chiều, mỗi khái niệm có
    một concept head Linear riêng dự đoán logits trên không gian trạng thái Ki của nó.
    Xác suất mềm (softmax) của 7 head được nối lại thành vector bottleneck 28 chiều
    (mỗi nhóm con tổng = 1), sau đó đưa qua đầu chẩn đoán g. g là Linear thuần (không
    có hidden layer) để đối chiếu công bằng với M2 (Logistic Regression trên concept
    one-hot Ground Truth): sự khác biệt hiệu năng giữa M2 và M3 phản ánh đúng phần
    thông tin bị mất khi concept phải được dự đoán từ ảnh thay vì lấy từ Ground Truth.
    Toàn bộ mạng (backbone + concept heads + g) được huấn luyện đồng thời (joint) bằng
    một hàm mất mát tổng hợp, gradient của L_diagnosis truyền ngược xuyên qua các
    concept probability (soft, differentiable) tới tận backbone.
    """

    def __init__(
        self,
        concept_names: List[str],
        concept_num_classes: Dict[str, int],
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.concept_names = list(concept_names)
        self.concept_num_classes = dict(concept_num_classes)
        self.total_concept_states = sum(self.concept_num_classes[name] for name in self.concept_names)
        self.num_classes = num_classes

        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features  # 1280 chiều

        self.features = nn.Sequential(
            backbone.features,
            backbone.avgpool,
        )
        self.dropout = nn.Dropout(p=dropout)

        # Mỗi khái niệm lâm sàng có một concept head Linear riêng: 1280 -> Ki
        self.concept_heads = nn.ModuleDict({
            name: nn.Linear(in_features, self.concept_num_classes[name])
            for name in self.concept_names
        })

        # Đầu chẩn đoán g: Linear thuần trên vector concept soft 28 chiều
        self.diagnosis_head = nn.Linear(self.total_concept_states, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        return feat

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        feat = self.dropout(self.extract_features(x))

        # Logits thô từng concept head (dùng cho Multi-Head Cross-Entropy)
        concept_logits: Dict[str, torch.Tensor] = {
            name: self.concept_heads[name](feat) for name in self.concept_names
        }

        # Xác suất mềm từng nhóm, nối lại thành vector bottleneck 28 chiều
        concept_vector = torch.cat(
            [torch.softmax(concept_logits[name], dim=-1) for name in self.concept_names],
            dim=-1,
        )

        diag_logits = self.diagnosis_head(concept_vector)
        return diag_logits, concept_logits, concept_vector


def load_soft_joint_cbm_state_dict(model: SoftJointCBM, state_dict: Mapping[str, torch.Tensor]) -> None:
    """Load M3 checkpoint weights (giữ đối xứng với load_blackbox_state_dict của M1)."""
    model.load_state_dict(state_dict, strict=True)


def get_soft_joint_cbm(
    concept_names: List[str],
    concept_num_classes: Dict[str, int],
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> SoftJointCBM:
    # Khởi tạo mô hình M3 - Soft Joint CBM
    return SoftJointCBM(
        concept_names=concept_names,
        concept_num_classes=concept_num_classes,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )
