# Định nghĩa chung các Concept Bottleneck Models: M3, M4-ST và M4-SG.
# Cả ba dùng EfficientNet-B0; mặc định head Linear, khác bottleneck/gradient.
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def make_diagnosis_head(input_dim: int, num_classes: int = 2,
                        architecture: str = "linear") -> nn.Module:
    # Head chỉ nhận concept: Linear là baseline; MLP128 thử tương tác phi tuyến.
    # MLP theo cấu trúc Nápoles: Linear -> LayerNorm -> ReLU -> dropout -> Linear.
    if architecture == "linear":
        return nn.Linear(input_dim, num_classes)
    if architecture == "mlp128":
        return nn.Sequential(nn.Linear(input_dim, 128), nn.LayerNorm(128),
                             nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes))
    raise ValueError("Unknown diagnosis head; expected linear or mlp128")


# M3: ảnh -> xác suất concept mềm -> logits chẩn đoán; huấn luyện joint.
class SoftJointCBM(nn.Module):
    def __init__(
        self,
        concept_names: List[str],
        concept_num_classes: Dict[str, int],
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
        diagnosis_head: str = "linear",
    ):
        super().__init__()
        self.concept_names = list(concept_names)
        self.concept_num_classes = dict(concept_num_classes)
        self.total_concept_states = sum(self.concept_num_classes[name] for name in self.concept_names)
        self.num_classes = num_classes

        # Backbone ImageNet dùng chung; Derm7pt có 7 concept groups, tổng 28 states.
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features  # 1280 chiều

        # Bỏ classifier gốc, giữ features và global average pooling để lấy vector 1280D.
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

        # Diagnosis head chỉ nhận bottleneck, không nhận trực tiếp đặc trưng ảnh.
        # M3 chọn Linear/MLP trên xác suất mềm; M4 dùng cùng head trên one-hot.
        self.diagnosis_head = make_diagnosis_head(self.total_concept_states, num_classes, diagnosis_head)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        # Chuyển output pooling (batch, 1280, 1, 1) thành (batch, 1280).
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

        # Softmax riêng mỗi group để tổng xác suất của group bằng 1.
        # Nối các groups thành bottleneck (batch, 28) trong cấu hình Derm7pt.
        concept_vector = torch.cat(
            [torch.softmax(concept_logits[name], dim=-1) for name in self.concept_names],
            dim=-1,
        )

        # Softmax khả vi: diagnosis loss truyền gradient về concept heads/backbone.
        diag_logits = self.diagnosis_head(concept_vector)
        # Trả thêm logits concept cho concept loss và vector cho đánh giá/intervention.
        return diag_logits, concept_logits, concept_vector


def hard_categorical(logits: torch.Tensor, straight_through: bool = False) -> torch.Tensor:
    # Chọn state xác suất cao nhất của một group và mã hóa thành one-hot.
    probabilities = logits.softmax(dim=-1)
    hard = F.one_hot(probabilities.argmax(dim=-1), logits.shape[-1]).to(logits.dtype)

    # Straight-through: forward dùng one-hot, backward dùng gradient xấp xỉ của softmax.
    # probabilities - probabilities.detach() bằng 0 ở forward nhưng còn gradient.
    # Trừ trước rồi mới cộng hard giữ các giá trị forward đúng 0/1, tránh sai số làm tròn.
    # Đây là surrogate gradient có bias, với temperature = 1; không dùng nhiễu Gumbel.
    return hard + (probabilities - probabilities.detach()) if straight_through else hard


# M4-ST kế thừa toàn bộ kiến trúc M3, chỉ thay bottleneck mềm bằng one-hot.
# Diagnosis loss vẫn cập nhật concept heads/backbone nhờ straight-through khi train.
# Với head mặc định, tên/shape tham số giữ như M3; runner kiểm tra protocol/head.
class HardJointCBM(SoftJointCBM):
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        features = self.dropout(self.extract_features(x))
        logits = {name: self.concept_heads[name](features) for name in self.concept_names}

        # self.training bật surrogate gradient khi train; eval dùng argmax thuần.
        # Mỗi group chọn đúng một state; vector Derm7pt có 28 chiều và đúng 7 bit 1.
        vector = torch.cat([hard_categorical(logits[name], self.training)
                            for name in self.concept_names], dim=-1)
        return self.diagnosis_head(vector), logits, vector


# M4-SG dùng cùng kiến trúc và hard forward như M4-ST, nhưng chặn diagnosis gradient.
# Concept loss vẫn cập nhật concept heads/backbone từ logits trả về trong cùng bước train.
class HardStopGradientCBM(SoftJointCBM):
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        features = self.dropout(self.extract_features(x))
        logits = {name: self.concept_heads[name](features) for name in self.concept_names}

        # Argmax/one-hot không dùng straight-through; detach tách bottleneck khỏi autograd.
        # Diagnosis loss chỉ cập nhật diagnosis head; concept logits vẫn giữ gradient.
        vector = torch.cat([hard_categorical(logits[name], straight_through=False)
                            for name in self.concept_names], dim=-1).detach()
        return self.diagnosis_head(vector), logits, vector


def get_hard_joint_cbm(concept_names: List[str], concept_num_classes: Dict[str, int],
                       num_classes: int = 2, pretrained: bool = True,
                       dropout: float = 0.2, diagnosis_head: str = "linear") -> HardJointCBM:
    # Factory M4-ST giữ nguyên giao diện khởi tạo để runners/tests dùng chung.
    return HardJointCBM(concept_names, concept_num_classes, num_classes, pretrained, dropout, diagnosis_head)


def get_hard_stop_gradient_cbm(concept_names: List[str], concept_num_classes: Dict[str, int],
                               num_classes: int = 2, pretrained: bool = True,
                               dropout: float = 0.2, diagnosis_head: str = "linear") -> HardStopGradientCBM:
    # Factory M4-SG có cùng cấu hình kiến trúc; khác cơ chế gradient trong forward.
    return HardStopGradientCBM(concept_names, concept_num_classes, num_classes, pretrained, dropout, diagnosis_head)
