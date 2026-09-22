import random
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch

try:
    import torchvision.transforms as T
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


class LetterboxResize:
    # Thay đổi kích thước ảnh giữ nguyên tỷ lệ khung hình và chèn viền đen đối xứng
    def __init__(self, target_size: Union[int, Tuple[int, int]] = (224, 224), fill_color=(0, 0, 0)):
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = target_size
        self.fill_color = fill_color

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        target_w, target_h = self.target_size
        scale = min(target_w / w, target_h / h)
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))

        resized_img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

        padded_img = Image.new("RGB", (target_w, target_h), self.fill_color)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        padded_img.paste(resized_img, (pad_x, pad_y))
        return padded_img


# Các lớp dự phòng dùng PIL thuần khi môi trường thiếu torchvision
class PILRandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class PILRandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class PILRandomRotation:
    def __init__(self, degrees: float = 15.0, fill_color=(0, 0, 0)):
        self.degrees = degrees
        self.fill_color = fill_color

    def __call__(self, img: Image.Image) -> Image.Image:
        angle = random.uniform(-self.degrees, self.degrees)
        return img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=self.fill_color)


class PILToTensor:
    def __call__(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)


class TensorNormalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std


class SimpleCompose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def get_transforms(
    split: str = "train",
    target_size: int = 224,
    augment: bool = True,
    augmentation_preset: str = "legacy_letterbox",
):
    # Pipeline biến đổi ảnh cho Derm7pt (chuẩn hóa ImageNet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    is_train = split.lower() in ["train", "training"]
    if augmentation_preset not in {"legacy_letterbox", "comparison"}:
        raise ValueError(f"Unknown augmentation preset: {augmentation_preset}")

    if is_train and augment and augmentation_preset == "comparison":
        if not HAS_TORCHVISION:
            raise ImportError("The comparison augmentation preset requires torchvision")
        # Cố định chính xác các tham số này cho M1/M3/M5 khi so sánh backbone.
        return T.Compose([
            T.RandomResizedCrop(
                (target_size, target_size),
                scale=(0.85, 1.0),
                ratio=(0.9, 1.1),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            T.RandomRotation(degrees=15, fill=0),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    if HAS_TORCHVISION:
        if is_train and augment:
            # Thứ tự: LetterboxResize trước RandomRotation là lựa chọn thiết kế có chủ đích.
            # Viền padding đen sẽ bị xoay theo, tạo ra vùng góc nhỏ lệch màu.
            # Ảnh hưởng không đáng kể với góc xoay nhỏ (15 độ) và đơn giản hóa pipeline.
            return T.Compose([
                LetterboxResize((target_size, target_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=15, fill=0),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])
        else:
            return T.Compose([
                LetterboxResize((target_size, target_size)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])
    else:
        if is_train and augment:
            return SimpleCompose([
                LetterboxResize((target_size, target_size)),
                PILRandomHorizontalFlip(p=0.5),
                PILRandomVerticalFlip(p=0.5),
                PILRandomRotation(degrees=15),
                PILToTensor(),
                TensorNormalize(mean=mean, std=std),
            ])
        else:
            return SimpleCompose([
                LetterboxResize((target_size, target_size)),
                PILToTensor(),
                TensorNormalize(mean=mean, std=std),
            ])


if __name__ == "__main__":
    dummy_img = Image.new("RGB", (768, 512), color=(200, 100, 50))
    letterbox = LetterboxResize((224, 224), fill_color=(128, 128, 128))
    out_img = letterbox(dummy_img)
    assert out_img.size == (224, 224)

    train_tf = get_transforms("train", 224, augment=True)
    tensor_out = train_tf(dummy_img)
    print(f"Output shape: {tuple(tensor_out.shape)}")
