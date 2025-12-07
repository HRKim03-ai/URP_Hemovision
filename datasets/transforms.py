from typing import Literal

from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(
    split: Literal["train", "val", "test"], image_size: int = 224, modality: str = "nail"
):
    """
    Build transforms for a given split and modality.

    Note: modality is currently unused but kept for future customization.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if split == "train":
        # On-the-fly augmentation: 매번 이미지 로드 시 랜덤 적용
        # Raw 옵션: 20% 확률로 augmentation 없이 원본만 사용
        # 나머지 80%는 하나 이상의 augmentation을 독립적으로 적용
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                # 80% 확률로 augmentation 적용, 20%는 Raw (원본만)
                transforms.RandomApply(
                    [
                        # 각 augmentation을 독립적으로 적용 (복수 적용 가능)
                        transforms.RandomApply(
                            [transforms.RandomHorizontalFlip()], p=0.5
                        ),
                        transforms.RandomApply(
                            [transforms.RandomVerticalFlip()], p=0.5
                        ),
                        transforms.RandomApply(
                            [
                                transforms.RandomResizedCrop(
                                    image_size, scale=(0.9, 1.0)
                                )
                            ],
                            p=0.5,
                        ),
                        transforms.RandomApply(
                            [transforms.RandomRotation(degrees=(-10, 0))], p=0.5
                        ),
                        transforms.RandomApply(
                            [transforms.RandomRotation(degrees=(0, 10))], p=0.5
                        ),
                    ],
                    p=0.8,  # 80% 확률로 augmentation 적용, 20%는 Raw
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )


