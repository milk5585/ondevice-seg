"""
Pascal VOC 2012 세그멘테이션 데이터 로더

데이터 위치는 셋 다 똑같이 맞춘다:
    C:/ondevice-seg/data/VOCdevkit/VOC2012
다른 위치(예: 학술부 재현 PC)면 환경변수 VOC_ROOT로 지정한다.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

VOC_ROOT = Path(os.environ.get("VOC_ROOT", "C:/ondevice-seg/data/VOCdevkit/VOC2012"))

# make_lists.py가 만든 고정 목록 (Git에 올려서 셋이 같은 목록을 쓴다)
LIST_DIR = Path(__file__).resolve().parent.parent / "lists"

NUM_CLASSES = 21
PERSON_CLASS = 15
# 정답 마스크에서 물체 테두리 등 '평가 제외' 픽셀
IGNORE = 255


def read_ids(split: str, root: Path = VOC_ROOT) -> list[str]:
    """
    이미지 ID 목록을 읽는다.
    - 'train' / 'val'  : VOC 공식 분할 (둘은 겹치지 않는다)
    - 그 외 이름        : lists/<이름>.txt (예: 'calib_train_person300')
    """
    fixed = LIST_DIR / f"{split}.txt"
    if fixed.exists():
        return fixed.read_text().split()
    official = root / "ImageSets" / "Segmentation" / f"{split}.txt"
    if not official.exists():
        raise FileNotFoundError(
            f"목록을 찾을 수 없습니다: {fixed} 또는 {official}\n"
            "VOC 데이터를 C:/ondevice-seg/data 에 받았는지 확인하세요."
        )
    return official.read_text().split()


def load_image(img_id: str, root: Path = VOC_ROOT) -> Image.Image:
    return Image.open(root / "JPEGImages" / f"{img_id}.jpg").convert("RGB")


def load_label(img_id: str, root: Path = VOC_ROOT) -> np.ndarray:
    """(H, W) uint8. 값은 0~20 클래스 번호, 255는 평가 제외."""
    return np.array(Image.open(root / "SegmentationClass" / f"{img_id}.png"))
