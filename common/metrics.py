"""
정확도 지표

혼동행렬(21×21) 하나만 쌓으면
  - 21클래스 mIoU (torchvision 공식 방식과 같은 계산)
  - person / 배경 2클래스 mIoU (우리 과제의 주 지표)
를 모델을 두 번 돌리지 않고 둘 다 얻을 수 있다.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .voc import IGNORE, NUM_CLASSES, PERSON_CLASS


def _iou(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, tp / union, np.nan)


class SegMetric:
    def __init__(self, num_classes: int = NUM_CLASSES):
        self.n = num_classes
        # cm[정답, 예측] = 픽셀 수
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, gt: np.ndarray, pred: np.ndarray) -> None:
        """gt, pred 모두 원본 이미지 크기 (H, W). 255 픽셀은 제외한다."""
        if gt.shape != pred.shape:
            raise ValueError(
                f"크기가 다릅니다: 정답 {gt.shape}, 예측 {pred.shape}. "
                "예측을 원본 크기로 복원한 뒤 넣어야 합니다."
            )
        valid = gt != IGNORE
        idx = gt[valid].astype(np.int64) * self.n + pred[valid].astype(np.int64)
        self.cm += np.bincount(idx, minlength=self.n**2).reshape(self.n, self.n)

    def person_cm(self) -> np.ndarray:
        """21×21을 [배경(=person 아닌 전부), person] 2×2로 합친다."""
        p, cm = PERSON_CLASS, self.cm
        pp = cm[p, p]
        p_as_bg = cm[p, :].sum() - pp  # 사람인데 사람 아니라고 함
        bg_as_p = cm[:, p].sum() - pp  # 사람 아닌데 사람이라고 함
        bb = cm.sum() - pp - p_as_bg - bg_as_p
        return np.array([[bb, bg_as_p], [p_as_bg, pp]])

    def summary(self) -> dict:
        iou21 = _iou(self.cm)
        iou2 = _iou(self.person_cm())
        return {
            "miou21": float(np.nanmean(iou21)),
            "miou_person": float(np.nanmean(iou2)),  # 배경·person 평균 (주 지표)
            "iou_person": float(iou2[1]),
            "iou_background": float(iou2[0]),
        }


def _boundary(mask: np.ndarray, ratio: float) -> np.ndarray:
    """
    마스크의 테두리 띠 (Boundary IoU 논문 공식 구현과 같은 방식).
    띠 두께 d = 이미지 대각선 × ratio. 마스크에서 d만큼 깎아낸(침식) 부분을 뺀다.
    공식 구현처럼 이미지 가장자리에 닿은 부분도 테두리로 친다.
    """
    h, w = mask.shape
    d = max(1, int(round(ratio * np.hypot(h, w))))
    # 바깥 1픽셀을 0으로 둘러야 이미지 가장자리도 테두리가 된다
    m = torch.from_numpy(np.pad(mask, 1).astype(np.float32))[None, None]
    # 침식 = 반전 → (2d+1) 최대값 풀링(=팽창) → 반전
    eroded = 1 - F.max_pool2d(1 - m, 2 * d + 1, stride=1, padding=d)
    eroded = eroded[0, 0, 1:-1, 1:-1].numpy().astype(bool)
    return mask & ~eroded


class BoundaryIoU:
    """
    person 테두리 IoU. 전체 이미지의 교집합·합집합 픽셀을 모두 더해서 마지막에 한 번 나눈다
    (mIoU와 같은 방식이라 작은 이미지 하나가 점수를 흔들지 않는다).
    ratio 0.02는 Boundary IoU 논문(Cheng et al., CVPR 2021) 기본값.
    """

    def __init__(self, cls: int = PERSON_CLASS, ratio: float = 0.02):
        self.cls, self.ratio = cls, ratio
        self.inter = 0
        self.union = 0

    def update(self, gt: np.ndarray, pred: np.ndarray) -> None:
        valid = gt != IGNORE
        g = _boundary(gt == self.cls, self.ratio) & valid
        p = _boundary(pred == self.cls, self.ratio) & valid
        self.inter += int((g & p).sum())
        self.union += int((g | p).sum())

    def summary(self) -> float:
        return self.inter / self.union if self.union else float("nan")
