"""
전처리 / 후처리 / 추론기

PyTorch와 ONNX가 '완전히 같은 입력'을 받도록 전처리를 여기 하나로 통일한다.
전처리가 사람마다 다르면 차이가 기법 때문인지 전처리 때문인지 구분이 안 된다.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image: Image.Image, size: int = 520, mode: str = "short") -> np.ndarray:
    """
    mode
      'short'  : 짧은 변을 size로 맞추고 비율 유지 (torchvision 공식과 같음)
      'square' : size × size로 고정 (입력 크기가 고정된 ONNX용)
    반환: (1, 3, H, W) float32
    """
    w, h = image.size
    if mode == "short":
        # torchvision F.resize(img, [size])와 같은 반올림 규칙
        if w <= h:
            new = (size, int(size * h / w))
        else:
            new = (int(size * w / h), size)
    elif mode == "square":
        new = (size, size)
    else:
        raise ValueError(f"mode는 'short' 또는 'square': {mode}")

    x = np.asarray(image.resize(new, Image.BILINEAR), dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def postprocess(logits: np.ndarray, orig_hw: tuple[int, int]) -> np.ndarray:
    """(1, C, h, w) 점수 → 원본 크기 (H, W) 클래스 맵. 해상도가 달라도 항상 원본 크기에서 채점한다."""
    t = torch.from_numpy(logits)
    t = F.interpolate(t, size=orig_hw, mode="bilinear", align_corners=False)
    return t.argmax(1)[0].numpy().astype(np.uint8)


class TorchPredictor:
    """
    PyTorch 모델용. model을 안 넘기면 사전학습 원본(베이스라인)을 쓴다.
    B는 OS16 등으로 바꾼 모델을 model=으로 넘기면 된다.
    """

    name = "torch"

    def __init__(self, model: torch.nn.Module | None = None):
        if model is None:
            from torchvision.models.segmentation import (
                DeepLabV3_ResNet50_Weights,
                deeplabv3_resnet50,
            )

            model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        # 보조 출력(aux)은 학습용이라 평가 때는 계산만 낭비. 'out' 결과는 동일하다.
        if getattr(model, "aux_classifier", None) is not None:
            model.aux_classifier = None
        self.model = model.eval()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return self.model(torch.from_numpy(x))["out"].numpy()


class OnnxPredictor:
    """ONNX 모델용. 첫 번째 출력이 (1, 21, H, W) 점수('out')여야 한다."""

    name = "onnx"

    def __init__(self, path: str | Path, threads: int = 0):
        import onnxruntime as ort

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.path = Path(path)
        self.sess = ort.InferenceSession(
            str(path), so, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.input_name: x})[0]
