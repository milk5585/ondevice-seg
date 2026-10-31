"""
비교군 MobileNetV3 두 모델을 FP32 ONNX(입력 크기 가변)로 내보내고, PyTorch 출력과 같은지 확인한다. (B)

    python export_mobilenet.py
결과: models/extra/mbv3_deeplab_fp32.onnx, models/extra/mbv3_lraspp_fp32.onnx
"""
from pathlib import Path

import numpy as np
import torch

from common.infer import OnnxPredictor, TorchPredictor, preprocess
from common.voc import load_image, read_ids
from compare_mobilenet import MODELS

OUT = Path(__file__).resolve().parent / "models" / "extra"
NAMES = {"deeplabv3_mbv3": "mbv3_deeplab_fp32", "lraspp_mbv3": "mbv3_lraspp_fp32"}


class OutOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = read_ids("val")[:5]
    for key, make in MODELS.items():
        tp = TorchPredictor(make())  # 보조 출력 제거 + eval
        path = OUT / f"{NAMES[key]}.onnx"
        torch.onnx.export(
            OutOnly(tp.model).eval(), torch.randn(1, 3, 384, 524), str(path),
            opset_version=17, input_names=["input"], output_names=["out"],
            dynamic_axes={"input": {2: "h", 3: "w"}, "out": {2: "h", 3: "w"}},
            do_constant_folding=True, dynamo=False,
        )
        op = OnnxPredictor(path)
        diff, agree = 0.0, []
        for img_id in ids:
            for size in (384, 520):
                x = preprocess(load_image(img_id), size, "short")
                a, b = tp(x), op(x)
                diff = max(diff, float(np.abs(a - b).max()))
                agree.append(float((a.argmax(1) == b.argmax(1)).mean()))
        print(f"{path.name}: {path.stat().st_size/2**20:.1f} MB, 최대 점수 차 {diff:.2e}, 픽셀 클래스 일치 {np.mean(agree)*100:.3f}%")


if __name__ == "__main__":
    main()
