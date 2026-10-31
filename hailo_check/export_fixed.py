"""Hailo 파싱 확인용 고정 입력 ONNX (제출용 아님). 설정은 9/27 파싱 성공 때와 동일 (opset 11, 레거시 exporter)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from structures import build


class OutOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


size = int(sys.argv[1]) if len(sys.argv) > 1 else 384
for name in ["os8", "os16_r6"]:
    path = Path(__file__).parent / f"{name}_{size}x{size}.onnx"
    torch.onnx.export(
        OutOnly(build(name)), torch.randn(1, 3, size, size), str(path),
        opset_version=11, input_names=["input"], output_names=["out"],
        do_constant_folding=True, dynamo=False,
    )
    print("exported", path.name, f"{path.stat().st_size / 1024**2:.1f} MB")
