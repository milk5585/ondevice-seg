"""
A 담당: 확정 구조(주력 os8 / 보조 os16_r6)를 FP32 ONNX로 변환하고 PyTorch와 일치하는지 확인

    python export_structs.py                    # os8, os16_r6 둘 다
    python export_structs.py --archs os8        # 하나만

결정 사항 (인수인계 문서 2장, B 구조 결정 문서 5장)
- 모델은 structures.build() 로 만든다 → 정확도·속도 측정과 같은 모델 보장
- 입력은 가변(H, W dynamic): 예선 평가는 '짧은 변 384, 비율 유지'라 사진마다 크기가 다름
- 출력은 'out' 하나 (1, 21, H, W)
- opset 18, **단일 파일**로 저장: 팀 공용 eval_voc.py / bench_speed.py 는 .onnx 크기만 재므로
  가중치를 외부 .onnx.data 로 빼면 크기가 0.4MB로 잘못 찍힘

결과
- models/os8_fp32.onnx, models/os16r6_fp32.onnx
- results/export_check.txt : PyTorch vs ONNX 오차·일치율 (보고서 '변환 검증' 근거)
"""

import argparse
import datetime
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from common.infer import preprocess
from common.voc import load_image, read_ids
from structures import build

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
LOG = ROOT / "results" / "export_check.txt"
OPSET = 18


class OutOnly(torch.nn.Module):
    """torchvision 모델은 dict를 반환 → ONNX 출력이 'out' 하나가 되도록 감싼다."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)["out"]


def file_name(arch):
    return MODEL_DIR / f"{arch.replace('_', '')}_fp32.onnx"


def is_dynamic(path):
    dims = onnx.load(str(path), load_external_data=False).graph.input[0].type.tensor_type.shape.dim
    return all(not d.HasField("dim_value") for d in dims[2:4])


def export(arch, out):
    model = OutOnly(build(arch)).eval()
    x = torch.randn(1, 3, 384, 524)  # 예선 평가의 대표 입력 크기 (VOC 가로 사진, 짧은 변 384)
    tmp = out.with_name(out.stem + "_tmp.onnx")

    try:
        from torch.export import Dim

        torch.onnx.export(
            model, (x,), str(tmp), dynamo=True, opset_version=OPSET,
            input_names=["input"], output_names=["out"],
            dynamic_shapes={"x": {2: Dim.DYNAMIC, 3: Dim.DYNAMIC}},
        )
        if not is_dynamic(tmp):
            raise RuntimeError("dynamo export 결과 입력 크기가 고정됨")
        how = "dynamo"
    except Exception as e:
        print(f"  dynamo export 실패 → 기존(TorchScript) 방식으로 재시도: {str(e)[:200]}")
        torch.onnx.export(
            model, (x,), str(tmp), dynamo=False, opset_version=OPSET,
            input_names=["input"], output_names=["out"],
            dynamic_axes={"input": {2: "h", 3: "w"}, "out": {2: "h", 3: "w"}},
        )
        how = "torchscript"

    # 외부 .data 로 저장됐더라도 다시 읽어서 단일 파일로 합쳐 저장
    m = onnx.load(str(tmp))
    if m.ir_version > 10:  # onnxruntime 1.30은 IR 13까지만 읽음. opset 18엔 IR 10이면 충분
        m.ir_version = 10
    onnx.save_model(m, str(out), save_as_external_data=False)
    onnx.checker.check_model(str(out))
    for p in (tmp, Path(str(tmp) + ".data")):
        if p.exists():
            p.unlink()
    assert is_dynamic(out), "입력 H, W가 dynamic이 아님"
    assert not Path(str(out) + ".data").exists()
    return model, how


def verify(model, out, n_images):
    """VOC val person 이미지 몇 장 + 520 크기 1장(가변 입력 동작 확인)으로 PyTorch와 비교."""
    try:
        ids = read_ids("val_person")[:n_images]
        inputs = [preprocess(load_image(i), 384, "short") for i in ids]
        inputs.append(preprocess(load_image(ids[0]), 520, "short"))
        src = f"VOC val_person 앞 {len(ids)}장 @384 + 1장 @520"
    except Exception as e:
        print(f"  (VOC 이미지 없음 → 랜덤 입력으로 검증: {e})")
        rng = np.random.default_rng(0)
        inputs = [rng.standard_normal((1, 3, 384, w), dtype=np.float32) for w in (524, 512, 384)]
        inputs.append(rng.standard_normal((1, 3, 520, 710), dtype=np.float32))
        src = "랜덤 입력 4개 (VOC 받은 뒤 다시 실행 권장)"

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    max_diff, agree, total = 0.0, 0, 0
    for x in inputs:
        with torch.inference_mode():
            ref = model(torch.from_numpy(x)).numpy()
        got = sess.run(None, {"input": x})[0]
        assert ref.shape == got.shape, f"출력 크기 불일치 {ref.shape} vs {got.shape}"
        max_diff = max(max_diff, float(np.abs(ref - got).max()))
        agree += int((ref.argmax(1) == got.argmax(1)).sum())
        total += ref.shape[2] * ref.shape[3]
    return src, max_diff, agree / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=["os8", "os16_r6"])
    ap.add_argument("--n", type=int, default=5, help="검증에 쓸 val 이미지 수")
    args = ap.parse_args()

    MODEL_DIR.mkdir(exist_ok=True)
    LOG.parent.mkdir(exist_ok=True)
    lines = []
    for arch in args.archs:
        out = file_name(arch)
        print(f"[{arch}] export → {out.relative_to(ROOT)}")
        model, how = export(arch, out)
        mb = out.stat().st_size / 1024**2
        src, diff, agree = verify(model, out, args.n)
        ok = agree >= 0.999 and diff < 1e-3
        line = (
            f"{datetime.datetime.now():%Y-%m-%d %H:%M} | {arch} | {out.name} | {mb:.1f}MB | "
            f"opset {OPSET} ({how}) | 입력 dynamic | {src} | 최대 오차 {diff:.2e} | "
            f"픽셀 클래스 일치 {agree*100:.4f}% | {'OK' if ok else '확인 필요'}"
        )
        print("  " + line)
        lines.append(line)

    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"# torch {torch.__version__} / onnx {onnx.__version__} / onnxruntime {ort.__version__}\n")
        f.write("\n".join(lines) + "\n")
    print(f"\n저장: {LOG.relative_to(ROOT)}")
    print("다음: python eval_voc.py --name os8_384_fp32 --onnx models/os8_fp32.onnx --size 384 --limit 50")


if __name__ == "__main__":
    main()
