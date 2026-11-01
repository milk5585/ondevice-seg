"""
[1101 추가 실험] ASPP 전역 평균 풀링: ReduceMean(axes=[-1,-2], keepdims=1) → GlobalAveragePool 교체

프로파일링(extra_profile_ops.py)에서 INT8 모델의 ReduceMean 1개가 11.9 ms(전체 8.3%)로 두 번째 병목이었다.
수학적으로 같은 GlobalAveragePool로 바꾸면 ORT가 정수 전역 평균 풀링 등으로 처리할 수 있는지 확인한다.
가중치·구조는 그대로이며 연산자 표기만 바꾼다 → FP32 출력이 원본과 같은지 검증한다.

    python extra_gap_swap.py
    → models/os8gap_fp32.onnx
    다음: python ptq_quantize.py --models models/os8gap_fp32.onnx --modes perchannel   (→ models/os8gap_int8_perchannel.onnx)
"""
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parent
SRC, DST = ROOT / "models/os8_fp32.onnx", ROOT / "models/os8gap_fp32.onnx"

m = onnx.load(str(SRC))
g = m.graph
init = {i.name: i for i in g.initializer}
n_swap = 0
for idx, n in enumerate(list(g.node)):
    if n.op_type != "ReduceMean":
        continue
    attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
    axes = sorted(numpy_helper.to_array(init[n.input[1]]).tolist())
    assert attrs.get("keepdims", 1) == 1 and axes in ([-2, -1], [2, 3]), (n.name, attrs, axes)
    new = helper.make_node("GlobalAveragePool", [n.input[0]], list(n.output), name=n.name + "_gap")
    g.node.remove(n)
    g.node.insert(idx, new)
    n_swap += 1
print(f"교체한 ReduceMean: {n_swap}개")
onnx.checker.check_model(m)
onnx.save_model(m, str(DST), save_as_external_data=False)

# 검증: 같은 입력에서 원본과 출력 비교 (두 가지 크기, 가변 입력 확인)
rng = np.random.default_rng(0)
a = ort.InferenceSession(str(SRC), providers=["CPUExecutionProvider"])
b = ort.InferenceSession(str(DST), providers=["CPUExecutionProvider"])
for hw in [(384, 524), (384, 384)]:
    x = rng.standard_normal((1, 3, *hw), dtype=np.float32)
    ya, yb = a.run(None, {"input": x})[0], b.run(None, {"input": x})[0]
    print(f"입력 {hw}: 최대 오차 {np.abs(ya - yb).max():.2e}, 픽셀 클래스 일치 {100*(ya.argmax(1) == yb.argmax(1)).mean():.4f}%")
print("저장:", DST)
