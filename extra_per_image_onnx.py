"""
[1101 추가 · 정확도, 어느 PC든 가능] 이미지별 교집합/합집합 기록 → 부트스트랩 신뢰구간용

person 포함 VOC val 446장(lists/val_person.txt)에 대해, 보고서와 같은 ONNX 모델 3개의
person / 배경 / Boundary IoU 교집합·합집합 픽셀 수를 이미지마다 기록한다.
  os8_520_fp32    : models/os8_fp32.onnx, 짧은 변 520 (Baseline)
  os8_384_fp32    : models/os8_fp32.onnx, 짧은 변 384 (해상도 축소만)
  os8_384_int8_pc : models/os8_int8_perchannel.onnx, 짧은 변 384 (최종)

    python extra_per_image_onnx.py        # → results/per_image_onnx.csv (약 15~25분)
"""
import csv
import time
from pathlib import Path

from common.infer import OnnxPredictor, postprocess, preprocess
from common.metrics import BoundaryIoU
from common.voc import IGNORE, PERSON_CLASS, load_image, load_label, read_ids

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "per_image_onnx.csv"
CONFIGS = [
    ("os8_520_fp32", "models/os8_fp32.onnx", 520),
    ("os8_384_fp32", "models/os8_fp32.onnx", 384),
    ("os8_384_int8_pc", "models/os8_int8_perchannel.onnx", 384),
]


def main():
    ids = read_ids("val_person")
    preds = {}
    for name, path, _ in CONFIGS:
        if path not in preds:
            preds[path] = OnnxPredictor(ROOT / path, 0)  # 정확도용이라 스레드 제한 없음
    OUT.parent.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "person_frac", "config", "p_inter", "p_union", "bg_inter", "bg_union", "b_inter", "b_union"])
        for i, img_id in enumerate(ids, 1):
            image, gt = load_image(img_id), load_label(img_id)
            valid = gt != IGNORE
            g = (gt == PERSON_CLASS) & valid
            gb = (gt != PERSON_CLASS) & valid
            frac = float(g.sum() / valid.sum())
            for name, path, size in CONFIGS:
                pred = postprocess(preds[path](preprocess(image, size, "short")), gt.shape)
                p = (pred == PERSON_CLASS) & valid
                pb = (pred != PERSON_CLASS) & valid
                b = BoundaryIoU()
                b.update(gt, pred)
                w.writerow([img_id, f"{frac:.5f}", name, int((g & p).sum()), int((g | p).sum()),
                            int((gb & pb).sum()), int((gb | pb).sum()), b.inter, b.union])
            if i % 50 == 0:
                print(f"{i}/{len(ids)} ({(time.perf_counter()-t0)/i:.2f}초/장)", flush=True)
    print("저장:", OUT)
    print("다음: python extra_bootstrap.py")


if __name__ == "__main__":
    main()
