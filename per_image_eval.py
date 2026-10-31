"""
입력별 취약성 분석 (B, 보고서 6.6 "어떤 입력에서 특히 취약했는지")

person이 있는 VOC val 446장(lists/val_person.txt)에 대해 이미지마다
person IoU / Boundary IoU의 교집합·합집합 픽셀 수를 기록한다.
비교: 원본 OS8 520 vs OS8 384 (FP32, PyTorch).
  - 최종 모델의 정확도 손실은 대부분 해상도 축소에서 왔으므로(INT8 추가 손실 -0.01/-0.09) 해상도 축소 손실이 어디에 몰리는지 본다.
입력 특성: 이미지에서 사람이 차지하는 면적 비율, 사람 수(SegmentationObject 기준).

    python per_image_eval.py            # results/per_image.csv
"""
import csv
import ctypes
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common.infer import TorchPredictor, postprocess, preprocess
from common.metrics import BoundaryIoU
from common.voc import IGNORE, PERSON_CLASS, VOC_ROOT, load_image, load_label, read_ids
from structures import build

OUT = Path(__file__).resolve().parent / "results" / "per_image.csv"
SIZES = [520, 384]


def no_efficiency_mode():
    """창 없이 실행돼도 Windows 효율 모드(CPU 제한)가 걸리지 않게 (정확도와 무관, 속도만)."""
    if sys.platform != "win32":
        return

    class PPT(ctypes.Structure):
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]

    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
    s = PPT(1, 1, 0)
    k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(s), ctypes.sizeof(s))


def n_persons(img_id, gt):
    obj = np.array(Image.open(VOC_ROOT / "SegmentationObject" / f"{img_id}.png"))
    n = 0
    for k in np.unique(obj):
        if k in (0, IGNORE):
            continue
        cls = gt[(obj == k) & (gt != IGNORE)]
        if cls.size and np.bincount(cls).argmax() == PERSON_CLASS:
            n += 1
    return n


def main():
    no_efficiency_mode()
    ids = read_ids("val_person")
    predictor = TorchPredictor(build("os8"))
    OUT.parent.mkdir(exist_ok=True)
    start = time.perf_counter()
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "h", "w", "person_frac", "n_persons", "size", "p_inter", "p_union", "b_inter", "b_union"])
        for i, img_id in enumerate(ids, 1):
            image, gt = load_image(img_id), load_label(img_id)
            valid = gt != IGNORE
            frac = float((gt == PERSON_CLASS).sum() / valid.sum())
            npers = n_persons(img_id, gt)
            for size in SIZES:
                with torch.inference_mode():
                    pred = postprocess(predictor(preprocess(image, size, "short")), gt.shape)
                g, p = (gt == PERSON_CLASS) & valid, (pred == PERSON_CLASS) & valid
                b = BoundaryIoU()
                b.update(gt, pred)
                w.writerow([img_id, gt.shape[0], gt.shape[1], f"{frac:.5f}", npers, size,
                            int((g & p).sum()), int((g | p).sum()), b.inter, b.union])
            if i % 50 == 0:
                print(f"{i}/{len(ids)} ({(time.perf_counter()-start)/i:.2f}초/장)", flush=True)
    print("done", OUT)


if __name__ == "__main__":
    main()
