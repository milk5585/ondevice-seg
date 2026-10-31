"""
고정 입력 크기에서의 정확도 (B, 본선 Hailo 대비)

Hailo는 입력 크기 하나로 컴파일하므로, 예선의 가변 입력(짧은 변 384)을 고정 크기로 바꿨을 때
정확도가 얼마나 달라지는지 미리 잰다.

입력 방식 (canvas = 모델에 들어가는 고정 크기 H×W)
  square        : 비율 무시하고 canvas로 늘리기 (common.infer의 'square'와 같음)
  letterbox     : 비율 유지해서 canvas 안에 최대로 줄이고, 남는 곳은 0(정규화 후 = 평균색)으로 채움.
                  결과에서 채운 부분을 잘라낸 뒤 원본 크기로 되돌려 채점한다.

채점은 기존과 똑같이 원본 크기에서, VOC val 1449장 전체, person mIoU / Boundary IoU / 21클래스 mIoU.

사용법 (학과 PC)
    python fixed_input_eval.py                       # 기본 6조합 전부 (이미 끝난 조합은 건너뜀)
    python fixed_input_eval.py --limit 20            # 빠른 동작 확인
    python fixed_input_eval.py --torch               # ONNX 대신 structures.py의 PyTorch 모델 사용
결과: results/fixed_input.csv
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common.infer import MEAN, STD, OnnxPredictor, TorchPredictor, postprocess, preprocess
from common.metrics import BoundaryIoU, SegMetric
from common.voc import load_image, load_label, read_ids

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "fixed_input.csv"
FIELDS = ["date", "model", "backend", "mode", "canvas_hw", "split", "n_images",
          "miou_person", "iou_person", "biou_person", "miou21", "sec_per_img"]

# (이름, ONNX 파일, structures.py 이름)
MODELS = [("os8", "models/os8_fp32.onnx", "os8"),
          ("os16r6", "models/os16r6_fp32.onnx", "os16_r6")]
# (mode, canvas H, canvas W)
CASES = [("square", 384, 384), ("letterbox", 384, 384), ("letterbox", 384, 512)]


def letterbox(image: Image.Image, ch: int, cw: int):
    """비율 유지 축소 + 가운데 배치. 반환: 입력 텐서, 이미지가 놓인 영역 (top, left, h, w)."""
    w, h = image.size
    s = min(ch / h, cw / w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    img = np.asarray(image.resize((nw, nh), Image.BILINEAR), dtype=np.float32) / 255.0
    img = (img - MEAN) / STD
    canvas = np.zeros((ch, cw, 3), dtype=np.float32)  # 정규화 후 0 = 평균색
    top, left = (ch - nh) // 2, (cw - nw) // 2
    canvas[top:top + nh, left:left + nw] = img
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None]), (top, left, nh, nw)


def run_case(predictor, mode, ch, cw, ids):
    metric, biou = SegMetric(), BoundaryIoU()
    start = time.perf_counter()
    for i, img_id in enumerate(ids, 1):
        image, gt = load_image(img_id), load_label(img_id)
        if mode == "square":
            logits = predictor(preprocess(image, ch, "square")) if ch == cw else None
            if logits is None:
                raise ValueError("square는 정사각 canvas만 지원")
            pred = postprocess(logits, gt.shape)
        else:
            x, (top, left, nh, nw) = letterbox(image, ch, cw)
            logits = predictor(x)[:, :, top:top + nh, left:left + nw]  # 채운 부분 잘라내기
            pred = postprocess(np.ascontiguousarray(logits), gt.shape)
        metric.update(gt, pred)
        biou.update(gt, pred)
        if i % 200 == 0:
            s = metric.summary()
            print(f"    {i}/{len(ids)}  person mIoU {s['miou_person']*100:.2f}  "
                  f"({(time.perf_counter()-start)/i:.2f}초/장)", flush=True)
    r = metric.summary()
    r["biou_person"] = biou.summary()
    r["sec_per_img"] = (time.perf_counter() - start) / len(ids)
    return r


def done_keys(n_expected):
    if not OUT.exists():
        return set()
    with open(OUT, encoding="utf-8") as f:
        return {(r["model"], r["backend"], r["mode"], r["canvas_hw"], r["split"])
                for r in csv.DictReader(f) if int(r["n_images"]) == n_expected}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torch", action="store_true", help="ONNX 대신 PyTorch(structures.py)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="0 = 기본값(정확도는 스레드와 무관)")
    args = ap.parse_args()

    ids = read_ids(args.split)[: args.limit or None]
    backend = "torch" if args.torch else "onnx"
    skip = done_keys(len(ids))
    OUT.parent.mkdir(exist_ok=True)
    new_file = not OUT.exists()

    for name, onnx_path, arch in MODELS:
        todo = [c for c in CASES if (name, backend, c[0], f"{c[1]}x{c[2]}", args.split) not in skip]
        if not todo:
            print(f"[건너뜀] {name}: 이미 끝남")
            continue
        if args.torch:
            from structures import build
            predictor = TorchPredictor(build(arch))
        else:
            predictor = OnnxPredictor(ROOT / onnx_path, args.threads)
        for mode, ch, cw in todo:
            print(f"[{name}] {mode} {ch}x{cw}  ({len(ids)}장)", flush=True)
            with torch.inference_mode():
                r = run_case(predictor, mode, ch, cw, ids)
            row = {"date": time.strftime("%Y-%m-%d %H:%M"), "model": name, "backend": backend,
                   "mode": mode, "canvas_hw": f"{ch}x{cw}", "split": args.split, "n_images": len(ids),
                   "miou_person": f"{r['miou_person']*100:.2f}", "iou_person": f"{r['iou_person']*100:.2f}",
                   "biou_person": f"{r['biou_person']*100:.2f}", "miou21": f"{r['miou21']*100:.2f}",
                   "sec_per_img": f"{r['sec_per_img']:.3f}"}
            with open(OUT, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, FIELDS)
                if new_file:
                    w.writeheader()
                    new_file = False
                w.writerow(row)
            print(f"  → person mIoU {row['miou_person']} / 경계 {row['biou_person']} / 21cls {row['miou21']}", flush=True)


if __name__ == "__main__":
    main()
