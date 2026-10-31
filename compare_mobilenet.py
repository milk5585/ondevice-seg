"""
비교군: 처음부터 가벼운 모델(MobileNetV3)을 쓰면? (B)

torchvision의 같은 학습 조건(COCO_WITH_VOC_LABELS_V1) 경량 모델 2개를
우리 모델과 '똑같은' 평가(VOC val 1449장, person mIoU / Boundary IoU / 21클래스)로 잰다.
입력은 원본과 같은 짧은 변 520, 우리 최종과 같은 384 두 가지.

    python compare_mobilenet.py              # 4조합 (이미 끝난 조합은 건너뜀)
    python compare_mobilenet.py --limit 20   # 빠른 확인
결과: results/mobilenet_compare.csv
"""
import argparse
import csv
import time
from pathlib import Path

from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    LRASPP_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
    lraspp_mobilenet_v3_large,
)

from common.evaluate import evaluate
from common.infer import TorchPredictor
from common.voc import read_ids

OUT = Path(__file__).resolve().parent / "results" / "mobilenet_compare.csv"
FIELDS = ["date", "model", "size", "split", "n_images", "miou_person", "iou_person", "biou_person", "miou21", "sec_per_img"]
MODELS = {
    "deeplabv3_mbv3": lambda: deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1),
    "lraspp_mbv3": lambda: lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1),
}
SIZES = [520, 384]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    n = len(read_ids(args.split)[: args.limit or None])

    done = set()
    if OUT.exists():
        with open(OUT, encoding="utf-8") as f:
            done = {(r["model"], r["size"], r["split"]) for r in csv.DictReader(f) if int(r["n_images"]) == n}

    for name, make in MODELS.items():
        todo = [s for s in SIZES if (name, str(s), args.split) not in done]
        if not todo:
            continue
        predictor = TorchPredictor(make())
        for size in todo:
            print(f"[{name}] 짧은 변 {size} ({n}장)", flush=True)
            r = evaluate(predictor, split=args.split, size=size, limit=args.limit, log_every=300)
            row = {"date": time.strftime("%Y-%m-%d %H:%M"), "model": name, "size": size, "split": args.split, "n_images": r["n_images"],
                   "miou_person": f"{r['miou_person']*100:.2f}", "iou_person": f"{r['iou_person']*100:.2f}",
                   "biou_person": f"{r['biou_person']*100:.2f}", "miou21": f"{r['miou21']*100:.2f}", "sec_per_img": f"{r['sec_per_img']:.3f}"}
            new = not OUT.exists()
            with open(OUT, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
            print(f"  → person mIoU {row['miou_person']} / 경계 {row['biou_person']} / 21cls {row['miou21']}", flush=True)


if __name__ == "__main__":
    main()
