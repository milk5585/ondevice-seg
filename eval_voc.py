"""
VOC 2012 val 정확도 측정 (팀 공용)

사용법:
    python eval_voc.py --name baseline                              # PyTorch 원본
    python eval_voc.py --name baseline_onnx --onnx models/baseline.onnx
    python eval_voc.py --name test --size 384 --limit 50            # 빠른 확인용

결과:
    화면 출력 + results/accuracy.csv 에 한 줄 추가
    (--limit 을 쓴 결과는 참고용. 보고서 숫자는 전체 1449장으로 잰다)
"""

import argparse
import csv
import datetime
from pathlib import Path

import torch

from common.evaluate import evaluate
from common.infer import OnnxPredictor, TorchPredictor

CSV_PATH = Path(__file__).resolve().parent / "results" / "accuracy.csv"
FIELDS = [
    "date", "name", "backend", "model_file", "size", "mode", "split",
    "n_images", "miou_person", "iou_person", "biou_person", "miou21", "model_mb", "sec_per_img",
]


def _upgrade_header() -> None:
    """옛 CSV에 새 열(예: biou_person)이 없으면 열을 추가해 다시 쓴다. 옛 줄의 새 칸은 빈칸."""
    if not CSV_PATH.exists():
        return
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == FIELDS:
            return
        rows = list(reader)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="결과 이름 (예: baseline, os16_384)")
    ap.add_argument("--onnx", help="ONNX 파일 경로. 없으면 PyTorch 원본 모델")
    ap.add_argument("--size", type=int, default=520)
    ap.add_argument("--mode", default="short", choices=["short", "square"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N장만 (0 = 전체)")
    ap.add_argument("--threads", type=int, default=0, help="CPU 스레드 수 (0 = 기본값)")
    args = ap.parse_args()

    if args.onnx:
        predictor = OnnxPredictor(args.onnx, args.threads)
        model_mb = Path(args.onnx).stat().st_size / 1024**2
    else:
        if args.threads:
            torch.set_num_threads(args.threads)
        predictor = TorchPredictor()
        model_mb = None

    print(f"[{args.name}] {predictor.name}, size={args.size}, mode={args.mode}")
    r = evaluate(predictor, args.split, args.size, args.mode, args.limit)

    print(f"\n이미지 수       : {r['n_images']}")
    print(f"person mIoU    : {r['miou_person']*100:.2f}   (주 지표)")
    print(f"person IoU     : {r['iou_person']*100:.2f}")
    print(f"person 테두리  : {r['biou_person']*100:.2f}   (Boundary IoU)")
    print(f"21클래스 mIoU  : {r['miou21']*100:.2f}")
    if args.limit:
        print("\n[주의] --limit 결과는 참고용입니다.")

    CSV_PATH.parent.mkdir(exist_ok=True)
    _upgrade_header()
    new_file = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "name": args.name,
            "backend": predictor.name,
            "model_file": args.onnx or "",
            "size": args.size,
            "mode": args.mode,
            "split": args.split,
            "n_images": r["n_images"],
            "miou_person": f"{r['miou_person']*100:.2f}",
            "iou_person": f"{r['iou_person']*100:.2f}",
            "biou_person": f"{r['biou_person']*100:.2f}",
            "miou21": f"{r['miou21']*100:.2f}",
            "model_mb": f"{model_mb:.1f}" if model_mb else "",
            "sec_per_img": f"{r['sec_per_img']:.3f}",
        })
    print(f"저장: {CSV_PATH}")


if __name__ == "__main__":
    main()
