"""
평가 코드 검증용: COCO val2017로 torchvision 공식 수치(mIoU 66.4, 픽셀 정확도 92.4) 재현

torchvision 공식 평가(references/segmentation)를 그대로 따라 한다.
  - COCO 80클래스 중 VOC와 겹치는 20클래스만 남기고 VOC 번호(1~20)로 바꾼다. 나머지는 배경(0)
  - 물체 두 개가 겹친 픽셀은 255(채점 제외)
  - 사진과 정답 모두 짧은 변을 520으로 줄여서 채점 (정답은 nearest)
    → VOC 평가(원본 크기 채점)와 다르다. 공식 숫자와 비교하려고 공식 방식에 맞춘 것
  - val2017 5000장 전부 사용 (정답이 비어 있는 사진도 빼지 않음)

전처리·모델·혼동행렬 계산은 VOC 평가와 같은 common/ 코드를 쓴다.
→ 여기서 66.4가 나오면 common/ 코드가 맞다는 근거가 된다.

사용법:
    python eval_coco.py                 # 전체 5000장
    python eval_coco.py --limit 20      # 빠른 확인용

결과: 화면 출력 + results/coco_check.csv 에 한 줄 추가
"""

import argparse
import csv
import datetime
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from common.infer import TorchPredictor, preprocess
from common.metrics import SegMetric

ROOT = Path(__file__).resolve().parent
COCO_DIR = ROOT / "data" / "coco"
CSV_PATH = ROOT / "results" / "coco_check.csv"

# torchvision references/segmentation/train.py 의 CAT_LIST
# 위치(0~20) = VOC 클래스 번호, 값 = COCO category_id
CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]

OFFICIAL_MIOU = 66.4
OFFICIAL_PIXACC = 92.4


def make_target(coco: COCO, img_id: int, h: int, w: int) -> np.ndarray:
    """torchvision ConvertCocoPolysToMask 와 같은 방식으로 (H, W) 정답 맵을 만든다."""
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
    anns = [a for a in anns if a["category_id"] in CAT_LIST]
    masks, cats = [], []
    for a in anns:
        rles = coco_mask.frPyObjects(a["segmentation"], h, w)
        m = coco_mask.decode(rles)
        if m.ndim < 3:
            m = m[..., None]
        masks.append(m.any(axis=2).astype(np.uint8))
        cats.append(CAT_LIST.index(a["category_id"]))
    if not masks:
        return np.zeros((h, w), dtype=np.uint8)
    masks = np.stack(masks)
    cats = np.array(cats, dtype=np.uint8)
    target = (masks * cats[:, None, None]).max(axis=0)
    target[masks.sum(0) > 1] = 255
    return target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=520)
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N장만 (0 = 전체)")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    coco = COCO(str(COCO_DIR / "annotations" / "instances_val2017.json"))
    ids = sorted(coco.getImgIds())
    if args.limit:
        ids = ids[: args.limit]

    predictor = TorchPredictor()
    metric = SegMetric()
    start = time.perf_counter()
    for i, img_id in enumerate(ids, 1):
        info = coco.loadImgs(img_id)[0]
        image = Image.open(COCO_DIR / "val2017" / info["file_name"]).convert("RGB")
        target = make_target(coco, img_id, info["height"], info["width"])

        x = preprocess(image, args.size, "short")
        rh, rw = x.shape[2], x.shape[3]
        gt = np.asarray(Image.fromarray(target).resize((rw, rh), Image.NEAREST))

        logits = torch.from_numpy(predictor(x))
        if logits.shape[2:] != (rh, rw):
            logits = F.interpolate(logits, size=(rh, rw), mode="bilinear", align_corners=False)
        pred = logits.argmax(1)[0].numpy().astype(np.uint8)
        metric.update(gt, pred)

        if i % 100 == 0 or i == len(ids):
            cm = metric.cm
            iou = np.diag(cm) / (cm.sum(0) + cm.sum(1) - np.diag(cm))
            el = time.perf_counter() - start
            print(
                f"  {i}/{len(ids)}  mIoU {np.nanmean(iou)*100:.2f}  "
                f"({el/i:.2f}초/장, 남은 시간 약 {el/i*(len(ids)-i)/60:.0f}분)",
                flush=True,
            )

    cm = metric.cm.astype(np.float64)
    tp = np.diag(cm)
    iou = tp / (cm.sum(0) + cm.sum(1) - tp)
    miou = np.nanmean(iou) * 100
    pixacc = tp.sum() / cm.sum() * 100

    print(f"\n이미지 수     : {len(ids)}")
    print(f"mIoU (21cls)  : {miou:.2f}   (공식 {OFFICIAL_MIOU}, 차이 {miou-OFFICIAL_MIOU:+.2f})")
    print(f"픽셀 정확도   : {pixacc:.2f}   (공식 {OFFICIAL_PIXACC}, 차이 {pixacc-OFFICIAL_PIXACC:+.2f})")
    print(f"person IoU    : {iou[15]*100:.2f}")
    if args.limit:
        print("\n[주의] --limit 결과는 참고용입니다.")

    CSV_PATH.parent.mkdir(exist_ok=True)
    new_file = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["date", "n_images", "size", "miou21", "pixacc", "person_iou",
                        "official_miou", "official_pixacc", "sec_per_img"])
        w.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), len(ids), args.size,
            f"{miou:.2f}", f"{pixacc:.2f}", f"{iou[15]*100:.2f}",
            OFFICIAL_MIOU, OFFICIAL_PIXACC, f"{(time.perf_counter()-start)/len(ids):.3f}",
        ])
    print(f"저장: {CSV_PATH}")


if __name__ == "__main__":
    main()
