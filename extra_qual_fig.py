"""
[1101 추가 · 정확도, 어느 PC든 가능] 정성 비교 그림: Baseline(520 FP32) vs 최종(384 INT8)

results/per_image_onnx.csv에서 두 사례를 자동으로 고른다 (또는 --ids로 직접 지정).
  ① 큰 인물(면적 20% 이상) 중 Boundary IoU 손실이 가장 큰 사진 → 경계(머리카락 등) 손실 확인
  ② 작은 인물(면적 5% 미만) 중 person IoU 손실이 가장 큰 사진 → 6.6절 '작은 인물 취약' 확인
※ 손실이 '가장 큰' 사례를 고르므로 그림 설명에 반드시 그렇게 적는다 (대표 사례가 아님).
열: 입력 | 정답 | Baseline 520 FP32 | 최종 384 INT8 | 차이 (빨강 = Baseline만 맞힘, 파랑 = 최종만 맞힘)

    python extra_qual_fig.py                      # → results/fig_qualitative.png
    python extra_qual_fig.py --ids 2007_000129 2008_000000
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.infer import OnnxPredictor, postprocess, preprocess
from common.voc import IGNORE, PERSON_CLASS, load_image, load_label

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "results" / "per_image_onnx.csv"
OUT = ROOT / "results" / "fig_qualitative.png"
plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def pick():
    d = defaultdict(dict)
    with open(SRC, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d[r["id"]][r["config"]] = r
    def iou(r, a, b):
        return int(r[a]) / int(r[b]) if int(r[b]) else 1.0
    big, small = [], []
    for i, c in d.items():
        a, b = c["os8_520_fp32"], c["os8_384_int8_pc"]
        frac = float(a["person_frac"])
        if frac >= 0.20:
            big.append((iou(b, "b_inter", "b_union") - iou(a, "b_inter", "b_union"), i))
        elif frac < 0.05:
            small.append((iou(b, "p_inter", "p_union") - iou(a, "p_inter", "p_union"), i))
    return [min(big)[1], min(small)[1]]


def crop_box(mask, h, w, margin=0.3):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0, h, 0, w
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    my, mx = int((y1 - y0) * margin) + 10, int((x1 - x0) * margin) + 10
    return max(0, y0 - my), min(h, y1 + my), max(0, x0 - mx), min(w, x1 + mx)


def tint(img, m, color=(0, 120, 255), a=0.45):
    out = img.astype(np.float32).copy()
    out[m] = out[m] * (1 - a) + np.array(color) * a
    return out.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+")
    args = ap.parse_args()
    ids = args.ids or pick()
    base = OnnxPredictor(ROOT / "models/os8_fp32.onnx", 0)
    final = OnnxPredictor(ROOT / "models/os8_int8_perchannel.onnx", 0)
    titles = ["입력", "정답 (person)", "Baseline 520 FP32", "최종 384 INT8", "차이"]
    fig, axes = plt.subplots(len(ids), 5, figsize=(16, 3.4 * len(ids)), squeeze=False)
    for row, img_id in enumerate(ids):
        image, gt = load_image(img_id), load_label(img_id)
        img = np.asarray(image)
        valid = gt != IGNORE
        g = (gt == PERSON_CLASS)
        pa = postprocess(base(preprocess(image, 520, "short")), gt.shape) == PERSON_CLASS
        pb = postprocess(final(preprocess(image, 384, "short")), gt.shape) == PERSON_CLASS
        y0, y1, x0, x1 = crop_box(g, *gt.shape)
        diff = np.full(gt.shape + (3,), 255, np.uint8)
        diff[valid & (pa == g) & (pb != g)] = (220, 40, 40)
        diff[valid & (pb == g) & (pa != g)] = (40, 90, 220)
        def iou(p):
            return 100 * (p & g & valid).sum() / max(1, ((p | g) & valid).sum())
        panels = [img, tint(img, g, (0, 180, 0)), tint(img, pa), tint(img, pb), diff]
        for col, p in enumerate(panels):
            ax = axes[row][col]
            ax.imshow(p[y0:y1, x0:x1])
            ax.set_xticks([]); ax.set_yticks([])
            t = titles[col]
            if col == 2: t += f"\nperson IoU {iou(pa):.1f}"
            if col == 3: t += f"\nperson IoU {iou(pb):.1f}"
            if col == 0: t += f"\n{img_id}"
            ax.set_title(t, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print("저장:", OUT, "사례:", ids)


if __name__ == "__main__":
    main()
