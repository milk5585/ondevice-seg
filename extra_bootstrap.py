"""
[1101 추가 · 정확도] 짝지은(paired) 부트스트랩 95% 신뢰구간

results/per_image_onnx.csv(extra_per_image_onnx.py)를 읽어, 같은 이미지 묶음을 10,000번 복원추출해
두 설정의 지표 차이 분포를 만든다. 지표는 묶음 전체의 교집합 합 / 합집합 합(데이터셋 기준 IoU).

    python extra_bootstrap.py        # → results/bootstrap_ci.csv
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "results" / "per_image_onnx.csv"
OUT = ROOT / "results" / "bootstrap_ci.csv"
PAIRS = [
    ("해상도 축소 (520→384, FP32)", "os8_520_fp32", "os8_384_fp32"),
    ("INT8 양자화 (384 FP32→INT8)", "os8_384_fp32", "os8_384_int8_pc"),
    ("전체 (Baseline→최종)", "os8_520_fp32", "os8_384_int8_pc"),
]
KEYS = ["p_inter", "p_union", "bg_inter", "bg_union", "b_inter", "b_union"]
B = 10000


def metrics(a):  # a: (..., 6) 합계
    p = a[..., 0] / a[..., 1]
    bg = a[..., 2] / a[..., 3]
    return {"person mIoU": 100 * (p + bg) / 2, "person IoU": 100 * p, "Boundary IoU": 100 * a[..., 4] / a[..., 5]}


def main():
    data = defaultdict(dict)
    with open(SRC, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            data[r["config"]][r["id"]] = [int(r[k]) for k in KEYS]
    ids = sorted(set.intersection(*(set(v) for v in data.values())))
    arr = {c: np.array([data[c][i] for i in ids], dtype=np.float64) for c in data}
    rng = np.random.default_rng(0)
    counts = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=B).astype(np.float64)  # (B, N)
    rows = []
    print(f"이미지 {len(ids)}장, 복원추출 {B}회 (seed 0)\n")
    for label, a, b in PAIRS:
        ma, mb = metrics(arr[a].sum(0)), metrics(arr[b].sum(0))
        ba, bb = metrics(counts @ arr[a]), metrics(counts @ arr[b])
        print(label)
        for m in ma:
            d = bb[m] - ba[m]
            lo, hi = np.percentile(d, [2.5, 97.5])
            sig = "유의" if (lo > 0 or hi < 0) else "유의하지 않음(0 포함)"
            print(f"  {m:13s} {ma[m]:6.2f} → {mb[m]:6.2f}  차이 {mb[m]-ma[m]:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  {sig}")
            rows.append([label, m, f"{ma[m]:.2f}", f"{mb[m]:.2f}", f"{mb[m]-ma[m]:.2f}", f"{lo:.2f}", f"{hi:.2f}", sig])
        print()
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["비교", "지표", "전", "후", "차이", "CI_low", "CI_high", "판정"])
        w.writerows(rows)
    print("저장:", OUT)


if __name__ == "__main__":
    main()
