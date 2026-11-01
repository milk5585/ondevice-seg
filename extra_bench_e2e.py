"""
[1101 추가 · 속도, 기준 PC에서만] end-to-end 지연: 전처리 + 추론 + 후처리(원본 크기 복원 + argmax)

bench_speed.py와 같은 조건(4스레드, 전원 '최고 성능', 유휴 검사, 효율 모드 해제, VOC val 첫 장,
warmup 20 / 측정 200회, 중앙값). 이미지 파일 읽기(JPEG 디코딩)는 제외 — 카메라 프레임이 이미 메모리에 있다고 가정.

    python extra_bench_e2e.py        # → results/e2e.csv
"""
import csv
import datetime
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

from bench_speed import disable_efficiency_mode, windows_power_info
from common.infer import OnnxPredictor, postprocess, preprocess
from common.voc import load_image, read_ids
from idle_check import wait_until_idle

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "e2e.csv"
CONFIGS = [("os8_520_fp32", "models/os8_fp32.onnx", 520), ("os8_384_int8_pc", "models/os8_int8_perchannel.onnx", 384)]
THREADS, WARMUP, RUNS = 4, 20, 200


def main():
    disable_efficiency_mode()
    mode, ac = windows_power_info()
    print(f"전원 모드: {mode} / 충전기: {ac}")
    if sys.platform == "win32" and not mode.startswith("최고 성능"):
        sys.exit("[중단] 전원 모드를 '최고 성능'으로 맞춰 주세요.")
    torch.set_num_threads(THREADS)  # 후처리(보간·argmax)도 같은 4스레드
    torch.set_num_interop_threads(1)
    image = load_image(read_ids("val")[0])
    hw = (image.size[1], image.size[0])
    rows = []
    for name, path, size in CONFIGS:
        pred = OnnxPredictor(ROOT / path, THREADS)
        if not wait_until_idle():
            sys.exit("[중단] PC가 조용하지 않습니다.")
        for _ in range(WARMUP):
            postprocess(pred(preprocess(image, size, "short")), hw)
        pre, inf, post, tot = [], [], [], []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            x = preprocess(image, size, "short")
            t1 = time.perf_counter()
            y = pred(x)
            t2 = time.perf_counter()
            postprocess(y, hw)
            t3 = time.perf_counter()
            pre.append((t1 - t0) * 1e3); inf.append((t2 - t1) * 1e3); post.append((t3 - t2) * 1e3); tot.append((t3 - t0) * 1e3)
        m = [statistics.median(v) for v in (pre, inf, post, tot)]
        p90 = float(np.percentile(tot, 90))
        print(f"[{name}] 전처리 {m[0]:.1f} / 추론 {m[1]:.1f} / 후처리 {m[2]:.1f} / 합계 {m[3]:.1f} ms (p90 {p90:.1f}), {1000/m[3]:.2f} FPS")
        rows.append([datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), name, size, THREADS, *(f"{v:.2f}" for v in m), f"{p90:.2f}", f"{1000/m[3]:.2f}", mode])
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "name", "size", "threads", "pre_ms", "infer_ms", "post_ms", "total_ms", "total_p90_ms", "fps", "power_mode"])
        w.writerows(rows)
    print("저장:", OUT)


if __name__ == "__main__":
    main()
