"""
ONNX 모델 추론 메모리 측정 (B). 방식은 mem_usage.py 설명 참고.

모델마다 새 프로세스로 따로 재서(다른 모델이 남긴 메모리 영향 없음) results/memory.csv에 한 줄씩 쌓는다.
  load_mb       : 모델 로드 후 상주 메모리 - 로드 전
  infer_peak_mb : 추론 10회 동안 5ms마다 잰 최대 메모리 - 로드 전  ← 보고서의 '피크 메모리'

사용법 (학과 PC)
    python mem_onnx.py --all --repeat 3        # 기본 7개 모델 × 3회
    python mem_onnx.py --name x --onnx models/os8_fp32.onnx --size 384   # 하나만
"""
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "memory.csv"
FIELDS = ["date", "name", "model_file", "size", "input_hw", "threads", "before_mb", "load_mb", "infer_peak_mb"]

ALL = [
    ("os8_520_fp32", "models/os8_fp32.onnx", 520),
    ("os8_384_fp32", "models/os8_fp32.onnx", 384),
    ("os8_384_int8_pc", "models/os8_int8_perchannel.onnx", 384),
    ("os8_384_int8_pt", "models/os8_int8_pertensor.onnx", 384),
    ("os16r6_384_fp32", "models/os16r6_fp32.onnx", 384),
    ("os16r6_384_int8_pc", "models/os16r6_int8_perchannel.onnx", 384),
    ("os16r6_384_int8_pt", "models/os16r6_int8_pertensor.onnx", 384),
]


def measure(name, onnx, size, mode, threads):
    from mem_usage import PeakSampler, current_mb
    from common.infer import OnnxPredictor, preprocess
    from common.voc import load_image, read_ids

    x = preprocess(load_image(read_ids("val")[0]), size, mode)  # bench_speed.py와 같은 입력
    before = current_mb()
    predictor = OnnxPredictor(ROOT / onnx, threads)
    loaded = current_mb()
    for _ in range(3):  # warmup
        predictor(x)
    with PeakSampler() as s:
        for _ in range(10):
            predictor(x)
    row = {"date": time.strftime("%Y-%m-%d %H:%M"), "name": name, "model_file": onnx, "size": size,
           "input_hw": f"{x.shape[2]}x{x.shape[3]}", "threads": threads, "before_mb": f"{before:.0f}",
           "load_mb": f"{loaded - before:.0f}", "infer_peak_mb": f"{s.peak_mb - before:.0f}"}
    OUT.parent.mkdir(exist_ok=True)
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"{name}: 로드 {row['load_mb']} MB / 추론 중 최대 {row['infer_peak_mb']} MB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--name")
    ap.add_argument("--onnx")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--mode", default="short", choices=["short", "square"])
    ap.add_argument("--threads", type=int, default=4, help="속도 측정과 같게 4")
    args = ap.parse_args()

    if args.all:
        for r in range(args.repeat):  # 모델을 번갈아 돌림 (시간에 따른 변화가 한 모델에 몰리지 않게)
            for name, onnx, size in ALL:
                subprocess.run([sys.executable, __file__, "--name", f"{name}_m{r+1}", "--onnx", onnx,
                                "--size", str(size), "--threads", str(args.threads)], check=True, cwd=ROOT)
    else:
        measure(args.name, args.onnx, args.size, args.mode, args.threads)


if __name__ == "__main__":
    main()
