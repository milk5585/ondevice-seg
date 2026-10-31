"""
파라미터 수와 연산량(GFLOPs) 세기 (C 담당, 팀 공용)

보고서 '결과 요약' 표의 파라미터 수(M) / 연산량(GFLOPs) 칸용.
속도 측정(bench_speed.py)과 같은 입력(VOC val 첫 장, 짧은 변 --size)으로 센다.

    python count_model.py                      # 원본 OS8 520 + 확정 후보 2개 (OS8 384, OS16r6 384)
    python count_model.py --arch os8 --size 320

세는 방법:
  - 파라미터: 모델 가중치 개수 전부 (aux 분류기는 빼고 → 평가·속도 측정 모델과 같음)
  - 연산량: torch.utils.flop_counter로 실제 forward 1회를 따라가며 셈.
            FLOPs = 곱셈+덧셈을 각각 1로 센 값 (= MACs × 2). 논문마다 MACs를 'FLOPs'라
            부르기도 하므로 보고서에는 둘 다 적고 어느 쪽인지 밝힌다.
            합성곱·행렬곱만 셈 (BatchNorm·ReLU·보간 등은 전체의 1% 미만이라 관례상 제외).
  - INT8 양자화는 파라미터 수·연산 횟수를 바꾸지 않는다 (숫자 하나를 담는 칸이 32비트→8비트로
    작아질 뿐). 그래서 양자화 모델은 같은 구조의 FP32 값과 같고, 줄어드는 건 용량·속도다.

결과: 화면 출력 + results/model_stats.csv
"""

import argparse
import csv
import datetime
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from common.infer import preprocess
from common.voc import load_image, read_ids
from structures import build

CSV_PATH = Path(__file__).resolve().parent / "results" / "model_stats.csv"
FIELDS = ["date", "arch", "size", "input_hw", "params_m", "gmacs", "gflops"]
DEFAULT_RUNS = [("os8", 520), ("os8", 384), ("os16_r6", 384)]


def count(arch: str, size: int) -> dict:
    model = build(arch)
    params = sum(p.numel() for p in model.parameters())
    x = torch.from_numpy(preprocess(load_image(read_ids("val")[0]), size, "short"))
    counter = FlopCounterMode(display=False)
    with counter, torch.inference_mode():
        model(x)
    flops = counter.get_total_flops()
    return {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "arch": arch,
        "size": size,
        "input_hw": f"{x.shape[2]}x{x.shape[3]}",
        "params_m": f"{params / 1e6:.2f}",
        "gmacs": f"{flops / 2e9:.1f}",
        "gflops": f"{flops / 1e9:.1f}",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", help="os8, os16_r12, os16_r6 (안 주면 기본 3개)")
    ap.add_argument("--size", type=int, default=520)
    args = ap.parse_args()

    runs = [(args.arch, args.size)] if args.arch else DEFAULT_RUNS
    CSV_PATH.parent.mkdir(exist_ok=True)
    new_file = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for arch, size in runs:
            r = count(arch, size)
            print(f"{arch:8s} {size}  입력 {r['input_hw']:8s}  파라미터 {r['params_m']} M  "
                  f"연산량 {r['gflops']} GFLOPs ({r['gmacs']} GMACs)", flush=True)
            w.writerow(r)
    print(f"저장: {CSV_PATH}")


if __name__ == "__main__":
    main()
