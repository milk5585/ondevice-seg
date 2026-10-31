"""
B 1주차 run: OS(8/16) × ASPP rate × 입력 해상도 조합별 정확도·속도 측정.
정확도는 C의 팀 공용 평가 코드(common.evaluate)로만 잰다.

사용법:
    python os_resolution_benchmark.py --limit 5              # 동작 확인용
    python os_resolution_benchmark.py --split val_person     # 빠른 1차 선별 (person 있는 446장)
    python os_resolution_benchmark.py                        # 공식 (val 전체 1449장)

결과: results/os_resolution.csv
속도(latency_ms_ref)는 이 PC에서 forward만 잰 참고값. 공식 속도는 기준 PC에서 따로 잰다.
--speed-runs 0 이면 속도 측정을 건너뛰고 정확도만 잰다.
중간에 멈춰도 같은 명령으로 다시 실행하면, 결과 CSV에 이미 있는 조합은 건너뛰고 이어서 잰다.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common.evaluate import evaluate
from common.infer import TorchPredictor
from common.voc import read_ids
from structures import CONFIGS, build


def keep_awake():
    # 수 시간 걸리는 측정 중 절전 진입 방지. 프로세스가 끝나면 자동 해제되고 시스템 설정은 바꾸지 않는다.
    # 창 없이 실행되면 Windows 11이 '효율 모드'로 CPU를 절반 이하로 제한하므로, 이 프로세스만 제외 요청한다.
    if sys.platform == "win32":
        import ctypes

        class PowerThrottlingState(ctypes.Structure):
            _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]

        kernel32 = ctypes.windll.kernel32
        kernel32.SetThreadExecutionState.argtypes = [ctypes.c_uint]
        kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
        state = PowerThrottlingState(1, 1, 0)
        if not kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state)):
            print(f"[경고] 효율 모드 해제 실패 (오류 {kernel32.GetLastError()}) — 창 없이 실행하면 느려질 수 있음", flush=True)


def measure_latency(model, res, warmup, runs):
    x = torch.randn(1, 3, res, round(res * 4 / 3))
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - start)
    return float(np.median(times)) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--resolutions", nargs="+", type=int, default=[520, 448, 384, 320])
    parser.add_argument("--split", default="val", help="val(1449장, 공식) 또는 val_person(446장, 선별용)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=torch.get_num_threads())
    parser.add_argument("--speed-warmup", type=int, default=10)
    parser.add_argument("--speed-runs", type=int, default=50)
    parser.add_argument("--out", default="results/os_resolution.csv")
    args = parser.parse_args()

    keep_awake()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["config", "output_stride", "aspp_rates", "resolution", "split", "n_images",
              "miou_person", "iou_person", "iou_background", "biou_person", "miou21",
              "latency_ms_ref", "speedup_vs_first", "sec_per_img", "threads", "torch"]

    n_expected = len(read_ids(args.split)[: args.limit or None])
    done = []
    if out.exists():
        with out.open(newline="", encoding="utf-8") as f:
            done = list(csv.DictReader(f))
    done_keys = {(r["config"], int(r["resolution"]), r["split"]) for r in done
                 if int(r["n_images"]) == n_expected}
    first_latency = float(done[0]["latency_ms_ref"]) if done and done[0]["latency_ms_ref"] else None
    if done_keys:
        print(f"이미 끝난 조합 {len(done_keys)}개는 건너뛰고 이어서 측정합니다: {sorted(done_keys)}", flush=True)

    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not done:
            writer.writeheader()
        for name in args.configs:
            pending = [res for res in args.resolutions if (name, res, args.split) not in done_keys]
            if not pending:
                continue
            output_stride, rates = CONFIGS[name]
            predictor = TorchPredictor(build(name))
            for res in pending:
                print(f"\n[{name} @ {res}px] split={args.split}", flush=True)
                r = evaluate(predictor, split=args.split, size=res, mode="short", limit=args.limit)
                latency = (measure_latency(predictor.model, res, args.speed_warmup, args.speed_runs)
                           if args.speed_runs else None)
                first_latency = first_latency or latency
                row = {
                    "config": name, "output_stride": output_stride,
                    "aspp_rates": "-".join(map(str, rates)), "resolution": res,
                    "split": args.split, "n_images": r["n_images"],
                    **{k: round(r[k] * 100, 2) for k in
                       ("miou_person", "iou_person", "iou_background", "biou_person", "miou21")},
                    "latency_ms_ref": round(latency, 1) if latency else "",
                    "speedup_vs_first": round(first_latency / latency, 2) if latency else "",
                    "sec_per_img": round(r["sec_per_img"], 3),
                    "threads": args.threads, "torch": torch.__version__,
                }
                writer.writerow(row)
                f.flush()
                speed = f"latency {row['latency_ms_ref']} ms (x{row['speedup_vs_first']})" if latency else "속도 측정 생략"
                print(f"  -> person mIoU {row['miou_person']}  boundary {row['biou_person']}  "
                      f"21cls {row['miou21']}  {speed}", flush=True)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
