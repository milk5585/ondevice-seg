"""
속도 반복 측정 묶기 (C 담당, 팀 공용)

같은 모델도 측정 시각에 따라 최대 ~3% 달라진다(B 확인). 그래서 최종 측정은 모델마다
bench_speed.py를 3번 따로 실행하고(같은 --name, 또는 이름 끝에 _r1/_r2/_r3), 이 스크립트로 묶어서 보고한다.

    순서는 모델을 번갈아 가며 (A B C → A B C → A B C) 돌린다.
    같은 모델을 연달아 3번 재면 그 시간대의 PC 상태가 한 모델에만 몰린다.

    python summarize_speed.py                 # speed.csv의 모든 이름
    python summarize_speed.py --since 2026-10-27   # 이 날짜 이후 측정만 (5주차 최종)

보고 값:
    latency  = 각 실행 중앙값들의 중앙값 (ms)
    범위      = 실행 중앙값의 최소~최대, 그리고 (최대-최소)/중앙값 = 재현 오차(%)
    메모리    = infer_mem_mb의 중앙값
기준(--runs 200, --warmup 20)이 아닌 줄과 test로 시작하는 이름은 뺀다.

결과: 화면 출력 + results/speed_summary.csv
"""

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
FIELDS = ["name", "backend", "size", "n_runs", "latency_ms", "min_ms", "max_ms",
          "spread_pct", "fps", "infer_mem_mb", "model_mb", "dates"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="", help="YYYY-MM-DD 이후 측정만")
    args = ap.parse_args()

    with (RESULTS / "speed.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    groups = defaultdict(list)
    for r in rows:
        if r["runs"] != "200" or r["warmup"] != "20" or r["name"].startswith("test"):
            continue
        if args.since and r["date"] < args.since:
            continue
        # B 방식: 이름 끝 _r1/_r2/_r3 → 한 묶음 (예: os8_520_fp32_r1 → os8_520_fp32 [3회])
        # 꼬리표 없는 단발 측정은 이름이 같아도 섞지 않는다
        m = re.fullmatch(r"(.+)_r\d+", r["name"])
        name = f"{m.group(1)} [라운드]" if m else r["name"]
        groups[(name, r["backend"], r["size"])].append(r)

    out = []
    for (name, backend, size), rs in groups.items():
        meds = [float(r["median_ms"]) for r in rs]
        mid = statistics.median(meds)
        mems = [float(r["infer_mem_mb"]) for r in rs if r.get("infer_mem_mb")]
        out.append({
            "name": name, "backend": backend, "size": size, "n_runs": len(rs),
            "latency_ms": f"{mid:.1f}", "min_ms": f"{min(meds):.1f}", "max_ms": f"{max(meds):.1f}",
            "spread_pct": f"{(max(meds) - min(meds)) / mid * 100:.1f}",
            "fps": f"{1000 / mid:.2f}",
            "infer_mem_mb": f"{statistics.median(mems):.0f}" if mems else "",
            "model_mb": rs[-1]["model_mb"],
            "dates": " / ".join(r["date"] for r in rs),
        })

    print(f"{'이름':24s} {'횟수':>4s} {'중앙값ms':>9s} {'범위':>17s} {'오차%':>6s} {'메모리MB':>8s}")
    for o in out:
        flag = "" if o["n_runs"] >= 3 else "  ← 3회 미만"
        print(f"{o['name']:24s} {o['n_runs']:>4d} {o['latency_ms']:>9s} "
              f"{o['min_ms'] + '~' + o['max_ms']:>17s} {o['spread_pct']:>6s} {o['infer_mem_mb']:>8s}{flag}")

    with (RESULTS / "speed_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    print(f"저장: {RESULTS / 'speed_summary.csv'}")


if __name__ == "__main__":
    main()
