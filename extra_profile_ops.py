"""
[1101 추가 · 속도, 기준 PC에서만] ONNX Runtime 연산자별 시간 프로파일링 (FP32 vs INT8)

목적: INT8이 왜 빨라졌는지를 연산자 단위로 확인한다.
  - ORT가 최적화한 그래프의 연산자 구성 (QDQ가 정수 Conv 커널로 합쳐졌는지)
  - 연산자 종류별 1회 추론당 시간과 비중 (Conv가 몇 %인지, Resize·ASPP 몫은 얼마인지)
조건: 보고서와 같게 CPU 4스레드(연산 간 병렬 1), 짧은 변 384, VOC val 첫 장. warmup 20회 뒤 50회 기록.
※ 프로파일링 자체가 약간 느리게 만들므로 '절대 시간'이 아니라 '비중'을 본다. 속도 숫자는 bench_speed.py 값을 쓴다.

    python extra_profile_ops.py      # → results/op_profile.csv, results/op_counts.csv
    python extra_profile_ops.py --tag gap --models os8_384_int8_pc=models/os8_int8_perchannel.onnx os8gap_384_int8_pc=models/os8gap_int8_perchannel.onnx
                                     # → results/op_profile_gap.csv, results/op_counts_gap.csv
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import onnx
import onnxruntime as ort

from bench_speed import disable_efficiency_mode, windows_power_info
from common.infer import preprocess
from common.voc import load_image, read_ids

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
MODELS = [("os8_384_fp32", "models/os8_fp32.onnx"), ("os8_384_int8_pc", "models/os8_int8_perchannel.onnx")]
THREADS, WARMUP, RUNS = 4, 20, 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", help="이름=경로 (기본: FP32 vs INT8 per-channel)")
    ap.add_argument("--tag", default="", help="결과 파일 이름 뒤에 붙일 표시")
    args = ap.parse_args()
    models = [tuple(a.split("=", 1)) for a in args.models] if args.models else MODELS
    sfx = f"_{args.tag}" if args.tag else ""
    disable_efficiency_mode()
    mode, ac = windows_power_info()
    print(f"전원 모드: {mode} / 충전기: {ac}")
    if sys.platform == "win32" and not mode.startswith("최고 성능"):
        sys.exit("[중단] 전원 모드를 '최고 성능'으로 맞춰 주세요.")
    x = preprocess(load_image(read_ids("val")[0]), 384, "short")
    RES.mkdir(exist_ok=True)
    prof_rows, count_rows = [], []
    for name, path in models:
        so = ort.SessionOptions()
        so.intra_op_num_threads = THREADS
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # ORT 기본값과 같음 (명시)
        opt_path = RES / f"_opt_{name}.onnx"
        so.optimized_model_filepath = str(opt_path)
        so.enable_profiling = True
        so.profile_file_prefix = str(RES / f"_prof_{name}")
        sess = ort.InferenceSession(str(ROOT / path), so, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        print(f"[{name}] warmup {WARMUP} + 기록 {RUNS}회 ...", flush=True)
        for _ in range(WARMUP + RUNS):
            sess.run(None, {inp: x})
        prof = sess.end_profiling()

        # 1) 최적화된 그래프의 연산자 구성
        try:
            g = onnx.load(str(opt_path), load_external_data=False).graph
            cnt = Counter(f"{n.domain + '.' if n.domain else ''}{n.op_type}" for n in g.node)
            for op, c in cnt.most_common():
                count_rows.append([name, op, c])
            print("  최적화 그래프 연산자:", ", ".join(f"{k}×{v}" for k, v in cnt.most_common(8)))
        except Exception as e:
            print("  (최적화 그래프 저장/읽기 실패:", str(e)[:120], ")")

        # 2) 연산자별 시간: warmup 이후 RUNS회의 model_run 구간 안의 커널 시간만
        ev = json.load(open(prof, encoding="utf-8"))
        runs = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev if e.get("cat") == "Session" and e.get("name") == "model_run")
        keep = runs[WARMUP:]
        lo, hi = keep[0][0], keep[-1][1]
        t = defaultdict(float)
        nodes = defaultdict(set)
        for e in ev:
            if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time") and lo <= e["ts"] <= hi:
                op = e.get("args", {}).get("op_name", "?")
                t[op] += e["dur"]
                nodes[op].add(e["name"])
        total = sum(t.values())
        run_ms = sum(b - a for a, b in keep) / len(keep) / 1000
        print(f"  1회 추론(프로파일링 중) {run_ms:.1f} ms, 커널 합 {total/len(keep)/1000:.1f} ms")
        for op, us in sorted(t.items(), key=lambda kv: -kv[1]):
            ms = us / len(keep) / 1000
            prof_rows.append([name, op, len(nodes[op]), f"{ms:.2f}", f"{100*us/total:.1f}"])
            if 100 * us / total >= 1:
                print(f"    {op:28s} 노드 {len(nodes[op]):3d}개  {ms:7.2f} ms  {100*us/total:5.1f}%")
        os.remove(prof)

    with open(RES / f"op_profile{sfx}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "op_type", "n_nodes", "ms_per_run", "share_pct"])
        w.writerows(prof_rows)
    with open(RES / f"op_counts{sfx}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "op_type", "count"])
        w.writerows(count_rows)
    print(f"저장: results/op_profile{sfx}.csv, results/op_counts{sfx}.csv")


if __name__ == "__main__":
    main()
