"""
추론 속도 측정 (C 담당, 팀 공용)

규칙: 기준 PC 1대에서만 잰다. warmup 20회 → 200회 측정 → 중앙값.
      스레드 수를 고정하고, 다른 측정(eval_voc.py 등)과 동시에 돌리지 않는다.
      재는 동안 크롬 등 다른 작업 금지.
      전원: 충전기 연결 + 전원 모드 "최고 성능". 아니면 측정을 거부한다
            (전원 모드만 바꿔도 2배 넘게 차이 남 — 팀원 A 확인).
      Windows 11 효율 모드: 창 없이 백그라운드로 돌리면 CPU가 절반 이하로 제한된다.
            → 이 스크립트가 시작할 때 자기 프로세스의 효율 모드를 직접 끈다.

      측정 전 idle_check로 PC가 조용한지 검사한다 (통과 못 하면 중단).

사용법:
    python bench_speed.py --name os8_520                             # PyTorch 원본 (OS8)
    python bench_speed.py --name os16_r6_384 --arch os16_r6 --size 384  # B의 구조 (structures.py)
    python bench_speed.py --name baseline_onnx --onnx models/baseline.onnx
    python bench_speed.py --name test --size 384 --runs 20 --warmup 5  # 빠른 확인용

결과:
    화면 출력 + results/speed.csv 에 한 줄 추가
    p95_ms     : 200회 중 느린 쪽 5% 경계 (보고서 양식의 p95)
    infer_mem_mb: 추론 중 메모리 최댓값 − 모델 로드 전 메모리 (보고서 '피크 메모리' = 이 값)
                  = 모델 가중치 + 추론 중 중간 결과. torch/onnxruntime import 몫은 빠진다.
                  속도 측정이 끝난 뒤 따로 10회 더 돌리며 재므로 속도 숫자에는 영향 없음 (mem_usage.py, B)
    peak_mem_mb / mem_before_mb: 위 값을 만든 원래 숫자 (프로세스 전체, 참고용)
    ※ 원본과 INT8을 비교할 때는 둘 다 --onnx(onnxruntime)로 잰 줄끼리 비교한다
    (--runs 200 / --warmup 20 이 아니면 참고용. 보고서 숫자는 기본값으로 잰다)
"""

import argparse
import csv
import ctypes
import datetime
import io
import platform
import sys
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from common.infer import OnnxPredictor, TorchPredictor, preprocess
from common.voc import load_image, read_ids
from mem_usage import PeakSampler, current_mb

CSV_PATH = Path(__file__).resolve().parent / "results" / "speed.csv"
FIELDS = [
    "date", "name", "backend", "model_file", "size", "mode", "input_hw",
    "threads", "warmup", "runs", "median_ms", "mean_ms", "p90_ms", "std_ms",
    "fps", "model_mb", "cpu", "torch", "power_mode", "ac_power", "arch",
    "cpu_before_pct", "p95_ms", "infer_mem_mb", "peak_mem_mb", "mem_before_mb",
]
# 이전 버전 열 (끝에 열 4개를 덧붙이기 전). 이 형식이면 버리지 않고 새 열을 빈칸으로 채워 이어 쓴다.
OLD_FIELDS = FIELDS[:-4]

# Windows 11 전원 모드 (설정 > 시스템 > 전원) GUID
# 클래식 전원 관리 옵션(제어판) 중 '고성능' 계열. 데스크톱은 이쪽으로 설정하는 경우가 많다.
HIGH_PERF_PLANS = {
    "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c": "고성능",
    "e9a42b02-d5df-448d-aa00-03f14749eb61": "최고의 성능",
}

POWER_MODES = {
    "ded574b5-45a0-4f42-8737-46345c09c238": "최고 성능",
    "00000000-0000-0000-0000-000000000000": "균형",
    "961cc777-2547-4f9d-8174-7d86181b8a7a": "최고 효율",
}


class _GUID(ctypes.Structure):
    _fields_ = [("d1", ctypes.c_uint32), ("d2", ctypes.c_uint16),
                ("d3", ctypes.c_uint16), ("d4", ctypes.c_ubyte * 8)]

    def __str__(self) -> str:
        d4 = bytes(self.d4).hex()
        return f"{self.d1:08x}-{self.d2:04x}-{self.d3:04x}-{d4[:4]}-{d4[4:]}"


def windows_power_info() -> tuple[str, str]:
    """
    (전원 모드, 충전기 연결 여부). Windows가 아니면 ('n/a', 'n/a').
    전원 모드: Windows 11 전원 모드가 '최고 성능'이거나, 전원 관리 옵션이 고성능 계열이면 '최고 성능'.
    충전기: 배터리가 없는 데스크톱은 'yes(배터리 없음)'으로 본다.
    """
    if sys.platform != "win32":
        return "n/a", "n/a"

    class SYSTEM_POWER_STATUS(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_uint32), ("BatteryFullLifeTime", ctypes.c_uint32)]

    st = SYSTEM_POWER_STATUS()
    ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st))
    if st.BatteryFlag & 128:  # 128 = 시스템 배터리 없음 (데스크톱)
        ac = "yes(배터리 없음)"
    else:
        ac = {0: "no", 1: "yes"}.get(st.ACLineStatus, "unknown")

    try:
        g = _GUID()
        ctypes.windll.powrprof.PowerGetEffectiveOverlayScheme(ctypes.byref(g))
        mode = POWER_MODES.get(str(g), str(g))
    except (AttributeError, OSError):
        mode = "unknown"

    if mode != "최고 성능":
        try:
            pg = ctypes.POINTER(_GUID)()
            if ctypes.windll.powrprof.PowerGetActiveScheme(None, ctypes.byref(pg)) == 0:
                plan = str(pg.contents)
                ctypes.windll.kernel32.LocalFree(pg)
                if plan in HIGH_PERF_PLANS:
                    mode = "최고 성능"
                    return f"{mode}({HIGH_PERF_PLANS[plan]})", ac
        except (AttributeError, OSError):
            pass
    return mode, ac


def disable_efficiency_mode() -> None:
    """
    Windows 11 효율 모드(EcoQoS) 해제 + 우선순위 '높음'.
    창 없이(백그라운드) 실행하면 Windows가 자동으로 효율 모드를 걸어 E코어로 몰고 클럭을 낮춘다.
    """
    if sys.platform != "win32":
        return

    class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong),
                    ("StateMask", ctypes.c_ulong)]

    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
    k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    h = k32.GetCurrentProcess()

    # ControlMask=EXECUTION_SPEED(1)를 '내가 정한다', StateMask=0 → 스로틀링 끔
    state = PROCESS_POWER_THROTTLING_STATE(1, 0x1, 0x0)
    ok1 = k32.SetProcessInformation(h, 4, ctypes.byref(state), ctypes.sizeof(state))  # 4 = ProcessPowerThrottling
    ok2 = k32.SetPriorityClass(h, 0x80)  # HIGH_PRIORITY_CLASS
    print(f"효율 모드 해제: {'OK' if ok1 else '실패'} / 우선순위 높음: {'OK' if ok2 else '실패'}")


def torch_model_mb(model: torch.nn.Module) -> float:
    """state_dict를 저장했을 때의 크기(MB). .pth 파일 크기와 같은 기준."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1024**2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="결과 이름 (예: baseline_fp32, os16_384)")
    ap.add_argument("--onnx", help="ONNX 파일 경로. 없으면 PyTorch 모델 (--arch)")
    ap.add_argument("--arch", default="os8", help="PyTorch 구조 (structures.py): os8, os16_r12, os16_r6")
    ap.add_argument("--size", type=int, default=520, help="입력 짧은 변 길이(px). 정확도 측정과 같은 방식")
    ap.add_argument("--mode", default="short", choices=["short", "square"])
    ap.add_argument("--threads", type=int, default=4, help="CPU 스레드 수 (모두 같은 값으로)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--skip-idle", action="store_true",
                    help="조용한 PC 검사 생략 (참고용, 보고서 숫자로 쓰지 말 것)")
    ap.add_argument("--ignore-power", action="store_true",
                    help="전원 조건이 안 맞아도 측정 (참고용, 보고서 숫자로 쓰지 말 것)")
    args = ap.parse_args()

    disable_efficiency_mode()
    power_mode, ac_power = windows_power_info()
    print(f"전원 모드: {power_mode} / 충전기: {ac_power}")
    if sys.platform == "win32" and (not power_mode.startswith("최고 성능")
                                    or not ac_power.startswith("yes")):
        msg = "전원 조건 불일치: 충전기 연결 + 전원 모드 '최고 성능'으로 맞춰 주세요."
        if not args.ignore_power:
            sys.exit(f"[중단] {msg}  (그래도 재려면 --ignore-power, 참고용)")
        print(f"[주의] {msg} → 참고용 숫자입니다.")

    # 스레드 고정: 연산 내부 병렬(intra)만 쓰고, 연산 간 병렬(inter)은 1로
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    if args.onnx:
        import onnxruntime  # noqa: F401  (import 몫은 '로드 전'에 포함시켜 빼기 위해 먼저)
    mem_before = current_mb()  # 모델 로드 '전' 메모리

    if args.onnx:
        predictor = OnnxPredictor(args.onnx, args.threads)
        model_mb = Path(args.onnx).stat().st_size / 1024**2
    else:
        from structures import build

        predictor = TorchPredictor(build(args.arch))
        model_mb = torch_model_mb(predictor.model)

    # 입력은 VOC val 첫 장으로 고정 → 누가 재도 같은 크기의 입력
    # (전처리·후처리 시간은 빼고 모델 추론만 잰다)
    x = preprocess(load_image(read_ids("val")[0]), args.size, args.mode)
    input_hw = f"{x.shape[2]}x{x.shape[3]}"

    arch = "" if args.onnx else args.arch
    print(f"[{args.name}] {predictor.name} {arch}, 입력 {input_hw}, 스레드 {args.threads}")

    # 모델 로딩이 끝난 뒤, 측정 바로 전에 PC가 조용한지 검사
    from idle_check import cpu_percent, wait_until_idle

    if args.skip_idle:
        print("[주의] 조용한 PC 검사를 건너뜁니다 → 참고용 숫자입니다.")
    elif not wait_until_idle():
        sys.exit("[중단] PC가 조용하지 않아 측정을 시작하지 않습니다.")
    cpu_before = cpu_percent(3.0)
    print(f"측정 직전 CPU 사용률: {cpu_before:.1f}%")

    print(f"warmup {args.warmup}회...", flush=True)
    for _ in range(args.warmup):
        predictor(x)

    print(f"측정 {args.runs}회...", flush=True)
    times = []
    for i in range(1, args.runs + 1):
        t0 = time.perf_counter()
        predictor(x)
        times.append((time.perf_counter() - t0) * 1000)
        if i % 50 == 0:
            print(f"  {i}/{args.runs}  지금까지 중앙값 {statistics.median(times):.1f} ms", flush=True)

    median = statistics.median(times)
    mean = statistics.mean(times)
    p90 = float(np.percentile(times, 90))
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    p95 = float(np.percentile(times, 95))

    # 메모리는 속도 측정이 끝난 뒤 따로 잰다 (감시 스레드가 속도에 끼어들지 않게)
    with PeakSampler() as s:
        for _ in range(10):
            predictor(x)
    peak_mem = s.peak_mb
    infer_mem = peak_mem - mem_before

    print(f"\n중앙값 : {median:.1f} ms   (주 지표)")
    print(f"평균   : {mean:.1f} ms")
    print(f"p90    : {p90:.1f} ms")
    print(f"p95    : {p95:.1f} ms")
    print(f"표준편차: {std:.1f} ms")
    print(f"FPS    : {1000/median:.2f}")
    print(f"모델   : {model_mb:.1f} MB")
    print(f"메모리 : 추론 중 {infer_mem:.0f} MB  (최대 {peak_mem:.0f} - 로드 전 {mem_before:.0f})")
    if args.runs != 200 or args.warmup != 20:
        print("\n[주의] 기본값(warmup 20, runs 200)이 아니라 참고용입니다.")

    CSV_PATH.parent.mkdir(exist_ok=True)
    if CSV_PATH.exists():
        with CSV_PATH.open(encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        if header == OLD_FIELDS:  # 열 4개 추가 전 형식 → 기존 줄은 그대로 두고 머리줄만 바꿈
            with CSV_PATH.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
                w.writeheader()
                w.writerows(rows)
            print("speed.csv에 p95_ms·메모리 열을 추가했습니다 (기존 줄은 빈칸).")
        elif header != FIELDS:  # 열이 바뀐 옛 파일은 따로 보관
            old = CSV_PATH.with_name(f"speed_old_{datetime.datetime.now():%m%d_%H%M}.csv")
            CSV_PATH.rename(old)
            print(f"옛 형식 CSV를 {old.name}(으)로 옮겼습니다.")
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
            "input_hw": input_hw,
            "threads": args.threads,
            "warmup": args.warmup,
            "runs": args.runs,
            "median_ms": f"{median:.2f}",
            "mean_ms": f"{mean:.2f}",
            "p90_ms": f"{p90:.2f}",
            "std_ms": f"{std:.2f}",
            "fps": f"{1000/median:.2f}",
            "model_mb": f"{model_mb:.1f}",
            "cpu": platform.processor(),
            "torch": torch.__version__,
            "power_mode": power_mode,
            "ac_power": ac_power,
            "arch": arch,
            "cpu_before_pct": f"{cpu_before:.1f}",
            "p95_ms": f"{p95:.2f}",
            "infer_mem_mb": f"{infer_mem:.0f}",
            "peak_mem_mb": f"{peak_mem:.0f}",
            "mem_before_mb": f"{mem_before:.0f}",
        })
    print(f"저장: {CSV_PATH}")


if __name__ == "__main__":
    main()
