"""
추론 중 메모리 사용량 측정 (B 작성, C의 bench_speed.py에 붙이는 용도)

왜 '프로세스 최대값(PeakWorkingSet)'을 안 쓰나
  모델을 불러오는 순간 임시로 메모리를 크게 쓰기 때문에, 프로세스 최대값은 '추론'이 아니라
  '로드'에서 결정된다. 실제로 재보니 OS8·OS16 둘 다 약 620MB로 똑같이 나와 차이가 안 보였다.
  그래서 추론하는 동안만 5ms마다 현재 메모리를 재서 최대값을 쓴다.

왜 '로드 전' 값을 빼나
  torch를 import 하는 것만으로 약 280MB를 쓴다. 이건 모델과 무관하므로 빼야 모델끼리 비교된다.

사용법 (bench_speed.py에서, 속도 측정이 끝난 뒤 별도로)
    from mem_usage import current_mb, PeakSampler
    before = current_mb()            # 모델 로드 '전'에 기록
    ... 모델 로드, warmup, 속도 측정 ...
    with PeakSampler() as s:          # 속도 측정과 따로 몇 번 더 돌린다 (샘플링이 속도에 영향 안 주게)
        for _ in range(10):
            predictor(x)
    infer_peak_mb = s.peak_mb - before
"""
import ctypes
import os
import sys
import threading
import time
from ctypes import wintypes


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def current_mb() -> float:
    """지금 이 프로세스가 쓰는 실제 메모리(MB). Windows=Working Set, Linux=RSS."""
    if sys.platform == "win32":
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMC), wintypes.DWORD]
        k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        c = _PMC()
        c.cb = ctypes.sizeof(_PMC)
        if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return float("nan")
        return c.WorkingSetSize / 2**20
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except OSError:
        return float("nan")


class PeakSampler:
    """with 블록 안에서 interval초마다 current_mb()를 재서 최대값을 peak_mb에 남긴다."""

    def __init__(self, interval: float = 0.005):
        self.interval = interval
        self.peak_mb = 0.0
        self._stop = threading.Event()

    def _run(self):
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, current_mb())
            time.sleep(self.interval)

    def __enter__(self):
        self.peak_mb = current_mb()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak_mb = max(self.peak_mb, current_mb())
        return False


if __name__ == "__main__":
    # 확인용: python mem_usage.py os8   (구조별 추론 중 메모리)
    import torch
    from structures import build

    name = sys.argv[1] if len(sys.argv) > 1 else "os8"
    before = current_mb()
    model = build(name)
    x = torch.randn(1, 3, 384, 524)
    with torch.inference_mode():
        model(x)
        with PeakSampler() as s:
            for _ in range(5):
                model(x)
    print(f"{name}: 로드 전 {before:.0f} MB / 추론 중 최대 {s.peak_mb:.0f} MB / 증가분 {s.peak_mb - before:.0f} MB")
