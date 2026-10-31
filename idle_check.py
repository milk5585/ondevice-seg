"""
속도 측정 전 '조용한 PC' 검사. 백그라운드 작업이 속도 숫자를 왜곡하지 않도록 측정 직전에 확인한다.
공용 PC라 프로세스를 자동 종료하지 않는다. CPU를 쓰는 프로그램 목록을 보여줄 뿐이고 종료는 사람이 한다.

사용법:
    python idle_check.py            # 지금 상태 검사 (통과 0 / 실패 1)
    python idle_check.py --wait     # 통과할 때까지 기다림 (최대 10분)

측정 코드에서:
    from idle_check import wait_until_idle
    if not wait_until_idle():
        sys.exit("PC가 조용하지 않아 측정을 시작하지 않습니다")

통과 기준 (기본값): 전체 CPU 사용률 평균 5% 미만이 30초 연속, 사용 가능 메모리 2GB 이상, 다른 로그인 사용자 없음
"""
import argparse
import ctypes
import subprocess
import time
from ctypes import wintypes


class _FILETIME(ctypes.Structure):
    _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def _system_times():
    idle, kernel, user = _FILETIME(), _FILETIME(), _FILETIME()
    ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
    as_int = lambda ft: (ft.high << 32) | ft.low
    return as_int(idle), as_int(kernel), as_int(user)


def cpu_percent(interval=1.0):
    """전체 CPU 사용률(%). kernel 시간에는 idle 시간이 포함되어 있어 빼서 계산한다."""
    i1, k1, u1 = _system_times()
    time.sleep(interval)
    i2, k2, u2 = _system_times()
    total = (k2 - k1) + (u2 - u1)
    return 100.0 * (total - (i2 - i1)) / total if total else 0.0


def available_memory_gb():
    m = _MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 1024**3


def other_user_sessions():
    """현재 로그인된 사용자 수. quser가 없는 Windows Home이면 None."""
    try:
        out = subprocess.run(["quser"], capture_output=True, text=True, timeout=10).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    sessions = [line for line in out.splitlines()[1:] if line.strip()]
    return max(0, len(sessions) - 1)


def top_processes(n=8):
    """CPU를 많이 쓰는 프로세스 (이름, 전체 대비 %). 확인용이라 실패하면 빈 목록."""
    ps = ("(Get-Counter '\\Process(*)\\% Processor Time' -SampleInterval 2 -MaxSamples 1).CounterSamples | "
          "Where-Object { $_.InstanceName -notin @('_total','idle') } | Sort-Object CookedValue -Descending | "
          f"Select-Object -First {n} | ForEach-Object {{ '{{0}}|{{1:N1}}' -f $_.InstanceName, "
          "($_.CookedValue / [Environment]::ProcessorCount) }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=30).stdout
    except subprocess.TimeoutExpired:
        return []
    rows = [line.rsplit("|", 1) for line in out.splitlines() if "|" in line]
    return [(name, float(pct)) for name, pct in rows]


def check(max_cpu=5.0, hold_sec=30, min_mem_gb=2.0, verbose=True):
    """hold_sec 동안 CPU를 재서 평균이 max_cpu 미만이고 메모리·로그인 조건을 만족하면 True."""
    samples = [cpu_percent(1.0) for _ in range(hold_sec)]
    cpu = sum(samples) / len(samples)
    mem = available_memory_gb()
    others = other_user_sessions()
    problems = []
    if cpu >= max_cpu:
        problems.append(f"CPU 사용률 평균 {cpu:.1f}% (기준 {max_cpu}% 미만)")
    if mem < min_mem_gb:
        problems.append(f"사용 가능 메모리 {mem:.1f}GB (기준 {min_mem_gb}GB 이상)")
    if others:
        problems.append(f"다른 로그인 사용자 {others}명 (quser로 확인)")
    if verbose:
        others_text = "확인 불가" if others is None else f"{others}명"
        print(f"CPU 평균 {cpu:.1f}% (최대 {max(samples):.1f}%) / 메모리 여유 {mem:.1f}GB / 다른 사용자 {others_text}")
        if problems:
            print("[미통과] " + " / ".join(problems))
            print("CPU를 많이 쓰는 프로그램 (직접 종료하세요. Windows 보안·업데이트·시스템 프로세스는 끄지 마세요):")
            for name, pct in top_processes():
                print(f"  {name:<30} {pct:5.1f}%")
        else:
            print("[통과] 측정을 시작해도 되는 상태입니다.")
    return not problems


def wait_until_idle(max_cpu=5.0, hold_sec=30, min_mem_gb=2.0, timeout_sec=600):
    deadline = time.time() + timeout_sec
    while True:
        if check(max_cpu, hold_sec, min_mem_gb):
            return True
        if time.time() >= deadline:
            print(f"[실패] {timeout_sec // 60}분 안에 조용한 상태가 되지 않았습니다.")
            return False
        print("30초 뒤 다시 확인합니다...\n")
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true", help="통과할 때까지 기다린다")
    parser.add_argument("--max-cpu", type=float, default=5.0)
    parser.add_argument("--hold", type=int, default=30, help="CPU를 재는 시간(초)")
    parser.add_argument("--min-mem", type=float, default=2.0)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.wait:
        ok = wait_until_idle(args.max_cpu, args.hold, args.min_mem, args.timeout)
    else:
        ok = check(args.max_cpu, args.hold, args.min_mem)
    raise SystemExit(0 if ok else 1)
