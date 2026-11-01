"""
A 담당: PTQ (INT8 정적 양자화) — 확정 구조 os8 / os16_r6 의 FP32 ONNX 대상

    python ptq_quantize.py                                   # 두 구조 × per-tensor/per-channel = 4개
    python ptq_quantize.py --models models/os8_fp32.onnx --modes perchannel
    python ptq_quantize.py --models models/os8_fp32.onnx --modes perchannel \
        --exclude results/exclude_candidates_os8_single_pc_n100.txt --top 3 --tag top3   # 혼합정밀도: 민감도(single) 상위 3개 FP32 유지

결정 사항 (인수인계 문서 2·5장, B 구조 결정 문서 5.1)
- calibration: lists/calib_train_person300.txt (C 공용, VOC train 중 person 포함 300장, val 겹침 0)
- 전처리: common.infer.preprocess(size=384, mode="short") — eval_voc.py 평가와 '완전히 같은' 입력
- QDQ 포맷, 활성 U8 / 가중치 S8, MinMax
- per-tensor / per-channel 두 가지 → 비교 (per-tensor 손실이 크면 민감도 분석의 근거)
- 단일 파일 저장 (팀 스크립트가 .onnx 크기만 잼)

결과
- models/os8_int8_pertensor.onnx, models/os8_int8_perchannel.onnx, (os16r6 도 동일)
- results/ptq_log.csv : 설정·크기·소요시간 (보고서 재현 정보)
"""

import argparse
import csv
import datetime
import time
from pathlib import Path

import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from common.infer import preprocess
from common.voc import load_image, read_ids

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "results" / "ptq_log.csv"
FIELDS = [
    "date", "fp32_model", "int8_model", "mode", "calib_list", "n_calib", "size", "resize_mode",
    "format", "act_type", "weight_type", "calib_method", "n_excluded", "fp32_mb", "int8_mb",
    "ratio", "sec", "onnxruntime",
]


class VOCCalibReader(CalibrationDataReader):
    def __init__(self, ids, input_name, size, mode):
        self.ids, self.name, self.size, self.mode = ids, input_name, size, mode
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.ids):
            return None
        img_id = self.ids[self.idx]
        self.idx += 1
        if self.idx % 50 == 0:
            print(f"    calib {self.idx}/{len(self.ids)}", flush=True)
        return {self.name: preprocess(load_image(img_id), self.size, self.mode)}

    def rewind(self):
        self.idx = 0


def prep_model(fp32):
    """ORT 권장 전처리 (shape 추론 + 그래프 최적화). 실패하면 symbolic shape 추론만 빼고 재시도."""
    prep = fp32.with_name(fp32.stem + "_prep.onnx")
    try:
        quant_pre_process(str(fp32), str(prep))
    except Exception as e:
        print(f"  symbolic shape 추론 실패 → skip 하고 재시도: {str(e)[:200]}")
        quant_pre_process(str(fp32), str(prep), skip_symbolic_shape=True)
    return prep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["models/os8_fp32.onnx", "models/os16r6_fp32.onnx"])
    ap.add_argument("--modes", nargs="+", default=["pertensor", "perchannel"],
                    choices=["pertensor", "perchannel"])
    ap.add_argument("--calib", default="calib_train_person300", help="lists/ 안의 목록 이름")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--mode", default="short", choices=["short", "square"])
    ap.add_argument("--exclude", help="FP32로 남길 노드 이름 목록 txt (한 줄에 하나, 4주차 혼합정밀도)")
    ap.add_argument("--top", type=int, default=0, help="--exclude 목록의 앞 N개만 사용 (0 = 전부)")
    ap.add_argument("--tag", default="", help="출력 파일 이름 뒤에 붙일 표시 (예: mixed)")
    args = ap.parse_args()

    ids = read_ids(args.calib)
    exclude = []
    if args.exclude:
        exclude = [l.strip() for l in Path(args.exclude).read_text(encoding="utf-8").splitlines() if l.strip()]
        if args.top:
            exclude = exclude[: args.top]
    print(f"calibration: lists/{args.calib}.txt ({len(ids)}장), 짧은 변 {args.size} ({args.mode})")
    if exclude:
        print(f"FP32 유지 노드: {len(exclude)}개 ({args.exclude})")

    LOG.parent.mkdir(exist_ok=True)
    for model in args.models:
        fp32 = (ROOT / model) if not Path(model).is_absolute() else Path(model)
        assert fp32.exists(), f"{fp32} 없음 → 먼저 python export_structs.py"
        assert not Path(str(fp32) + ".data").exists(), "외부 .data 파일이 있음 → export_structs.py로 다시 export"
        print(f"\n[{fp32.name}] pre-process")
        prep = prep_model(fp32)
        input_name = ort.InferenceSession(str(prep), providers=["CPUExecutionProvider"]).get_inputs()[0].name

        for mode in args.modes:
            suffix = f"_int8_{mode}" + (f"_{args.tag}" if args.tag else "")
            out = fp32.with_name(fp32.stem.replace("_fp32", "") + suffix + ".onnx")
            print(f"  quantize ({mode}) → {out.name}")
            t0 = time.time()
            quantize_static(
                str(prep),
                str(out),
                VOCCalibReader(ids, input_name, args.size, args.mode),
                quant_format=QuantFormat.QDQ,           # CPU EP 권장 포맷
                per_channel=(mode == "perchannel"),
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                calibrate_method=CalibrationMethod.MinMax,
                nodes_to_exclude=exclude or None,
                # CalibMaxIntermediateOutputs 쓰지 말 것: ORT 1.30은 N장마다 모은 값을 범위 계산 없이 버림
                # (300장 % 20 = 0 → "No data is collected", 나누어떨어지지 않으면 마지막 몇 장으로만 범위 결정)
            )
            sec = time.time() - t0
            fp32_mb = fp32.stat().st_size / 1024**2
            int8_mb = out.stat().st_size / 1024**2
            print(f"    {fp32_mb:.1f}MB → {int8_mb:.1f}MB ({fp32_mb/int8_mb:.2f}배 감소), {sec:.0f}초")

            new_file = not LOG.exists()
            with LOG.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new_file:
                    w.writeheader()
                w.writerow({
                    "date": f"{datetime.datetime.now():%Y-%m-%d %H:%M}",
                    "fp32_model": fp32.name, "int8_model": out.name, "mode": mode,
                    "calib_list": f"{args.calib}.txt", "n_calib": len(ids),
                    "size": args.size, "resize_mode": args.mode,
                    "format": "QDQ", "act_type": "U8", "weight_type": "S8", "calib_method": "MinMax",
                    "n_excluded": len(exclude), "fp32_mb": f"{fp32_mb:.1f}", "int8_mb": f"{int8_mb:.1f}",
                    "ratio": f"{fp32_mb/int8_mb:.2f}", "sec": f"{sec:.0f}", "onnxruntime": ort.__version__,
                })

    print(f"\n저장: {LOG.relative_to(ROOT)}")
    print("다음: python eval_voc.py --name os8_384_int8_pc --onnx models/os8_int8_perchannel.onnx --size 384")


if __name__ == "__main__":
    main()
