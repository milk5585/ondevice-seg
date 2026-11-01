"""
A 담당: 레이어별 양자화 민감도 분석 → 혼합정밀도(INT8 + FP32)에서 FP32로 남길 레이어 후보 찾기

    python sensitivity.py                          # 주력 os8, 한 레이어만 INT8 (single), per-channel
    python sensitivity.py --method loo             # 전부 INT8 + 한 레이어만 FP32 (leave-one-out)
    python sensitivity.py --n 30 --limit 5         # 빠른 동작 확인 (30장, 5개 레이어만)

두 가지 방식
- single : 한 Conv만 INT8, 나머지 FP32 → 그 레이어 '혼자' 만드는 손실 (B 문서 5.3의 정의)
- loo    : 전부 INT8, 한 Conv만 FP32 → 그 레이어를 '살렸을 때' 회복되는 정확도
           (혼합정밀도에 바로 쓰는 값. single은 레이어 간 상호작용을 못 봄 → 둘 다 재면 보고서 근거가 탄탄)

설정은 ptq_quantize.py와 동일 (QDQ, U8/S8, MinMax, calib = lists/calib_train_person300.txt, 짧은 변 384).
calibration은 처음 한 번만 하고 results/calib_cache_<모델>.json 에 저장해 레이어마다 재사용.

평가: val 중 person이 있는 이미지(lists/val_person.txt) 앞 N장 (기본 100장) — '순위'를 매기는 용도.
      최종 혼합정밀도 모델의 정확도는 반드시 eval_voc.py로 1449장 전체 측정.

결과 (중간에 끊겨도 다시 실행하면 이어서 진행)
- results/sensitivity_<모델>_<method>_<pc|pt>_n<장수>.csv  : 레이어별 person mIoU / 경계 IoU / FP32 대비 변화 / FP32 예측과 픽셀 일치율
- results/exclude_candidates_<모델>_<method>_<pc|pt>_n<장수>.txt : 민감한 순서대로 노드 이름 → ptq_quantize.py --exclude 로 사용
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from common import infer
from common.metrics import BoundaryIoU, SegMetric
from common.voc import load_image, load_label, read_ids

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
FIELDS = [
    "idx", "node", "op", "weight_shape", "params", "miou_person", "biou_person",
    "d_miou", "d_biou", "agree_fp32", "sec",
]


class CalibReader(CalibrationDataReader):
    def __init__(self, ids, input_name, size):
        self.ids, self.name, self.size, self.i = ids, input_name, size, 0

    def get_next(self):
        if self.i >= len(self.ids):
            return None
        img_id = self.ids[self.i]
        self.i += 1
        if self.i % 50 == 0:
            print(f"    calib {self.i}/{len(self.ids)}", flush=True)
        return {self.name: infer.preprocess(load_image(img_id), self.size, "short")}

    def rewind(self):
        self.i = 0


def get_prep(fp32):
    """ptq_quantize.py가 만든 _prep.onnx 재사용 (없으면 만든다)."""
    prep = fp32.with_name(fp32.stem + "_prep.onnx")
    if not prep.exists():
        try:
            quant_pre_process(str(fp32), str(prep))
        except Exception as e:
            print(f"  symbolic shape 추론 실패 → skip 하고 재시도: {str(e)[:200]}")
            quant_pre_process(str(fp32), str(prep), skip_symbolic_shape=True)
    return prep


def conv_layers(prep):
    """양자화 대상 Conv 노드 목록 (그래프 순서 = 입력 쪽부터)."""
    m = onnx.load(str(prep))
    inits = {t.name: tuple(t.dims) for t in m.graph.initializer}
    layers = []
    for n in m.graph.node:
        if n.op_type == "Conv":
            shape = inits.get(n.input[1], ())
            layers.append({"node": n.name, "op": n.op_type, "weight_shape": "x".join(map(str, shape)),
                           "params": int(np.prod(shape)) if shape else 0})
    assert layers, "Conv 노드를 못 찾음"
    assert all(l["node"] for l in layers), "이름 없는 Conv 노드가 있음 → quant_pre_process 결과 확인"
    return layers


class Evaluator:
    """평가 이미지를 한 번만 읽어 메모리에 두고, 모델마다 같은 입력으로 채점."""

    def __init__(self, ids, size):
        self.data = []
        for img_id in ids:
            gt = load_label(img_id)
            self.data.append((infer.preprocess(load_image(img_id), size, "short"), gt))
        self.ref = None  # FP32 예측 (픽셀 일치율 기준)

    def run(self, model_path):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        metric, biou, preds = SegMetric(), BoundaryIoU(), []
        for x, gt in self.data:
            pred = infer.postprocess(sess.run(None, {name: x})[0], gt.shape)
            metric.update(gt, pred)
            biou.update(gt, pred)
            preds.append(pred)
        agree = 1.0
        if self.ref is not None:
            same = sum(int((p == r).sum()) for p, r in zip(preds, self.ref))
            agree = same / sum(r.size for r in self.ref)
        return {"miou_person": metric.summary()["miou_person"], "biou_person": biou.summary(),
                "agree": agree, "preds": preds}


def quantize(prep, out, per_channel, cache, reader=None, nodes_to_quantize=None, nodes_to_exclude=None):
    quantize_static(
        str(prep), str(out), reader,
        quant_format=QuantFormat.QDQ,
        per_channel=per_channel,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_quantize=nodes_to_quantize,
        nodes_to_exclude=nodes_to_exclude,
        # CalibMaxIntermediateOutputs 금지 (ptq_quantize.py 주석 참고)
        calibration_cache_path=str(cache),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/os8_fp32.onnx")
    ap.add_argument("--method", default="single", choices=["single", "loo"])
    ap.add_argument("--pertensor", action="store_true", help="기본은 per-channel")
    ap.add_argument("--calib", default="calib_train_person300")
    ap.add_argument("--eval-list", default="val_person")
    ap.add_argument("--n", type=int, default=100, help="평가 이미지 수 (eval-list 앞에서부터)")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N개 레이어만 (동작 확인용)")
    args = ap.parse_args()

    fp32 = ROOT / args.model
    stem = fp32.stem.replace("_fp32", "")
    # 평가 장수가 다르면 기준(FP32) 값이 달라지므로 파일을 분리 (이어하기가 다른 n 결과와 섞이지 않게)
    tag = f"{stem}_{args.method}_{'pt' if args.pertensor else 'pc'}_n{args.n}"
    per_channel = not args.pertensor
    RES.mkdir(exist_ok=True)
    out_csv = RES / f"sensitivity_{tag}.csv"
    cache = RES / f"calib_cache_{stem}.json"
    tmp = fp32.with_name(f"_sens_tmp_{tag}.onnx")

    prep = get_prep(fp32)
    layers = conv_layers(prep)
    if args.limit:
        layers = layers[: args.limit]
    n_conv = len(conv_layers(prep))
    print(f"[{tag}] Conv {n_conv}개 중 {len(layers)}개 분석, 평가 {args.eval_list} 앞 {args.n}장 @짧은 변 {args.size}")

    # 1) calibration 캐시 (처음 한 번만)
    if not cache.exists():
        print(f"calibration 1회 실행 → {cache.relative_to(ROOT)}")
        name = ort.InferenceSession(str(prep), providers=["CPUExecutionProvider"]).get_inputs()[0].name
        quantize(prep, tmp, per_channel, cache, reader=CalibReader(read_ids(args.calib), name, args.size))

    # 2) 기준점: FP32, 전부 INT8
    ev = Evaluator(read_ids(args.eval_list)[: args.n], args.size)
    base = ev.run(prep)
    ev.ref = base["preds"]
    quantize(prep, tmp, per_channel, cache)
    full = ev.run(tmp)
    print(f"FP32       : person mIoU {base['miou_person']*100:.2f} / 경계 {base['biou_person']*100:.2f}")
    print(f"전부 INT8  : person mIoU {full['miou_person']*100:.2f} / 경계 {full['biou_person']*100:.2f} "
          f"/ FP32 일치 {full['agree']*100:.2f}%")

    # 3) 레이어별 (이미 한 레이어는 건너뜀 → 끊겨도 이어서)
    done = set()
    if out_csv.exists():
        with out_csv.open(encoding="utf-8") as f:
            done = {r["node"] for r in csv.DictReader(f)}
    new_file = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
            for label, r in (("FP32", base), ("ALL_INT8", full)):
                w.writerow({"idx": -1, "node": label, "op": "", "weight_shape": "", "params": "",
                            "miou_person": f"{r['miou_person']*100:.3f}", "biou_person": f"{r['biou_person']*100:.3f}",
                            "d_miou": f"{(r['miou_person']-base['miou_person'])*100:.3f}",
                            "d_biou": f"{(r['biou_person']-base['biou_person'])*100:.3f}",
                            "agree_fp32": f"{r['agree']*100:.3f}", "sec": ""})
        for i, layer in enumerate(layers):
            if layer["node"] in done:
                continue
            t0 = time.time()
            if args.method == "single":
                quantize(prep, tmp, per_channel, cache, nodes_to_quantize=[layer["node"]])
            else:
                quantize(prep, tmp, per_channel, cache, nodes_to_exclude=[layer["node"]])
            r = ev.run(tmp)
            sec = time.time() - t0
            d_m = (r["miou_person"] - base["miou_person"]) * 100
            d_b = (r["biou_person"] - base["biou_person"]) * 100
            w.writerow({**layer, "idx": i, "miou_person": f"{r['miou_person']*100:.3f}",
                        "biou_person": f"{r['biou_person']*100:.3f}", "d_miou": f"{d_m:.3f}",
                        "d_biou": f"{d_b:.3f}", "agree_fp32": f"{r['agree']*100:.3f}", "sec": f"{sec:.0f}"})
            f.flush()
            left = (len(layers) - i - 1) * sec / 60
            print(f"  [{i+1}/{len(layers)}] {layer['node'][:40]:40s} mIoU {d_m:+.3f} 경계 {d_b:+.3f} "
                  f"일치 {r['agree']*100:.2f}% ({sec:.0f}초, 남은 시간 약 {left:.0f}분)", flush=True)

    if tmp.exists():
        os.remove(tmp)

    # 4) 순위 → 혼합정밀도 후보 목록
    with out_csv.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["idx"] != "-1"]
    # single: 혼자 INT8일 때 FP32 일치율이 가장 많이 떨어지는 레이어 / loo: FP32로 살렸을 때 가장 많이 회복되는 레이어
    # (100장 mIoU 차이는 0.0x 단위라 순위가 흔들림 → 더 촘촘한 '픽셀 일치율'을 1순위, 경계 IoU를 2순위로)
    sign = 1 if args.method == "single" else -1
    rows.sort(key=lambda r: (sign * float(r["agree_fp32"]), sign * float(r["d_biou"])))
    cand = RES / f"exclude_candidates_{tag}.txt"
    cand.write_text("\n".join(r["node"] for r in rows) + "\n", encoding="utf-8")
    print("\n민감한 레이어 상위 10개:")
    for r in rows[:10]:
        print(f"  {r['node'][:45]:45s} {r['weight_shape']:>16s}  일치 {float(r['agree_fp32']):.2f}%  "
              f"mIoU {float(r['d_miou']):+.3f}  경계 {float(r['d_biou']):+.3f}")
    print(f"\n저장: {out_csv.relative_to(ROOT)}, {cand.relative_to(ROOT)}")
    print("혼합정밀도 예 (상위 3개만 FP32):")
    print(f"  python ptq_quantize.py --models {args.model} --modes {'pertensor' if args.pertensor else 'perchannel'} "
          f"--exclude {cand.relative_to(ROOT).as_posix()} --top 3 --tag top3")


if __name__ == "__main__":
    main()
