# 04다망 — 트랙2 예선 (자유주제): DeepLabV3-ResNet50 인물/배경 분리 온디바이스 최적화

torchvision `deeplabv3_resnet50` (COCO_WITH_VOC_LABELS_V1)에 **입력 해상도 축소(짧은 변 520→384) + INT8 PTQ(per-channel)** 를 적용했다.
기준 PC(i9-12900, ONNX Runtime CPU 4스레드)에서 측정한 결과:

| 항목 | Baseline (OS8 520 FP32) | 최적화 (OS8 384 INT8) |
| --- | --- | --- |
| person mIoU / Boundary IoU (VOC 2012 val 1,449장) | 94.51 / 59.08 | 94.33 / 56.89 |
| Latency (ms, 3회 중앙값) | 767.0 | 139.9 (5.48배) |
| 모델 크기 (MB) | 151.4 | 38.5 |
| 추론 메모리 (MB) | 484 | 154 |

## 1. 환경 구성

```
git clone https://github.com/milk5585/ondevice-seg.git
cd ondevice-seg
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

- Python 3.11.9, Windows 11 기준 (onnx 1.23.1, onnxruntime 1.30.0, torch 2.14.0+cpu)
- VOC 2012 (http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)를 받아 `data/VOCdevkit/VOC2012`에 풀기.
  다른 위치면 환경변수 `VOC_ROOT`로 지정.

## 2. 모델 파일

GitHub 파일당 100MB 제한 때문에 모델은 Google Drive로 제공한다. 받아서 `models/`에 넣는다.

- 베이스라인: `os8_fp32.onnx` — Google Drive: (공유 링크는 제출 메일 참고)
- 최적화 모델: `os8_int8_perchannel.onnx` — Google Drive: (공유 링크는 제출 메일 참고)

ONNX opset 18, IR 10 (torch.onnx.export dynamo=True, 입력 H·W 가변). 양자화: onnxruntime.quantization 1.30.0, QDQ, 활성 U8 / 가중치 S8, MinMax, calibration `lists/calib_train_person300.txt` 300장.

받은 뒤 SHA256 확인 — Windows `certutil -hashfile 파일명 SHA256`, Linux `sha256sum 파일명`

```
33b52973e5569ddc001bb03a3e8d5102d5bd5e6973fd4441b9a3ac9c3aa8daf4  os8_fp32.onnx
9b3778653c82fa3a5cfadce94fe06473e3efa0d6c865c19d820a6359094562b9  os8_int8_perchannel.onnx
46ecc8614b4db3bc5de5d75c2ad695084cc5ad93b57bd53a32b464a4abe1251f  os8_int8_pertensor.onnx
86cc494cf57279e0a223ba6b11c4bac5a832f62ab0d4f449acb84a3899e1b452  os16r6_fp32.onnx
02fe2d0b572444c015caf26394b0ebc5d7fd211e5403ddb76e06db18678c687e  os16r6_int8_perchannel.onnx
9b611e69298f11e186f3714a1669775359895f0acc425f584e0de8f59e182bc0  os16r6_int8_pertensor.onnx
84c90817ed75bbae287c173d04e910df7b08bf693f2efc9c5c46b43fd8c985b4  os8_int8_perchannel_top1.onnx
b9721cb465d2ea021b4e19569727845c638fbad294ecce444dbcd48b6bcade79  os8_int8_perchannel_top2.onnx
2b581103970f9e92f296ed35f4b98f882011a4e5695183a1967861cb3eaee3c2  os8_int8_perchannel_top3.onnx
```

(위 2개가 제출 모델, 나머지는 보고서 비교 실험용)

## 3. 실행

```
# 베이스라인 (원본 OS8 520, FP32 ONNX)
python eval_voc.py --name os8_520_onnx --onnx models/os8_fp32.onnx --size 520
python bench_speed.py --name os8_520_fp32 --onnx models/os8_fp32.onnx --size 520
python mem_onnx.py --name os8_520_fp32 --onnx models/os8_fp32.onnx --size 520 --repeat 3

# 최적화 모델 (OS8 384 INT8 per-channel)
python eval_voc.py --name os8_384_int8_pc --onnx models/os8_int8_perchannel.onnx --size 384
python bench_speed.py --name os8_384_int8_pc --onnx models/os8_int8_perchannel.onnx --size 384
python mem_onnx.py --name os8_384_int8_pc --onnx models/os8_int8_perchannel.onnx --size 384 --repeat 3

# 속도: bench_speed.py를 모델마다 3번 실행한 뒤 묶기 / 파라미터 수·연산량
python summarize_speed.py
python count_model.py
```

## 4. 출력 (results/)

| 파일 | 열 | 보고서 항목 |
| --- | --- | --- |
| accuracy.csv | miou_person, biou_person | 정확도 지표 1, 2 |
| speed.csv | median_ms, fps, model_mb | Latency, Throughput, 모델 크기 |
| speed_summary.csv | latency_ms, min_ms~max_ms | 3회 중앙값의 중앙값, 범위 |
| memory.csv | infer_peak_mb | 피크 메모리 (로드 전 대비 증가분) |
| model_stats.csv | params_m, gflops | 파라미터 수, 연산량 (FLOPs = MACs×2) |

팀이 측정한 원본 결과도 같은 폴더에 있다 (ptq_compare.csv, ptq_log.csv, sensitivity_*.csv, mobilenet_compare.csv, fixed_input.csv, os_resolution.csv).
`accuracy.csv`의 `sec_per_img` 열은 동시 실행 영향을 받아 속도 지표로 쓰지 않는다.

## 5. 유의사항

- `bench_speed.py`는 Windows 전원 모드가 '최고 성능'이 아니거나 PC가 유휴 상태가 아니면 측정을 거부한다 (`--ignore-power`, `--skip-idle`로 무시 가능, 참고용).
- 속도·메모리 절댓값은 PC마다 다르므로, 같은 PC에서 베이스라인 대비 배속으로 비교하길 권한다.
- PyTorch로 실행(참고용)하면 최초 1회 torchvision 사전학습 가중치(약 161 MB)를 자동으로 받는다.
- VOC val 1,449장 정확도 평가는 학과 PC 기준 모델 1개당 약 20~40분 걸린다.
