# A — ONNX 변환 · INT8 양자화 · 민감도 분석 · 혼합정밀도

담당: 황동규. 모든 수치는 학과 PC(i9-12900, Windows 11, ONNX Runtime 1.30.0)에서 측정. 보고서 5장·6.5·6.6의 근거.

## 파일

| 파일 | 하는 일 | 결과 |
| --- | --- | --- |
| `export_structs.py` | `structures.build("os8")`, `build("os16_r6")` → FP32 ONNX (opset 18, IR 10, 입력 H·W 가변, 단일 파일) + PyTorch 출력 일치 확인 | `models/os8_fp32.onnx`, `models/os16r6_fp32.onnx`, `results/export_check.txt` |
| `ptq_quantize.py` | INT8 정적 양자화 (QDQ, 활성 U8 / 가중치 S8, MinMax, per-tensor·per-channel). `--exclude --top K`로 혼합정밀도 | `models/*_int8_*.onnx`, `results/ptq_log.csv` |
| `sensitivity.py` | Conv 61개 레이어별 민감도. `single`(한 레이어만 INT8) / `loo`(전부 INT8 + 한 레이어만 FP32) | `results/sensitivity_os8_*_pc_n100.csv`, `results/exclude_candidates_*.txt` |
| `run_A.bat` | 위 과정을 순서대로 한 번에 실행 | — |

calibration: `lists/calib_train_person300.txt` (VOC train 중 person 300장, val과 겹침 0), 평가와 같은 전처리(짧은 변 384, 비율 유지). 민감도 분석은 calibration 범위를 `results/calib_cache_os8.json`에 저장해 재사용합니다.

## 실행

```powershell
venv\Scripts\activate
python export_structs.py                      # FP32 ONNX 2개
python ptq_quantize.py                        # INT8 4개 (제출 모델 = models/os8_int8_perchannel.onnx)
python sensitivity.py --method single         # 약 2시간
python sensitivity.py --method loo            # 약 70분
python ptq_quantize.py --models models/os8_fp32.onnx --modes perchannel --exclude results/exclude_candidates_os8_single_pc_n100.txt --top 1 --tag top1   # K=1~3
```

- `quant_pre_process`의 symbolic shape 추론은 가변 입력 때문에 "Incomplete symbolic shape inference"로 실패하고, 스크립트가 건너뛴 뒤 자동으로 다시 시도합니다(정상).
- 끝나면 작업 폴더에 GUID 이름의 `*.data`(약 158MB)와 `sym_shape_infer_temp.onnx` 임시 파일이 남을 수 있습니다. 지워도 됩니다.

## 결과 요약 (VOC val 1,449장)

| 모델 | 크기 (MB) | Latency (ms) | person mIoU | Boundary IoU |
| --- | --- | --- | --- | --- |
| OS8 520 FP32 (Baseline) | 151.4 | 767.0 | 94.51 | 59.08 |
| OS8 384 FP32 | 151.4 | 405.3 | 94.34 | 56.98 |
| **OS8 384 INT8 per-channel (제출)** | **38.5** | **139.9** | **94.33** | **56.89** |
| OS8 384 INT8 per-tensor | 38.2 | 139.9 | 94.26 | 56.88 |
| OS8 384 혼합정밀도 top1~3 (미채택) | 38.6 | 142.0~144.6* | 94.31~94.35 | 56.92~57.03 |

\* 10/16 별도 측정(같은 날 INT8 per-channel 140.6 ms). 혼합정밀도는 정확도 차이가 ±0.15%p(노이즈)이고 1~3% 느려 채택하지 않았습니다. loo 결과 어떤 한 레이어를 FP32로 되돌려도 FP32 일치율이 99.22~99.29%(전부 INT8 99.27%)로 회복되지 않아, INT8 오차가 여러 레이어에 분산돼 있음을 확인했습니다.
