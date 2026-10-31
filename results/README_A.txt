A(양자화) 최종 전달 — 2026-10-27, 학과 PC i9-12900 (C:\ondevice-seg)
보고서 초안(Claude Docs)의 2·5·6.5·6.6·7.1·8·9장 A 부분은 이 숫자로 채워 둠.

[최종 모델 제안] models/os8_int8_perchannel.onnx (38.5MB, SHA256 9b3778653c82…562b9)
  원본 OS8 520 FP32 대비: 크기 3.93배↓ / Latency 767.0→139.9ms (5.48배) / 피크 메모리 484→154MB
  정확도: person mIoU 94.51→94.33 (−0.18%p), Boundary IoU 59.08→56.89 (−2.19%p)
  양자화 자체 손실(같은 구조 FP32 384 대비): −0.01 / −0.09%p → 경계 손실은 대부분 해상도(384) 탓

1. 정확도 (eval_voc.py, VOC val 1449장) — accuracy.csv
   OS8 520 FP32 94.51 / 59.08 / mIoU21 77.84  ← B PyTorch 값과 일치 = ONNX 재현 확인
   OS8 384 FP32 94.34 / 56.98 / 77.75
   OS8 384 INT8 per-tensor 94.26 / 56.88 / 77.73,  per-channel 94.33 / 56.89 / 77.68
   OS16 384 FP32 92.38 / 44.06 / 74.22, INT8 pt 92.29 / 43.95, pc 92.34 / 44.01
   ⚠ sec_per_img 열은 동시 실행 영향 → 속도 지표로 쓰지 말 것

2. 속도·메모리 (B, bench_speed.py, ORT, 스레드 4, 3회 중앙값) — speed.csv, memory.csv
   10/15: 520 FP32 767.0 / 384 FP32 405.3 / INT8 pc 139.9 / OS16 FP32 128.4 / OS16 INT8 pc 46.6 ms
   10/16 혼합정밀도: INT8 pc 140.6 / top1 143.2 / top2 144.6 / top3 142.0 ms (같은 날 520 = 773.8)

3. 혼합정밀도 (single 상위 K개 Conv만 FP32) — ptq_log.csv
   K=1 94.35/57.03, K=2 94.34/56.98, K=3 94.31/56.92 (전부 ±0.15%p 노이즈), 속도 1~3% 손해 → 미채택

4. 민감도 (val_person 앞 100장, Conv 61개, FP32 픽셀 일치율 기준)
   single (한 레이어만 INT8): 입력 쪽 민감 — stem 7x7 99.18%, layer1 99.50~99.82%, layer4·ASPP 99.96~99.98%
   loo (전부 INT8 + 한 레이어만 FP32): 61개 모두 99.22~99.29% (전부 INT8 99.27%) → 어떤 레이어를 되돌려도 회복 없음
   해석: 개별로는 입력부가 민감하지만 전체 INT8 오차는 61개 레이어에 얇게 분산 → 혼합정밀도가 효과 없던 이유
   loo 상위 순위는 차이 0.01%p라 노이즈 (single과 stem 873만 겹침)

5. 버전: torch 2.14.0+cpu export(dynamo) opset 18, IR 10, 단일 파일 / onnxruntime.quantization 1.30.0
   QDQ, 활성 U8 / 가중치 S8, MinMax, calib = lists/calib_train_person300.txt 300장(짧은 변 384)
   onnx 1.23.1 / onnxruntime 1.30.0 / onnxscript 0.7.2 / Python 3.11.9

6. SHA256: sha256_models.txt (모델 9개). 드라이브엔 _prep.onnx 와 models/extra/ 제외
