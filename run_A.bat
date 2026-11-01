@echo off
chcp 65001 > nul
REM ============================================================
REM  A(양자화) 재현: FP32 ONNX 변환 -> INT8 PTQ -> 민감도 -> 혼합정밀도(top1~3)
REM  사전 조건: venv 활성화, data\VOCdevkit\VOC2012, lists\ 준비
REM  소요(학과 PC i9-12900 기준): 변환 수 분 / PTQ 4개 약 10분 /
REM        민감도 single 약 2시간, loo 약 70분 / 혼합정밀도 3개 약 30분
REM  제출 모델만 필요하면 1~2단계까지만 실행하면 됩니다.
REM ============================================================

echo [1/4] FP32 ONNX 변환 + PyTorch 일치 확인 (results\export_check.txt)
python export_structs.py || goto :err

echo [2/4] INT8 PTQ: os8/os16r6 x per-tensor/per-channel (results\ptq_log.csv)
python ptq_quantize.py || goto :err

echo [3/4] 레이어별 민감도 (val_person 앞 100장)
python sensitivity.py --method single || goto :err
python sensitivity.py --method loo || goto :err

echo [4/4] 혼합정밀도: 민감도(single) 상위 K개 Conv만 FP32 (실험용, 미채택)
for %%K in (1 2 3) do (
  python ptq_quantize.py --models models/os8_fp32.onnx --modes perchannel --exclude results/exclude_candidates_os8_single_pc_n100.txt --top %%K --tag top%%K || goto :err
)

echo.
echo 완료. 정확도는 eval_voc.py 로 1449장 전체 측정:
echo   python eval_voc.py --name os8_384_int8_pc --onnx models/os8_int8_perchannel.onnx --size 384
echo   python eval_voc.py --name os8_int8_pc_top1 --onnx models/os8_int8_perchannel_top1.onnx --size 384
goto :eof

:err
echo 오류로 중단했습니다. 위 메시지를 확인하세요.
exit /b 1
