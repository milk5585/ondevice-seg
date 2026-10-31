@echo off
rem B lab-PC batch: (0) checks  (1) speed x3 rounds  (2) memory x3  (3) fixed-input accuracy
rem Results: results\speed.csv, results\memory.csv, results\fixed_input.csv, results\pc_ver.txt
rem Speed rounds are round-robin (all models once, then again) so time drift is spread evenly.
setlocal
cd /d C:\ondevice-seg
call venv\Scripts\activate.bat
set "PY=python"
if defined DRYRUN set "PY=echo python"

echo ===== [0] checks =====
ver > results\pc_ver.txt
type results\pc_ver.txt
set MISSING=0
for %%F in (idle_check.py bench_speed.py mem_usage.py mem_onnx.py fixed_input_eval.py models\os8_fp32.onnx models\os16r6_fp32.onnx models\os8_int8_perchannel.onnx models\os8_int8_pertensor.onnx models\os16r6_int8_perchannel.onnx models\os16r6_int8_pertensor.onnx data\VOCdevkit\VOC2012\SegmentationClass\2007_000033.png) do (
  if not exist %%F (
    echo [MISSING] %%F
    set MISSING=1
  )
)
if "%MISSING%"=="1" (
  echo Some files are missing. Stopped.
  goto :end
)
echo All files found.

echo ===== [1] speed: 9 models x 3 rounds (about 40 min, do NOT use this PC) =====
for %%R in (1 2 3) do (
  echo ----- round %%R / 3 -----
  call :bench os8_520_fp32_r%%R models\os8_fp32.onnx 520 || goto :speedfail
  call :bench os8_384_fp32_r%%R models\os8_fp32.onnx 384 || goto :speedfail
  call :bench os8_320_fp32_r%%R models\os8_fp32.onnx 320 || goto :speedfail
  call :bench os8_384_int8_pc_r%%R models\os8_int8_perchannel.onnx 384 || goto :speedfail
  call :bench os8_384_int8_pt_r%%R models\os8_int8_pertensor.onnx 384 || goto :speedfail
  call :bench os16r6_520_fp32_r%%R models\os16r6_fp32.onnx 520 || goto :speedfail
  call :bench os16r6_384_fp32_r%%R models\os16r6_fp32.onnx 384 || goto :speedfail
  call :bench os16r6_384_int8_pc_r%%R models\os16r6_int8_perchannel.onnx 384 || goto :speedfail
  call :bench os16r6_384_int8_pt_r%%R models\os16r6_int8_pertensor.onnx 384 || goto :speedfail
)
echo Speed done.
goto :memory

:speedfail
echo [SPEED STOPPED] PC did not become idle. Run this file again later - finished rows stay in results\speed.csv.
echo Continuing with memory and accuracy (they do not need an idle PC).

:memory
echo ===== [2] memory: 7 models x 3 (about 5 min) =====
%PY% mem_onnx.py --all --repeat 3

echo ===== [3] fixed-input accuracy: 2 models x 3 inputs, VOC val 1449 (about 1 hour, PC can be used) =====
%PY% fixed_input_eval.py

:end
echo ===== ALL FINISHED =====
endlocal
pause
exit /b

:bench
rem %1 name  %2 onnx  %3 size. If idle check fails, wait for idle and retry once.
%PY% bench_speed.py --name %1 --onnx %2 --size %3 && exit /b 0
echo [retry] %1 - waiting until PC is idle (max 30 min)
%PY% idle_check.py --wait --timeout 1800 || exit /b 1
%PY% bench_speed.py --name %1 --onnx %2 --size %3
exit /b %errorlevel%
