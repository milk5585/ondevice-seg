#!/bin/bash
cd ~/workspace/check384
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hailo
for n in os8 os16_r6; do
  echo "=== $n"
  hailo parser onnx ${n}_384x384.onnx --hw-arch hailo10h --har-path ${n}_384.har -y > parse_${n}.log 2>&1
  echo "exit=$?"
  grep -E "Start nodes|End nodes|Translation completed|Saved HAR|rror|nsupported|Exception" parse_${n}.log
done
ls -la *.har
