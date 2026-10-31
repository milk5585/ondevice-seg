"""
B 1주차 check: OS16 변환이 제대로 되는지 사진 한 장으로 검증한다.

사용법:
    python check_os16.py test.jpg

통과 기준:
    1) OS16 백본에 사전학습 가중치가 strict=True로 로드된다
    2) 출력 마스크 크기 = 입력 크기
    3) OS16 마스크가 OS8 마스크와 크게 다르지 않다 (일치 IoU가 높다)
"""
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

from structures import build

PERSON_CLASS = 15
WEIGHTS = DeepLabV3_ResNet50_Weights.DEFAULT


def run(model, batch, warmup=3, runs=10):
    with torch.inference_mode():
        for _ in range(warmup):
            model(batch)
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            out = model(batch)["out"]
            times.append(time.perf_counter() - start)
    mask = (out.argmax(dim=1).squeeze(0) == PERSON_CLASS).numpy()
    return mask, float(np.median(times)), out.shape[-2:]


def agreement_iou(a, b):
    union = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / union if union else float("nan")


def main(image_path):
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    batch = WEIGHTS.transforms()(image).unsqueeze(0)
    print(f"원본 {image.size[0]}x{image.size[1]} -> 모델 입력 {batch.shape[-1]}x{batch.shape[-2]}")

    configs = [
        ("OS8  rates(12,24,36) [기본]", "os8"),
        ("OS16 rates(12,24,36)", "os16_r12"),
        ("OS16 rates(6,12,18)", "os16_r6"),
    ]
    results = {}
    for name, config in configs:
        mask, latency, out_hw = run(build(config), batch)
        size_ok = tuple(out_hw) == tuple(batch.shape[-2:])
        results[name] = (mask, latency, size_ok)

    base_mask, base_latency, _ = results[configs[0][0]]
    print(f"\n{'구성':<30}{'latency(중앙값)':>16}{'속도비':>8}{'person%':>9}{'OS8과 일치 IoU':>16}{'크기일치':>9}")
    for name, (mask, latency, size_ok) in results.items():
        print(f"{name:<30}{latency*1000:>13.0f} ms{base_latency/latency:>7.2f}x"
              f"{mask.mean()*100:>8.1f}%{agreement_iou(mask, base_mask):>16.3f}{str(size_ok):>9}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("사용법: python check_os16.py <이미지경로>")
    main(sys.argv[1])
