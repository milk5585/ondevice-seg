"""
1주차 스모크 테스트
사전학습된 DeepLabV3로 이미지 한 장에서 사람 영역을 분리해 저장한다.

사용법:
    python run_deeplab.py test.jpg

결과:
    out_mask.png    - 사람 영역만 흰색인 이진 마스크
    out_overlay.png - 원본 위에 마스크를 겹친 이미지
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)

# Pascal VOC 21개 클래스 중 person의 인덱스
PERSON_CLASS = 15


def main(image_path: str) -> None:
    src = Path(image_path)
    if not src.exists():
        sys.exit(f"이미지를 찾을 수 없습니다: {src}")

    # 1) 모델 로드 (처음 실행 시 가중치를 내려받으므로 몇 분 걸릴 수 있다)
    print("모델 로드 중...")
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights)
    model.eval()

    # 2) 전처리는 가중치에 딸린 것을 그대로 쓴다.
    #    직접 만들면 학습 때와 달라져 정확도가 떨어진다.
    preprocess = weights.transforms()

    image = Image.open(src).convert("RGB")
    batch = preprocess(image).unsqueeze(0)  # (1, 3, H, W)

    # 3) 추론
    print("추론 중...")
    start = time.perf_counter()
    with torch.inference_mode():
        # torchvision 세그멘테이션 모델은 dict를 반환한다.
        # 나중에 ONNX 변환에서 문제가 되는 지점이니 기억해 둘 것.
        output = model(batch)["out"]
    elapsed = time.perf_counter() - start

    # 4) 픽셀마다 가장 점수가 높은 클래스를 고른다
    pred = output.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)
    mask = (pred == PERSON_CLASS).astype(np.uint8)

    # 5) 저장 — 모델 입력 크기에서 원본 크기로 되돌린다
    mask_img = Image.fromarray(mask * 255).resize(image.size, Image.NEAREST)
    mask_img.save("out_mask.png")

    rgba = image.convert("RGBA")
    tint = Image.new("RGBA", image.size, (255, 0, 0, 110))
    overlay = Image.composite(
        Image.alpha_composite(rgba, tint), rgba, mask_img
    )
    overlay.convert("RGB").save("out_overlay.png")

    # 6) 확인용 출력
    ratio = mask.mean() * 100
    print(f"\n추론 시간: {elapsed:.2f}초")
    print(f"사람으로 분류된 픽셀: {ratio:.1f}%")
    print("저장 완료: out_mask.png, out_overlay.png")

    if ratio < 0.5:
        print("\n[주의] 사람이 거의 검출되지 않았습니다.")
        print("       사람이 크게 나온 사진으로 다시 시도해 보세요.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("사용법: python run_deeplab.py <이미지경로>")
    main(sys.argv[1])
