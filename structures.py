"""
B 담당: 출력 해상도(OS)와 ASPP rate를 바꾼 DeepLabV3-ResNet50 생성.
정확도 측정과 속도 측정 모두 이 함수로 만든 모델을 써야 같은 모델이 보장된다.

    from structures import build, CONFIGS
    model = build("os16_r6")    # 'os8'(사전학습 원본), 'os16_r12', 'os16_r6'

사전학습 가중치를 그대로 쓰고(파인튜닝 없음), stride/dilation 설정만 바꾼다.
반환 모델의 forward 출력은 torchvision과 같은 dict이며 결과는 ['out'] (1, 21, H, W).
"""
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

# 이름: (output stride, ASPP atrous rates)
CONFIGS = {
    "os8": (8, (12, 24, 36)),
    "os16_r12": (16, (12, 24, 36)),
    "os16_r6": (16, (6, 12, 18)),
}


def build(name):
    output_stride, rates = CONFIGS[name]
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.aux_classifier = None
    if output_stride == 16:
        backbone = IntermediateLayerGetter(
            resnet50(replace_stride_with_dilation=[False, False, True]),
            return_layers={"layer4": "out"},
        )
        backbone.load_state_dict(model.backbone.state_dict(), strict=True)
        model.backbone = backbone
    if tuple(rates) != (12, 24, 36):
        head = DeepLabHead(2048, 21, atrous_rates=rates)
        head.load_state_dict(model.classifier.state_dict(), strict=True)
        model.classifier = head
    return model.eval()
