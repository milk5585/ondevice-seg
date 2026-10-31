"""
VOC 정확도 평가 함수

코드에서 직접 부를 때:
    from common.evaluate import evaluate
    from common.infer import TorchPredictor
    result = evaluate(TorchPredictor(내_모델), size=384)
"""

import time

from .infer import postprocess, preprocess
from .metrics import BoundaryIoU, SegMetric
from .voc import load_image, load_label, read_ids


def evaluate(
    predictor,
    split: str = "val",
    size: int = 520,
    mode: str = "short",
    limit: int = 0,
    log_every: int = 100,
) -> dict:
    ids = read_ids(split)
    if limit:
        ids = ids[:limit]

    metric = SegMetric()
    biou = BoundaryIoU()
    start = time.perf_counter()
    for i, img_id in enumerate(ids, 1):
        image = load_image(img_id)
        gt = load_label(img_id)
        logits = predictor(preprocess(image, size, mode))
        pred = postprocess(logits, gt.shape)
        metric.update(gt, pred)
        biou.update(gt, pred)

        if log_every and i % log_every == 0:
            s = metric.summary()
            print(
                f"  {i}/{len(ids)}  person mIoU {s['miou_person']*100:.2f}  "
                f"21cls mIoU {s['miou21']*100:.2f}  "
                f"({(time.perf_counter()-start)/i:.2f}초/장)",
                flush=True,
            )

    result = metric.summary()
    result["biou_person"] = biou.summary()
    result["n_images"] = len(ids)
    # 참고용. 공식 속도는 기준 PC에서 따로 잰다.
    result["sec_per_img"] = (time.perf_counter() - start) / len(ids)
    return result
